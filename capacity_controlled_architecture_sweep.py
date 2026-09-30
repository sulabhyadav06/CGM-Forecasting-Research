"""
Capacity-controlled architecture sweep for Phase-2 Ohio 2018.

Purpose
-------
Compare GRU, GRU-TCN, GRU-Transformer, and Full (GRU+TCN+Transformer)
at approximately the SAME total parameter budget.

Budgets:
    ~100K
    ~160K
    ~256K

This is a development experiment on Patient 559. It is intended to answer:
"Does the architecture provide an advantage beyond raw parameter count?"

IMPORTANT:
- Uses the existing residual_ablation_experiment implementation.
- Uses identical data preprocessing/splits/training protocol.
- Reports actual parameter counts before training.
- Does NOT select a final model automatically.
- Patient 559 remains a development patient.

Because exact parameter counts depend on the current implementation and
feature dimension, this script searches a discrete configuration grid and
selects the configuration closest to each target budget for each variant.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import pandas as pd
import torch

from residual_ablation_experiment import (
    VARIANTS,
    set_seed,
    load_data,
    get_features,
    add_targets,
    fit_imputer,
    apply_imputer,
    fit_scaler,
    apply_scaler,
    make_split,
    run_variant,
    ResidualAblationModel,
    count_parameters,
)


TARGET_BUDGETS = {
    "100K": 100_000,
    "160K": 160_000,
    "256K": 256_000,
}

# Search grid. We keep the architectural choices sensible and use the same
# candidate dimensions across variants. The selection is by actual parameter
# count, not by an assumed formula.
D_MODELS = [24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112]
GRU_HIDDENS = [24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128]
GRU_LAYERS = [1, 2, 3]
TCN_LEVELS = [2, 3, 4]
HEADS = [2, 4, 8]
TRANSFORMER_LAYERS = [1, 2, 3]

# For computational sanity, only use the following candidates when matching
# the Transformer branch. d_model must be divisible by nhead.
def valid_heads(d_model):
    return [h for h in HEADS if d_model % h == 0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--lookback", type=int, default=120)
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[15, 30, 60, 90, 120],
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="output/phase2_capacity_sweep")
    return p.parse_args()


def model_params(
    variant,
    input_dim,
    horizons,
    lookback_steps,
    cfg,
):
    model = ResidualAblationModel(
        input_dim=input_dim,
        horizons=horizons,
        variant=variant,
        dropout=0.10,
        max_len=lookback_steps,
        **cfg,
    )
    return count_parameters(model)


def distance(actual, target):
    return abs(actual - target)


def choose_configs(input_dim, horizons, lookback_steps):
    """
    Find the closest configuration for every variant and budget.

    To avoid selecting pathological configurations, candidate dimensions are
    constrained to reasonable values. The final decision is based on actual
    trainable parameter count.
    """
    selected = {}

    for budget_name, target in TARGET_BUDGETS.items():
        selected[budget_name] = {}

        for variant in VARIANTS:
            best = None

            for d_model in D_MODELS:
                for gru_hidden in GRU_HIDDENS:
                    for gru_layers in GRU_LAYERS:
                        for tcn_levels in TCN_LEVELS:
                            for heads in valid_heads(d_model):
                                for transformer_layers in TRANSFORMER_LAYERS:

                                    # Avoid huge redundant models in the search.
                                    if gru_layers == 1 and gru_hidden > 96:
                                        continue
                                    if gru_layers == 3 and gru_hidden < 40:
                                        continue

                                    cfg = {
                                        "d_model": d_model,
                                        "gru_hidden": gru_hidden,
                                        "gru_layers": gru_layers,
                                        "tcn_levels": tcn_levels,
                                        "heads": heads,
                                        "transformer_layers": transformer_layers,
                                    }

                                    p = model_params(
                                        variant,
                                        input_dim,
                                        horizons,
                                        lookback_steps,
                                        cfg,
                                    )

                                    # For a pure GRU, branch-specific settings do
                                    # not matter, so keep the canonical small
                                    # settings in the stored configuration.
                                    if variant == "GRU":
                                        cfg["tcn_levels"] = 2
                                        cfg["heads"] = 2
                                        cfg["transformer_layers"] = 1

                                    score = distance(p, target)

                                    if best is None or score < best["distance"]:
                                        best = {
                                            "parameters": p,
                                            "distance": score,
                                            "config": cfg,
                                        }

            selected[budget_name][variant] = best

    return selected


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw, test_raw = load_data(args.data_root, args.patient)
    features = get_features(train_raw)

    train_raw = add_targets(train_raw, args.horizons)
    test_raw = add_targets(test_raw, args.horizons)

    targets = [f"target_{h}" for h in args.horizons]

    train_raw = train_raw.dropna(subset=targets).reset_index(drop=True)
    test_raw = test_raw.dropna(subset=targets).reset_index(drop=True)

    lookback_steps = args.lookback // 5
    max_horizon_steps = max(args.horizons) // 5

    train_part, val_part, purge = make_split(
        train_raw,
        lookback_steps,
        max_horizon_steps,
        args.val_fraction,
    )

    imputer = fit_imputer(train_part, features)
    train_part = apply_imputer(train_part, imputer)
    val_part = apply_imputer(val_part, imputer)
    test_part = apply_imputer(test_raw, imputer)

    scaler = fit_scaler(train_part, features + targets)
    train_part = apply_scaler(train_part, scaler)
    val_part = apply_scaler(val_part, scaler)
    test_part = apply_scaler(test_part, scaler)

    print("\n" + "=" * 90)
    print("CAPACITY-CONTROLLED ARCHITECTURE SWEEP")
    print("=" * 90)
    print(f"Patient: {args.patient}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {args.horizons}")
    print(f"Features: {len(features)}")
    print(f"Purged rows: {purge}")
    print(f"Device: {device}")

    print("\nSearching actual parameter counts...")

    selected = choose_configs(
        len(features),
        args.horizons,
        lookback_steps,
    )

    config_rows = []
    for budget_name, variants in selected.items():
        for variant, info in variants.items():
            row = {
                "budget": budget_name,
                "target_parameters": TARGET_BUDGETS[budget_name],
                "variant": variant,
                "actual_parameters": info["parameters"],
                "absolute_difference": info["distance"],
                **info["config"],
            }
            config_rows.append(row)

    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(
        os.path.join(args.output_dir, "selected_capacity_configs.csv"),
        index=False,
    )

    print("\n=== SELECTED CONFIGURATIONS ===")
    print(config_df.to_string(index=False))

    all_results = []
    all_diagnostics = []

    # Run each budget independently. Same seed is used within each budget.
    for budget_name in TARGET_BUDGETS:
        print("\n" + "=" * 90)
        print(f"RUNNING BUDGET: {budget_name}")
        print("=" * 90)

        for variant_index, variant in enumerate(VARIANTS):
            info = selected[budget_name][variant]
            cfg = info["config"]

            # Same seed for variants within a budget.
            set_seed(args.seed)

            class RunArgs:
                pass

            ra = RunArgs()
            ra.lookback = args.lookback
            ra.horizons = args.horizons
            ra.d_model = cfg["d_model"]
            ra.gru_hidden = cfg["gru_hidden"]
            ra.gru_layers = cfg["gru_layers"]
            ra.tcn_levels = cfg["tcn_levels"]
            ra.heads = cfg["heads"]
            ra.transformer_layers = cfg["transformer_layers"]
            ra.dropout = 0.10
            ra.batch_size = args.batch_size
            ra.epochs = args.epochs
            ra.patience = args.patience
            ra.val_fraction = args.val_fraction
            ra.lr = args.lr
            ra.weight_decay = args.weight_decay
            ra.grad_clip = args.grad_clip
            ra.patient = args.patient

            r, d, history, model = run_variant(
                variant,
                train_part,
                val_part,
                test_part,
                features,
                targets,
                ra,
                scaler,
                device,
            )

            if "variant" not in r.columns:
                r.insert(0, "variant", variant)
            if "patient" not in r.columns:
                r.insert(1, "patient", args.patient)

            r["budget"] = budget_name
            r["target_parameters"] = TARGET_BUDGETS[budget_name]
            r["actual_parameters"] = info["parameters"]
            r["parameter_difference"] = info["distance"]

            if "variant" not in d.columns:
                d.insert(0, "variant", variant)
            if "patient" not in d.columns:
                d.insert(1, "patient", args.patient)

            d["budget"] = budget_name
            d["target_parameters"] = TARGET_BUDGETS[budget_name]
            d["actual_parameters"] = info["parameters"]

            all_results.append(r)
            all_diagnostics.append(d)

            safe_variant = variant.lower().replace("-", "_")
            prefix = os.path.join(
                args.output_dir,
                f"{budget_name.lower()}_{safe_variant}",
            )
            r.to_csv(prefix + "_results.csv", index=False)
            d.to_csv(prefix + "_diagnostics.csv", index=False)
            history.to_csv(prefix + "_training.csv", index=False)
            torch.save(model.state_dict(), prefix + "_model.pt")

    results = pd.concat(all_results, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)

    results.to_csv(
        os.path.join(args.output_dir, "capacity_sweep_results.csv"),
        index=False,
    )
    diagnostics.to_csv(
        os.path.join(args.output_dir, "capacity_sweep_diagnostics.csv"),
        index=False,
    )

    # Compact comparison table.
    compact = (
        results[
            [
                "budget",
                "variant",
                "actual_parameters",
                "horizon",
                "mae_mgdl",
                "rmse_mgdl",
            ]
        ]
        .sort_values(["budget", "horizon", "mae_mgdl"])
        .reset_index(drop=True)
    )
    compact.to_csv(
        os.path.join(args.output_dir, "capacity_sweep_compact.csv"),
        index=False,
    )

    with open(
        os.path.join(args.output_dir, "capacity_sweep_config.json"),
        "w",
    ) as f:
        json.dump(
            {
                "args": vars(args),
                "features": features,
                "purge_rows": purge,
                "target_budgets": TARGET_BUDGETS,
                "selected_configs": selected,
            },
            f,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 90)
    print("=== CAPACITY SWEEP RESULTS ===")
    print("=" * 90)
    print(compact.to_string(index=False))

    print("\nFiles written to:", args.output_dir)


if __name__ == "__main__":
    main()
