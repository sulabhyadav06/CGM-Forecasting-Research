"""
Parameter-matched residual ablation experiment.

Uses the same data pipeline and training/evaluation code as
residual_ablation_experiment.py, but assigns approximately equal parameter
budgets to the four variants:

    GRU
    GRU-TCN
    GRU-Transformer
    Full (GRU + TCN + Transformer)

The configurations were selected against the current model implementation
for the 18-feature Phase-2 Ohio 2018 dataset and a ~100k parameter budget.
Always print the actual parameter count before interpreting results.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch

# Import the tested implementation/functions from the existing ablation runner.
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
    count_parameters,
    ResidualAblationModel,
)

# Approximately 100k parameters with the current 18-feature model:
#
# GRU              ~96,629
# GRU-TCN          ~99,972
# GRU-Transformer  ~99,861
# Full            ~100,900
#
# The small remaining difference is much smaller than the previous
# 59k / 152k / 163k / 256k comparison.
MATCHED_CONFIGS = {
    "GRU": {
        "d_model": 72,
        "gru_hidden": 72,
        "gru_layers": 2,
        "tcn_levels": 2,
        "heads": 2,
        "transformer_layers": 1,
    },
    "GRU-TCN": {
        "d_model": 40,
        "gru_hidden": 64,
        "gru_layers": 2,
        "tcn_levels": 2,
        "heads": 2,
        "transformer_layers": 1,
    },
    "GRU-Transformer": {
        "d_model": 48,
        "gru_hidden": 24,
        "gru_layers": 2,
        "tcn_levels": 2,
        "heads": 2,
        "transformer_layers": 1,
    },
    "Full": {
        "d_model": 32,
        "gru_hidden": 40,
        "gru_layers": 2,
        "tcn_levels": 2,
        "heads": 2,
        "transformer_layers": 2,
    },
}


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
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="output/phase2_parameter_matched")
    return p.parse_args()


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

    # Train-only imputation/scaling.
    imputer = fit_imputer(train_part, features)

    train_part = apply_imputer(train_part, imputer)
    val_part = apply_imputer(val_part, imputer)
    test_part = apply_imputer(test_raw, imputer)

    scaler = fit_scaler(train_part, features + targets)

    train_part = apply_scaler(train_part, scaler)
    val_part = apply_scaler(val_part, scaler)
    test_part = apply_scaler(test_part, scaler)

    print("\n" + "=" * 80)
    print("PARAMETER-MATCHED RESIDUAL ABLATION")
    print("=" * 80)
    print(f"Patient: {args.patient}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {args.horizons}")
    print(f"Features: {len(features)}")
    print(f"Temporal purge: {purge} rows")
    print(f"Device: {device}")
    print("\nActual parameter counts for this input dimension:")

    # Verify the budget before training.
    for variant in VARIANTS:
        cfg = MATCHED_CONFIGS[variant]
        probe = ResidualAblationModel(
            input_dim=len(features),
            horizons=args.horizons,
            variant=variant,
            dropout=args.dropout,
            max_len=lookback_steps,
            **cfg,
        )
        params = count_parameters(probe)
        print(f"  {variant:18s}: {params:,}")

    all_results = []
    all_diag = []

    for j, variant in enumerate(VARIANTS):
        cfg = MATCHED_CONFIGS[variant]

        # Keep deterministic but distinct initialization between variants.
        set_seed(args.seed + j)

        class VariantArgs:
            pass

        va = VariantArgs()
        va.lookback = args.lookback
        va.horizons = args.horizons
        va.dropout = args.dropout
        va.batch_size = args.batch_size
        va.epochs = args.epochs
        va.patience = args.patience
        va.val_fraction = args.val_fraction
        va.lr = args.lr
        va.weight_decay = args.weight_decay
        va.grad_clip = args.grad_clip
        va.patient = args.patient
        va.d_model = cfg["d_model"]
        va.gru_hidden = cfg["gru_hidden"]
        va.gru_layers = cfg["gru_layers"]
        va.tcn_levels = cfg["tcn_levels"]
        va.heads = cfg["heads"]
        va.transformer_layers = cfg["transformer_layers"]

        r, d, history, model = run_variant(
            variant,
            train_part,
            val_part,
            test_part,
            features,
            targets,
            va,
            scaler,
            device,
        )

        if "variant" not in r.columns:
         r.insert(0, "variant", variant)

        if "patient" not in r.columns:
         r.insert(1, "patient", args.patient)
         r.insert(2, "lookback_min", args.lookback)
         r.insert(3, "parameter_budget_target", 100000)

        if "variant" not in d.columns:
         d.insert(0, "variant", variant)

        if "patient" not in d.columns:
         d.insert(1, "patient", args.patient)

        all_results.append(r)
        all_diag.append(d)

        prefix = os.path.join(
            args.output_dir,
            f"patient_{args.patient}_{variant.lower().replace('-', '_')}",
        )

        r.to_csv(prefix + "_results.csv", index=False)
        d.to_csv(prefix + "_diagnostics.csv", index=False)
        history.to_csv(prefix + "_training.csv", index=False)
        torch.save(model.state_dict(), prefix + "_model.pt")

    results = pd.concat(all_results, ignore_index=True)
    diagnostics = pd.concat(all_diag, ignore_index=True)

    results.to_csv(
        os.path.join(
            args.output_dir,
            f"patient_{args.patient}_parameter_matched_results.csv",
        ),
        index=False,
    )
    diagnostics.to_csv(
        os.path.join(
            args.output_dir,
            f"patient_{args.patient}_parameter_matched_diagnostics.csv",
        ),
        index=False,
    )

    config = {
        "args": vars(args),
        "features": features,
        "purge_rows": purge,
        "matched_configs": MATCHED_CONFIGS,
        "parameter_budget_target": 100000,
        "imputer": imputer,
        "scaler": scaler,
    }

    with open(
        os.path.join(
            args.output_dir,
            f"patient_{args.patient}_parameter_matched_config.json",
        ),
        "w",
    ) as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("=== PARAMETER-MATCHED COMBINED RESULTS ===")
    print("=" * 80)
    print(results.to_string(index=False))

    print("\n=== DIAGNOSTICS ===")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
