import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.glunet.glunet_data import (
    load_split,
    build_sequences,
    FEATURES,
    LOOKBACK,
    N_CLASSES,
)
from experiments.glunet.glunet_model import GluNet


ROOT = Path("experiments/glunet/runs")
OUT = Path("experiments/glunet/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
PATIENT = 540
HORIZONS = (30, 60)

RUNS = {
    "baseline": ROOT / "pilot_patient540",
    "weighted": ROOT / "pilot_patient540_weighted",
}


def make_model(checkpoint):
    config = checkpoint["model_config"]
    model = GluNet(
        input_features=config["input_features"],
        channels=config["channels"],
        n_classes=config["n_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def entropy_rows(probabilities):
    # Natural-log entropy, in nats.
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def main():
    summaries = []

    test_df = load_split(PATIENT, "testing")

    for run_name, run_dir in RUNS.items():
        for horizon in HORIZONS:
            checkpoint_path = (
                run_dir / f"patient{PATIENT}_{horizon}min_best.pt"
            )
            prediction_path = (
                run_dir
                / f"patient{PATIENT}_{horizon}min_test_predictions.csv"
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)

            print(f"\nAuditing {run_name}, {horizon} minutes")

            # Explicit weights_only=False is needed for checkpoint
            # dictionaries containing non-tensor metadata on some versions.
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=DEVICE,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=DEVICE,
                )

            model = make_model(checkpoint)

            X, y, metadata = build_sequences(
                test_df,
                horizon_minutes=horizon,
            )

            if len(y) == 0:
                raise RuntimeError(
                    f"No test sequences for horizon {horizon}"
                )

            loader = DataLoader(
                TensorDataset(torch.from_numpy(X)),
                batch_size=128,
                shuffle=False,
                num_workers=0,
            )

            logits_list = []

            with torch.no_grad():
                for (xb,) in loader:
                    logits_list.append(model(xb.to(DEVICE)).cpu().numpy())

            logits = np.concatenate(logits_list, axis=0)

            # Numerically stable softmax.
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            probabilities = exp_logits / exp_logits.sum(
                axis=1, keepdims=True
            )

            if len(probabilities) != len(y):
                raise RuntimeError("Probability/label length mismatch")

            class_values = (
                np.arange(N_CLASSES, dtype=np.float64) + 1
            ) / 10.0 - 12.8

            argmax_classes = probabilities.argmax(axis=1)
            expected_delta = probabilities @ class_values
            argmax_delta = class_values[argmax_classes]
            expected_glucose = (
                metadata["current_glucose"].to_numpy(dtype=float)
                + expected_delta
            )

            true_classes = np.asarray(y, dtype=int)
            true_class_prob = probabilities[
                np.arange(len(y)), true_classes
            ]
            entropy = entropy_rows(probabilities)

            # Intermediate classes are indices 1 through 254.
            endpoint_mass = probabilities[:, 0] + probabilities[:, -1]
            intermediate_mass = probabilities[:, 1:-1].sum(axis=1)

            summary = {
                "run": run_name,
                "patient": PATIENT,
                "horizon_minutes": horizon,
                "n_test_sequences": len(y),
                "argmax_accuracy": float(
                    np.mean(argmax_classes == true_classes)
                ),
                "unique_argmax_classes": int(
                    np.unique(argmax_classes).size
                ),
                "mean_probability_class_0": float(
                    probabilities[:, 0].mean()
                ),
                "mean_probability_class_255": float(
                    probabilities[:, -1].mean()
                ),
                "mean_endpoint_probability_mass": float(
                    endpoint_mass.mean()
                ),
                "mean_intermediate_probability_mass": float(
                    intermediate_mass.mean()
                ),
                "mean_true_class_probability": float(
                    true_class_prob.mean()
                ),
                "median_true_class_probability": float(
                    np.median(true_class_prob)
                ),
                "mean_entropy_nats": float(entropy.mean()),
                "mean_max_probability": float(
                    probabilities.max(axis=1).mean()
                ),
                "mean_abs_expected_vs_argmax_delta": float(
                    np.abs(expected_delta - argmax_delta).mean()
                ),
                "median_abs_expected_vs_argmax_delta": float(
                    np.median(np.abs(expected_delta - argmax_delta))
                ),
                "expected_delta_min": float(expected_delta.min()),
                "expected_delta_median": float(np.median(expected_delta)),
                "expected_delta_max": float(expected_delta.max()),
                "checkpoint": str(checkpoint_path),
            }
            summaries.append(summary)

            prediction_audit = metadata.copy().reset_index(drop=True)
            prediction_audit["true_class"] = true_classes
            prediction_audit["argmax_class_recomputed"] = argmax_classes
            prediction_audit["true_class_probability"] = true_class_prob
            prediction_audit["class_0_probability"] = probabilities[:, 0]
            prediction_audit["class_255_probability"] = probabilities[:, -1]
            prediction_audit["endpoint_probability_mass"] = endpoint_mass
            prediction_audit["intermediate_probability_mass"] = intermediate_mass
            prediction_audit["entropy_nats"] = entropy
            prediction_audit["expected_delta_mg_dL"] = expected_delta
            prediction_audit["argmax_delta_mg_dL"] = argmax_delta
            prediction_audit["expected_glucose_mg_dL"] = expected_glucose

            # Save every class probability so we can inspect any example later.
            probability_columns = {
                f"prob_class_{i}": probabilities[:, i]
                for i in range(N_CLASSES)
            }
            probability_df = pd.DataFrame(probability_columns)
            prediction_audit = pd.concat(
                [prediction_audit, probability_df], axis=1
            )

            out_path = OUT / (
                f"{run_name}_patient{PATIENT}_{horizon}min_probabilities.csv"
            )
            prediction_audit.to_csv(out_path, index=False)

            # Cross-check reconstructed predictions against saved predictions.
            saved = pd.read_csv(prediction_path)
            if len(saved) != len(prediction_audit):
                print(
                    "WARNING: saved prediction count differs:",
                    len(saved),
                    len(prediction_audit),
                )
            else:
                delta_col = "predicted_delta_mg_dL"
                if delta_col in saved:
                    difference = np.abs(
                        saved[delta_col].to_numpy(dtype=float)
                        - expected_delta
                    )
                    print(
                        "Max absolute decoder difference:",
                        f"{difference.max():.10f}",
                    )

            print(json.dumps(summary, indent=2))
            print(f"Detailed probabilities saved: {out_path}")

    summary_path = OUT / "saved_probability_audit_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("PROBABILITY AUDIT COMPLETE")
    print(f"Summary: {summary_path}")
    print("=" * 72)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
