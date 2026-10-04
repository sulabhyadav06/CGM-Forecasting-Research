"""Run the corrected Phase-2 hybrid experiments across authorized OhioT1DM CSVs.

Expected CSV layout is either:
  data/train/<patient>.csv and data/test/<patient>.csv
or:
  data/<patient>_training_multimodal.csv and data/<patient>_testing_multimodal.csv

The OhioT1DM files are not distributed with this repository. Run this script only
in an environment where you have authorized access to the dataset.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PATIENTS = (
    "540", "544", "552", "567", "584", "596",
    "559", "563", "570", "575", "588", "591",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="output/phase2_hybrid_corrected")
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Run the main hybrid model only; skip the multi-architecture ablation grid.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "gru", "tcn", "transformer", "tcn_gru", "gru_transformer",
            "tcn_transformer", "tcn_gru_transformer",
            "adaptive_tcn_gru_transformer",
        ],
    )
    return parser.parse_args()


def run(command):
    print("\n>>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent

    for patient in args.patients:
        print(f"\n{'=' * 72}\nPatient {patient}\n{'=' * 72}", flush=True)
        hybrid_cmd = [
            sys.executable, str(script_dir / "hybrid_experiments.py"),
            "--data-root", args.data_root,
            "--patient", patient,
            "--lookback", str(args.lookback),
            "--horizons", *map(str, args.horizons),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--output-dir", str(root / "hybrid" / patient),
        ]
        run(hybrid_cmd)

        if not args.skip_ablation:
            ablation_cmd = [
                sys.executable, str(script_dir / "hybrid_ablation_experiments.py"),
                "--data-root", args.data_root,
                "--patient", patient,
                "--lookback", str(args.lookback),
                "--horizons", *map(str, args.horizons),
                "--architectures", *args.architectures,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
                "--output-dir", str(root / "ablations" / patient),
            ]
            run(ablation_cmd)

    print(
        "\nCompleted all requested patient runs. Inspect each output directory, "
        "then aggregate and independently review the saved results before reporting them.",
        flush=True,
    )


if __name__ == "__main__":
    main()
