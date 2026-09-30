#!/usr/bin/env python3
"""
Multi-patient Transformer lookback study.

Purpose:
    Evaluate lookback windows of 60, 120, 180, and 240 minutes
    for the parameter-matched Transformer on Ohio 2018.

Fixed:
    horizon = 30 minutes
    d_model = 56
    heads = 2
    layers = 1
    pooling = attention

The script is designed to use the existing transformer_experiments.py
runner, preserving its preprocessing, temporal validation, scaling,
metrics, and output conventions.

It launches one lookback at a time and writes each run to a separate
output directory so results cannot overwrite one another.

Run from the repository root:

python transformer_lookback_study.py
"""

from pathlib import Path
import argparse
import subprocess
import sys


def run_command(command, cwd):
    print("\n" + "=" * 78)
    print("Running:")
    print(" ".join(command))
    print("=" * 78 + "\n")

    result = subprocess.run(
        command,
        cwd=str(cwd),
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nExperiment failed with exit code "
            f"{result.returncode}."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
        help="CGM Forecasting Research repository root.",
    )

    parser.add_argument(
        "--runner",
        default="transformer_experiments.py",
        help="Transformer experiment runner.",
    )

    parser.add_argument(
        "--data-root",
        default="data/phase2/ohio2018",
        help="Ohio 2018 Phase-2 data root.",
    )

    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=[60, 120, 180, 240],
        help="Lookback windows in minutes.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon in minutes.",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=56,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--pooling",
        default="attention",
        choices=["last", "mean", "attention"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-root",
        default="output/phase2_transformer_lookback_study",
    )

    args = parser.parse_args()

    repo = (
        Path(args.repo_root)
        .expanduser()
        .resolve()
    )

    runner = (
        repo / args.runner
    )

    if not runner.exists():
        raise FileNotFoundError(
            f"Transformer runner not found:\n{runner}"
        )

    data_root = (
        repo / args.data_root
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found:\n{data_root}"
        )

    output_root = (
        repo / args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 78)
    print("MULTI-PATIENT TRANSFORMER LOOKBACK STUDY")
    print("=" * 78)
    print(f"Data root:       {data_root}")
    print(f"Horizons:        [{args.horizon}]")
    print(f"Lookbacks:       {args.lookbacks}")
    print(f"d_model:         {args.d_model}")
    print(f"Heads:           {args.heads}")
    print(f"Layers:          {args.layers}")
    print(f"Pooling:         {args.pooling}")
    print(f"Epochs:          {args.epochs}")
    print(f"Output root:     {output_root}")
    print("=" * 78)

    for lookback in args.lookbacks:

        output_dir = (
            output_root
            / f"lookback_{lookback}min"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            sys.executable,
            str(runner),
            "--data-root",
            str(data_root),
            "--lookback",
            str(lookback),
            "--horizon",
            str(args.horizon),
            "--d-model",
            str(args.d_model),
            "--heads",
            str(args.heads),
            "--layers",
            str(args.layers),
            "--pooling",
            args.pooling,
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            str(output_dir),
        ]

        run_command(
            command,
            repo,
        )

    print("\n" + "=" * 78)
    print("LOOKBACK STUDY COMPLETE")
    print("=" * 78)
    print(f"Results are under:\n{output_root}")
    print("\nExpected directories:")

    for lookback in args.lookbacks:
        print(
            f"  {output_root / f'lookback_{lookback}min'}"
        )

    print(
        "\nSend me the Mean ± SD tables from the four runs."
    )


if __name__ == "__main__":
    main()
