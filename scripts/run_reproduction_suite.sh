#!/usr/bin/env bash
# Local experiment launcher. Never uploads raw OhioT1DM data.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${CGM_OHIO_DATA_ROOT:-$HOME/Desktop/OhioT1DM_data_backup}"
OUT="$ROOT/output/reproduction_suite_$(date +%Y%m%d_%H%M%S)"
LOG="$OUT/logs"
mkdir -p "$LOG"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: dataset directory not found: $DATA_ROOT" >&2
  echo "Set CGM_OHIO_DATA_ROOT to the directory containing patient training/testing multimodal CSVs." >&2
  exit 2
fi
for pid in 540 544 552 559 563 567 570 575 584 588 591 596; do
  for split in training testing; do
    [[ -f "$DATA_ROOT/${pid}_${split}_multimodal.csv" ]] || { echo "ERROR: missing $DATA_ROOT/${pid}_${split}_multimodal.csv" >&2; exit 2; }
  done
done
CRNN_EPOCHS="${CRNN_EPOCHS:-50}"
GLUNET_EPOCHS="${GLUNET_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GLUNET_BATCH_SIZE="${GLUNET_BATCH_SIZE:-128}"
run_logged() {
  local label="$1"; shift
  local logfile="$LOG/${label}.log"
  echo
  echo "========== $label =========="
  echo "Log: $logfile"
  "$@" 2>&1 | tee "$logfile"
}
echo "Repository: $ROOT"
echo "Dataset root: $DATA_ROOT"
echo "Outputs: $OUT"
# Stage 1: initial CRNN at 30/60 minutes.
for h in 30 60; do
  run_logged "crnn_h${h}" python -m experiments.crnn.train_crnn --horizon "$h" --epochs "$CRNN_EPOCHS" --batch-size "$BATCH_SIZE" --output-root "$OUT/crnn_initial"
done
# Stage 2: GluNet 256-class candidate at 30/60 minutes.
run_logged "glunet_30_60" python experiments/glunet/train_glunet_candidate_all_patients.py --data-root "$DATA_ROOT" --output-dir "$OUT/glunet_initial" --horizons 30 60 --epochs "$GLUNET_EPOCHS" --batch-size "$GLUNET_BATCH_SIZE"
# Stage 3: extended horizons.
for h in 15 90 120; do
  run_logged "crnn_h${h}" python -m experiments.crnn.train_crnn --horizon "$h" --epochs "$CRNN_EPOCHS" --batch-size "$BATCH_SIZE" --output-root "$OUT/crnn_extended"
done
run_logged "glunet_15_90_120" python experiments/glunet/train_glunet_candidate_all_patients.py --data-root "$DATA_ROOT" --output-dir "$OUT/glunet_extended" --horizons 15 90 120 --epochs "$GLUNET_EPOCHS" --batch-size "$GLUNET_BATCH_SIZE"
echo
echo "Candidate runs finished. Logs: $LOG"
echo "Results: $OUT"
echo "Successful execution does not prove exact paper reproduction."
