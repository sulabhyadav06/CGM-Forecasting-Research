#!/bin/bash
set -euo pipefail

REPO="/Users/sulabhyadav/Desktop/CGM Forecasting Research"
cd "$REPO"

EXPECTED_BRANCH="phase2-hybrid-update"
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: Current branch is '$BRANCH', expected '$EXPECTED_BRANCH'."
  exit 1
fi

echo "[1/7] Checking remote and working tree..."
git fetch origin
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  LOCAL="$(git rev-parse HEAD)"
  REMOTE="$(git rev-parse "origin/$BRANCH")"
  if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "ERROR: Local HEAD and origin/$BRANCH differ."
    echo "       Local : $LOCAL"
    echo "       Remote: $REMOTE"
    echo "No automatic rebase/merge will be attempted."
    exit 1
  fi
fi

echo "[2/7] Backing up and patching phase2_feature_ablation.py..."
STAMP="$(date +%Y%m%d_%H%M%S)"
cp phase2_feature_ablation.py "phase2_feature_ablation.py.bak_${STAMP}"
python3 - <<'PY'
from pathlib import Path

p = Path("phase2_feature_ablation.py")
s = p.read_text()

# Patch the data-preparation block using stable code lines rather than a
# comment/indentation-sensitive full-block match.
if 'tr2["glucose_target_raw"] = tr2["glucose"]' not in s:
    needle = "                tr2 = tr.copy()\n                te2 = te.copy()\n"
    insert = needle + '                tr2["glucose_target_raw"] = tr2["glucose"]\n                te2["glucose_target_raw"] = te2["glucose"]\n'
    if needle not in s:
        raise SystemExit("Could not find train/test dataframe-copy lines; no code was changed.")
    s = s.replace(needle, insert, 1)

# All sequence targets must use the unscaled glucose copy; model input features
# still use the standardized `glucose` column.
s = s.replace('target_col="glucose",', 'target_col="glucose_target_raw",')

# Also ensure the standalone run_case path has a raw target column.
run_needle = '    df = load_df(path)\n\n    missing = [c for c in features if c not in df.columns]\n'
run_insert = '    df = load_df(path)\n    df["glucose_target_raw"] = df["glucose"]\n\n    missing = [c for c in features if c not in df.columns]\n'
if 'df["glucose_target_raw"] = df["glucose"]' not in s:
    if run_needle not in s:
        raise SystemExit("Could not find run_case dataframe load block; no code was changed.")
    s = s.replace(run_needle, run_insert, 1)

p.write_text(s)
PY
python3 -m py_compile phase2_feature_ablation.py

echo "[3/7] Running the corrected feature-ablation experiment..."
python3 phase2_feature_ablation.py --data-root . --output output/phase2_feature_ablation

echo "[4/7] Validating regenerated metrics and creating aggregate common-feature summary..."
python3 - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

out = Path("output/phase2_feature_ablation")
summary_path = out / "feature_ablation_summary.csv"
patient_path = out / "patient_horizon_results.csv"
if not summary_path.exists() or not patient_path.exists():
    raise SystemExit("Expected experiment outputs were not created.")

summary = pd.read_csv(summary_path)
patient = pd.read_csv(patient_path)
required = {"cohort", "feature_set", "horizon_min", "mae_mean", "rmse_mean", "mard_mean", "n_patients"}
missing = required - set(summary.columns)
if missing:
    raise SystemExit(f"Summary missing columns: {sorted(missing)}")

num_cols = ["mae_mean", "rmse_mean", "mard_mean", "clarke_A_mean", "clarke_B_mean", "clarke_C_mean", "clarke_D_mean", "clarke_E_mean"]
if not np.isfinite(summary[num_cols].to_numpy(dtype=float)).all():
    raise SystemExit("Non-finite values found in summary metrics.")

ceg_sum = summary[["clarke_A_mean", "clarke_B_mean", "clarke_C_mean", "clarke_D_mean", "clarke_E_mean"]].sum(axis=1)
if not np.allclose(ceg_sum.to_numpy(), 100.0, atol=1e-5):
    raise SystemExit("Clarke zone percentages do not sum to 100%.")

# A standardized-target bug would typically leave RMSE/MAE around ~1.0.
# Refuse publication if all regenerated RMSE means remain suspiciously tiny.
if float(summary["rmse_mean"].median()) < 5.0:
    raise SystemExit("Regenerated RMSE is still suspiciously small; refusing to publish results.")

expected_common = {"glucose", "glucose_carbs", "glucose_insulin", "glucose_carbs_insulin"}
common = patient[patient["feature_set"].isin(expected_common)].copy()
common_summary = (
    common.groupby(["feature_set", "horizon_min"])
    .agg(
        mae_mean=("mae_mgdl", "mean"), mae_sd=("mae_mgdl", "std"),
        rmse_mean=("rmse_mgdl", "mean"), rmse_sd=("rmse_mgdl", "std"),
        n_patients=("patient", "nunique"),
    )
    .reset_index()
    .sort_values(["feature_set", "horizon_min"])
)
if common_summary.empty:
    raise SystemExit("No common-feature results found.")
if common_summary["n_patients"].min() != 12:
    bad = common_summary.loc[common_summary["n_patients"] != 12, ["feature_set", "horizon_min", "n_patients"]]
    raise SystemExit("Common-feature experiment does not contain all 12 patients:\n" + bad.to_string(index=False))

common_summary.to_csv(out / "common_feature_all12_summary.csv", index=False)

print("Validated feature-ablation summary:")
print(summary.to_string(index=False))
print("\nValidated all-12 common-feature summary:")
print(common_summary.to_string(index=False))
PY

echo "[5/7] Updating curated public results..."
mkdir -p reports/public_results
cp output/phase2_feature_ablation/feature_ablation_summary.csv reports/public_results/feature_ablation_summary.csv
cp output/phase2_feature_ablation/common_feature_all12_summary.csv reports/public_results/common_feature_all12_summary.csv

# The older baseline comparison file has duplicate metric column names and is not safe to publish yet.
if [[ -f reports/public_results/hybrid_vs_baselines_summary.csv ]]; then
  mkdir -p output/review_pending_publication
  mv reports/public_results/hybrid_vs_baselines_summary.csv output/review_pending_publication/
fi

# Remove the extra EOF blank line that previously triggered git diff --check.
python3 - <<'PY'
from pathlib import Path
p = Path("mini_paper.md")
if p.exists():
    p.write_text(p.read_text().rstrip() + "\n")
PY

echo "[6/7] Staging all tracked changes plus curated aggregate results..."
git add -u
for f in reports/public_results/*.csv; do
  [[ -e "$f" ]] && git add "$f"
done

# Never stage these classes of files even if they somehow appear as untracked.
git reset --quiet -- reports/public_results/hybrid_vs_baselines_summary.csv 2>/dev/null || true

git diff --cached --check

echo "\nStaged changes:"
git diff --cached --stat

echo "\nStaged files:"
git diff --cached --name-only

echo "\nLarge staged files (>50MB):"
FOUND_LARGE=0
while IFS= read -r f; do
  if [[ -f "$f" ]]; then
    bytes=$(wc -c < "$f")
    if (( bytes > 50 * 1024 * 1024 )); then
      echo "$f ($bytes bytes)"
      FOUND_LARGE=1
    fi
  fi
done < <(git diff --cached --name-only)
if (( FOUND_LARGE )); then
  echo "ERROR: Large file detected in staged changes."
  exit 1
fi

# Keep the temporary backup local and outside Git.
mkdir -p output/local_backups
mv "phase2_feature_ablation.py.bak_${STAMP}" output/local_backups/

echo "\n[7/7] Committing and pushing..."
git status --short

git commit -m "Phase 2: fix feature ablation target scaling"
git push -u origin "$BRANCH"

echo "\n=============================="
echo "DONE: Phase-2 changes pushed to $BRANCH"
echo "=============================="
git status --short
