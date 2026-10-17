from pathlib import Path
import pandas as pd

DATA_ROOT = Path.home() / "Desktop" / "OhioT1DM_data_backup"

PATIENTS = [
    "540", "544", "552", "567", "584", "596",
    "559", "563", "570", "575", "588", "591",
]

OUTPUT = Path("output/crnn_initial/timestamp_audit.csv")


def audit_file(patient, split):
    path = DATA_ROOT / f"{patient}_{split}_multimodal.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    delta = ts.diff().dt.total_seconds() / 60.0

    # Treat intervals within 30 seconds of five minutes as
    # nominally regular for this audit only. This does not
    # resample data or establish that the interval is acceptable.
    jitter = delta.notna() & ((delta - 5.0).abs() <= 0.5)
    short_or_irregular = delta.notna() & ~jitter

    return {
        "patient": patient,
        "split": split,
        "rows": len(df),
        "invalid_timestamps": int(ts.isna().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "nominal_5min_intervals": int(jitter.sum()),
        "other_intervals": int(short_or_irregular.sum()),
        "median_interval_min": float(delta.median()),
        "intervals_under_5min": int((delta < 5.0 - 0.5).sum()),
        "intervals_over_5min": int((delta > 5.0 + 0.5).sum()),
        "missing_glucose": int(df["Glucose"].isna().sum()),
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        audit_file(patient, split)
        for patient in PATIENTS
        for split in ["training", "testing"]
    ]

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved audit: {OUTPUT}")


if __name__ == "__main__":
    main()
