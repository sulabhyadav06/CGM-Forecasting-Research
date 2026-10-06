"""Lightweight regression tests for the Phase-2 forecasting protocol.

These tests validate target-offset arithmetic and critical source invariants.
They do not replace model training or clinical-metric evaluation on the datasets.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ("hybrid_experiments.py", "hybrid_ablation_experiments.py")


class ForecastAlignmentTests(unittest.TestCase):
    def test_horizon_offsets_match_input_window_convention(self):
        # Input rows are [i-lookback, ..., i-1], so the last input is i-1.
        # The target at i + h/5 - 1 is exactly h minutes after that input.
        for horizon in (15, 30, 60):
            steps = horizon // 5
            target_offset = steps - 1
            elapsed_minutes = (target_offset + 1) * 5
            self.assertEqual(elapsed_minutes, horizon)

    def test_hybrid_runners_use_correct_shift(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("shift(-(steps - 1))", source)
                self.assertNotIn("shift(-steps)", source)

    def test_chronological_split_excludes_cross_boundary_training_targets(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("train_end = split_idx - max_offset", source)
                self.assertIn("end_anchor=train_end", source)
                self.assertIn("start_anchor=split_idx", source)

    def test_all_patient_runner_includes_all_twelve_patient_ids(self):
        source = (ROOT / "run_phase2_hybrid_all_patients.py").read_text(encoding="utf-8")
        for patient in ("540", "544", "552", "567", "584", "596",
                        "559", "563", "570", "575", "588", "591"):
            with self.subTest(patient=patient):
                self.assertIn(f'"{patient}"', source)
        self.assertIn('"--horizons"', source)
        self.assertIn('default=[15, 30, 60]', source)
        self.assertIn('"hybrid_experiments.py"', source)
        self.assertIn('"hybrid_ablation_experiments.py"', source)

    def test_loaders_support_both_csv_layouts_and_header_case(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn('os.path.join(root, split, f"{patient}.csv")', source)
                self.assertIn('f"{patient}_{suffix}_multimodal.csv"', source)
                self.assertIn('str(column).strip().lower()', source)
                self.assertIn('pd.to_datetime(df["timestamp"], errors="raise")', source)

    def test_windows_check_five_minute_continuity(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn('data["timestamp"]', source)
                self.assertIn("pd.Timedelta(minutes=5)", source)
                self.assertIn("pd.Timedelta(minutes=h)", source)


if __name__ == "__main__":
    unittest.main()
