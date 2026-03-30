import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import edge_infer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "edge_infer.py"


class EdgeInferCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_simulation_mode_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_cli("--simulate", "--output-dir", tmpdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            json_reports = list(Path(tmpdir).glob("edgeinfer_*.json"))
            csv_reports = list(Path(tmpdir).glob("edgeinfer_*.csv"))
            self.assertEqual(len(json_reports), 1)
            self.assertEqual(len(csv_reports), 1)

            report = json.loads(json_reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "simulation")
            self.assertEqual(report["summary"]["best_quality_variant"], "fp32")
            self.assertEqual(report["dataset"]["split_counts"]["calibration"], 64)
            self.assertTrue(any(item["name"] == "fp32" for item in report["results"]))
            self.assertIn("macro_f1", report["results"][0])

    def test_compare_and_history_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = self.run_cli("--simulate", "--output-dir", tmpdir, "--seed", "11")
            second = self.run_cli("--simulate", "--output-dir", tmpdir, "--seed", "22")

            self.assertEqual(first.returncode, 0, msg=first.stderr or first.stdout)
            self.assertEqual(second.returncode, 0, msg=second.stderr or second.stdout)

            json_reports = list(Path(tmpdir).glob("edgeinfer_*.json"))
            csv_reports = list(Path(tmpdir).glob("edgeinfer_*.csv"))
            self.assertEqual(len(json_reports), 2)
            self.assertEqual(len(csv_reports), 2)

            compare = self.run_cli("compare", "--latest", "--output-dir", tmpdir)
            self.assertEqual(compare.returncode, 0, msg=compare.stderr or compare.stdout)
            self.assertIn('"mode": "comparison"', compare.stdout)

            comparison_reports = list(Path(tmpdir).glob("edgeinfer_compare_*.json"))
            self.assertEqual(len(comparison_reports), 1)

            history = self.run_cli("history", "--output-dir", tmpdir, "--limit", "2")
            self.assertEqual(history.returncode, 0, msg=history.stderr or history.stdout)
            self.assertIn('"mode": "history"', history.stdout)
            self.assertIn("simulation_only", history.stdout)

    @unittest.skipIf(edge_infer.TORCH_AVAILABLE, "Only relevant when PyTorch is unavailable.")
    def test_real_mode_requires_torch_when_missing(self):
        result = self.run_cli("--skip-report")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PyTorch is required", result.stderr + result.stdout)


class EdgeInferHelpersTests(unittest.TestCase):
    def test_tokenizer_normalizes_basic_text(self):
        self.assertEqual(
            edge_infer.tokenize_text("GPU-ready, Edge AI!"),
            ["gpu", "ready", "edge", "ai"],
        )

    def test_compute_classification_metrics(self):
        metrics = edge_infer.compute_classification_metrics(
            targets=[0, 0, 1, 1],
            predictions=[0, 1, 1, 1],
            index_to_label=["negative", "positive"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["macro_f1"], 0.7333)
        self.assertEqual(metrics["weighted_f1"], 0.7333)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
