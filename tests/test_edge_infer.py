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
    def test_simulation_mode_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--simulate",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            json_reports = list(Path(tmpdir).glob("*.json"))
            csv_reports = list(Path(tmpdir).glob("*.csv"))
            self.assertEqual(len(json_reports), 1)
            self.assertEqual(len(csv_reports), 1)

            report = json.loads(json_reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "simulation")
            self.assertIn("results", report)
            self.assertTrue(any(item["name"] == "fp32" for item in report["results"]))

    @unittest.skipIf(edge_infer.TORCH_AVAILABLE, "Only relevant when PyTorch is unavailable.")
    def test_real_mode_requires_torch_when_missing(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--skip-report"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PyTorch is required", result.stderr + result.stdout)


class EdgeInferHelpersTests(unittest.TestCase):
    def test_tokenizer_normalizes_basic_text(self):
        self.assertEqual(
            edge_infer.tokenize_text("GPU-ready, Edge AI!"),
            ["gpu", "ready", "edge", "ai"],
        )


if __name__ == "__main__":
    unittest.main()
