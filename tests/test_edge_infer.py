import json
import operator
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            result = self.run_cli(
                "--simulate",
                "--output-dir",
                tmpdir,
                "--export-formats",
                "onnx",
            )
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
            self.assertTrue(any(item["name"] == "static_int8" for item in report["results"]))
            self.assertIn("macro_f1", report["results"][0])
            self.assertEqual(len(report["exports"]), 1)
            self.assertEqual(report["exports"][0]["format"], "onnx")
            self.assertTrue(report["exports"][0]["simulated"])

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

    def test_compare_thresholds_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            candidate_path = Path(tmpdir) / "candidate.json"

            baseline_report = {
                "generated_at": "2026-03-30T00:00:00+00:00",
                "dataset": {"name": "demo"},
                "results": [
                    {
                        "name": "fp32",
                        "mean_ms": 4.0,
                        "size_mb": 1.0,
                        "accuracy": 0.90,
                        "macro_f1": 0.89,
                        "weighted_f1": 0.89,
                    }
                ],
                "summary": {},
            }
            candidate_report = {
                "generated_at": "2026-03-30T00:05:00+00:00",
                "dataset": {"name": "demo"},
                "results": [
                    {
                        "name": "fp32",
                        "mean_ms": 4.8,
                        "size_mb": 1.1,
                        "accuracy": 0.86,
                        "macro_f1": 0.84,
                        "weighted_f1": 0.84,
                    }
                ],
                "summary": {},
            }
            baseline_path.write_text(json.dumps(baseline_report), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")

            result = self.run_cli(
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--max-latency-regression-pct",
                "10",
                "--max-accuracy-drop",
                "0.01",
                "--max-macro-f1-drop",
                "0.01",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("Regression Gates", result.stdout)
            self.assertIn('"regressions"', result.stdout)

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

    def test_evaluate_regression_thresholds(self):
        comparison = {
            "variant_comparisons": {
                "fp32": {
                    "latency_change_pct": 12.5,
                    "size_change_pct": 3.0,
                    "accuracy_delta": -0.03,
                    "macro_f1_delta": -0.02,
                    "weighted_f1_delta": -0.02,
                }
            }
        }
        args = type(
            "Args",
            (),
            {
                "max_latency_regression_pct": 10.0,
                "max_size_regression_pct": 5.0,
                "max_accuracy_drop": 0.01,
                "max_macro_f1_drop": 0.01,
            },
        )()

        regressions = edge_infer.evaluate_regression_thresholds(comparison, args)
        metrics = {item["metric"] for item in regressions}

        self.assertEqual(metrics, {"latency_change_pct", "accuracy_drop", "macro_f1_drop"})

    def test_allocate_split_counts_keeps_train_example_and_non_zero_requested_splits(self):
        train_count, counts = edge_infer.allocate_split_counts(
            12,
            {"calibration": 0.1, "validation": 0.1, "test": 0.1},
        )

        self.assertGreaterEqual(train_count, 1)
        self.assertEqual(train_count + sum(counts.values()), 12)
        self.assertGreaterEqual(counts["calibration"], 1)
        self.assertGreaterEqual(counts["validation"], 1)
        self.assertGreaterEqual(counts["test"], 1)

    def test_split_records_rejects_invalid_ratios(self):
        records = [{"text": f"row {idx}", "label": idx % 2} for idx in range(12)]

        with self.assertRaisesRegex(ValueError, "must sum to less than 1"):
            edge_infer.split_records(records, 0.4, 0.4, 0.2, seed=1)

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            edge_infer.split_records(records, -0.1, 0.1, 0.1, seed=1)

    def test_build_jsonl_dataset_with_patched_tensorize(self):
        records = [
            {"text": f"great battery life {idx}", "label": "positive"} for idx in range(6)
        ] + [
            {"text": f"screen flicker issue {idx}", "label": "negative"} for idx in range(6)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "reviews.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            with patch.object(edge_infer, "tensorize_examples", side_effect=lambda inputs, labels: (inputs, labels)):
                dataset = edge_infer.build_jsonl_dataset(
                    str(dataset_path),
                    seq_len=8,
                    calibration_ratio=0.1,
                    validation_ratio=0.1,
                    test_ratio=0.1,
                    seed=42,
                )

        self.assertEqual(dataset["name"], "jsonl:reviews.jsonl")
        self.assertEqual(dataset["num_classes"], 2)
        self.assertEqual(sum(dataset["split_counts"].values()), 12)
        self.assertEqual(dataset["split_counts"]["calibration"], 2)
        self.assertEqual(dataset["split_counts"]["validation"], 2)
        self.assertEqual(dataset["split_counts"]["test"], 2)
        self.assertEqual(len(dataset["train"][0]), dataset["split_counts"]["train"])
        self.assertEqual(len(dataset["train"][1]), dataset["split_counts"]["train"])

    def test_build_simulated_export_results_reports_missing_source_variant(self):
        args = SimpleNamespace(
            seed=7,
            export_formats=["onnx"],
            export_variants=["fp32", "static_int8"],
            batch_size=2,
            seq_len=16,
            n_runs=5,
            warmup_runs=1,
        )
        benchmark_results = [
            {
                "name": "fp32",
                "mean_ms": 5.0,
                "metrics": {
                    "accuracy": 0.9,
                    "macro_precision": 0.9,
                    "macro_recall": 0.9,
                    "macro_f1": 0.9,
                    "weighted_f1": 0.9,
                    "balanced_accuracy": 0.9,
                },
            }
        ]

        exports, errors = edge_infer.build_simulated_export_results(args, benchmark_results)

        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0]["name"], "onnx:fp32")
        self.assertTrue(exports[0]["simulated"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["source_variant"], "static_int8")

    def test_summarize_results_picks_best_latency_and_quality(self):
        summary = edge_infer.summarize_results(
            [
                {"name": "fp32", "mean_ms": 5.0, "size_mb": 1.0, "accuracy": 0.90, "macro_f1": 0.91},
                {"name": "dynamic_int8", "mean_ms": 3.0, "size_mb": 0.4, "accuracy": 0.88, "macro_f1": 0.89},
                {"name": "dynamic_float16", "mean_ms": 4.0, "size_mb": 0.7, "accuracy": 0.91, "macro_f1": 0.93},
            ]
        )

        self.assertEqual(summary["best_latency_variant"], "dynamic_int8")
        self.assertEqual(summary["best_quality_variant"], "dynamic_float16")
        self.assertEqual(summary["comparisons"]["dynamic_int8"]["speedup_vs_fp32"], 1.67)
        self.assertEqual(summary["comparisons"]["dynamic_float16"]["macro_f1_delta_vs_fp32"], 0.02)

    def test_get_benchmark_report_files_ignores_non_benchmark_json_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            benchmark_old = tmpdir_path / "edgeinfer_20260330_000000_000001.json"
            benchmark_new = tmpdir_path / "edgeinfer_20260330_000000_000002.json"
            comparison = tmpdir_path / "edgeinfer_compare_20260330_000000_000003.json"
            invalid = tmpdir_path / "broken.json"

            benchmark_payload = {"results": [{"name": "fp32"}], "summary": {"best_latency_variant": "fp32"}}
            benchmark_old.write_text(json.dumps(benchmark_payload), encoding="utf-8")
            benchmark_new.write_text(json.dumps(benchmark_payload), encoding="utf-8")
            comparison.write_text(json.dumps({"mode": "comparison", "variant_comparisons": {}}), encoding="utf-8")
            invalid.write_text("{not valid json", encoding="utf-8")

            report_files = edge_infer.get_benchmark_report_files(tmpdir)

        self.assertEqual([path.name for path in report_files], [benchmark_new.name, benchmark_old.name])

    def test_parse_cli_args_defaults_to_benchmark(self):
        args = edge_infer.parse_cli_args(["--simulate", "--skip-report"])
        self.assertEqual(args.command, "benchmark")
        self.assertTrue(args.simulate)
        self.assertTrue(args.skip_report)

    def test_validate_benchmark_args_rejects_export_variant_not_in_variants(self):
        args = SimpleNamespace(
            seq_len=32,
            batch_size=1,
            train_batch_size=32,
            num_heads=4,
            train_epochs=1,
            embed_dim=64,
            ff_dim=128,
            dataset_jsonl=None,
            train_samples=10,
            calibration_samples=2,
            validation_samples=2,
            test_samples=2,
            vocab_size=100,
            calibration_ratio=0.1,
            validation_ratio=0.1,
            test_ratio=0.1,
            export_opset=17,
            export_formats=["onnx"],
            export_variants=["static_int8"],
            variants=["fp32"],
            simulate=True,
        )

        with self.assertRaisesRegex(SystemExit, "Export variants must also be benchmarked variants"):
            edge_infer.validate_benchmark_args(args)

    def test_validate_benchmark_args_rejects_unsupported_onnx_export_variant(self):
        args = SimpleNamespace(
            seq_len=32,
            batch_size=1,
            train_batch_size=32,
            num_heads=4,
            train_epochs=1,
            embed_dim=64,
            ff_dim=128,
            dataset_jsonl=None,
            train_samples=10,
            calibration_samples=2,
            validation_samples=2,
            test_samples=2,
            vocab_size=100,
            calibration_ratio=0.1,
            validation_ratio=0.1,
            test_ratio=0.1,
            export_opset=17,
            export_formats=["onnx"],
            export_variants=["dynamic_int8"],
            variants=["fp32", "dynamic_int8"],
            simulate=True,
        )

        with self.assertRaisesRegex(SystemExit, "ONNX export currently supports only these variants"):
            edge_infer.validate_benchmark_args(args)

    @unittest.skipUnless(edge_infer.TORCH_AVAILABLE, "Requires PyTorch.")
    def test_build_static_qconfig_mapping_disables_bool_mask_observers(self):
        try:
            backend = edge_infer.select_quantized_engine()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        mapping = edge_infer.build_static_qconfig_mapping(backend)

        self.assertIsNone(mapping.object_type_qconfigs["eq"])
        self.assertIsNone(mapping.object_type_qconfigs["unsqueeze"])
        self.assertIsNone(mapping.object_type_qconfigs["float"])
        self.assertIsNone(mapping.object_type_qconfigs["sum"])
        self.assertIsNone(mapping.object_type_qconfigs["clamp"])
        self.assertIsNone(mapping.object_type_qconfigs[operator.invert])

    @unittest.skipUnless(edge_infer.TORCH_AVAILABLE, "Requires PyTorch.")
    def test_apply_static_quantization_runs_with_transformer_mask_path(self):
        try:
            backend = edge_infer.select_quantized_engine()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        if not hasattr(edge_infer.torch_quantization, "get_default_qconfig_mapping"):
            self.skipTest("Static FX quantization is unavailable in this PyTorch build.")

        dataset = edge_infer.build_keyword_dataset(
            seq_len=8,
            train_samples=8,
            calibration_samples=4,
            validation_samples=4,
            test_samples=4,
            vocab_size=64,
            seed=5,
        )
        model = edge_infer.SmallTextClassifier(
            vocab_size=dataset["vocab_size"],
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            ff_dim=16,
            num_classes=dataset["num_classes"],
            max_len=8,
            pad_token_id=edge_infer.PAD_TOKEN_ID,
        ).eval()

        quantized_model = edge_infer.apply_static_quantization(
            model,
            calibration_dataset=dataset["calibration"],
            batch_size=2,
        )
        inference = edge_infer.benchmark_inference(
            quantized_model,
            dataset["test"][0][:2],
            n_runs=1,
            warmup_runs=1,
        )
        metrics = edge_infer.evaluate_model(
            quantized_model,
            dataset["test"],
            batch_size=2,
            index_to_label=dataset["index_to_label"],
        )

        self.assertEqual(getattr(quantized_model, "_edgeinfer_quantized_engine", None), backend)
        self.assertGreaterEqual(inference["mean_ms"], 0.0)
        self.assertIn("accuracy", metrics)

    def test_resolve_compare_paths_uses_latest_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            older = tmpdir_path / "edgeinfer_20260330_000000_000001.json"
            newer = tmpdir_path / "edgeinfer_20260330_000000_000002.json"
            payload = {"results": [{"name": "fp32"}], "summary": {"best_latency_variant": "fp32"}}
            older.write_text(json.dumps(payload), encoding="utf-8")
            newer.write_text(json.dumps(payload), encoding="utf-8")

            args = SimpleNamespace(
                latest=True,
                baseline=None,
                candidate=None,
                output_dir=tmpdir,
            )
            baseline_path, candidate_path = edge_infer.resolve_compare_paths(args)

        self.assertEqual(Path(baseline_path).name, older.name)
        self.assertEqual(Path(candidate_path).name, newer.name)

    def test_build_history_entries_returns_most_recent_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            first = tmpdir_path / "edgeinfer_20260330_000000_000001.json"
            second = tmpdir_path / "edgeinfer_20260330_000000_000002.json"
            for idx, path in enumerate((first, second), start=1):
                path.write_text(
                    json.dumps(
                        {
                            "generated_at": f"2026-03-30T00:00:0{idx}+00:00",
                            "mode": "simulation",
                            "dataset": {"name": f"demo{idx}"},
                            "results": [{"name": "fp32"}],
                            "summary": {
                                "best_latency_variant": "fp32",
                                "best_quality_variant": "fp32",
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            entries = edge_infer.build_history_entries(tmpdir, limit=1)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["dataset"], "demo2")


if __name__ == "__main__":
    unittest.main()
