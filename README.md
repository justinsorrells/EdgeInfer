# EdgeInfer

EdgeInfer is a CPU-first benchmarking tool for small text classifiers, focused on the kinds of model compression tradeoffs that matter for edge deployment.

It now does more than print a toy before/after comparison:

- trains a compact transformer before benchmarking, so quality metrics are meaningful
- uses separate train, calibration, validation, and test splits
- benchmarks multiple deployment variants: `fp32`, `dynamic_int8`, `dynamic_float16`, and `static_int8`
- reports accuracy, macro F1, weighted F1, balanced accuracy, and confusion matrices
- can export and benchmark ONNX runtimes from selected variants
- saves JSON and CSV reports for each run
- compares saved runs and shows local report history
- requires explicit `--simulate` mode when PyTorch is unavailable, so fake metrics are clearly labeled

## Motivation

Deploying AI in bandwidth-limited, power-constrained environments requires models that are fast, small, and still accurate enough to trust. EdgeInfer is meant to show the workflow engineers use to compare baseline and optimized model variants before shipping to CPU-bound devices.

## What It Does

1. Builds and trains a small transformer text classifier
2. Splits data into train, calibration, validation, and test sets
3. Evaluates validation metrics during training
4. Benchmarks repeated CPU inference with warmup runs
5. Applies dynamic and static quantization variants
6. Re-benchmarks latency, throughput, size, and held-out test metrics
7. Optionally exports variants to ONNX and benchmarks the exported runtime
8. Writes machine-readable JSON and CSV artifacts to `reports/`
9. Compares past benchmark runs with built-in CLI commands

## Installation

```bash
pip install torch onnxruntime onnxscript
```

## Benchmark Usage

Run the default built-in dataset benchmark:

```bash
python edge_infer.py
```

Run a clearly labeled demo without PyTorch:

```bash
python edge_infer.py --simulate
```

Benchmark only FP32 and dynamic INT8:

```bash
python edge_infer.py --variants fp32 dynamic_int8
```

Include static quantization in the run:

```bash
python edge_infer.py --variants fp32 dynamic_int8 static_int8
```

Tune the experiment:

```bash
python edge_infer.py \
  --train-epochs 4 \
  --seq-len 48 \
  --batch-size 2 \
  --n-runs 75 \
  --warmup-runs 15
```

Use a local JSONL dataset:

```bash
python edge_infer.py --dataset-jsonl data/reviews.jsonl
```

Export and benchmark the FP32 ONNX runtime:

```bash
python edge_infer.py --export-formats onnx
```

Current ONNX export support is limited to the eager `fp32` variant for this transformer benchmark. Quantized variants are still benchmarked in-process, but are not exported to ONNX by this CLI yet.

Expected JSONL format:

```json
{"text": "battery lasts all day", "label": 1}
{"text": "screen flickers after boot", "label": 0}
```

## Compare And History

Compare the latest two saved benchmark runs:

```bash
python edge_infer.py compare --latest
```

Fail CI if latency regresses by more than 10% or macro F1 drops by more than 0.01:

```bash
python edge_infer.py compare --latest \
  --max-latency-regression-pct 10 \
  --max-macro-f1-drop 0.01
```

Compare two explicit report files:

```bash
python edge_infer.py compare \
  --baseline reports/edgeinfer_20260329_120000_000001.json \
  --candidate reports/edgeinfer_20260329_121500_000002.json
```

List recent reports:

```bash
python edge_infer.py history --limit 5
```

## Output Artifacts

Each benchmark run writes:

- `reports/edgeinfer_<timestamp>.json`
- `reports/edgeinfer_<timestamp>.csv`
- `reports/edgeinfer_export_<timestamp>_<variant>.onnx` when ONNX export is requested

Each comparison run writes:

- `reports/edgeinfer_compare_<timestamp>.json`

Benchmark reports include:

- runtime mode (`benchmark` or `simulation`)
- system metadata
- config values
- dataset summary and split counts
- training history
- per-variant latency, throughput, size, and quality metrics
- optional export/runtime benchmark entries
- confusion matrices and per-class scores
- comparison summaries versus FP32

## Example Console Flow

```text
=== EdgeInfer: Quantization & Edge Inference Benchmark ===

[ Dataset ] keyword_signal
  Train examples:       512
  Calibration examples: 64
  Validation examples:  128
  Test examples:        128

[ Training ]
  Epoch 1: loss=0.4120 val_acc=92.2% val_macro_f1=92.1%
  Epoch 2: loss=0.1337 val_acc=98.4% val_macro_f1=98.4%
  Epoch 3: loss=0.0518 val_acc=99.2% val_macro_f1=99.2%

[ fp32 ]
  Size:     0.56 MB
  Latency:  4.31 ms mean, 4.07 ms median, 5.14 ms p95
  Throughput: 231.92 samples/s
  Quality:  acc=99.2%, macro_f1=99.2%, weighted_f1=99.2%
```

## Useful Flags

- `--simulate`: run demo metrics intentionally and transparently
- `--dataset-jsonl PATH`: benchmark on a local labeled text dataset
- `--calibration-ratio`, `--validation-ratio`, `--test-ratio`: JSONL split controls
- `--calibration-samples`, `--validation-samples`, `--test-samples`: built-in dataset split sizes
- `--variants ...`: choose which deployment variants to compare
- `--export-formats onnx`: export and benchmark ONNX runtimes
- `--export-variants ...`: choose which eager variants to export
- `--export-opset`: select the ONNX opset version
- `--output-dir DIR`: choose where reports are saved
- `--skip-report`: skip writing JSON/CSV artifacts
- `--embed-dim`, `--num-heads`, `--num-layers`, `--ff-dim`: adjust model size

## Extending

- Expand ONNX support to additional providers and quantized export paths
- Swap the local tokenizer/model for a real fine-tuned checkpoint
- Add pruning or sparsity-aware comparisons
- Wire compare thresholds into CI regression checks

## Tech Stack

- Python 3.11+
- PyTorch
- CPU-only benchmarking

## Relevance

This project is aligned with edge AI and model optimization work where deployment constraints matter as much as raw model quality.
