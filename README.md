# EdgeInfer

EdgeInfer is a CPU-first benchmarking tool for small text classifiers, focused on the kinds of model compression tradeoffs that matter for edge deployment.

It now does more than print a toy before/after comparison:

- trains a compact transformer before benchmarking, so accuracy is meaningful
- benchmarks multiple deployment variants: `fp32`, `dynamic_int8`, and `dynamic_float16`
- saves JSON and CSV reports for each run
- supports either a built-in learnable dataset or a user-provided JSONL text dataset
- requires explicit `--simulate` mode when PyTorch is unavailable, so fake metrics are clearly labeled

## Motivation

Deploying AI in bandwidth-limited, power-constrained environments requires models that are fast, small, and still accurate enough to trust. EdgeInfer is meant to show the workflow engineers use to compare baseline and optimized model variants before shipping to CPU-bound devices.

## What It Does

1. Builds and trains a small transformer text classifier
2. Evaluates baseline FP32 accuracy
3. Benchmarks repeated CPU inference with warmup runs
4. Applies dynamic quantization variants
5. Re-benchmarks latency, throughput, model size, and accuracy deltas
6. Writes machine-readable JSON and CSV artifacts to `reports/`

## Installation

```bash
pip install torch
```

## Usage

Run the default built-in dataset benchmark:

```bash
python edge_infer.py
```

Run a clearly labeled demo without PyTorch:

```bash
python edge_infer.py --simulate
```

Benchmark only FP32 and INT8:

```bash
python edge_infer.py --variants fp32 dynamic_int8
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

Expected JSONL format:

```json
{"text": "battery lasts all day", "label": 1}
{"text": "screen flickers after boot", "label": 0}
```

## Output Artifacts

Each run writes:

- `reports/edgeinfer_<timestamp>.json`
- `reports/edgeinfer_<timestamp>.csv`

The report includes:

- runtime mode (`benchmark` or `simulation`)
- system metadata
- config values
- dataset summary
- training history
- per-variant latency, throughput, size, and accuracy
- comparison summaries versus FP32

## Example Console Flow

```text
=== EdgeInfer: Quantization & Edge Inference Benchmark ===

[ Dataset ] keyword_signal
  Train examples: 512
  Eval examples:  128

[ Training ]
  Epoch 1: loss=0.4120 eval_acc=92.2%
  Epoch 2: loss=0.1337 eval_acc=98.4%
  Epoch 3: loss=0.0518 eval_acc=99.2%

[ fp32 ]
  Size:     0.56 MB
  Latency:  4.31 ms mean, 4.07 ms median, 5.14 ms p95
  Throughput: 231.92 samples/s
  Accuracy: 99.2%
```

## Useful Flags

- `--simulate`: run demo metrics intentionally and transparently
- `--dataset-jsonl PATH`: benchmark on a local labeled text dataset
- `--variants ...`: choose which deployment variants to compare
- `--output-dir DIR`: choose where reports are saved
- `--skip-report`: skip writing JSON/CSV artifacts
- `--embed-dim`, `--num-heads`, `--num-layers`, `--ff-dim`: adjust model size

## Extending

- Add static post-training quantization with a calibration split
- Support ONNX export for hardware-agnostic deployment
- Swap the local tokenizer/model for a real fine-tuned checkpoint
- Add pruning or sparsity-aware comparisons
- Add CI or dashboards to compare benchmark history across runs

## Tech Stack

- Python 3.11+
- PyTorch
- CPU-only benchmarking

## Relevance

This project is aligned with edge AI and model optimization work where deployment constraints matter as much as raw model quality.
