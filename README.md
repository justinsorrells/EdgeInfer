# EdgeInfer

A PyTorch pipeline demonstrating **post-training quantization** of a small transformer model for **edge inference** — benchmarking latency and accuracy tradeoff on CPU-only hardware.

## Motivation

Deploying AI in bandwidth-limited, power-constrained environments (edge devices, drone platforms, embedded hardware) requires models that are fast, small, and accurate. This project demonstrates the quantization workflow used to prepare models for edge deployment.

## What It Does

1. Builds a small transformer encoder classifier (FP32)
2. Benchmarks baseline latency and memory footprint
3. Applies **dynamic INT8 quantization** (`torch.quantization.quantize_dynamic`)
4. Re-benchmarks the quantized model
5. Reports speedup, size reduction, and accuracy delta

## Usage

```bash
pip install torch
python edge_infer.py
```

## Example Output

```
[ FP32 model ]
  Size:     0.54 MB
  Latency:  4.2 ms mean, 5.1 ms p95

[ INT8 quantized model ]
  Size:     0.18 MB
  Latency:  1.9 ms mean, 2.3 ms p95

[ Summary ]
  Speedup:        2.2x faster inference
  Size reduction: 66.7%
  Accuracy delta: 0.001
```

## Extending

- Swap the toy model for a real fine-tuned checkpoint (e.g. DistilBERT, TinyLlama)
- Add static quantization with calibration dataset for better accuracy preservation
- Export to ONNX for hardware-agnostic deployment
- Add pruning to further reduce model size before quantization

## Tech Stack

- Python 3.11+
- PyTorch (quantization module)
- No GPU required — benchmarks on CPU to simulate edge hardware

## Relevance

Built as a portfolio project aligned with Tyto Athene's TALON lab work on edge AI deployment for DoD environments.
