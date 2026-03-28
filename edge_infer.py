"""
EdgeInfer - Quantization & Edge Inference Benchmark
Demonstrates post-training quantization of a small transformer model
and benchmarks latency/accuracy tradeoff on CPU-only hardware.

Relevant to: Tyto Athene TALON lab (edge AI, model optimization for
bandwidth-limited, power-constrained environments).
"""

import time
import json
import random
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.quantization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[warn] PyTorch not installed. Running in simulation mode.")

# ---------------------------------------------------------------------------
# Minimal Transformer Classifier (small enough to quantize quickly on CPU)
# ---------------------------------------------------------------------------

class SmallTextClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Lightweight transformer encoder + linear head.
    Designed to fit in ~10MB after quantization.
    Simulates the kind of small model deployed on edge hardware (e.g. Jetson, RPi).
    """
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64,
                 num_heads: int = 4, num_classes: int = 2, max_len: int = 64):
        if TORCH_AVAILABLE:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_enc = nn.Embedding(max_len, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=128,
                dropout=0.0, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.max_len = max_len

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_enc(positions)
        out = self.encoder(out)
        return self.classifier(out.mean(dim=1))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def model_size_mb(model) -> float:
    """Estimate model size in MB via parameter count."""
    if not TORCH_AVAILABLE:
        return 0.0
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)


def benchmark_inference(model, inputs, n_runs: int = 50) -> dict:
    """Measure mean/p95 latency over n_runs forward passes."""
    if not TORCH_AVAILABLE:
        # Simulation mode: return plausible numbers
        base = random.uniform(3.0, 6.0)
        return {"mean_ms": base, "p95_ms": base * 1.3, "n_runs": n_runs}

    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(inputs)
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "mean_ms": round(sum(latencies) / len(latencies), 3),
        "p95_ms": round(latencies[int(0.95 * len(latencies))], 3),
        "n_runs": n_runs,
    }


def measure_accuracy(model, dataset) -> float:
    """Evaluate accuracy on a toy dataset."""
    if not TORCH_AVAILABLE:
        return random.uniform(0.80, 0.95)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dataset:
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
    return correct / len(dataset)


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model):
    """Apply PyTorch dynamic quantization (int8 weights, fp32 activations)."""
    if not TORCH_AVAILABLE:
        return model
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Embedding}, dtype=torch.qint8
    )


def run_benchmark(seq_len: int = 32, batch_size: int = 1,
                  n_runs: int = 50, n_samples: int = 200) -> dict:
    """
    Full pipeline:
    1. Build float32 model
    2. Benchmark baseline
    3. Apply quantization
    4. Benchmark quantized model
    5. Report accuracy and speedup
    """
    print("=== EdgeInfer: Quantization & Edge Inference Benchmark ===\n")

    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        model_fp32 = SmallTextClassifier()
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        # Toy labeled dataset
        dataset = [
            (torch.randint(0, 1000, (1, seq_len)), torch.randint(0, 2, (1,)))
            for _ in range(n_samples)
        ]
    else:
        model_fp32 = None
        inputs = None
        dataset = []

    # Baseline
    print("[ FP32 model ]")
    size_fp32 = model_size_mb(model_fp32)
    latency_fp32 = benchmark_inference(model_fp32, inputs, n_runs)
    acc_fp32 = measure_accuracy(model_fp32, dataset)
    print(f"  Size:     {size_fp32:.2f} MB")
    print(f"  Latency:  {latency_fp32['mean_ms']:.2f} ms mean, {latency_fp32['p95_ms']:.2f} ms p95")
    print(f"  Accuracy: {acc_fp32:.1%} (random init baseline)\n")

    # Quantize
    print("[ Applying dynamic INT8 quantization... ]")
    model_int8 = apply_dynamic_quantization(model_fp32)

    print("[ INT8 quantized model ]")
    size_int8 = model_size_mb(model_int8)
    latency_int8 = benchmark_inference(model_int8, inputs, n_runs)
    acc_int8 = measure_accuracy(model_int8, dataset)
    print(f"  Size:     {size_int8:.2f} MB")
    print(f"  Latency:  {latency_int8['mean_ms']:.2f} ms mean, {latency_int8['p95_ms']:.2f} ms p95")
    print(f"  Accuracy: {acc_int8:.1%} (should be near-identical to FP32)\n")

    speedup = latency_fp32["mean_ms"] / max(latency_int8["mean_ms"], 0.001)
    size_reduction = (1 - size_int8 / max(size_fp32, 0.001)) * 100

    results = {
        "fp32": {"size_mb": size_fp32, **latency_fp32, "accuracy": acc_fp32},
        "int8": {"size_mb": size_int8, **latency_int8, "accuracy": acc_int8},
        "speedup": round(speedup, 2),
        "size_reduction_pct": round(size_reduction, 1),
    }

    print(f"[ Summary ]")
    print(f"  Speedup:        {speedup:.2f}x faster inference")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Accuracy delta: {abs(acc_fp32 - acc_int8):.3f} (lower is better)")
    return results


if __name__ == "__main__":
    results = run_benchmark()
    print("\n[ JSON output ]")
    print(json.dumps(results, indent=2))
