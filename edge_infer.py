"""
EdgeInfer - Quantization & Edge Inference Benchmark
Demonstrates post-training optimization of a small transformer model
and benchmarks latency/accuracy tradeoffs on CPU-only hardware.
"""

import argparse
import csv
import io
import json
import math
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    import torch.nn as nn

    try:
        import torch.ao.quantization as torch_quantization
    except ImportError:
        import torch.quantization as torch_quantization

    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    torch = None
    nn = None
    torch_quantization = None
    TORCH_AVAILABLE = False
    TORCH_VERSION = None

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
DEFAULT_VARIANTS = ("fp32", "dynamic_int8", "dynamic_float16")


class SmallTextClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Lightweight transformer encoder + linear head.
    Designed to stay small enough for quick CPU-side experiments.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        num_classes: int = 2,
        max_len: int = 64,
        pad_token_id: int = PAD_TOKEN_ID,
    ):
        if TORCH_AVAILABLE:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
            self.pos_enc = nn.Embedding(max_len, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.pad_token_id = pad_token_id

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None

        mask = x.eq(self.pad_token_id)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_enc(positions)
        out = self.encoder(out, src_key_padding_mask=mask)

        valid_tokens = (~mask).unsqueeze(-1).float()
        pooled = (out * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def tokenize_text(text: str) -> list[str]:
    """Simple tokenizer for JSONL text datasets."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def read_jsonl_dataset(dataset_path: str) -> list[dict]:
    """Read JSONL rows shaped like {'text': '...', 'label': ...}."""
    records = []
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if "text" not in record or "label" not in record:
                raise ValueError(
                    f"Line {line_number} must include both 'text' and 'label' fields."
                )

            records.append({"text": str(record["text"]), "label": record["label"]})

    if len(records) < 10:
        raise ValueError("JSONL dataset must contain at least 10 labeled examples.")

    return records


def build_text_vocab(records: list[dict], vocab_limit: int = 5000) -> dict[str, int]:
    """Build a tiny whitespace-ish vocabulary for local text files."""
    counter = Counter()
    for record in records:
        counter.update(tokenize_text(record["text"]))

    vocab = {"<pad>": PAD_TOKEN_ID, "<unk>": UNK_TOKEN_ID}
    for token, _ in counter.most_common(max(vocab_limit - len(vocab), 0)):
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], seq_len: int) -> list[int]:
    tokens = tokenize_text(text)
    encoded = [vocab.get(token, UNK_TOKEN_ID) for token in tokens[:seq_len]]
    if len(encoded) < seq_len:
        encoded.extend([PAD_TOKEN_ID] * (seq_len - len(encoded)))
    return encoded


def shuffle_split(records: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    eval_count = max(1, int(len(shuffled) * eval_ratio))
    train_count = len(shuffled) - eval_count
    if train_count < 1:
        raise ValueError("Dataset split left no training examples. Add more data.")

    return shuffled[:train_count], shuffled[train_count:]


def tensorize_examples(inputs: list[list[int]], labels: list[int]):
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def generate_keyword_examples(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Generate a deterministic, learnable dataset.
    Labels are determined by whether positive or negative signal tokens dominate.
    """
    rng = random.Random(seed)
    positive_tokens = list(range(2, 20))
    negative_tokens = list(range(20, 38))
    neutral_tokens = list(range(38, max(vocab_size, 39)))

    inputs = []
    labels = []
    for _ in range(n_samples):
        label = rng.randint(0, 1)
        sequence = [rng.choice(neutral_tokens) for _ in range(seq_len)]

        signal_count = rng.randint(max(2, seq_len // 10), max(3, seq_len // 4))
        decoy_count = rng.randint(0, max(1, signal_count // 3))
        positions = list(range(seq_len))
        rng.shuffle(positions)

        signal_pool = positive_tokens if label == 1 else negative_tokens
        decoy_pool = negative_tokens if label == 1 else positive_tokens

        for pos in positions[:signal_count]:
            sequence[pos] = rng.choice(signal_pool)
        for pos in positions[signal_count:signal_count + decoy_count]:
            sequence[pos] = rng.choice(decoy_pool)

        inputs.append(sequence)
        labels.append(label)

    return inputs, labels


def build_keyword_dataset(
    seq_len: int,
    train_samples: int,
    eval_samples: int,
    vocab_size: int,
    seed: int,
) -> dict:
    train_inputs, train_labels = generate_keyword_examples(
        n_samples=train_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    eval_inputs, eval_labels = generate_keyword_examples(
        n_samples=eval_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed + 1,
    )

    return {
        "name": "keyword_signal",
        "description": "Synthetic but learnable signal-vs-noise classification dataset.",
        "vocab_size": vocab_size,
        "num_classes": 2,
        "train_examples": train_samples,
        "eval_examples": eval_samples,
        "train": tensorize_examples(train_inputs, train_labels),
        "eval": tensorize_examples(eval_inputs, eval_labels),
    }


def build_jsonl_dataset(dataset_path: str, seq_len: int, seed: int) -> dict:
    records = read_jsonl_dataset(dataset_path)
    train_records, eval_records = shuffle_split(records, eval_ratio=0.2, seed=seed)
    vocab = build_text_vocab(train_records)

    label_values = sorted({record["label"] for record in records}, key=str)
    label_to_index = {label: index for index, label in enumerate(label_values)}

    train_inputs = [encode_text(record["text"], vocab, seq_len) for record in train_records]
    train_labels = [label_to_index[record["label"]] for record in train_records]
    eval_inputs = [encode_text(record["text"], vocab, seq_len) for record in eval_records]
    eval_labels = [label_to_index[record["label"]] for record in eval_records]

    return {
        "name": f"jsonl:{Path(dataset_path).name}",
        "description": "User-provided JSONL text dataset.",
        "source_path": str(Path(dataset_path).resolve()),
        "vocab_size": len(vocab),
        "num_classes": len(label_to_index),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "label_map": {str(key): value for key, value in label_to_index.items()},
        "train": tensorize_examples(train_inputs, train_labels),
        "eval": tensorize_examples(eval_inputs, eval_labels),
    }


def iterate_batches(inputs, labels, batch_size: int, shuffle: bool, seed: int):
    indices = list(range(len(labels)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield inputs[batch_indices], labels[batch_indices]


def percentile(sorted_values: list[float], pct: float) -> float:
    index = max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1)
    return sorted_values[index]


def model_size_mb(model) -> float | None:
    """Estimate serialized model size in MB via state_dict bytes."""
    if not TORCH_AVAILABLE:
        return None

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / (1024 ** 2)


def benchmark_inference(model, inputs, n_runs: int = 50, warmup_runs: int = 10) -> dict:
    """Measure mean/median/p95 latency across repeated forward passes."""
    model.eval()
    latencies = []

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(inputs)

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    sorted_latencies = sorted(latencies)
    mean_ms = statistics.mean(latencies)

    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(statistics.median(latencies), 3),
        "stddev_ms": round(statistics.pstdev(latencies), 3),
        "p95_ms": round(percentile(sorted_latencies, 95), 3),
        "samples_per_sec": round((inputs.size(0) * 1000) / max(mean_ms, 1e-9), 2),
        "tokens_per_sec": round((inputs.numel() * 1000) / max(mean_ms, 1e-9), 2),
        "n_runs": n_runs,
        "warmup_runs": warmup_runs,
    }


def measure_accuracy(model, dataset, batch_size: int = 32) -> float:
    """Evaluate accuracy on an eval dataset."""
    inputs, labels = dataset
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch_inputs, batch_labels in iterate_batches(
            inputs,
            labels,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            predictions = model(batch_inputs).argmax(dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

    return correct / max(total, 1)


def train_model(
    model,
    train_dataset,
    eval_dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    """Quick CPU training pass so eval accuracy is meaningful before quantization."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()

    train_inputs, train_labels = train_dataset
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        example_count = 0

        for batch_inputs, batch_labels in iterate_batches(
            train_inputs,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_labels)
            example_count += len(batch_labels)

        eval_accuracy = measure_accuracy(model, eval_dataset, batch_size=batch_size)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(running_loss / max(example_count, 1), 4),
                "eval_accuracy": round(eval_accuracy, 4),
            }
        )

    return {
        "epochs": history,
        "train_seconds": round(time.perf_counter() - start, 3),
        "final_eval_accuracy": history[-1]["eval_accuracy"] if history else None,
    }


def apply_dynamic_quantization(model, variant_name: str):
    """Apply dynamic quantization with the requested weight dtype."""
    dtype_lookup = {
        "dynamic_int8": torch.qint8,
        "dynamic_float16": torch.float16,
    }
    dtype = dtype_lookup[variant_name]
    return torch_quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


def build_variant_model(model, variant_name: str):
    if variant_name == "fp32":
        return deepcopy(model)
    if variant_name in {"dynamic_int8", "dynamic_float16"}:
        return apply_dynamic_quantization(deepcopy(model), variant_name)
    raise ValueError(f"Unsupported variant: {variant_name}")


def summarize_results(results: list[dict]) -> dict:
    fp32 = next((item for item in results if item["name"] == "fp32"), None)
    fastest = min(results, key=lambda item: item["mean_ms"])
    summary = {"best_latency_variant": fastest["name"], "comparisons": {}}

    if not fp32:
        return summary

    for item in results:
        if item["name"] == "fp32":
            continue

        size_reduction = None
        if fp32["size_mb"] and item["size_mb"] is not None:
            size_reduction = round((1 - item["size_mb"] / fp32["size_mb"]) * 100, 1)

        summary["comparisons"][item["name"]] = {
            "speedup_vs_fp32": round(fp32["mean_ms"] / max(item["mean_ms"], 1e-9), 2),
            "size_reduction_vs_fp32_pct": size_reduction,
            "accuracy_delta_vs_fp32": round(abs(fp32["accuracy"] - item["accuracy"]), 4),
        }

    return summary


def build_system_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "torch_available": TORCH_AVAILABLE,
        "torch_version": TORCH_VERSION,
    }


def build_simulation_report(args) -> dict:
    print("=== EdgeInfer: Simulation Mode ===\n")
    print("[warn] No real PyTorch execution happened. Metrics below are labeled demo values.\n")

    rng = random.Random(args.seed)
    base_latency = rng.uniform(4.2, 6.0)
    base_accuracy = rng.uniform(0.88, 0.93)
    base_size = 0.52
    results = []

    for variant_name in args.variants:
        if variant_name == "fp32":
            size_mb = base_size
            mean_ms = base_latency
            accuracy = base_accuracy
        elif variant_name == "dynamic_int8":
            size_mb = round(base_size * rng.uniform(0.32, 0.42), 3)
            mean_ms = base_latency * rng.uniform(0.5, 0.7)
            accuracy = max(base_accuracy - rng.uniform(0.0, 0.01), 0.0)
        else:
            size_mb = round(base_size * rng.uniform(0.55, 0.7), 3)
            mean_ms = base_latency * rng.uniform(0.78, 0.92)
            accuracy = max(base_accuracy - rng.uniform(0.0, 0.005), 0.0)

        result = {
            "name": variant_name,
            "size_mb": round(size_mb, 3),
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(mean_ms * 0.97, 3),
            "stddev_ms": round(mean_ms * 0.05, 3),
            "p95_ms": round(mean_ms * 1.18, 3),
            "samples_per_sec": round((args.batch_size * 1000) / mean_ms, 2),
            "tokens_per_sec": round((args.batch_size * args.seq_len * 1000) / mean_ms, 2),
            "n_runs": args.n_runs,
            "warmup_runs": args.warmup_runs,
            "accuracy": round(accuracy, 4),
        }
        results.append(result)
        print_variant_summary(result)

    return {
        "mode": "simulation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": build_system_info(),
        "config": vars(args),
        "dataset": {
            "name": "simulation_only",
            "description": "Demo metrics only. No training or inference executed.",
        },
        "training": {"executed": False},
        "results": results,
        "summary": summarize_results(results),
        "variant_errors": [],
    }


def prepare_dataset(args) -> dict:
    if args.dataset_jsonl:
        return build_jsonl_dataset(args.dataset_jsonl, seq_len=args.seq_len, seed=args.seed)

    return build_keyword_dataset(
        seq_len=args.seq_len,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )


def print_variant_summary(result: dict):
    print(f"[ {result['name']} ]")
    print(f"  Size:     {result['size_mb']:.2f} MB")
    print(
        f"  Latency:  {result['mean_ms']:.2f} ms mean, "
        f"{result['median_ms']:.2f} ms median, {result['p95_ms']:.2f} ms p95"
    )
    print(f"  Throughput: {result['samples_per_sec']:.2f} samples/s")
    print(f"  Accuracy: {result['accuracy']:.1%}\n")


def write_report_files(report: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"edgeinfer_{timestamp}.json"
    csv_path = output_path / f"edgeinfer_{timestamp}.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_fields = [
        "name",
        "size_mb",
        "mean_ms",
        "median_ms",
        "stddev_ms",
        "p95_ms",
        "samples_per_sec",
        "tokens_per_sec",
        "accuracy",
        "n_runs",
        "warmup_runs",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in report["results"]:
            writer.writerow({field: result.get(field) for field in csv_fields})

    return {
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve()),
    }


def run_real_benchmark(args) -> dict:
    print("=== EdgeInfer: Quantization & Edge Inference Benchmark ===\n")
    torch.manual_seed(args.seed)

    dataset = prepare_dataset(args)
    print(f"[ Dataset ] {dataset['name']}")
    print(f"  Train examples: {dataset['train_examples']}")
    print(f"  Eval examples:  {dataset['eval_examples']}")
    print(f"  Vocab size:     {dataset['vocab_size']}")
    print(f"  Classes:        {dataset['num_classes']}\n")

    model = SmallTextClassifier(
        vocab_size=dataset["vocab_size"],
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        num_classes=dataset["num_classes"],
        max_len=args.seq_len,
        pad_token_id=PAD_TOKEN_ID,
    )

    print("[ Training ]")
    training = train_model(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    for epoch_entry in training["epochs"]:
        print(
            f"  Epoch {epoch_entry['epoch']}: "
            f"loss={epoch_entry['loss']:.4f} "
            f"eval_acc={epoch_entry['eval_accuracy']:.1%}"
        )
    print(f"  Train time: {training['train_seconds']:.2f} s\n")

    eval_inputs, _ = dataset["eval"]
    benchmark_inputs = eval_inputs[:max(1, min(args.batch_size, eval_inputs.size(0)))]

    results = []
    variant_errors = []
    for variant_name in args.variants:
        try:
            variant_model = build_variant_model(model, variant_name)
            result = {
                "name": variant_name,
                "size_mb": round(model_size_mb(variant_model), 3),
                **benchmark_inference(
                    variant_model,
                    benchmark_inputs,
                    n_runs=args.n_runs,
                    warmup_runs=args.warmup_runs,
                ),
                "accuracy": round(
                    measure_accuracy(
                        variant_model,
                        dataset["eval"],
                        batch_size=args.train_batch_size,
                    ),
                    4,
                ),
            }
            results.append(result)
            print_variant_summary(result)
        except Exception as exc:
            error = {"name": variant_name, "error": str(exc)}
            variant_errors.append(error)
            print(f"[warn] Skipping {variant_name}: {exc}\n")

    if not results:
        raise RuntimeError("All benchmark variants failed.")

    return {
        "mode": "benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": build_system_info(),
        "config": vars(args),
        "dataset": {
            key: value
            for key, value in dataset.items()
            if key not in {"train", "eval"}
        },
        "training": training,
        "results": results,
        "summary": summarize_results(results),
        "variant_errors": variant_errors,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a small text classifier for edge-oriented CPU inference."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in clearly labeled demo mode when PyTorch is unavailable.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        help="Optional JSONL dataset with {'text': '...', 'label': ...} records.",
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length used for training and inference.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size for latency benchmarking.")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size used during training and eval.")
    parser.add_argument("--train-samples", type=int, default=512, help="Training examples for the built-in dataset.")
    parser.add_argument("--eval-samples", type=int, default=128, help="Eval examples for the built-in dataset.")
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs before benchmarking.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Learning rate for the quick training pass.")
    parser.add_argument("--n-runs", type=int, default=50, help="Measured inference runs per variant.")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Warmup forward passes before timing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help="Variant(s) to benchmark.",
    )
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON and CSV benchmark reports.")
    parser.add_argument("--skip-report", action="store_true", help="Skip writing JSON/CSV artifacts.")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size for the built-in dataset.")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension for the transformer.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--ff-dim", type=int, default=128, help="Feed-forward hidden dimension.")
    return parser


def validate_args(args):
    if args.seq_len < 4:
        raise SystemExit("--seq-len must be at least 4.")
    if args.batch_size < 1 or args.train_batch_size < 1:
        raise SystemExit("Batch sizes must be at least 1.")
    if args.num_heads < 1:
        raise SystemExit("--num-heads must be at least 1.")
    if args.train_epochs < 1:
        raise SystemExit("--train-epochs must be at least 1.")
    if args.embed_dim < 4 or args.ff_dim < 4:
        raise SystemExit("Model dimensions must be at least 4.")
    if args.embed_dim % args.num_heads != 0:
        raise SystemExit("--embed-dim must be divisible by --num-heads.")
    if not args.dataset_jsonl and (args.train_samples < 1 or args.eval_samples < 1):
        raise SystemExit("Training and eval sample counts must be at least 1.")
    if not args.dataset_jsonl and args.vocab_size < 64:
        raise SystemExit("--vocab-size must be at least 64 for the built-in dataset.")
    if args.dataset_jsonl and not Path(args.dataset_jsonl).exists():
        raise SystemExit(f"Dataset file not found: {args.dataset_jsonl}")
    if not TORCH_AVAILABLE and not args.simulate:
        raise SystemExit(
            "PyTorch is required for real benchmarks. Install it or rerun with --simulate "
            "for clearly labeled demo metrics."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    report = build_simulation_report(args) if args.simulate else run_real_benchmark(args)

    if not args.skip_report:
        report["artifacts"] = write_report_files(report, args.output_dir)
        print("[ Artifacts ]")
        print(f"  JSON: {report['artifacts']['json_path']}")
        print(f"  CSV:  {report['artifacts']['csv_path']}\n")

    print("[ Summary ]")
    print(f"  Best latency variant: {report['summary']['best_latency_variant']}")
    for name, comparison in report["summary"]["comparisons"].items():
        size_reduction_text = (
            "n/a"
            if comparison["size_reduction_vs_fp32_pct"] is None
            else f"{comparison['size_reduction_vs_fp32_pct']}%"
        )
        print(
            f"  {name}: "
            f"{comparison['speedup_vs_fp32']:.2f}x speedup, "
            f"{size_reduction_text} size reduction, "
            f"{comparison['accuracy_delta_vs_fp32']:.4f} accuracy delta"
        )

    print("\n[ JSON output ]")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "summary": report["summary"],
                "artifacts": report.get("artifacts"),
                "variant_errors": report["variant_errors"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
