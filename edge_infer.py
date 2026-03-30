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
import operator
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
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

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
    ONNXRUNTIME_VERSION = ort.__version__
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False
    ONNXRUNTIME_VERSION = None

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
DEFAULT_VARIANTS = ("fp32", "dynamic_int8", "dynamic_float16", "static_int8")
EXPORT_FORMATS = ("onnx",)
ONNX_SUPPORTED_EXPORT_VARIANTS = ("fp32",)
DEFAULT_COMMAND = "benchmark"


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
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
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

    if len(records) < 12:
        raise ValueError("JSONL dataset must contain at least 12 labeled examples.")

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


def build_label_metadata(label_values) -> tuple[dict, list[str]]:
    label_values = list(label_values)
    label_to_index = {label: index for index, label in enumerate(label_values)}
    index_to_label = [str(label) for label in label_values]
    return label_to_index, index_to_label


def allocate_split_counts(total: int, ratios: dict[str, float]) -> tuple[int, dict[str, int]]:
    """Allocate non-train split counts and leave the remainder for training."""
    if total < 4:
        raise ValueError("Need at least 4 examples to create train/validation/test splits.")

    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(raw_counts[name]) for name in ratios}
    remaining_for_non_train = max(total - 1 - sum(counts.values()), 0)
    fractional_parts = sorted(
        ((raw_counts[name] - counts[name], name) for name in ratios),
        reverse=True,
    )

    for _, name in fractional_parts:
        if remaining_for_non_train <= 0:
            break
        counts[name] += 1
        remaining_for_non_train -= 1

    desired_non_zero = [name for name, ratio in ratios.items() if ratio > 0]
    if total >= len(desired_non_zero) + 1:
        for name in desired_non_zero:
            if counts[name] > 0:
                continue

            donor = max(
                (split_name for split_name in counts if counts[split_name] > 1),
                key=counts.get,
                default=None,
            )
            if donor is not None:
                counts[donor] -= 1
                counts[name] += 1

    train_count = total - sum(counts.values())
    if train_count < 1:
        raise ValueError("Split ratios leave no training examples. Lower the non-train ratios.")

    return train_count, counts


def split_records(
    records: list[dict],
    calibration_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    ratios = {
        "calibration": calibration_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    if any(ratio < 0 for ratio in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    if sum(ratios.values()) >= 1.0:
        raise ValueError("Calibration, validation, and test ratios must sum to less than 1.")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_count, counts = allocate_split_counts(len(shuffled), ratios)
    calibration_end = train_count + counts["calibration"]
    validation_end = calibration_end + counts["validation"]

    return {
        "train": shuffled[:train_count],
        "calibration": shuffled[train_count:calibration_end],
        "validation": shuffled[calibration_end:validation_end],
        "test": shuffled[validation_end:],
    }


def tensorize_examples(inputs: list[list[int]], labels: list[int]):
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def encode_records(records: list[dict], vocab: dict[str, int], label_to_index: dict, seq_len: int):
    inputs = [encode_text(record["text"], vocab, seq_len) for record in records]
    labels = [label_to_index[record["label"]] for record in records]
    return tensorize_examples(inputs, labels)


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
    calibration_samples: int,
    validation_samples: int,
    test_samples: int,
    vocab_size: int,
    seed: int,
) -> dict:
    label_map = {"negative_signal": 0, "positive_signal": 1}
    index_to_label = ["negative_signal", "positive_signal"]

    split_specs = {
        "train": (train_samples, seed),
        "calibration": (calibration_samples, seed + 1),
        "validation": (validation_samples, seed + 2),
        "test": (test_samples, seed + 3),
    }

    tensor_splits = {}
    split_counts = {}
    for split_name, (sample_count, split_seed) in split_specs.items():
        split_inputs, split_labels = generate_keyword_examples(
            n_samples=sample_count,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=split_seed,
        )
        tensor_splits[split_name] = tensorize_examples(split_inputs, split_labels)
        split_counts[split_name] = sample_count

    return {
        "name": "keyword_signal",
        "description": "Synthetic but learnable signal-vs-noise classification dataset.",
        "source_path": None,
        "vocab_size": vocab_size,
        "num_classes": 2,
        "label_map": label_map,
        "index_to_label": index_to_label,
        "split_counts": split_counts,
        **tensor_splits,
    }


def build_jsonl_dataset(
    dataset_path: str,
    seq_len: int,
    calibration_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    records = read_jsonl_dataset(dataset_path)
    record_splits = split_records(
        records,
        calibration_ratio=calibration_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    vocab = build_text_vocab(record_splits["train"])

    label_values = sorted({record["label"] for record in records}, key=str)
    label_to_index, index_to_label = build_label_metadata(label_values)

    tensor_splits = {
        split_name: encode_records(split_records_, vocab, label_to_index, seq_len)
        for split_name, split_records_ in record_splits.items()
    }

    return {
        "name": f"jsonl:{Path(dataset_path).name}",
        "description": "User-provided JSONL text dataset.",
        "source_path": str(Path(dataset_path).resolve()),
        "vocab_size": len(vocab),
        "num_classes": len(label_to_index),
        "label_map": {str(key): value for key, value in label_to_index.items()},
        "index_to_label": index_to_label,
        "split_counts": {split_name: len(split_records_) for split_name, split_records_ in record_splits.items()},
        **tensor_splits,
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


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_timing_metrics(
    latencies_ms: list[float],
    batch_size: int,
    token_count: int,
    n_runs: int,
    warmup_runs: int,
) -> dict:
    sorted_latencies = sorted(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "stddev_ms": round(statistics.pstdev(latencies_ms), 3),
        "p95_ms": round(percentile(sorted_latencies, 95), 3),
        "samples_per_sec": round((batch_size * 1000) / max(mean_ms, 1e-9), 2),
        "tokens_per_sec": round((token_count * 1000) / max(mean_ms, 1e-9), 2),
        "n_runs": n_runs,
        "warmup_runs": warmup_runs,
    }


def select_quantized_engine() -> str:
    supported_engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    for candidate in ("x86", "fbgemm", "qnnpack"):
        if candidate in supported_engines:
            return candidate

    current_engine = getattr(torch.backends.quantized, "engine", "")
    if current_engine and str(current_engine).lower() not in {"none", "noqengine"}:
        return current_engine
    raise RuntimeError("No supported quantized backend found for static quantization.")


def has_restorable_quantized_engine(engine) -> bool:
    return engine is not None and str(engine).lower() not in {"none", "noqengine", ""}


@contextmanager
def quantized_engine_context(engine: str | None):
    original_engine = getattr(torch.backends.quantized, "engine", None) if TORCH_AVAILABLE else None
    try:
        if TORCH_AVAILABLE and engine:
            torch.backends.quantized.engine = engine
        yield
    finally:
        if TORCH_AVAILABLE and has_restorable_quantized_engine(original_engine):
            torch.backends.quantized.engine = original_engine


@contextmanager
def mha_fastpath_context(disabled: bool):
    if not TORCH_AVAILABLE or not hasattr(torch.backends, "mha") or not hasattr(torch.backends.mha, "get_fastpath_enabled"):
        yield
        return

    original_state = torch.backends.mha.get_fastpath_enabled()
    try:
        if disabled:
            torch.backends.mha.set_fastpath_enabled(False)
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(original_state)


def calibrate_model(model, calibration_dataset, batch_size: int):
    calibration_inputs, calibration_labels = calibration_dataset
    if len(calibration_labels) < 1:
        raise RuntimeError("Static quantization needs at least one calibration example.")

    model.eval()
    with torch.inference_mode():
        for batch_inputs, _ in iterate_batches(
            calibration_inputs,
            calibration_labels,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            _ = model(batch_inputs)


def build_static_qconfig_mapping(backend: str):
    """
    Skip observers on bool- and shape-only ops used for padding masks.
    FX static quantization otherwise tries to histogram a bool tensor, which fails
    during calibration on the transformer masking path.
    """
    qconfig_mapping = torch_quantization.get_default_qconfig_mapping(backend)
    for method_name in ("eq", "unsqueeze", "float", "sum", "clamp"):
        qconfig_mapping = qconfig_mapping.set_object_type(method_name, None)
    return qconfig_mapping.set_object_type(operator.invert, None)


def model_size_mb(model) -> float | None:
    """Estimate serialized model size in MB via state_dict bytes."""
    if not TORCH_AVAILABLE:
        return None

    buffer = io.BytesIO()
    with quantized_engine_context(getattr(model, "_edgeinfer_quantized_engine", None)):
        torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / (1024 ** 2)


def benchmark_inference(model, inputs, n_runs: int = 50, warmup_runs: int = 10) -> dict:
    """Measure mean/median/p95 latency across repeated forward passes."""
    model.eval()
    latencies = []

    with quantized_engine_context(getattr(model, "_edgeinfer_quantized_engine", None)):
        with mha_fastpath_context(getattr(model, "_edgeinfer_disable_mha_fastpath", False)):
            with torch.inference_mode():
                for _ in range(warmup_runs):
                    _ = model(inputs)

                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = model(inputs)
                    latencies.append((time.perf_counter() - start) * 1000)

    return build_timing_metrics(
        latencies_ms=latencies,
        batch_size=inputs.size(0),
        token_count=inputs.numel(),
        n_runs=n_runs,
        warmup_runs=warmup_runs,
    )


def collect_predictions(model, dataset, batch_size: int = 32) -> tuple[list[int], list[int]]:
    inputs, labels = dataset
    predictions = []
    targets = []

    model.eval()
    with quantized_engine_context(getattr(model, "_edgeinfer_quantized_engine", None)):
        with mha_fastpath_context(getattr(model, "_edgeinfer_disable_mha_fastpath", False)):
            with torch.inference_mode():
                for batch_inputs, batch_labels in iterate_batches(
                    inputs,
                    labels,
                    batch_size=batch_size,
                    shuffle=False,
                    seed=0,
                ):
                    logits = model(batch_inputs)
                    predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                    targets.extend(batch_labels.cpu().tolist())

    return targets, predictions


def compute_classification_metrics(
    targets: list[int],
    predictions: list[int],
    index_to_label: list[str] | None = None,
) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same length.")
    if not targets:
        raise ValueError("Need at least one prediction to compute metrics.")

    inferred_classes = max(max(targets), max(predictions)) + 1
    if index_to_label is None:
        index_to_label = [f"class_{idx}" for idx in range(inferred_classes)]
    num_classes = max(len(index_to_label), inferred_classes)
    if len(index_to_label) < num_classes:
        index_to_label = list(index_to_label) + [
            f"class_{idx}" for idx in range(len(index_to_label), num_classes)
        ]

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        confusion_matrix[target][prediction] += 1

    per_class = []
    for class_index in range(num_classes):
        tp = confusion_matrix[class_index][class_index]
        fp = sum(confusion_matrix[row][class_index] for row in range(num_classes) if row != class_index)
        fn = sum(confusion_matrix[class_index][col] for col in range(num_classes) if col != class_index)
        support = sum(confusion_matrix[class_index])

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_class.append(
            {
                "label": index_to_label[class_index],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )

    accuracy = safe_divide(sum(int(target == prediction) for target, prediction in zip(targets, predictions)), len(targets))
    macro_precision = statistics.mean(class_metrics["precision"] for class_metrics in per_class)
    macro_recall = statistics.mean(class_metrics["recall"] for class_metrics in per_class)
    macro_f1 = statistics.mean(class_metrics["f1"] for class_metrics in per_class)
    weighted_f1 = safe_divide(
        sum(class_metrics["f1"] * class_metrics["support"] for class_metrics in per_class),
        len(targets),
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(macro_recall, 4),
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
        "support": len(targets),
    }


def evaluate_model(model, dataset, batch_size: int = 32, index_to_label: list[str] | None = None) -> dict:
    targets, predictions = collect_predictions(model, dataset, batch_size=batch_size)
    return compute_classification_metrics(targets, predictions, index_to_label=index_to_label)


def train_model(
    model,
    train_dataset,
    validation_dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    index_to_label: list[str],
) -> dict:
    """Quick CPU training pass so held-out metrics are meaningful before quantization."""
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

        validation_metrics = evaluate_model(
            model,
            validation_dataset,
            batch_size=batch_size,
            index_to_label=index_to_label,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(running_loss / max(example_count, 1), 4),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )

    return {
        "epochs": history,
        "train_seconds": round(time.perf_counter() - start, 3),
        "final_validation_accuracy": history[-1]["validation_accuracy"] if history else None,
        "final_validation_macro_f1": history[-1]["validation_macro_f1"] if history else None,
    }


def apply_dynamic_quantization(model, variant_name: str):
    """Apply dynamic quantization with the requested weight dtype."""
    dtype_lookup = {
        "dynamic_int8": torch.qint8,
        "dynamic_float16": torch.float16,
    }
    dtype = dtype_lookup[variant_name]
    original_engine = getattr(torch.backends.quantized, "engine", None)
    selected_engine = select_quantized_engine()
    try:
        if has_restorable_quantized_engine(original_engine) or selected_engine:
            torch.backends.quantized.engine = selected_engine
        quantized_model = torch_quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)
        setattr(quantized_model, "_edgeinfer_quantized_engine", selected_engine)
        setattr(quantized_model, "_edgeinfer_disable_mha_fastpath", True)
        return quantized_model
    except Exception as exc:
        raise RuntimeError(
            f"Dynamic quantization failed with engine '{selected_engine}': {exc}"
        ) from exc
    finally:
        if has_restorable_quantized_engine(original_engine):
            torch.backends.quantized.engine = original_engine


def apply_static_quantization(model, calibration_dataset, batch_size: int):
    """Apply post-training static quantization using the calibration split."""
    backend = select_quantized_engine()
    original_engine = getattr(torch.backends.quantized, "engine", None)

    try:
        if has_restorable_quantized_engine(original_engine) or backend:
            torch.backends.quantized.engine = backend

        if not hasattr(torch_quantization, "get_default_qconfig_mapping"):
            raise RuntimeError("Installed PyTorch build does not expose static FX qconfig mappings.")

        try:
            from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        except ImportError as exc:
            raise RuntimeError("Static FX quantization is unavailable in this PyTorch build.") from exc

        calibration_inputs, _ = calibration_dataset
        example_inputs = (calibration_inputs[:1],)
        qconfig_mapping = build_static_qconfig_mapping(backend)
        prepared = prepare_fx(deepcopy(model).eval(), qconfig_mapping, example_inputs)
        calibrate_model(prepared, calibration_dataset, batch_size=batch_size)
        converted = convert_fx(prepared)
        setattr(converted, "_edgeinfer_quantized_engine", backend)
        setattr(converted, "_edgeinfer_disable_mha_fastpath", True)
        return converted
    except Exception as exc:
        raise RuntimeError(f"Static quantization failed: {exc}") from exc
    finally:
        if has_restorable_quantized_engine(original_engine):
            torch.backends.quantized.engine = original_engine


def build_variant_model(model, variant_name: str, calibration_dataset=None, calibration_batch_size: int = 32):
    if variant_name == "fp32":
        return deepcopy(model)
    if variant_name in {"dynamic_int8", "dynamic_float16"}:
        return apply_dynamic_quantization(deepcopy(model), variant_name)
    if variant_name == "static_int8":
        if calibration_dataset is None:
            raise ValueError("Static quantization requires a calibration dataset.")
        return apply_static_quantization(
            deepcopy(model),
            calibration_dataset=calibration_dataset,
            batch_size=calibration_batch_size,
        )
    raise ValueError(f"Unsupported variant: {variant_name}")


def export_model_to_onnx(
    model,
    sample_inputs,
    output_dir: str,
    variant_name: str,
    export_opset: int,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"edgeinfer_export_{unique_timestamp_token()}_{variant_name}.onnx"

    model.eval()
    with mha_fastpath_context(True):
        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample_inputs,
                str(artifact_path),
                input_names=["input_ids"],
                output_names=["logits"],
                opset_version=export_opset,
                dynamo=False,
            )

    return str(artifact_path.resolve())


def validate_export_variant_support(format_name: str, source_variant: str):
    if format_name == "onnx" and source_variant not in ONNX_SUPPORTED_EXPORT_VARIANTS:
        supported_text = ", ".join(ONNX_SUPPORTED_EXPORT_VARIANTS)
        raise RuntimeError(
            "ONNX export currently supports only these eager variants for this transformer benchmark: "
            f"{supported_text}. Requested: {source_variant}."
        )


def benchmark_onnx_runtime(onnx_path: str, inputs, n_runs: int, warmup_runs: int) -> dict:
    if not ONNXRUNTIME_AVAILABLE:
        raise RuntimeError("onnxruntime is not installed.")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    numpy_inputs = inputs.cpu().numpy()
    latencies = []

    for _ in range(warmup_runs):
        session.run(None, {input_name: numpy_inputs})

    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: numpy_inputs})
        latencies.append((time.perf_counter() - start) * 1000)

    return build_timing_metrics(
        latencies_ms=latencies,
        batch_size=int(numpy_inputs.shape[0]),
        token_count=int(numpy_inputs.size),
        n_runs=n_runs,
        warmup_runs=warmup_runs,
    )


def collect_onnx_predictions(onnx_path: str, dataset, batch_size: int = 32) -> tuple[list[int], list[int]]:
    if not ONNXRUNTIME_AVAILABLE:
        raise RuntimeError("onnxruntime is not installed.")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    inputs, labels = dataset
    predictions = []
    targets = []

    for batch_inputs, batch_labels in iterate_batches(
        inputs,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        logits = session.run(None, {input_name: batch_inputs.cpu().numpy()})[0]
        predictions.extend(logits.argmax(axis=-1).tolist())
        targets.extend(batch_labels.cpu().tolist())

    return targets, predictions


def build_result_record(name: str, size_mb, inference_metrics: dict, evaluation_metrics: dict) -> dict:
    return {
        "name": name,
        "size_mb": None if size_mb is None else round(size_mb, 3),
        **inference_metrics,
        "accuracy": evaluation_metrics["accuracy"],
        "macro_precision": evaluation_metrics["macro_precision"],
        "macro_recall": evaluation_metrics["macro_recall"],
        "macro_f1": evaluation_metrics["macro_f1"],
        "weighted_f1": evaluation_metrics["weighted_f1"],
        "balanced_accuracy": evaluation_metrics["balanced_accuracy"],
        "metrics": evaluation_metrics,
    }


def build_export_result(
    format_name: str,
    source_variant: str,
    artifact_path: str | None,
    inference_metrics: dict,
    evaluation_metrics: dict,
    simulated: bool,
) -> dict:
    return {
        "name": f"{format_name}:{source_variant}",
        "format": format_name,
        "source_variant": source_variant,
        "artifact_path": artifact_path,
        "simulated": simulated,
        **inference_metrics,
        "accuracy": evaluation_metrics["accuracy"],
        "macro_precision": evaluation_metrics["macro_precision"],
        "macro_recall": evaluation_metrics["macro_recall"],
        "macro_f1": evaluation_metrics["macro_f1"],
        "weighted_f1": evaluation_metrics["weighted_f1"],
        "balanced_accuracy": evaluation_metrics["balanced_accuracy"],
        "metrics": evaluation_metrics,
    }


def summarize_results(results: list[dict]) -> dict:
    fp32 = next((item for item in results if item["name"] == "fp32"), None)
    fastest = min(results, key=lambda item: item["mean_ms"])
    best_quality = max(results, key=lambda item: item.get("macro_f1", -1))
    summary = {
        "best_latency_variant": fastest["name"],
        "best_quality_variant": best_quality["name"],
        "comparisons": {},
    }

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
            "accuracy_delta_vs_fp32": round(item["accuracy"] - fp32["accuracy"], 4),
            "macro_f1_delta_vs_fp32": round(item["macro_f1"] - fp32["macro_f1"], 4),
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
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "onnxruntime_version": ONNXRUNTIME_VERSION,
    }


def unique_timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def simulated_metrics_from_accuracy(accuracy: float, index_to_label: list[str], rng: random.Random) -> dict:
    total = 1000
    correct = int(round(total * accuracy))
    errors = total - correct
    class_count = len(index_to_label)
    base_support = total // class_count
    support_counts = [base_support for _ in range(class_count)]
    support_counts[-1] += total - sum(support_counts)

    confusion_matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    remaining_errors = errors
    for class_index in range(class_count):
        class_errors = min(remaining_errors, max(0, support_counts[class_index] // 4))
        confusion_matrix[class_index][class_index] = support_counts[class_index] - class_errors
        if class_errors and class_count > 1:
            target_prediction = (class_index + 1 + rng.randint(0, class_count - 2)) % class_count
            confusion_matrix[class_index][target_prediction] = class_errors
        remaining_errors -= class_errors

    if remaining_errors > 0:
        confusion_matrix[0][0] -= remaining_errors
        confusion_matrix[0][1 % class_count] += remaining_errors

    targets = []
    predictions = []
    for target_index, row in enumerate(confusion_matrix):
        for prediction_index, count in enumerate(row):
            targets.extend([target_index] * count)
            predictions.extend([prediction_index] * count)

    return compute_classification_metrics(targets, predictions, index_to_label=index_to_label)


def build_simulated_export_results(args, benchmark_results: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed + 10_000)
    exports = []
    export_errors = []
    results_by_name = {result["name"]: result for result in benchmark_results}

    for format_name in args.export_formats:
        for source_variant in args.export_variants:
            source_result = results_by_name.get(source_variant)
            if source_result is None:
                export_errors.append(
                    {
                        "format": format_name,
                        "source_variant": source_variant,
                        "error": "Source variant was not benchmarked successfully.",
                    }
                )
                continue

            mean_ms = source_result["mean_ms"] * rng.uniform(0.82, 0.96)
            inference_metrics = {
                "mean_ms": round(mean_ms, 3),
                "median_ms": round(mean_ms * 0.98, 3),
                "stddev_ms": round(mean_ms * 0.04, 3),
                "p95_ms": round(mean_ms * 1.15, 3),
                "samples_per_sec": round((args.batch_size * 1000) / mean_ms, 2),
                "tokens_per_sec": round((args.batch_size * args.seq_len * 1000) / mean_ms, 2),
                "n_runs": args.n_runs,
                "warmup_runs": args.warmup_runs,
            }
            evaluation_metrics = source_result["metrics"]
            exports.append(
                build_export_result(
                    format_name=format_name,
                    source_variant=source_variant,
                    artifact_path=None,
                    inference_metrics=inference_metrics,
                    evaluation_metrics=evaluation_metrics,
                    simulated=True,
                )
            )

    return exports, export_errors


def build_simulation_report(args) -> dict:
    print("=== EdgeInfer: Simulation Mode ===\n")
    print("[warn] No real PyTorch execution happened. Metrics below are labeled demo values.\n")

    rng = random.Random(args.seed)
    base_latency = rng.uniform(4.2, 6.0)
    base_accuracy = rng.uniform(0.88, 0.93)
    base_size = 0.52
    index_to_label = ["negative_signal", "positive_signal"]
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
        elif variant_name == "static_int8":
            size_mb = round(base_size * rng.uniform(0.28, 0.38), 3)
            mean_ms = base_latency * rng.uniform(0.45, 0.62)
            accuracy = max(base_accuracy - rng.uniform(0.0, 0.015), 0.0)
        else:
            size_mb = round(base_size * rng.uniform(0.55, 0.7), 3)
            mean_ms = base_latency * rng.uniform(0.78, 0.92)
            accuracy = max(base_accuracy - rng.uniform(0.0, 0.005), 0.0)

        inference_metrics = {
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(mean_ms * 0.97, 3),
            "stddev_ms": round(mean_ms * 0.05, 3),
            "p95_ms": round(mean_ms * 1.18, 3),
            "samples_per_sec": round((args.batch_size * 1000) / mean_ms, 2),
            "tokens_per_sec": round((args.batch_size * args.seq_len * 1000) / mean_ms, 2),
            "n_runs": args.n_runs,
            "warmup_runs": args.warmup_runs,
        }
        evaluation_metrics = simulated_metrics_from_accuracy(accuracy, index_to_label, rng)
        result = build_result_record(variant_name, size_mb, inference_metrics, evaluation_metrics)
        results.append(result)
        print_variant_summary(result)

    exports, export_errors = build_simulated_export_results(args, results)
    return {
        "mode": "simulation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": build_system_info(),
        "command": "benchmark",
        "config": vars(args),
        "dataset": {
            "name": "simulation_only",
            "description": "Demo metrics only. No training or inference executed.",
            "label_map": {"negative_signal": 0, "positive_signal": 1},
            "index_to_label": index_to_label,
            "split_counts": {
                "train": args.train_samples,
                "calibration": args.calibration_samples,
                "validation": args.validation_samples,
                "test": args.test_samples,
            },
        },
        "training": {"executed": False},
        "results": results,
        "exports": exports,
        "summary": summarize_results(results),
        "variant_errors": [],
        "export_errors": export_errors,
    }


def prepare_dataset(args) -> dict:
    if args.dataset_jsonl:
        return build_jsonl_dataset(
            args.dataset_jsonl,
            seq_len=args.seq_len,
            calibration_ratio=args.calibration_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    return build_keyword_dataset(
        seq_len=args.seq_len,
        train_samples=args.train_samples,
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
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
    print(
        f"  Quality:  acc={result['accuracy']:.1%}, "
        f"macro_f1={result['macro_f1']:.1%}, weighted_f1={result['weighted_f1']:.1%}\n"
    )


def print_export_summary(export_result: dict):
    print(f"[ export {export_result['format']} from {export_result['source_variant']} ]")
    artifact_text = export_result["artifact_path"] or "simulation-only"
    print(f"  Artifact:  {artifact_text}")
    print(
        f"  Runtime:   {export_result['mean_ms']:.2f} ms mean, "
        f"{export_result['median_ms']:.2f} ms median, {export_result['p95_ms']:.2f} ms p95"
    )
    print(
        f"  Quality:   acc={export_result['accuracy']:.1%}, "
        f"macro_f1={export_result['macro_f1']:.1%}\n"
    )


def dataset_report_metadata(dataset: dict) -> dict:
    return {
        "name": dataset["name"],
        "description": dataset["description"],
        "source_path": dataset["source_path"],
        "vocab_size": dataset["vocab_size"],
        "num_classes": dataset["num_classes"],
        "label_map": dataset["label_map"],
        "index_to_label": dataset["index_to_label"],
        "split_counts": dataset["split_counts"],
        "benchmark_split": "test",
        "model_selection_split": "validation",
        "calibration_split": "calibration",
    }


def write_report_files(report: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = unique_timestamp_token()
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
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
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


def run_export_benchmarks(
    args,
    variant_models: dict[str, object],
    benchmark_results: list[dict],
    dataset: dict,
    benchmark_inputs,
) -> tuple[list[dict], list[dict]]:
    exports = []
    export_errors = []
    benchmark_results_by_name = {result["name"]: result for result in benchmark_results}

    for format_name in args.export_formats:
        for source_variant in args.export_variants:
            variant_model = variant_models.get(source_variant)
            source_result = benchmark_results_by_name.get(source_variant)
            if variant_model is None or source_result is None:
                export_errors.append(
                    {
                        "format": format_name,
                        "source_variant": source_variant,
                        "error": "Source variant was not benchmarked successfully.",
                    }
                )
                continue

            artifact_path = None
            try:
                if format_name != "onnx":
                    raise RuntimeError(f"Unsupported export format: {format_name}")

                validate_export_variant_support(format_name, source_variant)
                artifact_path = export_model_to_onnx(
                    variant_model,
                    benchmark_inputs,
                    output_dir=args.output_dir,
                    variant_name=source_variant,
                    export_opset=args.export_opset,
                )
                inference_metrics = benchmark_onnx_runtime(
                    artifact_path,
                    benchmark_inputs,
                    n_runs=args.n_runs,
                    warmup_runs=args.warmup_runs,
                )
                targets, predictions = collect_onnx_predictions(
                    artifact_path,
                    dataset["test"],
                    batch_size=int(benchmark_inputs.size(0)),
                )
                evaluation_metrics = compute_classification_metrics(
                    targets,
                    predictions,
                    index_to_label=dataset["index_to_label"],
                )
                export_result = build_export_result(
                    format_name=format_name,
                    source_variant=source_variant,
                    artifact_path=artifact_path,
                    inference_metrics=inference_metrics,
                    evaluation_metrics=evaluation_metrics,
                    simulated=False,
                )
                exports.append(export_result)
                print_export_summary(export_result)
            except Exception as exc:
                error = {
                    "format": format_name,
                    "source_variant": source_variant,
                    "error": str(exc),
                    "artifact_path": artifact_path,
                }
                export_errors.append(error)
                print(f"[warn] Skipping export {format_name}:{source_variant}: {exc}\n")

    return exports, export_errors


def run_real_benchmark(args) -> dict:
    print("=== EdgeInfer: Quantization & Edge Inference Benchmark ===\n")
    torch.manual_seed(args.seed)

    dataset = prepare_dataset(args)
    print(f"[ Dataset ] {dataset['name']}")
    print(f"  Train examples:       {dataset['split_counts']['train']}")
    print(f"  Calibration examples: {dataset['split_counts']['calibration']}")
    print(f"  Validation examples:  {dataset['split_counts']['validation']}")
    print(f"  Test examples:        {dataset['split_counts']['test']}")
    print(f"  Vocab size:           {dataset['vocab_size']}")
    print(f"  Classes:              {dataset['num_classes']}\n")

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
        validation_dataset=dataset["validation"],
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        index_to_label=dataset["index_to_label"],
    )
    for epoch_entry in training["epochs"]:
        print(
            f"  Epoch {epoch_entry['epoch']}: "
            f"loss={epoch_entry['loss']:.4f} "
            f"val_acc={epoch_entry['validation_accuracy']:.1%} "
            f"val_macro_f1={epoch_entry['validation_macro_f1']:.1%}"
        )
    print(f"  Train time: {training['train_seconds']:.2f} s\n")

    benchmark_inputs, _ = dataset["test"]
    benchmark_inputs = benchmark_inputs[:max(1, min(args.batch_size, benchmark_inputs.size(0)))]

    results = []
    variant_models = {}
    variant_errors = []
    for variant_name in args.variants:
        try:
            variant_model = build_variant_model(
                model,
                variant_name,
                calibration_dataset=dataset["calibration"],
                calibration_batch_size=args.train_batch_size,
            )
            inference_metrics = benchmark_inference(
                variant_model,
                benchmark_inputs,
                n_runs=args.n_runs,
                warmup_runs=args.warmup_runs,
            )
            evaluation_metrics = evaluate_model(
                variant_model,
                dataset["test"],
                batch_size=args.train_batch_size,
                index_to_label=dataset["index_to_label"],
            )
            result = build_result_record(
                variant_name,
                model_size_mb(variant_model),
                inference_metrics,
                evaluation_metrics,
            )
            results.append(result)
            variant_models[variant_name] = variant_model
            print_variant_summary(result)
        except Exception as exc:
            error = {"name": variant_name, "error": str(exc)}
            variant_errors.append(error)
            print(f"[warn] Skipping {variant_name}: {exc}\n")

    if not results:
        raise RuntimeError("All benchmark variants failed.")

    exports, export_errors = run_export_benchmarks(
        args,
        variant_models=variant_models,
        benchmark_results=results,
        dataset=dataset,
        benchmark_inputs=benchmark_inputs,
    )

    return {
        "mode": "benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": build_system_info(),
        "command": "benchmark",
        "config": vars(args),
        "dataset": dataset_report_metadata(dataset),
        "training": training,
        "results": results,
        "exports": exports,
        "summary": summarize_results(results),
        "variant_errors": variant_errors,
        "export_errors": export_errors,
    }


def load_json_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_benchmark_report_files(output_dir: str) -> list[Path]:
    report_paths = []
    for path in Path(output_dir).glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if isinstance(report, dict) and "results" in report and "summary" in report:
            report_paths.append(path)

    return sorted(report_paths, key=lambda item: item.name, reverse=True)


def report_variants_by_name(report: dict) -> dict[str, dict]:
    return {result["name"]: result for result in report.get("results", [])}


def compare_reports(baseline_report: dict, candidate_report: dict) -> dict:
    baseline_variants = report_variants_by_name(baseline_report)
    candidate_variants = report_variants_by_name(candidate_report)
    shared_variants = sorted(set(baseline_variants) & set(candidate_variants))

    if not shared_variants:
        raise ValueError("Reports do not share any variant names to compare.")

    comparisons = {}
    for variant_name in shared_variants:
        baseline = baseline_variants[variant_name]
        candidate = candidate_variants[variant_name]

        latency_change_pct = None
        if baseline["mean_ms"]:
            latency_change_pct = round(((candidate["mean_ms"] / baseline["mean_ms"]) - 1) * 100, 1)

        size_change_pct = None
        if baseline["size_mb"] not in (None, 0) and candidate["size_mb"] is not None:
            size_change_pct = round(((candidate["size_mb"] / baseline["size_mb"]) - 1) * 100, 1)

        comparisons[variant_name] = {
            "latency_delta_ms": round(candidate["mean_ms"] - baseline["mean_ms"], 3),
            "latency_change_pct": latency_change_pct,
            "size_delta_mb": None if baseline["size_mb"] is None or candidate["size_mb"] is None else round(candidate["size_mb"] - baseline["size_mb"], 3),
            "size_change_pct": size_change_pct,
            "accuracy_delta": round(candidate["accuracy"] - baseline["accuracy"], 4),
            "macro_f1_delta": round(candidate["macro_f1"] - baseline["macro_f1"], 4),
            "weighted_f1_delta": round(candidate["weighted_f1"] - baseline["weighted_f1"], 4),
        }

    return {
        "mode": "comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_generated_at": baseline_report.get("generated_at"),
        "candidate_generated_at": candidate_report.get("generated_at"),
        "baseline_summary": baseline_report.get("summary"),
        "candidate_summary": candidate_report.get("summary"),
        "baseline_dataset": baseline_report.get("dataset", {}).get("name"),
        "candidate_dataset": candidate_report.get("dataset", {}).get("name"),
        "shared_variants": shared_variants,
        "variant_comparisons": comparisons,
    }


def evaluate_regression_thresholds(comparison: dict, args) -> list[dict]:
    regressions = []

    for variant_name, diff in comparison["variant_comparisons"].items():
        if args.max_latency_regression_pct is not None and diff["latency_change_pct"] is not None:
            if diff["latency_change_pct"] > args.max_latency_regression_pct:
                regressions.append(
                    {
                        "variant": variant_name,
                        "metric": "latency_change_pct",
                        "observed": diff["latency_change_pct"],
                        "threshold": args.max_latency_regression_pct,
                    }
                )

        if args.max_size_regression_pct is not None and diff["size_change_pct"] is not None:
            if diff["size_change_pct"] > args.max_size_regression_pct:
                regressions.append(
                    {
                        "variant": variant_name,
                        "metric": "size_change_pct",
                        "observed": diff["size_change_pct"],
                        "threshold": args.max_size_regression_pct,
                    }
                )

        if args.max_accuracy_drop is not None:
            accuracy_drop = -diff["accuracy_delta"]
            if accuracy_drop > args.max_accuracy_drop:
                regressions.append(
                    {
                        "variant": variant_name,
                        "metric": "accuracy_drop",
                        "observed": round(accuracy_drop, 4),
                        "threshold": args.max_accuracy_drop,
                    }
                )

        if args.max_macro_f1_drop is not None:
            macro_f1_drop = -diff["macro_f1_delta"]
            if macro_f1_drop > args.max_macro_f1_drop:
                regressions.append(
                    {
                        "variant": variant_name,
                        "metric": "macro_f1_drop",
                        "observed": round(macro_f1_drop, 4),
                        "threshold": args.max_macro_f1_drop,
                    }
                )

    return regressions


def write_comparison_report(comparison: dict, output_path: str | None, output_dir: str) -> str:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = unique_timestamp_token()
        path = output_dir_path / f"edgeinfer_compare_{timestamp}.json"

    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return str(path.resolve())


def print_comparison_summary(comparison: dict):
    print("=== EdgeInfer: Report Comparison ===\n")
    print(f"Baseline run:  {comparison['baseline_generated_at']}")
    print(f"Candidate run: {comparison['candidate_generated_at']}")
    print(f"Shared variants: {', '.join(comparison['shared_variants'])}\n")

    for variant_name in comparison["shared_variants"]:
        diff = comparison["variant_comparisons"][variant_name]
        latency_pct = "n/a" if diff["latency_change_pct"] is None else f"{diff['latency_change_pct']}%"
        size_pct = "n/a" if diff["size_change_pct"] is None else f"{diff['size_change_pct']}%"
        size_delta = "n/a" if diff["size_delta_mb"] is None else f"{diff['size_delta_mb']:+.3f} MB"
        print(f"[ {variant_name} ]")
        print(f"  Latency:    {diff['latency_delta_ms']:+.3f} ms ({latency_pct})")
        print(f"  Size:       {size_delta} ({size_pct})")
        print(f"  Accuracy:   {diff['accuracy_delta']:+.4f}")
        print(f"  Macro F1:   {diff['macro_f1_delta']:+.4f}")
        print(f"  Weighted F1:{diff['weighted_f1_delta']:+.4f}\n")


def build_history_entries(output_dir: str, limit: int) -> list[dict]:
    entries = []
    for path in get_benchmark_report_files(output_dir)[:limit]:
        report = load_json_file(str(path))
        entries.append(
            {
                "path": str(path.resolve()),
                "generated_at": report.get("generated_at"),
                "mode": report.get("mode"),
                "dataset": report.get("dataset", {}).get("name"),
                "best_latency_variant": report.get("summary", {}).get("best_latency_variant"),
                "best_quality_variant": report.get("summary", {}).get("best_quality_variant"),
                "variant_count": len(report.get("results", [])),
            }
        )
    return entries


def print_history(entries: list[dict], output_dir: str):
    print("=== EdgeInfer: Report History ===\n")
    print(f"Source directory: {Path(output_dir).resolve()}\n")

    if not entries:
        print("No benchmark reports found.\n")
        return

    for entry in entries:
        print(f"[ {entry['generated_at']} ]")
        print(f"  Dataset: {entry['dataset']}")
        print(f"  Best latency: {entry['best_latency_variant']}")
        print(f"  Best quality: {entry['best_quality_variant']}")
        print(f"  Variants: {entry['variant_count']}")
        print(f"  File: {entry['path']}\n")


def add_benchmark_arguments(parser: argparse.ArgumentParser):
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
    parser.add_argument("--calibration-samples", type=int, default=64, help="Calibration examples for the built-in dataset.")
    parser.add_argument(
        "--eval-samples",
        "--validation-samples",
        dest="validation_samples",
        type=int,
        default=128,
        help="Validation examples for the built-in dataset.",
    )
    parser.add_argument("--test-samples", type=int, default=128, help="Test examples for the built-in dataset.")
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
    parser.add_argument("--calibration-ratio", type=float, default=0.1, help="Calibration ratio for JSONL datasets.")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation ratio for JSONL datasets.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio for JSONL datasets.")
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=EXPORT_FORMATS,
        default=[],
        help="Optional export/runtime benchmarks to run after eager benchmarking.",
    )
    parser.add_argument(
        "--export-variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=["fp32"],
        help="Variant(s) to export and benchmark in the requested export format(s).",
    )
    parser.add_argument("--export-opset", type=int, default=17, help="ONNX opset version for exported models.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark and compare small text classifiers for edge-oriented CPU inference."
    )
    subparsers = parser.add_subparsers(dest="command")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Train, benchmark, and report model variants.",
    )
    add_benchmark_arguments(benchmark_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two saved benchmark reports.",
    )
    compare_parser.add_argument("--baseline", help="Path to the baseline benchmark JSON report.")
    compare_parser.add_argument("--candidate", help="Path to the candidate benchmark JSON report.")
    compare_parser.add_argument(
        "--latest",
        action="store_true",
        help="Compare the two newest benchmark reports in --output-dir.",
    )
    compare_parser.add_argument("--output-dir", default="reports", help="Directory containing benchmark reports.")
    compare_parser.add_argument("--output", help="Optional path for a saved comparison JSON report.")
    compare_parser.add_argument("--max-latency-regression-pct", type=float, help="Fail if latency increases by more than this percentage.")
    compare_parser.add_argument("--max-size-regression-pct", type=float, help="Fail if model size increases by more than this percentage.")
    compare_parser.add_argument("--max-accuracy-drop", type=float, help="Fail if accuracy drops by more than this absolute amount.")
    compare_parser.add_argument("--max-macro-f1-drop", type=float, help="Fail if macro F1 drops by more than this absolute amount.")

    history_parser = subparsers.add_parser(
        "history",
        help="List recent benchmark reports.",
    )
    history_parser.add_argument("--output-dir", default="reports", help="Directory containing benchmark reports.")
    history_parser.add_argument("--limit", type=int, default=5, help="Number of recent reports to show.")

    return parser


def parse_cli_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    known_commands = {"benchmark", "compare", "history"}
    if not raw_argv or raw_argv[0] not in known_commands:
        raw_argv = [DEFAULT_COMMAND, *raw_argv]

    parser = build_argument_parser()
    return parser.parse_args(raw_argv)


def validate_benchmark_args(args):
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
    if not args.dataset_jsonl:
        if min(args.train_samples, args.calibration_samples, args.validation_samples, args.test_samples) < 1:
            raise SystemExit("Built-in dataset split sizes must all be at least 1.")
        if args.vocab_size < 64:
            raise SystemExit("--vocab-size must be at least 64 for the built-in dataset.")
    else:
        if not Path(args.dataset_jsonl).exists():
            raise SystemExit(f"Dataset file not found: {args.dataset_jsonl}")
        if args.calibration_ratio < 0 or args.validation_ratio < 0 or args.test_ratio < 0:
            raise SystemExit("JSONL split ratios must be non-negative.")
        if args.calibration_ratio + args.validation_ratio + args.test_ratio >= 1:
            raise SystemExit("JSONL split ratios must sum to less than 1.")
    if args.export_opset < 11:
        raise SystemExit("--export-opset must be at least 11.")
    if args.export_formats:
        missing_export_variants = sorted(set(args.export_variants) - set(args.variants))
        if missing_export_variants:
            raise SystemExit(
                "Export variants must also be benchmarked variants. Missing from --variants: "
                + ", ".join(missing_export_variants)
            )
        if "onnx" in args.export_formats:
            unsupported_onnx_variants = sorted(
                set(args.export_variants) - set(ONNX_SUPPORTED_EXPORT_VARIANTS)
            )
            if unsupported_onnx_variants:
                raise SystemExit(
                    "ONNX export currently supports only these variants for this transformer benchmark: "
                    + ", ".join(ONNX_SUPPORTED_EXPORT_VARIANTS)
                    + ". Unsupported export variant(s): "
                    + ", ".join(unsupported_onnx_variants)
                )
    if not TORCH_AVAILABLE and not args.simulate:
        raise SystemExit(
            "PyTorch is required for real benchmarks. Install it or rerun with --simulate "
            "for clearly labeled demo metrics."
        )


def resolve_compare_paths(args) -> tuple[str, str]:
    if args.latest or (not args.baseline and not args.candidate):
        report_paths = get_benchmark_report_files(args.output_dir)
        if len(report_paths) < 2:
            raise SystemExit("Need at least two benchmark reports in the output directory to compare latest runs.")
        return str(report_paths[1]), str(report_paths[0])

    if not args.baseline or not args.candidate:
        raise SystemExit("Provide both --baseline and --candidate, or use --latest.")
    if not Path(args.baseline).exists():
        raise SystemExit(f"Baseline report not found: {args.baseline}")
    if not Path(args.candidate).exists():
        raise SystemExit(f"Candidate report not found: {args.candidate}")
    return args.baseline, args.candidate


def run_compare_command(args) -> int:
    baseline_path, candidate_path = resolve_compare_paths(args)
    baseline_report = load_json_file(baseline_path)
    candidate_report = load_json_file(candidate_path)
    comparison = compare_reports(baseline_report, candidate_report)
    comparison["baseline_path"] = str(Path(baseline_path).resolve())
    comparison["candidate_path"] = str(Path(candidate_path).resolve())
    comparison["regressions"] = evaluate_regression_thresholds(comparison, args)

    print_comparison_summary(comparison)
    saved_path = write_comparison_report(comparison, args.output, args.output_dir)
    print("[ Comparison Artifact ]")
    print(f"  JSON: {saved_path}\n")

    exit_code = 0
    if comparison["regressions"]:
        exit_code = 2
        print("[ Regression Gates ]")
        for regression in comparison["regressions"]:
            print(
                f"  {regression['variant']} {regression['metric']}: "
                f"observed={regression['observed']} threshold={regression['threshold']}"
            )
        print()

    print("[ JSON output ]")
    print(json.dumps(comparison, indent=2))
    return exit_code


def run_history_command(args) -> int:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    entries = build_history_entries(args.output_dir, args.limit)
    print_history(entries, args.output_dir)
    print("[ JSON output ]")
    print(json.dumps({"mode": "history", "entries": entries}, indent=2))
    return 0


def run_benchmark_command(args) -> int:
    validate_benchmark_args(args)
    report = build_simulation_report(args) if args.simulate else run_real_benchmark(args)

    if report.get("exports"):
        print("[ Export Benchmarks ]")
        for export_result in report["exports"]:
            print_export_summary(export_result)
    for export_error in report.get("export_errors", []):
        print(
            f"[warn] Export error {export_error['format']}:{export_error['source_variant']}: "
            f"{export_error['error']}"
        )

    if not args.skip_report:
        report["artifacts"] = write_report_files(report, args.output_dir)
        print("[ Artifacts ]")
        print(f"  JSON: {report['artifacts']['json_path']}")
        print(f"  CSV:  {report['artifacts']['csv_path']}\n")

    print("[ Summary ]")
    print(f"  Best latency variant: {report['summary']['best_latency_variant']}")
    print(f"  Best quality variant: {report['summary']['best_quality_variant']}")
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
            f"{comparison['accuracy_delta_vs_fp32']:+.4f} accuracy delta, "
            f"{comparison['macro_f1_delta_vs_fp32']:+.4f} macro F1 delta"
        )

    print("\n[ JSON output ]")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "summary": report["summary"],
                "artifacts": report.get("artifacts"),
                "variant_errors": report["variant_errors"],
                "exports": report.get("exports", []),
                "export_errors": report.get("export_errors", []),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)

    if args.command == "benchmark":
        return run_benchmark_command(args)
    if args.command == "compare":
        return run_compare_command(args)
    if args.command == "history":
        return run_history_command(args)

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
