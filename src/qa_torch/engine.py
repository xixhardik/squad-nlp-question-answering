"""Batched forward passes that collect start and end logits.

Written as an explicit loop rather than delegated to ``Trainer.predict`` because the
evaluation path needs plain Python floats to hand to :mod:`qa_core`, and because the
whole point of the project is that the inference path is inspectable.

Logits are converted to float32 before leaving the GPU. Under bf16 or fp16 the raw
tensors carry reduced mantissa precision, and span scoring sums two logits and then
exponentiates them, so the conversion happens once here rather than being an implicit
detail of the decoder.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import PreTrainedModel

from qa_torch.device import resolve_device

logger = logging.getLogger(__name__)

__all__ = ["MODEL_INPUT_KEYS", "collect_qa_logits", "count_features"]

#: Tokenizer outputs a question answering encoder may consume. ``token_type_ids`` is
#: absent for DistilBERT and RoBERTa, so presence is checked rather than assumed.
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def _present_input_keys(features: Any) -> list[str]:
    """Return the model input columns actually present in ``features``."""
    if hasattr(features, "column_names"):
        available = set(features.column_names)
    else:
        available = set(features.keys())
    keys = [key for key in MODEL_INPUT_KEYS if key in available]
    if "input_ids" not in keys:
        raise ValueError(
            "Features must contain 'input_ids'. "
            f"Found columns: {sorted(available)}."
        )
    return keys


def count_features(features: Any) -> int:
    """Return the number of feature rows.

    Args:
        features: A ``datasets.Dataset`` or a mapping of column name to list.

    Returns:
        The row count.
    """
    if hasattr(features, "__len__") and hasattr(features, "column_names"):
        return len(features)
    first_key = next(iter(features))
    return len(features[first_key])


def _slice_batch(features: Any, keys: Sequence[str], start: int, end: int) -> dict[str, list]:
    """Extract one batch of model inputs as plain Python lists."""
    if hasattr(features, "column_names"):
        # datasets.Dataset: a single range lookup is far cheaper than per-column slicing.
        rows = features[start:end]
        return {key: rows[key] for key in keys}
    return {key: list(features[key][start:end]) for key in keys}


def collect_qa_logits(
    model: PreTrainedModel,
    features: Mapping[str, Sequence[Any]] | Any,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    log_every: int | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Run the model over every feature and collect start/end logits.

    Args:
        model: A question answering model emitting ``start_logits``/``end_logits``.
        features: A ``datasets.Dataset`` or mapping of column name to list, holding
            ``input_ids`` and optionally ``attention_mask``/``token_type_ids``.
        batch_size: Features per forward pass. Evaluation stores no activations, so
            this can exceed the training batch size.
        device: Target device. Resolved automatically when ``None``.
        log_every: Log progress every N batches. ``None`` disables progress logging.

    Returns:
        ``(start_logits, end_logits)``, each a list with one list of float32 values
        per feature, in the same order as ``features``.

    Raises:
        ValueError: If ``features`` lacks ``input_ids`` or ``batch_size`` is not
            positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    keys = _present_input_keys(features)
    total = count_features(features)
    if total == 0:
        return [], []

    resolved_device = resolve_device() if device is None else torch.device(device)
    model.to(resolved_device)
    model.eval()

    start_logits: list[list[float]] = []
    end_logits: list[list[float]] = []

    # inference_mode is stronger than no_grad: it also disables version counter
    # bookkeeping, which measurably reduces overhead on long evaluation loops.
    with torch.inference_mode():
        for batch_index, offset in enumerate(range(0, total, batch_size)):
            batch = _slice_batch(features, keys, offset, min(offset + batch_size, total))
            tensors = {
                key: torch.tensor(value, dtype=torch.long, device=resolved_device)
                for key, value in batch.items()
            }
            outputs = model(**tensors)
            # .float() normalises bf16/fp16 to float32 before the decoder sees it.
            start_logits.extend(outputs.start_logits.float().cpu().tolist())
            end_logits.extend(outputs.end_logits.float().cpu().tolist())

            if log_every and (batch_index + 1) % log_every == 0:
                logger.info(
                    "Collected logits for %d/%d features", len(start_logits), total
                )

    if len(start_logits) != total:  # pragma: no cover - guards a silent truncation
        raise RuntimeError(
            f"Collected {len(start_logits)} logit rows for {total} features. "
            "Feature/logit alignment is broken; decoding would attribute spans to the "
            "wrong examples."
        )

    return start_logits, end_logits
