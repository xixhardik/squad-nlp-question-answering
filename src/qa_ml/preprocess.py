"""Turning SQuAD splits into model-ready datasets.

A thin ``datasets``-aware layer over :class:`qa_torch.features.SquadFeatureBuilder`.
The windowing and alignment logic lives there so inference can reuse it; this module
only handles the dataset mechanics: batched ``map`` calls, column separation and
reporting.

Column separation is the part worth reading carefully. ``Trainer`` is configured with
``remove_unused_columns=False``, which means every column in the dataset it receives is
forwarded to the model's ``forward``. A stray ``example_id`` or ``offset_mapping``
column would raise a ``TypeError`` deep inside the training loop. So the feature sets
are split explicitly:

- **model inputs** go to ``Trainer``: ``input_ids``, ``attention_mask``, optionally
  ``token_type_ids``, plus ``start_positions``/``end_positions`` for training.
- **metadata** stays here: ``example_id``, ``offset_mapping``, ``context_mask`` and
  ``alignment_status``. Evaluation needs it to map windows back to examples and to
  recover character spans, but the model must never see it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qa_torch.features import (
    ALIGNMENT_STATUS_COLUMN,
    AlignmentReport,
    SquadFeatureBuilder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EVAL_MODEL_COLUMNS",
    "TRAIN_MODEL_COLUMNS",
    "EvalFeatureBundle",
    "TrainFeatureBundle",
    "build_eval_features",
    "build_train_features",
]

#: Columns ``Trainer`` may forward to the model during training.
TRAIN_MODEL_COLUMNS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "start_positions",
    "end_positions",
)

#: Columns the model may receive during evaluation. No labels: evaluation compares
#: decoded text against gold answers, not token positions.
EVAL_MODEL_COLUMNS = ("input_ids", "attention_mask", "token_type_ids")


@dataclass(frozen=True, slots=True)
class TrainFeatureBundle:
    """Training features plus the alignment report describing their quality.

    Attributes:
        dataset: Model-input-only dataset, safe to hand to ``Trainer``.
        alignment_report: Counts of alignment outcomes, including how many examples
            had their answer found in no window at all.
        num_examples: Source examples the features were built from.
    """

    dataset: Any
    alignment_report: AlignmentReport
    num_examples: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary for the experiment record."""
        return {
            "num_examples": self.num_examples,
            "num_features": len(self.dataset),
            "features_per_example": round(len(self.dataset) / max(self.num_examples, 1), 4),
            "alignment": self.alignment_report.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvalFeatureBundle:
    """Evaluation features, split into model inputs and decoding metadata.

    Attributes:
        dataset: Model-input-only dataset for the forward pass.
        example_ids: Source example id per feature, in feature order.
        offset_mappings: Per-feature token offsets into the original context.
        context_masks: Per-feature ``1``/``0`` flags marking context tokens.
        contexts: Original context string per example id.
        references: Gold answers per example id, for scoring.
    """

    dataset: Any
    example_ids: list[str]
    offset_mappings: list[list[list[int]]]
    context_masks: list[list[int]]
    contexts: dict[str, str]
    references: dict[str, list[str]] = field(default_factory=dict)

    @property
    def num_features(self) -> int:
        """Number of feature windows."""
        return len(self.example_ids)

    @property
    def num_examples(self) -> int:
        """Number of distinct source examples."""
        return len(self.contexts)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary for the experiment record."""
        return {
            "num_examples": self.num_examples,
            "num_features": self.num_features,
            "features_per_example": round(
                self.num_features / max(self.num_examples, 1), 4
            ),
        }


def _keep_columns(dataset: Any, columns: tuple[str, ...]) -> Any:
    """Drop every column not in ``columns``."""
    unwanted = [name for name in dataset.column_names if name not in columns]
    if unwanted:
        dataset = dataset.remove_columns(unwanted)
    return dataset


def build_train_features(
    dataset: Any,
    builder: SquadFeatureBuilder,
    *,
    num_proc: int | None = None,
    batch_size: int = 100,
    load_from_cache_file: bool = True,
) -> TrainFeatureBundle:
    """Build training features from a SQuAD split.

    Args:
        dataset: A ``datasets.Dataset`` with the SQuAD schema.
        builder: The configured feature builder.
        num_proc: Worker processes for ``map``. Leave ``None`` on Windows, where
            processes are spawned rather than forked and startup dominates.
        batch_size: Examples per ``map`` batch.
        load_from_cache_file: Reuse a previously computed cache when available.

    Returns:
        The :class:`TrainFeatureBundle`.
    """
    num_examples = len(dataset)
    features = dataset.map(
        builder.build_train_features,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="Building train features",
    )

    report = AlignmentReport.from_columns(
        features[ALIGNMENT_STATUS_COLUMN], features["example_id"]
    )

    if report.examples_with_no_aligned_feature:
        # Surfaced loudly rather than dropped: these examples contribute no learnable
        # signal, and a rising count is the first sign that windowing is misconfigured.
        logger.warning(
            "%d of %d examples had their answer in NO window and will train on a [CLS] "
            "label only. Check max_seq_length and doc_stride.",
            report.examples_with_no_aligned_feature,
            num_examples,
        )
    logger.info(
        "Train features: %d from %d examples (%.1f%% aligned)",
        len(features),
        num_examples,
        100.0 * report.aligned_fraction,
    )

    return TrainFeatureBundle(
        dataset=_keep_columns(features, TRAIN_MODEL_COLUMNS),
        alignment_report=report,
        num_examples=num_examples,
    )


def build_eval_features(
    dataset: Any,
    builder: SquadFeatureBuilder,
    *,
    num_proc: int | None = None,
    batch_size: int = 100,
    load_from_cache_file: bool = True,
) -> EvalFeatureBundle:
    """Build evaluation features from a SQuAD split.

    Args:
        dataset: A ``datasets.Dataset`` with the SQuAD schema.
        builder: The configured feature builder.
        num_proc: Worker processes for ``map``.
        batch_size: Examples per ``map`` batch.
        load_from_cache_file: Reuse a previously computed cache when available.

    Returns:
        The :class:`EvalFeatureBundle`, with decoding metadata pulled out of the
        model-input dataset.
    """
    from qa_ml.data import build_reference_answers

    features = dataset.map(
        builder.build_eval_features,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="Building eval features",
    )

    example_ids = list(features["example_id"])
    offset_mappings = list(features["offset_mapping"])
    context_masks = list(features["context_mask"])

    contexts = dict(zip(dataset["id"], dataset["context"], strict=True))
    references = (
        build_reference_answers(dataset) if "answers" in dataset.column_names else {}
    )

    logger.info(
        "Eval features: %d from %d examples", len(features), len(contexts)
    )

    return EvalFeatureBundle(
        dataset=_keep_columns(features, EVAL_MODEL_COLUMNS),
        example_ids=example_ids,
        offset_mappings=offset_mappings,
        context_masks=context_masks,
        contexts=contexts,
        references=references,
    )


@dataclass(frozen=True, slots=True)
class ValidationFeatureBundle:
    """Validation features that serve both loss computation and text scoring.

    Built from a single windowing pass. Two different things are wanted from the
    validation split during training:

    - **eval loss**, which needs ``start_positions``/``end_positions`` labels
    - **Exact Match and F1**, which need offsets to decode predicted text

    Producing them from separate passes would tokenize the split twice and, worse,
    allow the two feature sets to drift out of alignment, at which point decoded
    spans would be attributed to the wrong examples.

    Attributes:
        labelled_dataset: Model inputs **with** labels, for ``Trainer``.
        eval_bundle: The same features as model-inputs-only, plus decoding metadata.
        alignment_report: Alignment outcomes for the validation features.
    """

    labelled_dataset: Any
    eval_bundle: EvalFeatureBundle
    alignment_report: AlignmentReport

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary for the experiment record."""
        payload = dict(self.eval_bundle.as_dict())
        payload["alignment"] = self.alignment_report.as_dict()
        return payload


def build_validation_features(
    dataset: Any,
    builder: SquadFeatureBuilder,
    *,
    num_proc: int | None = None,
    batch_size: int = 100,
    load_from_cache_file: bool = True,
) -> ValidationFeatureBundle:
    """Build validation features carrying both labels and decoding metadata.

    Args:
        dataset: A ``datasets.Dataset`` with the SQuAD schema.
        builder: The configured feature builder.
        num_proc: Worker processes for ``map``.
        batch_size: Examples per ``map`` batch.
        load_from_cache_file: Reuse a previously computed cache when available.

    Returns:
        The :class:`ValidationFeatureBundle`.
    """
    from qa_ml.data import build_reference_answers

    features = dataset.map(
        builder.build_validation_features,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="Building validation features",
    )

    report = AlignmentReport.from_columns(
        features[ALIGNMENT_STATUS_COLUMN], features["example_id"]
    )
    example_ids = list(features["example_id"])
    offset_mappings = list(features["offset_mapping"])
    context_masks = list(features["context_mask"])
    contexts = dict(zip(dataset["id"], dataset["context"], strict=True))
    references = (
        build_reference_answers(dataset) if "answers" in dataset.column_names else {}
    )

    logger.info(
        "Validation features: %d from %d examples (%.1f%% aligned)",
        len(features),
        len(contexts),
        100.0 * report.aligned_fraction,
    )

    return ValidationFeatureBundle(
        labelled_dataset=_keep_columns(features, TRAIN_MODEL_COLUMNS),
        eval_bundle=EvalFeatureBundle(
            dataset=_keep_columns(features, EVAL_MODEL_COLUMNS),
            example_ids=example_ids,
            offset_mappings=offset_mappings,
            context_masks=context_masks,
            contexts=contexts,
            references=references,
        ),
        alignment_report=report,
    )
