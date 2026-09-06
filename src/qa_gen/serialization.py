"""Rebuilding qa_gen objects from plain mappings.

Why both directions are needed
-----------------------------
``as_dict()`` makes a corpus and a run storable. The arrow has to run both ways: a prepared
dataset is written once and read by the trainer, by the evaluator and by whatever inspects a
finished run weeks later, and each of those needs the objects back rather than the dicts. One
parser, kept correct, rather than three that agree until they do not.

It also makes the schema testable before any of those consumers exist. ``from_dict(x.as_dict())
== x`` is the property that says the serialized form is lossless, and it is asserted for every
type here.

Derived keys are ignored, unknown keys are rejected
--------------------------------------------------
``as_dict()`` emits computed values for the benefit of readers -- ``target_count``,
``is_traceable``, ``grounded_rate``, ``trainable_fraction``. Those are not constructor
arguments, so they are skipped on the way back in, which is what makes the round trip hold.

Anything else unrecognised raises. Same reasoning as
:mod:`qa_paper.serialization`: a producer that wrote ``"choices"`` instead of ``"options"``
should fail at the boundary, not silently yield an option-less MCQ that every downstream
metric then reports as a model failure.
"""

from __future__ import annotations

from typing import Any

from qa_gen.examples import QuestionGenerationExample, QuestionGenerationTarget
from qa_gen.metadata import EvaluationMetrics, TrainingRunMetadata
from qa_gen.statistics import DatasetStatistics, LengthSummary
from qa_paper.enums import Difficulty, QuestionType
from qa_paper.serialization import SerializationError, grounding_from_dict

__all__ = [
    "SerializationError",
    "evaluation_metrics_from_dict",
    "example_from_dict",
    "run_metadata_from_dict",
    "statistics_from_dict",
    "target_from_dict",
]

#: Keys the ``as_dict()`` methods in this package emit for readability but that are not
#: constructor arguments. Ignored on the way back in rather than rejected.
_DERIVED_KEYS = frozenset(
    {
        "correct_option",
        "grounded_rate",
        "is_reproducible",
        "is_traceable",
        "json_validity",
        "marks_accuracy",
        "mean_marks",
        "difficulty_accuracy",
        "answer_exact_match",
        "answer_token_f1",
        "question_token_f1",
        "question_type_accuracy",
        "schema_validity",
        "target_count",
        "trainable_fraction",
    }
)


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    """Return ``value`` as a dict or raise."""
    if not isinstance(value, dict):
        raise SerializationError(f"{what} must be a mapping, got {type(value).__name__}.")
    return value


def _check_keys(data: dict[str, Any], allowed: set[str], what: str) -> None:
    """Reject keys that are neither constructor arguments nor known derived values."""
    unknown = sorted(set(data) - allowed - _DERIVED_KEYS)
    if unknown:
        raise SerializationError(
            f"unknown key(s) for {what}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}."
        )


def _require(data: dict[str, Any], key: str, what: str) -> Any:
    """Return ``data[key]`` or raise."""
    if key not in data:
        raise SerializationError(f"{what} is missing required key {key!r}.")
    return data[key]


def _to_enum(value: Any, enum_type: type, what: str) -> Any:
    """Coerce ``value`` into ``enum_type`` or raise a readable error."""
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        valid = [member.value for member in enum_type]  # type: ignore[attr-defined]
        raise SerializationError(
            f"{what} has invalid value {value!r}; expected one of {valid}."
        ) from exc


def target_from_dict(data: dict[str, Any]) -> QuestionGenerationTarget:
    """Rebuild a :class:`~qa_gen.examples.QuestionGenerationTarget`.

    Distinct from :func:`qa_gen.examples.target_from_json`, and the difference is the point.
    That function parses *model output*, so it tolerates missing optional keys and is lenient
    about what a generation might omit. This one reads *our own* serialized form, so it
    demands the full record and rejects anything unexpected. Using the lenient parser for
    stored data would let a truncated file load as a valid-looking target.

    Args:
        data: Mapping as produced by ``QuestionGenerationTarget.as_dict()``.

    Returns:
        The reconstructed target.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    data = _require_mapping(data, "target")
    allowed = {
        "question_type",
        "question",
        "answer",
        "options",
        "correct_option_index",
        "difficulty",
        "marks",
        "explanation",
    }
    _check_keys(data, allowed, "target")

    index = data.get("correct_option_index")
    return QuestionGenerationTarget(
        question_type=_to_enum(
            _require(data, "question_type", "target"), QuestionType, "target.question_type"
        ),
        question=str(_require(data, "question", "target")),
        answer=str(_require(data, "answer", "target")),
        options=tuple(data.get("options") or ()),
        correct_option_index=None if index is None else int(index),
        difficulty=_to_enum(
            _require(data, "difficulty", "target"), Difficulty, "target.difficulty"
        ),
        marks=_require(data, "marks", "target"),
        explanation=data.get("explanation"),
    )


def example_from_dict(data: dict[str, Any]) -> QuestionGenerationExample:
    """Rebuild a :class:`~qa_gen.examples.QuestionGenerationExample`.

    Args:
        data: Mapping as produced by ``QuestionGenerationExample.as_dict()``.

    Returns:
        The reconstructed example, satisfying ``example_from_dict(e.as_dict()) == e``.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    data = _require_mapping(data, "example")
    allowed = {
        "id",
        "context",
        "targets",
        "source",
        "topic",
        "grounding",
        "group_key",
        "metadata",
    }
    _check_keys(data, allowed, "example")

    grounding_data = data.get("grounding")
    return QuestionGenerationExample(
        id=str(_require(data, "id", "example")),
        context=str(_require(data, "context", "example")),
        targets=tuple(target_from_dict(raw) for raw in data.get("targets") or ()),
        source=str(data.get("source", "unknown")),
        topic=data.get("topic"),
        grounding=grounding_from_dict(grounding_data) if grounding_data is not None else None,
        group_key=data.get("group_key"),
        metadata=dict(data.get("metadata") or {}),
    )


def _length_summary_from_dict(data: Any, what: str) -> LengthSummary:
    """Rebuild a :class:`~qa_gen.statistics.LengthSummary` from its abbreviated form."""
    data = _require_mapping(data, what)
    _check_keys(data, {"min", "mean", "max", "count"}, what)
    return LengthSummary(
        minimum=int(data.get("min", 0)),
        mean=float(data.get("mean", 0.0)),
        maximum=int(data.get("max", 0)),
        count=int(data.get("count", 0)),
    )


def statistics_from_dict(data: dict[str, Any]) -> DatasetStatistics:
    """Rebuild a :class:`~qa_gen.statistics.DatasetStatistics`.

    Args:
        data: Mapping as produced by ``DatasetStatistics.as_dict()``.

    Returns:
        The reconstructed statistics.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "statistics")
    allowed = {
        "total_examples",
        "total_targets",
        "examples_by_source",
        "targets_by_question_type",
        "targets_by_difficulty",
        "examples_by_topic",
        "distinct_topics",
        "examples_without_topic",
        "context_chars",
        "question_chars",
        "answer_chars",
        "targets_per_example",
        "mcq_option_counts",
        "grounded_examples",
        "distinct_group_keys",
        "marks_total",
    }
    _check_keys(data, allowed, "statistics")

    def summary(key: str) -> LengthSummary:
        raw = data.get(key)
        if raw is None:
            return LengthSummary()
        return _length_summary_from_dict(raw, f"statistics.{key}")

    return DatasetStatistics(
        total_examples=int(data.get("total_examples", 0)),
        total_targets=int(data.get("total_targets", 0)),
        examples_by_source=dict(data.get("examples_by_source") or {}),
        targets_by_question_type=dict(data.get("targets_by_question_type") or {}),
        targets_by_difficulty=dict(data.get("targets_by_difficulty") or {}),
        examples_by_topic=dict(data.get("examples_by_topic") or {}),
        distinct_topics=int(data.get("distinct_topics", 0)),
        examples_without_topic=int(data.get("examples_without_topic", 0)),
        context_chars=summary("context_chars"),
        question_chars=summary("question_chars"),
        answer_chars=summary("answer_chars"),
        targets_per_example=summary("targets_per_example"),
        mcq_option_counts=summary("mcq_option_counts"),
        grounded_examples=int(data.get("grounded_examples", 0)),
        distinct_group_keys=int(data.get("distinct_group_keys", 0)),
        marks_total=int(data.get("marks_total", 0)),
    )


def evaluation_metrics_from_dict(data: dict[str, Any]) -> EvaluationMetrics:
    """Rebuild an :class:`~qa_gen.metadata.EvaluationMetrics` from its serialized form.

    The rates in the mapping are derived and therefore ignored; the counts under ``"counts"``
    are what get restored, and the rates recompute from them. Reading the rates back instead
    would let a hand-edited file hold a rate that disagrees with its own denominator.

    ``answer_f1_total`` and ``question_f1_total`` are recovered by multiplying the mean by the
    total, which is exact enough to reproduce the mean and is the only lossy step in this
    module -- recorded here rather than discovered later.

    Args:
        data: Mapping as produced by ``EvaluationMetrics.as_dict()``.

    Returns:
        The reconstructed metrics.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "evaluation metrics")
    allowed = {"total", "counts", "notes", "per_source", "per_question_type"}
    _check_keys(data, allowed, "evaluation metrics")

    total = int(data.get("total", 0))
    counts = _require_mapping(data.get("counts") or {}, "evaluation metrics counts")
    return EvaluationMetrics(
        total=total,
        json_valid=int(counts.get("json_valid", 0)),
        schema_valid=int(counts.get("schema_valid", 0)),
        question_type_matches=int(counts.get("question_type_matches", 0)),
        difficulty_matches=int(counts.get("difficulty_matches", 0)),
        marks_matches=int(counts.get("marks_matches", 0)),
        answer_exact_matches=int(counts.get("answer_exact_matches", 0)),
        answer_f1_total=float(data.get("answer_token_f1", 0.0)) * total,
        question_f1_total=float(data.get("question_token_f1", 0.0)) * total,
        mcq_option_count_matches=int(counts.get("mcq_option_count_matches", 0)),
        per_source={
            name: evaluation_metrics_from_dict(raw)
            for name, raw in (data.get("per_source") or {}).items()
        },
        per_question_type={
            name: evaluation_metrics_from_dict(raw)
            for name, raw in (data.get("per_question_type") or {}).items()
        },
        notes=tuple(data.get("notes") or ()),
    )


def run_metadata_from_dict(data: dict[str, Any]) -> TrainingRunMetadata:
    """Rebuild a :class:`~qa_gen.metadata.TrainingRunMetadata`.

    Args:
        data: Mapping as produced by ``TrainingRunMetadata.as_dict()``.

    Returns:
        The reconstructed record. A record whose ``status`` is ``failed`` round-trips with its
        error intact, because a failed run's record is the only evidence of what happened and
        losing it on reload would defeat the point of writing it.

    Raises:
        SerializationError: If required keys are missing.
    """
    data = _require_mapping(data, "run metadata")
    allowed = {
        "run_id",
        "experiment_name",
        "phase",
        "status",
        "started_at",
        "finished_at",
        "config",
        "config_hash",
        "base_model",
        "base_model_revision",
        "adapter_path",
        "dataset",
        "environment",
        "seeding",
        "training",
        "evaluation",
        "trainable_parameters",
        "total_parameters",
        "error",
        "notes",
    }
    _check_keys(data, allowed, "run metadata")

    record = TrainingRunMetadata(
        run_id=str(_require(data, "run_id", "run metadata")),
        experiment_name=str(_require(data, "experiment_name", "run metadata")),
        phase=str(data.get("phase", "17")),
        status=str(data.get("status", "pending")),
        config=dict(data.get("config") or {}),
        config_hash=str(data.get("config_hash", "")),
        base_model=str(data.get("base_model", "")),
        base_model_revision=str(data.get("base_model_revision", "")),
        adapter_path=data.get("adapter_path"),
        dataset=dict(data.get("dataset") or {}),
        environment=dict(data.get("environment") or {}),
        seeding=dict(data.get("seeding") or {}),
        training=dict(data.get("training") or {}),
        evaluation=dict(data.get("evaluation") or {}),
        trainable_parameters=data.get("trainable_parameters"),
        total_parameters=data.get("total_parameters"),
        error=data.get("error"),
        notes=list(data.get("notes") or ()),
    )
    # started_at defaults to "now" on construction, so an absent value in the mapping must
    # not be silently replaced with the time of reading.
    if "started_at" in data:
        record.started_at = str(data["started_at"])
    record.finished_at = data.get("finished_at")
    return record
