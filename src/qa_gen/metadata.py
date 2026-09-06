"""Run and dataset metadata: what was trained, on what, and how well.

The record is the deliverable
----------------------------
Same principle as :mod:`qa_ml.experiment`: terminal output is not a record. It is not
machine-readable, it is lost when a Lightning session ends, and it cannot be diffed. A run
produces a JSON document holding the resolved config, the dataset fingerprint, the split
sizes, the adapter description and the metrics, and that document is what a claim about the
model is made from.

Environment capture is injected, not imported
--------------------------------------------
:class:`TrainingRunMetadata` takes ``environment`` as a plain mapping. It does **not** call
:func:`qa_ml.environment.collect_environment`, even though that is exactly the payload
wanted, because doing so would drag ``torch`` and the whole ML stack into a package whose
tests currently run in milliseconds with no GPU and no network. Phase 17B, which has the
stack loaded anyway, passes the mapping in.

The inversion is not a compromise. It also means a run record can be constructed in a test
with a fixed environment stub, which is what makes the serialization round-trip assertable
rather than dependent on the machine it runs on.

What EvaluationMetrics measures, and what it cannot
--------------------------------------------------
:func:`score_predictions` compares predicted targets against references using only the
standard library and :mod:`qa_core`. It measures whether the model emitted parseable JSON,
whether that JSON matched the schema, whether the controllable fields -- type, difficulty,
marks -- came back as requested, and how close the answer and question text are by SQuAD
Exact Match and token F1, reusing :mod:`qa_core.metrics` so the numbers mean the same thing
they do for the extractive model.

It does not measure whether a question is *good*. Answerability from the context, distractor
plausibility and pedagogical value are not computable from string comparison, and inventing a
proxy for them would produce a number that looks like quality and is not. Those need either
the extractive model as a judge or human review, and both are deferred with that stated
rather than approximated.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qa_core.metrics import exact_match_score, token_f1_score
from qa_gen.adapters import AdapterSpec
from qa_gen.examples import QuestionGenerationTarget, TargetParseError, target_from_json
from qa_gen.statistics import DatasetStatistics

__all__ = [
    "DatasetMetadata",
    "EvaluationMetrics",
    "TrainingRunMetadata",
    "score_predictions",
    "utc_now",
]


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """The provenance of one prepared corpus.

    Written next to a run so the dataset a checkpoint was trained on is identified by
    content, not by a directory name that may since have been overwritten.

    Attributes:
        fingerprint: Content digest of the corpus, from
            :func:`qa_gen.splitting.compute_dataset_fingerprint`. The identity of the data.
        created_at: UTC ISO timestamp of preparation.
        config_hash: Digest of the dataset configuration that produced it.
        sources: Source ids included, sorted.
        adapter_specs: The declarative description of every adapter used, so "mapped how"
            is answerable from the artifact even if the adapter code later changes.
        split_summary: Output of :meth:`qa_gen.splitting.DatasetSplits.as_dict`.
        statistics: Composition of the whole corpus.
        split_statistics: Composition per split, keyed by split name. Kept separately
            because a validation split whose difficulty mix differs from train is a real
            experimental problem and invisible in the aggregate.
        validation_summary: Output of
            :meth:`qa_gen.validation.DatasetValidationReport.as_dict`, minus the issue list
            when the caller trims it. Recorded so the share of rejected examples is part of
            the run rather than a number someone remembers.
        notes: Free-form remarks, e.g. which corpora were unavailable.
    """

    fingerprint: str
    created_at: str = field(default_factory=utc_now)
    config_hash: str = ""
    sources: tuple[str, ...] = ()
    adapter_specs: tuple[AdapterSpec, ...] = ()
    split_summary: dict[str, Any] = field(default_factory=dict)
    statistics: DatasetStatistics | None = None
    split_statistics: dict[str, DatasetStatistics] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "adapter_specs", tuple(self.adapter_specs))
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "sources": list(self.sources),
            "adapter_specs": [spec.as_dict() for spec in self.adapter_specs],
            "split_summary": dict(self.split_summary),
            "statistics": self.statistics.as_dict() if self.statistics else None,
            "split_statistics": {
                name: stats.as_dict() for name, stats in sorted(self.split_statistics.items())
            },
            "validation_summary": dict(self.validation_summary),
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the record as a JSON document."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Scores for one evaluation pass.

    Every rate lies in [0, 1]. Counts are carried alongside them because a rate without a
    denominator cannot be compared between runs: 100% of four generations is not a result.

    Attributes:
        total: Generations scored.
        json_valid: Generations that parsed as JSON.
        schema_valid: Generations that parsed *and* matched the canonical target schema.
        question_type_matches: Generations whose type matched the request. The measure of
            whether the model is controllable, which is the point of conditioning the prompt
            on a type at all.
        difficulty_matches: Generations whose difficulty matched the request.
        marks_matches: Generations whose marks matched the request.
        answer_exact_matches: Generations whose answer exactly matched the reference under
            SQuAD normalization.
        answer_f1_total: Summed answer token F1, divided by :attr:`total` for the mean.
        question_f1_total: Summed question token F1.
        mcq_option_count_matches: MCQ generations with the requested number of options.
        per_source: Metrics broken down by dataset source.
        per_question_type: Metrics broken down by requested question type.
        notes: Remarks about what this pass did and did not measure.
    """

    total: int = 0
    json_valid: int = 0
    schema_valid: int = 0
    question_type_matches: int = 0
    difficulty_matches: int = 0
    marks_matches: int = 0
    answer_exact_matches: int = 0
    answer_f1_total: float = 0.0
    question_f1_total: float = 0.0
    mcq_option_count_matches: int = 0
    per_source: dict[str, EvaluationMetrics] = field(default_factory=dict)
    per_question_type: dict[str, EvaluationMetrics] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce ``notes`` to a tuple."""
        object.__setattr__(self, "notes", tuple(self.notes))

    def _rate(self, numerator: float) -> float:
        """Return ``numerator / total``, rounded, or 0.0 for an empty pass."""
        if not self.total:
            return 0.0
        return round(numerator / self.total, 6)

    @property
    def json_validity(self) -> float:
        """Share of generations that were valid JSON."""
        return self._rate(self.json_valid)

    @property
    def schema_validity(self) -> float:
        """Share of generations that matched the canonical schema.

        Never exceeds :attr:`json_validity`: schema validity is a strictly stronger
        condition, and the gap between the two says whether the model is producing malformed
        JSON or well-formed JSON with the wrong fields. Those need different fixes.
        """
        return self._rate(self.schema_valid)

    @property
    def question_type_accuracy(self) -> float:
        """Share of generations that honoured the requested question type."""
        return self._rate(self.question_type_matches)

    @property
    def difficulty_accuracy(self) -> float:
        """Share of generations that honoured the requested difficulty."""
        return self._rate(self.difficulty_matches)

    @property
    def marks_accuracy(self) -> float:
        """Share of generations that honoured the requested marks."""
        return self._rate(self.marks_matches)

    @property
    def answer_exact_match(self) -> float:
        """Share of generations whose answer exactly matched the reference."""
        return self._rate(self.answer_exact_matches)

    @property
    def answer_token_f1(self) -> float:
        """Mean token F1 between generated and reference answers."""
        return self._rate(self.answer_f1_total)

    @property
    def question_token_f1(self) -> float:
        """Mean token F1 between generated and reference questions.

        Low values are not necessarily bad. A model that writes a different, valid question
        about the same passage scores near zero here, and that is the desired behaviour --
        which is precisely why this is reported as one signal among several rather than as a
        headline number.
        """
        return self._rate(self.question_f1_total)

    def as_dict(self, *, include_breakdowns: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Args:
            include_breakdowns: Include the per-source and per-type tables. Disabled when
                serializing a nested breakdown, which would otherwise recurse.
        """
        payload: dict[str, Any] = {
            "total": self.total,
            "json_validity": self.json_validity,
            "schema_validity": self.schema_validity,
            "question_type_accuracy": self.question_type_accuracy,
            "difficulty_accuracy": self.difficulty_accuracy,
            "marks_accuracy": self.marks_accuracy,
            "answer_exact_match": self.answer_exact_match,
            "answer_token_f1": self.answer_token_f1,
            "question_token_f1": self.question_token_f1,
            "counts": {
                "json_valid": self.json_valid,
                "schema_valid": self.schema_valid,
                "question_type_matches": self.question_type_matches,
                "difficulty_matches": self.difficulty_matches,
                "marks_matches": self.marks_matches,
                "answer_exact_matches": self.answer_exact_matches,
                "mcq_option_count_matches": self.mcq_option_count_matches,
            },
            "notes": list(self.notes),
        }
        if include_breakdowns:
            payload["per_source"] = {
                name: metrics.as_dict(include_breakdowns=False)
                for name, metrics in sorted(self.per_source.items())
            }
            payload["per_question_type"] = {
                name: metrics.as_dict(include_breakdowns=False)
                for name, metrics in sorted(self.per_question_type.items())
            }
        return payload


def _score_one(
    prediction: str | dict[str, Any] | QuestionGenerationTarget,
    reference: QuestionGenerationTarget,
) -> dict[str, float]:
    """Score one generation against its reference.

    Returns a mapping of tally field names to their increments, so the caller can add them
    into the overall pass and into each breakdown without re-deriving anything.
    """
    increments: dict[str, float] = {"total": 1.0}

    if isinstance(prediction, QuestionGenerationTarget):
        parsed: QuestionGenerationTarget | None = prediction
        increments["json_valid"] = 1.0
        increments["schema_valid"] = 1.0
    else:
        try:
            if isinstance(prediction, str):
                json.loads(prediction)
            increments["json_valid"] = 1.0
        except json.JSONDecodeError:
            return increments
        try:
            parsed = target_from_json(prediction)
            increments["schema_valid"] = 1.0
        except TargetParseError:
            return increments

    if parsed.question_type == reference.question_type:
        increments["question_type_matches"] = 1.0
    if parsed.difficulty == reference.difficulty:
        increments["difficulty_matches"] = 1.0
    if parsed.marks == reference.marks:
        increments["marks_matches"] = 1.0
    # qa_core's scorers take a *sequence* of gold answers and return the best match, because
    # a SQuAD dev example carries several annotator-accepted answers. There is exactly one
    # reference here, so it is wrapped in a one-element list. Passing the bare string would
    # make the function iterate its characters and score zero for everything.
    if exact_match_score(parsed.answer, [reference.answer]):
        increments["answer_exact_matches"] = 1.0
    increments["answer_f1_total"] = token_f1_score(parsed.answer, [reference.answer])
    increments["question_f1_total"] = token_f1_score(parsed.question, [reference.question])
    if reference.options and len(parsed.options) == len(reference.options):
        increments["mcq_option_count_matches"] = 1.0
    return increments


def _accumulate(tally: dict[str, float], increments: dict[str, float]) -> None:
    """Add ``increments`` into ``tally`` in place."""
    for key, value in increments.items():
        tally[key] = tally.get(key, 0.0) + value


def _to_metrics(
    tally: dict[str, float],
    *,
    notes: Sequence[str] = (),
    per_source: dict[str, EvaluationMetrics] | None = None,
    per_question_type: dict[str, EvaluationMetrics] | None = None,
) -> EvaluationMetrics:
    """Build an :class:`EvaluationMetrics` from an accumulated tally."""
    return EvaluationMetrics(
        total=int(tally.get("total", 0)),
        json_valid=int(tally.get("json_valid", 0)),
        schema_valid=int(tally.get("schema_valid", 0)),
        question_type_matches=int(tally.get("question_type_matches", 0)),
        difficulty_matches=int(tally.get("difficulty_matches", 0)),
        marks_matches=int(tally.get("marks_matches", 0)),
        answer_exact_matches=int(tally.get("answer_exact_matches", 0)),
        answer_f1_total=tally.get("answer_f1_total", 0.0),
        question_f1_total=tally.get("question_f1_total", 0.0),
        mcq_option_count_matches=int(tally.get("mcq_option_count_matches", 0)),
        per_source=dict(per_source or {}),
        per_question_type=dict(per_question_type or {}),
        notes=tuple(notes),
    )


def score_predictions(
    predictions: Sequence[str | dict[str, Any] | QuestionGenerationTarget],
    references: Sequence[QuestionGenerationTarget],
    *,
    sources: Sequence[str] | None = None,
    notes: Sequence[str] = (),
) -> EvaluationMetrics:
    """Score generated targets against references.

    Deterministic and model-free: it compares strings and counts agreements, so it runs in a
    unit test without a checkpoint. Phase 17B supplies real generations to the same function.

    A generation that fails to parse scores zero on every downstream field rather than being
    skipped. Skipping would make ``answer_token_f1`` the mean over the *parseable* subset,
    which rises as the model gets worse at emitting JSON -- a metric that improves under
    degradation is worse than no metric.

    Args:
        predictions: Model outputs, as JSON text, decoded mappings, or already-parsed
            targets. Parsed targets count as valid JSON by definition; the mixed input type
            exists so a caller that already parsed for its own reasons need not re-serialize.
        references: The expected targets, parallel to ``predictions``.
        sources: Dataset source per prediction, for the per-source breakdown. Omit to skip it.
        notes: Remarks recorded on the result.

    Returns:
        The :class:`EvaluationMetrics`, with per-source and per-question-type breakdowns when
        derivable.

    Raises:
        ValueError: If the input sequences differ in length. Silently zipping to the shorter
            one would report a score over a subset while claiming it covered everything.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must be the same length, got {len(predictions)} "
            f"and {len(references)}."
        )
    if sources is not None and len(sources) != len(predictions):
        raise ValueError(
            f"sources must be the same length as predictions, got {len(sources)} and "
            f"{len(predictions)}."
        )

    overall: dict[str, float] = {}
    by_source: dict[str, dict[str, float]] = {}
    by_type: dict[str, dict[str, float]] = {}

    for index, (prediction, reference) in enumerate(
        zip(predictions, references, strict=True)
    ):
        increments = _score_one(prediction, reference)
        _accumulate(overall, increments)

        if sources is not None:
            _accumulate(by_source.setdefault(sources[index], {}), increments)

        type_key = str(getattr(reference.question_type, "value", reference.question_type))
        _accumulate(by_type.setdefault(type_key, {}), increments)

    return _to_metrics(
        overall,
        notes=notes,
        per_source={name: _to_metrics(tally) for name, tally in sorted(by_source.items())},
        per_question_type={
            name: _to_metrics(tally) for name, tally in sorted(by_type.items())
        },
    )


@dataclass
class TrainingRunMetadata:
    """The complete record of one fine-tuning run.

    Deliberately mutable, exactly like :class:`qa_ml.experiment.ExperimentRecord`: it is
    created before training starts and filled in as results arrive, then written once at the
    end -- and again on failure, so a crashed run still leaves a record explaining what
    happened. An undocumented gap in the record is worse than a recorded failure.

    Attributes:
        run_id: Unique run identifier, from
            :meth:`qa_gen.config.GenerationExperimentConfig.run_id`.
        experiment_name: Short experiment name from the config.
        phase: Project phase that produced the run.
        status: ``pending``, ``running``, ``completed`` or ``failed``.
        started_at: UTC ISO timestamp when the record was created.
        finished_at: UTC ISO timestamp when the run ended.
        config: The fully resolved configuration.
        config_hash: Deterministic digest of ``config``.
        base_model: Model id that was fine-tuned.
        base_model_revision: Pinned revision of those weights.
        adapter_path: Where the trained adapter was written. ``None`` until it exists.
        dataset: Output of :meth:`DatasetMetadata.as_dict`.
        environment: Environment and git provenance, injected by the caller. See the module
            docstring for why it is not collected here.
        seeding: What was seeded.
        training: Runtime, throughput, peak memory and logged losses.
        evaluation: Output of :meth:`EvaluationMetrics.as_dict`.
        trainable_parameters: Parameters the adapter actually trains.
        total_parameters: Parameters in the base model.
        error: Exception summary when ``status`` is ``failed``.
        notes: Free-form remarks.
    """

    run_id: str
    experiment_name: str
    phase: str = "17"
    status: str = "pending"
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    base_model: str = ""
    base_model_revision: str = ""
    adapter_path: str | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    seeding: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    error: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_reproducible(self) -> bool:
        """Whether the run started from a clean git tree.

        ``False`` when provenance is absent or the tree was dirty: the recorded commit then
        does not describe the code that ran, so the run cannot be reproduced from it. Same
        definition as :attr:`qa_ml.experiment.ExperimentRecord.is_reproducible`, so the two
        halves of the project agree on what the word means.
        """
        git = self.environment.get("git") or {}
        if not git.get("available"):
            return False
        return not git.get("dirty", True)

    @property
    def trainable_fraction(self) -> float | None:
        """Share of parameters that train, rounded.

        The headline claim of parameter-efficient fine-tuning, so it is derived from measured
        counts rather than from the LoRA config's arithmetic. ``None`` until both counts are
        recorded.
        """
        if not self.trainable_parameters or not self.total_parameters:
            return None
        return round(self.trainable_parameters / self.total_parameters, 6)

    def mark_running(self) -> None:
        """Mark the run started."""
        self.status = "running"

    def mark_completed(self) -> None:
        """Mark the run finished successfully."""
        self.status = "completed"
        self.finished_at = utc_now()

    def mark_failed(self, exc: BaseException) -> None:
        """Mark the run failed and record why.

        Args:
            exc: The exception that ended the run.
        """
        self.status = "failed"
        self.finished_at = utc_now()
        self.error = {"type": type(exc).__name__, "message": str(exc)}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "is_reproducible": self.is_reproducible,
            "config_hash": self.config_hash,
            "config": self.config,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "adapter_path": self.adapter_path,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
            "dataset": self.dataset,
            "environment": self.environment,
            "seeding": self.seeding,
            "training": self.training,
            "evaluation": self.evaluation,
            "notes": list(self.notes),
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Return the record as a JSON document."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False, default=str)
