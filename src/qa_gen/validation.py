"""Checks on canonical examples before they are trained on.

Reports, does not raise
-----------------------
The same asymmetry :mod:`qa_paper.validation` uses, for the same reason and at a different
boundary. An adapter raises when a corpus is not the corpus it was written for -- that is a
configuration fault. This module reports, because it inspects *content*, and some proportion
of any real corpus is unusable. The useful response is a list of which examples are bad and
why, so the good ones can be trained on and the rate can be recorded, not an exception that
discards a corpus over one blank answer.

Silently dropping them would be worse than either. A run that trained on 60% of what it
claimed, with no record of the other 40%, is not reproducible and its data volume is a
fiction.

Why not reuse qa_paper's report types
-------------------------------------
:class:`qa_paper.validation.ValidationIssue` types its ``code`` field as
:class:`qa_paper.validation.IssueCode`, whose members are about assembled papers -- section
homogeneity, blueprint quotas, mark totals. None of them can occur here, and none of these
can occur there. Widening that enum to cover both would force every consumer to handle codes
it can never see and would couple dataset ingestion to paper assembly for no benefit. So the
report machinery is deliberately parallel, and the codes are disjoint by construction.

Question and answer text are checked, difficulty and marks are not judged
------------------------------------------------------------------------
Validation checks that a difficulty is *a* difficulty and that marks are a positive integer.
It does not check whether a question labelled ``easy`` is easy, because the adapters assign
those labels from the corpus's character rather than reading them, and a rule here would
just re-encode the same guess as if it were a finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qa_gen.config import QuestionGenerationDatasetConfig
from qa_gen.examples import QuestionGenerationExample, QuestionGenerationTarget
from qa_gen.splitting import find_duplicate_examples
from qa_paper.enums import Difficulty, QuestionType
from qa_paper.validation import MINIMUM_MCQ_OPTIONS

__all__ = [
    "MINIMUM_MCQ_OPTIONS",
    "DatasetIssue",
    "DatasetIssueCode",
    "DatasetValidationError",
    "DatasetValidationReport",
    "Severity",
    "validate_dataset",
    "validate_example",
]


class Severity(str, Enum):
    """How serious a finding is.

    Attributes:
        ERROR: The example is not usable for training as it stands.
        WARNING: Usable but worth a human look.
    """

    ERROR = "error"
    WARNING = "warning"


class DatasetIssueCode(str, Enum):
    """Stable identifiers for every condition this module detects.

    Assert on these rather than on message wording, which is free to improve.

    Attributes:
        MISSING_CONTEXT: The context is absent or whitespace only.
        MISSING_QUESTION: A target's question text is absent or whitespace only.
        MISSING_ANSWER: A target's answer is absent or whitespace only.
        INVALID_QUESTION_TYPE: A target's type is not a
            :class:`~qa_paper.enums.QuestionType` member.
        INVALID_DIFFICULTY: A target's difficulty is not a
            :class:`~qa_paper.enums.Difficulty` member.
        INVALID_MARKS: Marks are not a positive integer.
        INVALID_MCQ_OPTION_COUNT: An MCQ has fewer than
            :data:`~qa_paper.validation.MINIMUM_MCQ_OPTIONS` options, or a blank or repeated
            option.
        MCQ_INDEX_OUT_OF_RANGE: The correct-option index does not address an option.
        MCQ_ANSWER_MISMATCH: The stated answer is not the correct option's text.
        OPTIONS_ON_NON_MCQ: A type that takes no options carries some.
        EMPTY_TARGETS: The example has no targets, so there is nothing to learn from it.
        DUPLICATE_EXAMPLE_ID: Two examples share an id.
        DUPLICATE_EXAMPLE_CONTENT: Two examples share a content fingerprint.
        CONTEXT_TOO_SHORT: The context is shorter than the configured minimum.
        CONTEXT_TOO_LONG: The context is longer than the configured maximum.
        QUESTION_TYPE_NOT_ALLOWED: The type is outside the configured allow-list.
        DIFFICULTY_NOT_ALLOWED: The difficulty is outside the configured allow-list.
        UNGROUNDED_EXAMPLE: The example carries no traceable provenance.
        ANSWER_NOT_IN_CONTEXT: An extractive-looking answer does not appear in the context.
        EMPTY_DATASET: The corpus contains no examples at all.
        UNKNOWN_SOURCE: The example's source is not one the configuration asked for.
    """

    MISSING_CONTEXT = "missing_context"
    MISSING_QUESTION = "missing_question"
    MISSING_ANSWER = "missing_answer"
    INVALID_QUESTION_TYPE = "invalid_question_type"
    INVALID_DIFFICULTY = "invalid_difficulty"
    INVALID_MARKS = "invalid_marks"
    INVALID_MCQ_OPTION_COUNT = "invalid_mcq_option_count"
    MCQ_INDEX_OUT_OF_RANGE = "mcq_index_out_of_range"
    MCQ_ANSWER_MISMATCH = "mcq_answer_mismatch"
    OPTIONS_ON_NON_MCQ = "options_on_non_mcq"
    EMPTY_TARGETS = "empty_targets"
    DUPLICATE_EXAMPLE_ID = "duplicate_example_id"
    DUPLICATE_EXAMPLE_CONTENT = "duplicate_example_content"
    CONTEXT_TOO_SHORT = "context_too_short"
    CONTEXT_TOO_LONG = "context_too_long"
    QUESTION_TYPE_NOT_ALLOWED = "question_type_not_allowed"
    DIFFICULTY_NOT_ALLOWED = "difficulty_not_allowed"
    UNGROUNDED_EXAMPLE = "ungrounded_example"
    ANSWER_NOT_IN_CONTEXT = "answer_not_in_context"
    EMPTY_DATASET = "empty_dataset"
    UNKNOWN_SOURCE = "unknown_source"


class DatasetValidationError(ValueError):
    """Raised by :meth:`DatasetValidationReport.raise_if_invalid` when errors are present."""


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    """One problem found in a corpus.

    Attributes:
        code: Stable identifier for the condition.
        message: Human-readable explanation naming the offending value.
        severity: Whether this blocks use of the example.
        example_id: Identifier of the offending example, when applicable.
        target_index: Which target within the example, when applicable. A QAG example with
            eight pairs needs to say which one is broken.
        source: Dataset source of the offending example, so a per-corpus failure rate is
            derivable from the report alone.
    """

    code: DatasetIssueCode
    message: str
    severity: Severity = Severity.ERROR
    example_id: str | None = None
    target_index: int | None = None
    source: str | None = None

    @property
    def is_error(self) -> bool:
        """Whether this issue has ``ERROR`` severity."""
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "example_id": self.example_id,
            "target_index": self.target_index,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """The accumulated result of validating examples or a whole corpus.

    Attributes:
        issues: Every finding, in discovery order.
        examples_checked: How many examples were inspected. Carried so a failure *rate* is
            computable from the report without the corpus alongside it -- 400 errors out of
            500 examples and out of 500,000 are different situations.
    """

    issues: tuple[DatasetIssue, ...] = field(default_factory=tuple)
    examples_checked: int = 0

    def __post_init__(self) -> None:
        """Coerce ``issues`` to a tuple."""
        object.__setattr__(self, "issues", tuple(self.issues))

    def __len__(self) -> int:
        """Return the total number of issues."""
        return len(self.issues)

    def __bool__(self) -> bool:
        """Return :attr:`ok`.

        Defined explicitly because the default truthiness of a dataclass would make an empty
        report falsy, inverting the intuitive reading of ``if report:``.
        """
        return self.ok

    @property
    def errors(self) -> tuple[DatasetIssue, ...]:
        """Issues with ``ERROR`` severity."""
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[DatasetIssue, ...]:
        """Issues with ``WARNING`` severity."""
        return tuple(issue for issue in self.issues if not issue.is_error)

    @property
    def ok(self) -> bool:
        """Whether the corpus is usable, i.e. there are no errors."""
        return not self.errors

    @property
    def codes(self) -> tuple[DatasetIssueCode, ...]:
        """Every code present, in discovery order, including repeats."""
        return tuple(issue.code for issue in self.issues)

    @property
    def invalid_example_ids(self) -> frozenset[str]:
        """Ids of examples with at least one error.

        The practical output: the caller filters these out and trains on the rest, having
        recorded how many there were and why.
        """
        return frozenset(
            issue.example_id for issue in self.errors if issue.example_id is not None
        )

    def has(self, code: DatasetIssueCode) -> bool:
        """Whether ``code`` appears in this report."""
        return any(issue.code is code for issue in self.issues)

    def count(self, code: DatasetIssueCode) -> int:
        """How many times ``code`` appears."""
        return sum(1 for issue in self.issues if issue.code is code)

    def code_counts(self) -> dict[str, int]:
        """Return issue counts by code, sorted by code for stable serialization."""
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code.value] = counts.get(issue.code.value, 0) + 1
        return dict(sorted(counts.items()))

    def merged_with(self, other: DatasetValidationReport) -> DatasetValidationReport:
        """Return a new report combining this one's findings with ``other``'s."""
        return DatasetValidationReport(
            issues=(*self.issues, *other.issues),
            examples_checked=self.examples_checked + other.examples_checked,
        )

    def raise_if_invalid(self) -> None:
        """Raise when errors are present, for callers that want fail-fast behaviour.

        Raises:
            DatasetValidationError: If :attr:`errors` is non-empty. At most twenty errors
                are listed; a corpus with thousands of them produces an unreadable message
                and the report itself is the right place to look.
        """
        errors = self.errors
        if not errors:
            return
        shown = errors[:20]
        detail = "\n".join(f"  - [{issue.code.value}] {issue.message}" for issue in shown)
        suffix = "" if len(errors) == len(shown) else f"\n  ... and {len(errors) - 20} more"
        raise DatasetValidationError(
            f"{len(errors)} dataset validation error(s) across {self.examples_checked} "
            f"example(s):\n{detail}{suffix}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "examples_checked": self.examples_checked,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "invalid_example_count": len(self.invalid_example_ids),
            "code_counts": self.code_counts(),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _issue(
    code: DatasetIssueCode,
    message: str,
    *,
    example: QuestionGenerationExample | None = None,
    target_index: int | None = None,
    severity: Severity = Severity.ERROR,
) -> DatasetIssue:
    """Build a :class:`DatasetIssue`, filling the example fields from ``example``."""
    return DatasetIssue(
        code=code,
        message=message,
        severity=severity,
        example_id=example.id if example is not None else None,
        target_index=target_index,
        source=example.source if example is not None else None,
    )


def _validate_marks(
    example: QuestionGenerationExample, target: QuestionGenerationTarget, index: int
) -> list[DatasetIssue]:
    """Check that marks are a positive integer.

    ``bool`` is excluded explicitly: it is a subclass of ``int``, so ``True`` would
    otherwise pass as one mark, and a corpus that produced booleans for marks is one nobody
    should be training on unnoticed.
    """
    marks = target.marks
    if not isinstance(marks, int) or isinstance(marks, bool):
        return [
            _issue(
                DatasetIssueCode.INVALID_MARKS,
                f"example {example.id!r} target {index} has non-integer marks {marks!r} "
                f"({type(marks).__name__}).",
                example=example,
                target_index=index,
            )
        ]
    if marks <= 0:
        return [
            _issue(
                DatasetIssueCode.INVALID_MARKS,
                f"example {example.id!r} target {index} has marks {marks}; marks must be a "
                "positive integer.",
                example=example,
                target_index=index,
            )
        ]
    return []


def _validate_mcq(
    example: QuestionGenerationExample, target: QuestionGenerationTarget, index: int
) -> list[DatasetIssue]:
    """Check option count, distinctness, index range and answer agreement for an MCQ."""
    issues: list[DatasetIssue] = []
    options = target.options

    if len(options) < MINIMUM_MCQ_OPTIONS:
        issues.append(
            _issue(
                DatasetIssueCode.INVALID_MCQ_OPTION_COUNT,
                f"example {example.id!r} target {index} is an MCQ with {len(options)} "
                f"option(s); at least {MINIMUM_MCQ_OPTIONS} are required.",
                example=example,
                target_index=index,
            )
        )

    blanks = [position for position, option in enumerate(options) if not option.strip()]
    if blanks:
        issues.append(
            _issue(
                DatasetIssueCode.INVALID_MCQ_OPTION_COUNT,
                f"example {example.id!r} target {index} has blank option(s) at {blanks}.",
                example=example,
                target_index=index,
            )
        )

    normalized = [option.strip().casefold() for option in options if option.strip()]
    if len(set(normalized)) != len(normalized):
        issues.append(
            _issue(
                DatasetIssueCode.INVALID_MCQ_OPTION_COUNT,
                f"example {example.id!r} target {index} repeats an option, so the item has "
                "more than one correct answer or a redundant distractor.",
                example=example,
                target_index=index,
            )
        )

    position = target.correct_option_index
    if position is None or not 0 <= position < len(options):
        issues.append(
            _issue(
                DatasetIssueCode.MCQ_INDEX_OUT_OF_RANGE,
                f"example {example.id!r} target {index} has correct_option_index "
                f"{position!r} but {len(options)} option(s).",
                example=example,
                target_index=index,
            )
        )
    else:
        correct = target.correct_option
        if (
            correct is not None
            and target.answer.strip()
            and correct.strip().casefold() != target.answer.strip().casefold()
        ):
            issues.append(
                _issue(
                    DatasetIssueCode.MCQ_ANSWER_MISMATCH,
                    f"example {example.id!r} target {index} answer {target.answer!r} does "
                    f"not match the option at index {position} ({correct!r}).",
                    example=example,
                    target_index=index,
                )
            )

    return issues


def _validate_target(
    example: QuestionGenerationExample, target: QuestionGenerationTarget, index: int
) -> list[DatasetIssue]:
    """Check one target of one example."""
    issues: list[DatasetIssue] = []

    if not isinstance(target.question, str) or not target.question.strip():
        issues.append(
            _issue(
                DatasetIssueCode.MISSING_QUESTION,
                f"example {example.id!r} target {index} has no question text.",
                example=example,
                target_index=index,
            )
        )

    if not isinstance(target.answer, str) or not target.answer.strip():
        issues.append(
            _issue(
                DatasetIssueCode.MISSING_ANSWER,
                f"example {example.id!r} target {index} has no answer, so there is nothing "
                "to supervise the answer field with.",
                example=example,
                target_index=index,
            )
        )

    if not isinstance(target.question_type, QuestionType):
        issues.append(
            _issue(
                DatasetIssueCode.INVALID_QUESTION_TYPE,
                f"example {example.id!r} target {index} has unsupported question type "
                f"{target.question_type!r}; expected one of "
                f"{[member.value for member in QuestionType]}.",
                example=example,
                target_index=index,
            )
        )
    elif target.question_type is QuestionType.MCQ:
        issues.extend(_validate_mcq(example, target, index))
    elif target.options:
        issues.append(
            _issue(
                DatasetIssueCode.OPTIONS_ON_NON_MCQ,
                f"example {example.id!r} target {index} is of type "
                f"{target.question_type.value!r} but carries {len(target.options)} "
                "option(s). Only an MCQ has options.",
                example=example,
                target_index=index,
            )
        )

    if not isinstance(target.difficulty, Difficulty):
        issues.append(
            _issue(
                DatasetIssueCode.INVALID_DIFFICULTY,
                f"example {example.id!r} target {index} has invalid difficulty "
                f"{target.difficulty!r}; expected one of "
                f"{[member.value for member in Difficulty]}.",
                example=example,
                target_index=index,
            )
        )

    issues.extend(_validate_marks(example, target, index))
    return issues


def validate_example(
    example: QuestionGenerationExample,
    *,
    config: QuestionGenerationDatasetConfig | None = None,
    check_answer_in_context: bool = False,
) -> DatasetValidationReport:
    """Validate one example in isolation.

    Cross-example conditions such as duplicate ids are not detectable here; see
    :func:`validate_dataset`.

    Args:
        example: The example to check.
        config: When supplied, also enforces the configured context length bounds, type and
            difficulty allow-lists and grounding requirement. Without it, only the
            content-integrity rules run, which is what a unit test of one example wants.
        check_answer_in_context: Report a warning when a short answer does not appear
            verbatim in the context. Off by default because it is only meaningful for
            extractive corpora: a long-answer target is *supposed* to be a synthesis rather
            than a quotation, and flagging those would bury the real findings.

    Returns:
        A :class:`DatasetValidationReport` for this example.
    """
    issues: list[DatasetIssue] = []

    if not isinstance(example.context, str) or not example.context.strip():
        issues.append(
            _issue(
                DatasetIssueCode.MISSING_CONTEXT,
                f"example {example.id!r} has no context, so nothing grounds the question.",
                example=example,
            )
        )
    elif config is not None:
        length = len(example.context)
        if length < config.min_context_chars:
            issues.append(
                _issue(
                    DatasetIssueCode.CONTEXT_TOO_SHORT,
                    f"example {example.id!r} has a {length}-character context, below the "
                    f"configured minimum of {config.min_context_chars}.",
                    example=example,
                )
            )
        elif length > config.max_context_chars:
            issues.append(
                _issue(
                    DatasetIssueCode.CONTEXT_TOO_LONG,
                    f"example {example.id!r} has a {length}-character context, above the "
                    f"configured maximum of {config.max_context_chars}. It would be "
                    "truncated, potentially cutting off the answer.",
                    example=example,
                )
            )

    if not example.targets:
        issues.append(
            _issue(
                DatasetIssueCode.EMPTY_TARGETS,
                f"example {example.id!r} has no targets; there is nothing for the model to "
                "learn to produce.",
                example=example,
            )
        )

    for index, target in enumerate(example.targets):
        issues.extend(_validate_target(example, target, index))

    if config is not None:
        issues.extend(_apply_config_filters(example, config))

    if check_answer_in_context and example.context:
        issues.extend(_check_answers_appear_in_context(example))

    return DatasetValidationReport(issues=tuple(issues), examples_checked=1)


def _apply_config_filters(
    example: QuestionGenerationExample, config: QuestionGenerationDatasetConfig
) -> list[DatasetIssue]:
    """Apply the configured allow-lists, source list and grounding requirement."""
    issues: list[DatasetIssue] = []

    if config.sources and example.source not in config.sources:
        issues.append(
            _issue(
                DatasetIssueCode.UNKNOWN_SOURCE,
                f"example {example.id!r} comes from source {example.source!r}, which is not "
                f"among the configured sources {list(config.sources)}.",
                example=example,
                severity=Severity.WARNING,
            )
        )

    if config.allowed_question_types:
        allowed = set(config.allowed_question_types)
        for index, target in enumerate(example.targets):
            value = getattr(target.question_type, "value", target.question_type)
            if value not in allowed:
                issues.append(
                    _issue(
                        DatasetIssueCode.QUESTION_TYPE_NOT_ALLOWED,
                        f"example {example.id!r} target {index} has type {value!r}, which is "
                        f"not among the configured types {sorted(allowed)}.",
                        example=example,
                        target_index=index,
                        severity=Severity.WARNING,
                    )
                )

    if config.allowed_difficulties:
        allowed = set(config.allowed_difficulties)
        for index, target in enumerate(example.targets):
            value = getattr(target.difficulty, "value", target.difficulty)
            if value not in allowed:
                issues.append(
                    _issue(
                        DatasetIssueCode.DIFFICULTY_NOT_ALLOWED,
                        f"example {example.id!r} target {index} has difficulty {value!r}, "
                        f"which is not among the configured difficulties {sorted(allowed)}.",
                        example=example,
                        target_index=index,
                        severity=Severity.WARNING,
                    )
                )

    if config.require_grounding and not example.is_grounded:
        issues.append(
            _issue(
                DatasetIssueCode.UNGROUNDED_EXAMPLE,
                f"example {example.id!r} has no traceable grounding, so a question generated "
                "from it could not be checked against the source.",
                example=example,
            )
        )

    return issues


def _check_answers_appear_in_context(
    example: QuestionGenerationExample,
) -> list[DatasetIssue]:
    """Warn when a short-answer target's answer is not a substring of the context.

    A warning, never an error, and only for the types where a verbatim answer is expected.
    Case-insensitive: a corpus that capitalises the start of an answer has not made it
    unanswerable.
    """
    issues: list[DatasetIssue] = []
    haystack = example.context.casefold()
    extractive = {QuestionType.SHORT_ANSWER, QuestionType.FILL_BLANK}
    for index, target in enumerate(example.targets):
        if target.question_type not in extractive:
            continue
        answer = target.answer.strip()
        if answer and answer.casefold() not in haystack:
            issues.append(
                _issue(
                    DatasetIssueCode.ANSWER_NOT_IN_CONTEXT,
                    f"example {example.id!r} target {index} answer {answer!r} does not appear "
                    "in the context, so the question may not be answerable from it.",
                    example=example,
                    target_index=index,
                    severity=Severity.WARNING,
                )
            )
    return issues


def validate_dataset(
    examples: Sequence[QuestionGenerationExample],
    *,
    config: QuestionGenerationDatasetConfig | None = None,
    check_answer_in_context: bool = False,
    allow_empty: bool = False,
) -> DatasetValidationReport:
    """Validate a corpus, including the cross-example checks.

    Args:
        examples: The corpus to check.
        config: Forwarded to :func:`validate_example`.
        check_answer_in_context: Forwarded to :func:`validate_example`.
        allow_empty: Treat an empty corpus as acceptable. ``False`` reports
            ``EMPTY_DATASET``, because "the filters removed everything" is otherwise a
            perfectly quiet way for a training run to start on nothing.

    Returns:
        A combined :class:`DatasetValidationReport` covering each example individually plus
        duplicate ids and duplicate content across the corpus.
    """
    issues: list[DatasetIssue] = []

    if not examples and not allow_empty:
        issues.append(
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_DATASET,
                message=(
                    "the corpus contains no examples. Either no source produced any, or the "
                    "configured filters removed all of them."
                ),
            )
        )

    for example in examples:
        issues.extend(
            validate_example(
                example,
                config=config,
                check_answer_in_context=check_answer_in_context,
            ).issues
        )

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.id] = counts.get(example.id, 0) + 1
    for example_id, count in sorted(counts.items()):
        if count > 1:
            issues.append(
                DatasetIssue(
                    code=DatasetIssueCode.DUPLICATE_EXAMPLE_ID,
                    message=(
                        f"example id {example_id!r} appears {count} times. Ids must be unique "
                        "so a split assignment identifies exactly one example; adapters "
                        "derive them from record content, so a collision usually means the "
                        "same record was ingested twice."
                    ),
                    example_id=example_id,
                )
            )

    for fingerprint, ids in find_duplicate_examples(examples).items():
        issues.append(
            DatasetIssue(
                code=DatasetIssueCode.DUPLICATE_EXAMPLE_CONTENT,
                message=(
                    f"examples {list(ids)} share a content fingerprint "
                    f"({fingerprint[:24]}...), so they teach the same thing from the same "
                    "passage. Training on both inflates their weight and risks the same "
                    "content reaching two splits."
                ),
                severity=Severity.WARNING,
                example_id=ids[0],
            )
        )

    return DatasetValidationReport(
        issues=tuple(issues), examples_checked=len(examples)
    )
