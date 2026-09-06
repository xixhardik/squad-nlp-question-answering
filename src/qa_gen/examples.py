"""The canonical training example and target for generative question generation.

One representation, many datasets
--------------------------------
Every corpus this project will train on -- SQuAD reformulated for question generation,
LMQG's SQuAD QAG, LearningQ, an educational MCQ set -- arrives in a different shape. If
each shape reached the trainer, then dataset mixing, split determinism, validation and
statistics would each need per-dataset branches, and the model would be learning from a
format nobody could describe in one place.

So there is exactly one canonical form. :mod:`qa_gen.adapters` converts external records
into it and nothing downstream knows where an example came from beyond its
:attr:`QuestionGenerationExample.source` label.

The vocabulary is borrowed, not redefined
----------------------------------------
:class:`~qa_paper.enums.QuestionType`, :class:`~qa_paper.enums.Difficulty` and
:class:`~qa_paper.grounding.ContentGrounding` come from :mod:`qa_paper`. That is the
whole point: the model is being trained to produce questions that
:func:`qa_paper.validation.validate_question` will accept and
:func:`qa_paper.assembly.assemble_paper` will place. A second enumeration of question
types -- even an identical one -- would drift, and the drift would show up as a trained
model emitting a type the paper assembler rejects.

:meth:`QuestionGenerationTarget.to_question` makes that compatibility executable rather
than aspirational: a target converts into a real :class:`qa_paper.questions.Question`,
so the training objective and the production domain object are checked against each
other by the test suite.

Why a target is structured, not a string
---------------------------------------
The model is trained to emit JSON with named fields, not prose. Free-form text would
have to be parsed with heuristics at inference time, and every failure mode -- a missing
answer, four options where three were asked for, a difficulty that is not a difficulty --
would surface as a regex that quietly matched the wrong thing. With a structured target,
a malformed generation is a JSON or schema error at a known boundary.
:meth:`QuestionGenerationTarget.to_json` is the exact string the model is trained to
produce, and :func:`target_from_json` is the only thing that has to parse it.

Why an example holds several targets
------------------------------------
QAG corpora give a paragraph and every question-answer pair drawn from it. Forcing those
into one example per pair would either duplicate the paragraph -- which then has to be
de-duplicated for split leakage -- or throw pairs away. A tuple of targets keeps the
paragraph once and the pairs intact, and single-target corpora are simply the
one-element case.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from qa_core.normalize import normalize_answer
from qa_paper.enums import Difficulty, QuestionType
from qa_paper.fingerprint import normalize_question_text
from qa_paper.grounding import ContentGrounding
from qa_paper.questions import (
    CaseScenarioPayload,
    FillBlankPayload,
    McqPayload,
    Question,
    QuestionPayload,
    TrueFalsePayload,
)

__all__ = [
    "TARGET_JSON_FIELDS",
    "QuestionGenerationExample",
    "QuestionGenerationTarget",
    "TargetParseError",
    "target_from_json",
]

#: The field names, in emission order, of the JSON object the model is trained to
#: produce. Ordered rather than alphabetical: the type constrains everything after it,
#: and a model that has committed to ``"mcq"`` before writing options is less likely to
#: produce options for a short-answer question. Exposed because the prompt's
#: output-format contract and the parser must agree, and agreeing by accident is not
#: good enough -- ``tests`` asserts they are generated from this one tuple.
TARGET_JSON_FIELDS: tuple[str, ...] = (
    "question_type",
    "question",
    "answer",
    "options",
    "correct_option_index",
    "difficulty",
    "marks",
    "explanation",
)

#: Length of the digest embedded in a derived example id.
_ID_HASH_LENGTH = 12


class TargetParseError(ValueError):
    """Raised when model output cannot be read as a :class:`QuestionGenerationTarget`.

    A distinct type because the caller's response differs from other failures: a parse
    error means *this generation* is unusable and should be retried or dropped, not that
    the pipeline is misconfigured.
    """


@dataclass(frozen=True, slots=True)
class QuestionGenerationTarget:
    """One question the model is trained to produce, as structured fields.

    Permissive on construction, exactly like :class:`qa_paper.questions.Question` and for
    the same reason: a target arriving from an external dataset or from a generation may
    be malformed, and :mod:`qa_gen.validation` has to be able to *report* that rather than
    crash the ingestion of a whole corpus.

    Attributes:
        question_type: Which kind of question this is.
        question: The question text as presented to the candidate.
        answer: The model answer. Required for every type, including MCQ, so the answer
            key is renderable without reaching into the options.
        options: Answer options, for MCQ. Empty for every other type.
        correct_option_index: Zero-based index into ``options``. ``None`` when the type
            has no options.
        difficulty: Concrete difficulty. Never "mixed"; see :mod:`qa_paper.enums`.
        marks: Marks a fully correct answer earns. Integer, matching
            :mod:`qa_paper.blueprint`.
        explanation: Optional rationale. Useful for MCQ distractor analysis and as a
            supervision signal, but never required.
    """

    question_type: QuestionType = QuestionType.SHORT_ANSWER
    question: str = ""
    answer: str = ""
    options: tuple[str, ...] = ()
    correct_option_index: int | None = None
    difficulty: Difficulty = Difficulty.MEDIUM
    marks: int = 1
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Coerce ``options`` to a tuple so a frozen instance is truly immutable."""
        object.__setattr__(self, "options", tuple(self.options))

    @property
    def has_options(self) -> bool:
        """Whether this target carries answer options."""
        return bool(self.options)

    @property
    def correct_option(self) -> str | None:
        """The text of the correct option, or ``None`` when the index does not address one."""
        index = self.correct_option_index
        if index is None or not 0 <= index < len(self.options):
            return None
        return self.options[index]

    def fingerprint(self) -> str:
        """Return a normalized key for duplicate and leakage detection.

        Delegates to :func:`qa_paper.fingerprint.normalize_question_text`, so a numeric or
        symbol-heavy question is not collapsed into a different one. The answer is folded
        in because the same question asked of two passages with different answers is two
        training examples, not a duplicate.

        Returns:
            A stable string key.
        """
        type_key = getattr(self.question_type, "value", self.question_type)
        return (
            f"{type_key}:{normalize_question_text(self.question)}"
            f"|{normalize_answer(self.answer)}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible mapping, in :data:`TARGET_JSON_FIELDS` order.

        This is the *training target representation*. Whatever appears here is what the
        model is taught to emit, so the ordering and the key names are part of the
        contract rather than a formatting detail.
        """
        return {
            "question_type": getattr(self.question_type, "value", self.question_type),
            "question": self.question,
            "answer": self.answer,
            "options": list(self.options),
            "correct_option_index": self.correct_option_index,
            "difficulty": getattr(self.difficulty, "value", self.difficulty),
            "marks": self.marks,
            "explanation": self.explanation,
        }

    def to_json(self, *, indent: int | None = None, compact: bool = True) -> str:
        """Return the exact JSON string the model is trained to produce.

        Args:
            indent: Indentation for pretty output. ``None`` keeps it on one line.
            compact: Drop keys that carry no information for this question type --
                ``options`` and ``correct_option_index`` for a type without options, and
                ``explanation`` when absent. Defaults to ``True`` because teaching a
                model to emit ``"options": []`` on every short-answer question spends
                tokens on nothing and invites it to fill them in.

        Returns:
            A JSON object string with keys in :data:`TARGET_JSON_FIELDS` order.
        """
        payload = self.as_dict()
        if compact:
            if not self.options:
                payload.pop("options", None)
                payload.pop("correct_option_index", None)
            if self.explanation is None:
                payload.pop("explanation", None)
        separators = None if indent is not None else (",", ":")
        return json.dumps(payload, indent=indent, separators=separators, ensure_ascii=False)

    def to_payload(self) -> QuestionPayload | None:
        """Return the :mod:`qa_paper` payload this target implies.

        Returns:
            The payload class required by :attr:`question_type`, or ``None`` for the
            free-text types that take none. ``MATCH_FOLLOWING`` also returns ``None``:
            two columns and their pairings are not representable in this target schema,
            which is a deliberate limitation recorded in :meth:`to_question`.
        """
        if self.question_type is QuestionType.MCQ:
            return McqPayload(
                options=self.options,
                correct_index=self.correct_option_index or 0,
            )
        if self.question_type is QuestionType.TRUE_FALSE:
            return TrueFalsePayload(
                correct_answer=normalize_answer(self.answer) in {"true", "t", "yes", "correct"}
            )
        if self.question_type is QuestionType.FILL_BLANK:
            return FillBlankPayload(accepted_answers=(self.answer,))
        if self.question_type is QuestionType.CASE_SCENARIO:
            # A case is one scenario plus the parts asked about it. This target schema
            # carries a single question, so the case has exactly one part. Multi-part
            # cases need a richer target and are deferred, not approximated.
            return CaseScenarioPayload(scenario="", sub_questions=(self.question,))
        return None

    def to_question(
        self,
        question_id: str,
        *,
        topic: str | None = None,
        grounding: ContentGrounding | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Question:
        """Convert this target into a :class:`qa_paper.questions.Question`.

        The bridge that makes "the model is trained to produce what the paper system
        consumes" a checkable statement. Nothing in this phase calls it in anger; the
        tests call it to prove the two schemas have not drifted, and the generator
        adapter in a later phase will call it for real.

        ``MATCH_FOLLOWING`` converts without a payload and will therefore be reported by
        :func:`qa_paper.validation.validate_question` as missing one. That is the honest
        outcome: this target schema cannot express two columns, so a match-the-following
        question cannot be produced from it and the validation report says so rather than
        a silent empty payload passing.

        Args:
            question_id: Identifier to assign, unique within the paper it joins.
            topic: Syllabus topic label.
            grounding: Provenance for the question.
            metadata: Free-form extras, e.g. the generator and dataset that produced it.

        Returns:
            The equivalent :class:`~qa_paper.questions.Question`.
        """
        return Question(
            id=question_id,
            question_type=self.question_type,
            text=self.question,
            marks=self.marks,
            difficulty=self.difficulty,
            answer=self.answer,
            topic=topic,
            payload=self.to_payload(),
            grounding=grounding,
            metadata=dict(metadata or {}),
        )


def _coerce_enum(value: Any, enum_type: type, field_name: str) -> Any:
    """Coerce ``value`` into ``enum_type``, raising :class:`TargetParseError` on failure."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        valid = [member.value for member in enum_type]  # type: ignore[attr-defined]
        raise TargetParseError(
            f"{field_name} has invalid value {value!r}; expected one of {valid}."
        ) from exc


def target_from_json(payload: str | dict[str, Any]) -> QuestionGenerationTarget:
    """Parse a canonical target from JSON text or an already-decoded mapping.

    The single place model output is interpreted. Unknown keys are rejected rather than
    ignored: a model that emitted ``"choices"`` instead of ``"options"`` has not learned
    the format, and silently producing an option-less MCQ would hide that from every
    metric.

    Args:
        payload: JSON object text, or the mapping it decodes to.

    Returns:
        The parsed target. Absent optional keys take their schema defaults, so a compact
        emission from :meth:`QuestionGenerationTarget.to_json` round-trips.

    Raises:
        TargetParseError: If the text is not JSON, is not an object, carries an unknown
            key, or holds an invalid question type or difficulty.
    """
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TargetParseError(f"target is not valid JSON: {exc}") from exc
    else:
        decoded = payload

    if not isinstance(decoded, dict):
        raise TargetParseError(
            f"target must be a JSON object, got {type(decoded).__name__}."
        )

    unknown = sorted(set(decoded) - set(TARGET_JSON_FIELDS))
    if unknown:
        raise TargetParseError(
            f"unknown target key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(TARGET_JSON_FIELDS)}."
        )

    # marks is passed through without coercion. int("2") would turn a model that emitted
    # a string into a silent success, and "the model does not respect the marks type" is
    # exactly the kind of thing INVALID_MARKS exists to report.
    index = decoded.get("correct_option_index")
    return QuestionGenerationTarget(
        question_type=_coerce_enum(
            decoded.get("question_type", QuestionType.SHORT_ANSWER.value),
            QuestionType,
            "question_type",
        ),
        question=str(decoded.get("question", "")),
        answer=str(decoded.get("answer", "")),
        options=tuple(decoded.get("options") or ()),
        correct_option_index=None if index is None else int(index),
        difficulty=_coerce_enum(
            decoded.get("difficulty", Difficulty.MEDIUM.value), Difficulty, "difficulty"
        ),
        marks=decoded.get("marks", 1),
        explanation=decoded.get("explanation"),
    )


@dataclass(frozen=True, slots=True)
class QuestionGenerationExample:
    """One canonical training example: a passage and the questions drawn from it.

    Attributes:
        id: Unique identifier within the dataset. Adapters derive it deterministically
            from the source record so re-ingesting a corpus yields the same ids and the
            same split assignment.
        context: The source passage the questions are answerable from. Used verbatim, so
            grounding offsets index it exactly.
        targets: The questions drawn from ``context``, in dataset order. Several for a
            QAG corpus, one for everything else. Must be non-empty;
            :mod:`qa_gen.validation` reports ``EMPTY_TARGETS`` otherwise rather than the
            constructor raising.
        source: Identifier of the dataset this came from, e.g. ``"squad-qg"``. Kept so
            statistics, split metadata and per-source evaluation are all possible, and so
            a corpus mix is auditable after the fact.
        topic: Syllabus topic or article title, when the source provides one.
        grounding: Where in the source document the context came from. Populated when the
            corpus carries offsets; SQuAD does, most do not.
        group_key: Leakage-control key. Examples sharing it are never split apart. ``None``
            means "derive one from the context", which is the behaviour that stops the
            same paragraph appearing in both train and test. See :mod:`qa_gen.splitting`.
        metadata: Free-form extras from the adapter, e.g. the original record id.
    """

    id: str
    context: str
    targets: tuple[QuestionGenerationTarget, ...] = ()
    source: str = "unknown"
    topic: str | None = None
    grounding: ContentGrounding | None = None
    group_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``targets`` to a tuple."""
        object.__setattr__(self, "targets", tuple(self.targets))

    def __len__(self) -> int:
        """Return the number of targets."""
        return len(self.targets)

    @property
    def primary_target(self) -> QuestionGenerationTarget | None:
        """The first target, or ``None`` when there are none.

        The single-target view every downstream grouping uses. A QAG example with four
        pairs still counts once in the by-difficulty table, because the example is what
        gets split and prompted, not the individual pair.
        """
        return self.targets[0] if self.targets else None

    @property
    def question_type(self) -> QuestionType | None:
        """Question type of the primary target."""
        target = self.primary_target
        return target.question_type if target else None

    @property
    def difficulty(self) -> Difficulty | None:
        """Difficulty of the primary target."""
        target = self.primary_target
        return target.difficulty if target else None

    @property
    def marks(self) -> int | None:
        """Marks of the primary target."""
        target = self.primary_target
        return target.marks if target else None

    @property
    def question(self) -> str | None:
        """Question text of the primary target."""
        target = self.primary_target
        return target.question if target else None

    @property
    def answer(self) -> str | None:
        """Answer of the primary target."""
        target = self.primary_target
        return target.answer if target else None

    @property
    def options(self) -> tuple[str, ...]:
        """Options of the primary target, empty when it has none."""
        target = self.primary_target
        return target.options if target else ()

    @property
    def is_grounded(self) -> bool:
        """Whether this example's provenance can be followed to a specific passage."""
        return self.grounding is not None and self.grounding.is_traceable

    @property
    def context_fingerprint(self) -> str:
        """Digest of the normalized context, used as the default leakage group.

        Normalized before hashing, so two copies of a paragraph differing only in
        whitespace or punctuation still group together. This is the mechanism that keeps
        the same passage out of two splits.
        """
        normalized = normalize_answer(self.context)
        return hashlib.sha256(normalized.encode()).hexdigest()[:_ID_HASH_LENGTH]

    def effective_group_key(self) -> str:
        """Return the key that decides which split this example joins.

        Returns:
            :attr:`group_key` when the adapter set one -- a SQuAD article title, say,
            which groups more aggressively than a paragraph -- otherwise
            :attr:`context_fingerprint`.
        """
        if self.group_key and self.group_key.strip():
            return self.group_key
        return f"ctx:{self.context_fingerprint}"

    def fingerprint(self) -> str:
        """Return a normalized key identifying this example's content.

        Combines the normalized context with every target fingerprint, so two examples
        collide only when they teach the same thing from the same passage. Used for
        duplicate reporting and for the dataset fingerprint.

        Returns:
            A stable string key.
        """
        parts = [self.context_fingerprint, *(t.fingerprint() for t in self.targets)]
        return "|".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "context": self.context,
            "targets": [target.as_dict() for target in self.targets],
            "target_count": len(self.targets),
            "source": self.source,
            "topic": self.topic,
            "grounding": self.grounding.as_dict() if self.grounding is not None else None,
            "group_key": self.group_key,
            "metadata": dict(self.metadata),
        }
