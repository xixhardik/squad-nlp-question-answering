"""Deterministic descriptive statistics for a canonical corpus.

Why these numbers, and why deterministic
---------------------------------------
A fine-tuning corpus assembled from four sources with assigned difficulties and assigned
marks is a construction, not a given. Its composition is a decision, and an undocumented
decision is indistinguishable from an accident: "the model never produces hard questions" and
"the corpus was 96% easy" are the same finding, but only one of them is actionable, and only
if the composition was recorded.

So this module is the counterpart to :func:`qa_ml.data.summarize_split` for the generative
side, and it exists to be written into a run record rather than printed.

Determinism is not decoration. Every mapping is returned with sorted keys and every mean is
rounded at a fixed precision, so two runs over the same corpus produce byte-identical
statistics and a diff between two corpora shows only what actually changed. Insertion-ordered
counters would produce a different JSON ordering for the same data and make that diff useless.

Counting units, stated because it would otherwise be ambiguous
-------------------------------------------------------------
An example with eight question-answer pairs is **one** example and **eight** targets. Both
are reported: :attr:`DatasetStatistics.total_examples` is what gets split and prompted, while
:attr:`DatasetStatistics.total_targets` is how many questions the model actually sees. The
by-type and by-difficulty tables count *targets*, because a QAG example's eight pairs are
eight instances of their type; the by-source and by-topic tables count *examples*, because a
source and a topic are properties of the passage rather than of each pair.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from qa_gen.examples import QuestionGenerationExample

__all__ = [
    "LengthSummary",
    "DatasetStatistics",
    "compute_statistics",
]

#: Decimal places kept for every mean. Fixed so statistics are byte-stable across runs and
#: platforms; float repr differences would otherwise show up as spurious diffs.
_PRECISION = 2

#: Most topics listed in :attr:`DatasetStatistics.examples_by_topic`. SQuAD alone has several
#: hundred article titles, and an unbounded table would dominate a run record while telling a
#: reader nothing they could act on. The count of distinct topics is always reported in full.
DEFAULT_TOPIC_LIMIT = 50


@dataclass(frozen=True, slots=True)
class LengthSummary:
    """Minimum, mean and maximum of a length distribution, in characters.

    Characters rather than tokens, deliberately. Tokenising here would mean importing a
    tokenizer, which would mean a network fetch and a heavy dependency in a package that has
    neither. Characters are a stable proxy that needs no model, and the token budget that
    actually matters is enforced against
    :attr:`~qa_gen.config.GeneratorModelConfig.max_seq_length` at training time, where a real
    tokenizer is present.

    Attributes:
        minimum: Shortest observed length.
        mean: Arithmetic mean, rounded.
        maximum: Longest observed length.
        count: How many values contributed.
    """

    minimum: int = 0
    mean: float = 0.0
    maximum: int = 0
    count: int = 0

    @classmethod
    def from_values(cls, values: Sequence[int]) -> LengthSummary:
        """Summarise a sequence of lengths.

        Args:
            values: The lengths.

        Returns:
            The summary. An empty sequence yields all zeros rather than raising: a corpus
            with no MCQ options is a legitimate corpus, and a statistics function that
            crashed on it would be useless precisely when the composition is unexpected.
        """
        if not values:
            return cls()
        return cls(
            minimum=min(values),
            mean=round(sum(values) / len(values), _PRECISION),
            maximum=max(values),
            count=len(values),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "min": self.minimum,
            "mean": self.mean,
            "max": self.maximum,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Composition of one corpus or split.

    Attributes:
        total_examples: Number of examples, i.e. of passages.
        total_targets: Number of question-answer targets across all examples.
        examples_by_source: Example count per dataset source, sorted by source.
        targets_by_question_type: Target count per question type, sorted by type value.
        targets_by_difficulty: Target count per difficulty, in easy/medium/hard order rather
            than alphabetical, because the ordered reading is the informative one.
        examples_by_topic: Example count per topic, highest first, truncated to
            :data:`DEFAULT_TOPIC_LIMIT`.
        distinct_topics: How many distinct topics exist, reported in full even when the table
            above is truncated.
        examples_without_topic: Examples carrying no topic label.
        context_chars: Context length distribution.
        question_chars: Question length distribution, over targets.
        answer_chars: Answer length distribution, over targets.
        targets_per_example: How many targets each example carries.
        mcq_option_counts: Option count distribution, over MCQ targets only.
        grounded_examples: Examples with traceable provenance.
        distinct_group_keys: How many leakage groups the corpus contains. The ceiling on how
            finely it can be split: a corpus of 10,000 examples in 3 groups cannot produce a
            5% validation split without giving one group away entirely.
        marks_total: Sum of marks across all targets.
    """

    total_examples: int = 0
    total_targets: int = 0
    examples_by_source: dict[str, int] = field(default_factory=dict)
    targets_by_question_type: dict[str, int] = field(default_factory=dict)
    targets_by_difficulty: dict[str, int] = field(default_factory=dict)
    examples_by_topic: dict[str, int] = field(default_factory=dict)
    distinct_topics: int = 0
    examples_without_topic: int = 0
    context_chars: LengthSummary = field(default_factory=LengthSummary)
    question_chars: LengthSummary = field(default_factory=LengthSummary)
    answer_chars: LengthSummary = field(default_factory=LengthSummary)
    targets_per_example: LengthSummary = field(default_factory=LengthSummary)
    mcq_option_counts: LengthSummary = field(default_factory=LengthSummary)
    grounded_examples: int = 0
    distinct_group_keys: int = 0
    marks_total: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the corpus holds no examples."""
        return self.total_examples == 0

    @property
    def grounded_rate(self) -> float:
        """Share of examples with traceable provenance, rounded.

        Expected to be low while SQuAD is the only corpus that supplies offsets. Reported so
        that stays a known quantity rather than a surprise when grounding is eventually
        required.
        """
        if not self.total_examples:
            return 0.0
        return round(self.grounded_examples / self.total_examples, 4)

    @property
    def mean_marks(self) -> float:
        """Mean marks per target, rounded."""
        if not self.total_targets:
            return 0.0
        return round(self.marks_total / self.total_targets, _PRECISION)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable key ordering."""
        return {
            "total_examples": self.total_examples,
            "total_targets": self.total_targets,
            "examples_by_source": dict(self.examples_by_source),
            "targets_by_question_type": dict(self.targets_by_question_type),
            "targets_by_difficulty": dict(self.targets_by_difficulty),
            "examples_by_topic": dict(self.examples_by_topic),
            "distinct_topics": self.distinct_topics,
            "examples_without_topic": self.examples_without_topic,
            "context_chars": self.context_chars.as_dict(),
            "question_chars": self.question_chars.as_dict(),
            "answer_chars": self.answer_chars.as_dict(),
            "targets_per_example": self.targets_per_example.as_dict(),
            "mcq_option_counts": self.mcq_option_counts.as_dict(),
            "grounded_examples": self.grounded_examples,
            "grounded_rate": self.grounded_rate,
            "distinct_group_keys": self.distinct_group_keys,
            "marks_total": self.marks_total,
            "mean_marks": self.mean_marks,
        }


#: Difficulty order for the by-difficulty table. Hard-coded rather than read from the enum so
#: the table reads easy-to-hard even if the enum is ever reordered.
_DIFFICULTY_ORDER = ("easy", "medium", "hard")


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    """Return ``counts`` ordered by key."""
    return dict(sorted(counts.items()))


def _ranked_counts(counts: dict[str, int], limit: int) -> dict[str, int]:
    """Return the ``limit`` largest entries, highest first, ties broken by key."""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:limit])


def compute_statistics(
    examples: Iterable[QuestionGenerationExample],
    *,
    topic_limit: int = DEFAULT_TOPIC_LIMIT,
) -> DatasetStatistics:
    """Compute the composition of a corpus.

    Args:
        examples: The corpus or split to summarise. Consumed once, so a generator is fine.
        topic_limit: How many topics to list. The distinct-topic count is always complete.

    Returns:
        The :class:`DatasetStatistics`. An empty corpus yields a zeroed instance rather than
        an error.
    """
    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    group_keys: set[str] = set()

    context_lengths: list[int] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    targets_per: list[int] = []
    option_counts: list[int] = []

    total_examples = 0
    total_targets = 0
    grounded = 0
    untopiced = 0
    marks_total = 0

    for example in examples:
        total_examples += 1
        by_source[example.source] = by_source.get(example.source, 0) + 1
        context_lengths.append(len(example.context))
        targets_per.append(len(example.targets))
        group_keys.add(example.effective_group_key())
        if example.is_grounded:
            grounded += 1

        topic = (example.topic or "").strip()
        if topic:
            by_topic[topic] = by_topic.get(topic, 0) + 1
        else:
            untopiced += 1

        for target in example.targets:
            total_targets += 1
            type_key = str(getattr(target.question_type, "value", target.question_type))
            by_type[type_key] = by_type.get(type_key, 0) + 1
            difficulty_key = str(getattr(target.difficulty, "value", target.difficulty))
            by_difficulty[difficulty_key] = by_difficulty.get(difficulty_key, 0) + 1
            question_lengths.append(len(target.question))
            answer_lengths.append(len(target.answer))
            if target.options:
                option_counts.append(len(target.options))
            if isinstance(target.marks, int) and not isinstance(target.marks, bool):
                marks_total += target.marks

    ordered_difficulty = {
        key: by_difficulty[key] for key in _DIFFICULTY_ORDER if key in by_difficulty
    }
    # Anything outside the known order -- which validation reports as INVALID_DIFFICULTY --
    # is still counted rather than dropped, so the statistics do not quietly disagree with
    # the validation report about how many targets exist.
    ordered_difficulty.update(
        _sorted_counts(
            {key: value for key, value in by_difficulty.items() if key not in ordered_difficulty}
        )
    )

    return DatasetStatistics(
        total_examples=total_examples,
        total_targets=total_targets,
        examples_by_source=_sorted_counts(by_source),
        targets_by_question_type=_sorted_counts(by_type),
        targets_by_difficulty=ordered_difficulty,
        examples_by_topic=_ranked_counts(by_topic, topic_limit),
        distinct_topics=len(by_topic),
        examples_without_topic=untopiced,
        context_chars=LengthSummary.from_values(context_lengths),
        question_chars=LengthSummary.from_values(question_lengths),
        answer_chars=LengthSummary.from_values(answer_lengths),
        targets_per_example=LengthSummary.from_values(targets_per),
        mcq_option_counts=LengthSummary.from_values(option_counts),
        grounded_examples=grounded,
        distinct_group_keys=len(group_keys),
        marks_total=marks_total,
    )
