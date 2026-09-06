"""Deterministic, leakage-aware train/validation/test partitioning.

No random number generator at all
--------------------------------
There is no ``random`` import here, no ``numpy.random``, no shuffle and no global state.
Every assignment is a pure function of ``(seed, group key)``: the pair is hashed with
SHA-256, and groups are ordered by that digest. Two consequences follow, and both matter
more than they sound.

**The partition does not depend on input order.** Reading the corpus in a different order,
loading two sources in a different sequence, or a dataset library changing its iteration
does not move a single example. A seeded shuffle gives reproducibility only if the input
order is also reproducible, which is a much weaker guarantee than it appears and one that
quietly fails the first time a filter is added upstream.

**Nothing else in the process is disturbed.** ``random.seed()`` mutates interpreter-global
state; calling it inside a data-loading function changes the behaviour of unrelated code
that happens to run afterwards. That is exactly the class of bug that makes a run
irreproducible for reasons nobody can find.

Leakage control is the point, not a feature
-------------------------------------------
Question generation leaks in a way extractive QA does not. If the same paragraph appears in
train and in test, the model has seen the exact passage it is being asked to write a novel
question about, and the test score measures memorisation. Worse, the leak is invisible in
the numbers: it makes them *better*.

So examples are never assigned individually. They are assigned in **groups**, and a group
lands wholly in one split:

- ``group_by="context"`` groups by a digest of the normalized context, so the same passage
  cannot be split. The default.
- ``group_by="topic"`` groups by the example's own ``group_key`` -- a SQuAD article title, a
  LearningQ document id -- which is stricter, because paragraphs from one article overlap
  heavily even when no paragraph is shared. This is how the original SQuAD splits were
  built and what a headline number should use.
- ``group_by="example"`` disables grouping. Offered only so the size of the effect can be
  measured; it is not a reasonable choice for a reported result.

Near-duplicate detection is separate and reported, not silently applied:
:func:`find_duplicate_examples` uses
:meth:`~qa_gen.examples.QuestionGenerationExample.fingerprint`, which normalizes through
:mod:`qa_paper.fingerprint` and so does not collapse ``2 + 2`` into ``2 - 2``.

Exact sizes, and the trade that buys them
-----------------------------------------
Groups are ordered by digest and then assigned to whichever split is furthest below its
target count. That gives split sizes close to the requested ratios and a fully determined
result.

The cost, stated because it is a real one: adding examples to the corpus can move existing
groups between splits. The alternative -- thresholding each group's digest against the
ratios -- is stable under insertion but gives sizes that drift from the ratios on small
corpora. Exact and reportable was chosen over stable, and
:attr:`DatasetSplits.dataset_fingerprint` is what makes a partition change *detectable*: if
the corpus changed, the fingerprint changed, and the recorded split is known to describe a
different dataset.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from qa_gen.examples import QuestionGenerationExample

__all__ = [
    "DatasetSplits",
    "DeterministicGroupSplitter",
    "GroupAssignment",
    "SplitError",
    "SplitName",
    "SplitRatios",
    "compute_dataset_fingerprint",
    "find_duplicate_examples",
]


class SplitError(ValueError):
    """Raised when a split request is impossible or its inputs are inconsistent."""


class SplitName(str, Enum):
    """The three partitions.

    Subclasses ``(str, Enum)`` rather than ``enum.StrEnum`` for the same reason as
    :class:`qa_paper.enums.QuestionType`: ``StrEnum`` requires Python 3.11 and this project
    declares ``>=3.10``.

    Attributes:
        TRAIN: Used to fit the adapter.
        VALIDATION: Used to rank checkpoints during training.
        TEST: Held out. Read once, at the end.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Target shares for the three splits.

    Attributes:
        train: Share assigned to training.
        validation: Share assigned to validation.
        test: Share assigned to test.
    """

    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05

    def validate(self) -> None:
        """Check the ratios.

        Raises:
            SplitError: If a ratio is outside [0, 1], the three do not sum to one, or the
                training share is zero.
        """
        values = {"train": self.train, "validation": self.validation, "test": self.test}
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise SplitError(f"split ratio {name} must lie in [0, 1], got {value}.")
        total = sum(values.values())
        if abs(total - 1.0) > 1e-9:
            raise SplitError(
                f"split ratios must sum to 1.0, got {total} ({values}). "
                "Adjust one of them so the partition is well defined."
            )
        if self.train <= 0.0:
            raise SplitError(
                "the train ratio must be greater than zero; there would be nothing to "
                "fine-tune on."
            )

    def target_counts(self, total: int) -> dict[SplitName, int]:
        """Return the target example count for each split.

        Computed with integer arithmetic and a remainder pass rather than by rounding three
        floats independently, so the three counts always sum to ``total`` exactly. Rounding
        each share separately loses or invents an example depending on the corpus size,
        which then shows up as an off-by-one nobody can explain.

        Args:
            total: Number of examples to partition.

        Returns:
            A mapping with an entry for every split, summing to ``total``.
        """
        if total <= 0:
            return dict.fromkeys(SplitName, 0)
        shares = {
            SplitName.TRAIN: self.train,
            SplitName.VALIDATION: self.validation,
            SplitName.TEST: self.test,
        }
        floors = {name: int(total * share) for name, share in shares.items()}
        remainder = total - sum(floors.values())
        # Distribute the remainder to the largest fractional parts, breaking ties by split
        # order so the result is deterministic rather than dict-order dependent.
        fractions = sorted(
            ((total * shares[name]) - floors[name], index, name)
            for index, name in enumerate(SplitName)
        )
        for _, _, name in reversed(fractions[-remainder:] if remainder else []):
            floors[name] += 1
        return floors

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """Which split one leakage group was assigned to, and why.

    Kept so a partition is auditable. "Why is this paragraph in test?" is answerable from
    the recorded assignment rather than by re-deriving the algorithm.

    Attributes:
        group_key: The leakage key shared by every example in the group.
        split: The split the whole group joined.
        example_count: How many examples the group holds.
        example_ids: Ids in the group, sorted.
        digest: The ``(seed, group_key)`` digest that ordered this group.
        sources: Dataset sources represented in the group, sorted. Usually one; more than
            one means two corpora share a passage, which is itself worth seeing.
    """

    group_key: str
    split: SplitName
    example_count: int
    example_ids: tuple[str, ...] = ()
    digest: str = ""
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "example_ids", tuple(self.example_ids))
        object.__setattr__(self, "sources", tuple(self.sources))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "group_key": self.group_key,
            "split": self.split.value,
            "example_count": self.example_count,
            "example_ids": list(self.example_ids),
            "digest": self.digest,
            "sources": list(self.sources),
        }


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """The result of partitioning a corpus.

    Attributes:
        train: Training examples, in deterministic order.
        validation: Validation examples.
        test: Test examples.
        seed: The seed that produced this partition.
        group_by: The grouping strategy used.
        ratios: The requested ratios.
        dataset_fingerprint: Digest of the corpus that was partitioned. Two partitions with
            the same seed, strategy and fingerprint are the same partition; a differing
            fingerprint means the corpus changed and the recorded split describes other
            data.
        assignments: Per-group record of where each group went.
        splitter: Identifier of the splitter, versioned so a change of algorithm is visible
            in the run record rather than inferred from shifting numbers.
    """

    train: tuple[QuestionGenerationExample, ...] = ()
    validation: tuple[QuestionGenerationExample, ...] = ()
    test: tuple[QuestionGenerationExample, ...] = ()
    seed: int = 0
    group_by: str = "context"
    ratios: SplitRatios = field(default_factory=SplitRatios)
    dataset_fingerprint: str = ""
    assignments: tuple[GroupAssignment, ...] = ()
    splitter: str = ""

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        for name in ("train", "validation", "test", "assignments"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def __len__(self) -> int:
        """Return the total number of examples across all splits."""
        return len(self.train) + len(self.validation) + len(self.test)

    def __getitem__(self, split: SplitName | str) -> tuple[QuestionGenerationExample, ...]:
        """Return one split by name.

        Args:
            split: The split to fetch, as a :class:`SplitName` or its string value.

        Returns:
            The examples in that split.

        Raises:
            SplitError: If the name is not one of the three splits.
        """
        try:
            name = SplitName(split)
        except ValueError as exc:
            valid = [member.value for member in SplitName]
            raise SplitError(f"unknown split {split!r}; expected one of {valid}.") from exc
        return {
            SplitName.TRAIN: self.train,
            SplitName.VALIDATION: self.validation,
            SplitName.TEST: self.test,
        }[name]

    @property
    def sizes(self) -> dict[str, int]:
        """Example count per split."""
        return {
            SplitName.TRAIN.value: len(self.train),
            SplitName.VALIDATION.value: len(self.validation),
            SplitName.TEST.value: len(self.test),
        }

    def source_breakdown(self) -> dict[str, dict[str, int]]:
        """Return per-split example counts by dataset source.

        The source-aware metadata that makes a mixed corpus interpretable: a test split
        that happens to contain no MCQ examples explains an MCQ metric of zero, and without
        this table that would look like a model failure.

        Returns:
            ``{split: {source: count}}`` with keys sorted for stable serialization.
        """
        breakdown: dict[str, dict[str, int]] = {}
        for name in SplitName:
            counts: dict[str, int] = {}
            for example in self[name]:
                counts[example.source] = counts.get(example.source, 0) + 1
            breakdown[name.value] = dict(sorted(counts.items()))
        return breakdown

    def group_keys(self, split: SplitName | str) -> frozenset[str]:
        """Return the leakage keys present in one split."""
        return frozenset(example.effective_group_key() for example in self[split])

    def leaked_group_keys(self) -> frozenset[str]:
        """Return leakage keys appearing in more than one split.

        Always empty for a partition this module produced, which is exactly why it is worth
        asserting: it turns the central invariant into something a test can check rather
        than something the docstring claims.
        """
        train, validation, test = (self.group_keys(name) for name in SplitName)
        return frozenset(
            (train & validation) | (train & test) | (validation & test)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary.

        The examples themselves are omitted: a split summary belongs in a run record and
        inlining an entire corpus into it would make the record unreadable and enormous.
        Ids are recorded per group in :attr:`assignments`, which is enough to reconstruct
        the partition exactly.
        """
        return {
            "splitter": self.splitter,
            "seed": self.seed,
            "group_by": self.group_by,
            "ratios": self.ratios.as_dict(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "sizes": self.sizes,
            "total": len(self),
            "group_count": len(self.assignments),
            "source_breakdown": self.source_breakdown(),
            "leaked_group_keys": sorted(self.leaked_group_keys()),
        }


@runtime_checkable
class DatasetSplitter(Protocol):
    """The contract for partitioning a corpus.

    Structural, like the other replaceable boundaries in this project. An implementation
    must be deterministic: the same examples, seed and strategy must give the same
    partition, because the split is recorded by fingerprint rather than by storing every
    example, and a non-reproducible splitter makes that record a lie.
    """

    @property
    def name(self) -> str:
        """Identifier recorded on the partitions this splitter produces."""
        ...

    def split(
        self,
        examples: Sequence[QuestionGenerationExample],
        *,
        seed: int,
        ratios: SplitRatios,
        group_by: str = "context",
    ) -> DatasetSplits:
        """Partition ``examples``.

        Args:
            examples: The corpus to partition.
            seed: Partition seed.
            ratios: Target shares.
            group_by: Leakage-control strategy.

        Returns:
            The partition, with its per-group audit trail.
        """
        ...


def compute_dataset_fingerprint(
    examples: Iterable[QuestionGenerationExample], *, length: int = 16
) -> str:
    """Return a digest identifying a corpus by content.

    Order-independent: example fingerprints are sorted before hashing, so reading the same
    corpus in a different order gives the same value. That is the property that makes the
    fingerprint usable as "is this the same dataset?" rather than "was it read the same
    way?".

    Args:
        examples: The corpus.
        length: Number of leading hex characters to return.

    Returns:
        Truncated SHA-256 hex digest. Empty corpora hash consistently rather than specially.
    """
    digest = hashlib.sha256()
    for fingerprint in sorted(example.fingerprint() for example in examples):
        digest.update(fingerprint.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:length]


def find_duplicate_examples(
    examples: Sequence[QuestionGenerationExample],
) -> dict[str, tuple[str, ...]]:
    """Group example ids by content fingerprint, keeping only collisions.

    Reports rather than removes. Whether a duplicate should be dropped depends on why it is
    there -- two corpora legitimately overlapping is different from one corpus containing
    the same item twice -- so the decision belongs to the caller and
    :attr:`~qa_gen.config.QuestionGenerationDatasetConfig.drop_duplicates`.

    Args:
        examples: The corpus to inspect.

    Returns:
        A mapping of fingerprint to the ids sharing it, for fingerprints shared by two or
        more examples. Ids are sorted, and the mapping is ordered by fingerprint, so the
        report is stable.
    """
    by_fingerprint: dict[str, list[str]] = {}
    for example in examples:
        by_fingerprint.setdefault(example.fingerprint(), []).append(example.id)
    return {
        fingerprint: tuple(sorted(ids))
        for fingerprint, ids in sorted(by_fingerprint.items())
        if len(ids) > 1
    }


@dataclass(frozen=True, slots=True)
class DeterministicGroupSplitter:
    """Partitions a corpus by hashing ``(seed, group key)``. No RNG involved.

    Attributes:
        name: Identifier recorded on the partition. Versioned: if the algorithm changes,
            previously recorded splits are still identifiable as having come from the old
            one.
    """

    name: str = "deterministic-group-splitter-v1"

    def split(
        self,
        examples: Sequence[QuestionGenerationExample],
        *,
        seed: int = 42,
        ratios: SplitRatios | None = None,
        group_by: str = "context",
    ) -> DatasetSplits:
        """Partition ``examples`` into train, validation and test.

        Args:
            examples: The corpus. May be empty, which yields three empty splits rather than
                an error: an empty corpus is a validation finding, not a splitting failure.
            seed: Partition seed. Folded into every group digest, so changing it reshuffles
                the whole partition and the same seed always reproduces it.
            ratios: Target shares. Defaults to 90/5/5.
            group_by: ``"context"``, ``"topic"`` or ``"example"``. See the module docstring.

        Returns:
            The :class:`DatasetSplits`, with per-group assignments recorded.

        Raises:
            SplitError: If the ratios are invalid, the strategy is unknown, or two examples
                share an id -- an ambiguous corpus cannot be partitioned reproducibly, and
                proceeding would make the recorded assignment unusable for tracing.
        """
        resolved = ratios or SplitRatios()
        resolved.validate()
        if group_by not in ("context", "topic", "example"):
            raise SplitError(
                f"unknown group_by {group_by!r}; expected 'context', 'topic' or 'example'."
            )

        self._assert_unique_ids(examples)

        groups = self._build_groups(examples, group_by=group_by)
        ordered = sorted(
            groups.items(), key=lambda item: (_group_digest(seed, item[0]), item[0])
        )

        targets = resolved.target_counts(len(examples))
        buckets: dict[SplitName, list[QuestionGenerationExample]] = {
            name: [] for name in SplitName
        }
        assignments: list[GroupAssignment] = []

        for group_key, members in ordered:
            chosen = self._choose_split(buckets, targets)
            buckets[chosen].extend(members)
            assignments.append(
                GroupAssignment(
                    group_key=group_key,
                    split=chosen,
                    example_count=len(members),
                    example_ids=tuple(sorted(member.id for member in members)),
                    digest=_group_digest(seed, group_key)[:12],
                    sources=tuple(sorted({member.source for member in members})),
                )
            )

        return DatasetSplits(
            train=tuple(_ordered(buckets[SplitName.TRAIN])),
            validation=tuple(_ordered(buckets[SplitName.VALIDATION])),
            test=tuple(_ordered(buckets[SplitName.TEST])),
            seed=seed,
            group_by=group_by,
            ratios=resolved,
            dataset_fingerprint=compute_dataset_fingerprint(examples),
            assignments=tuple(assignments),
            splitter=self.name,
        )

    @staticmethod
    def _assert_unique_ids(examples: Sequence[QuestionGenerationExample]) -> None:
        """Refuse a corpus with repeated ids."""
        seen: dict[str, int] = {}
        for example in examples:
            seen[example.id] = seen.get(example.id, 0) + 1
        repeated = sorted(key for key, count in seen.items() if count > 1)
        if repeated:
            raise SplitError(
                f"cannot partition a corpus with duplicate example id(s): {repeated[:5]}"
                f"{' ...' if len(repeated) > 5 else ''}. Ids must be unique so a recorded "
                "assignment identifies exactly one example. Run qa_gen.validation first."
            )

    @staticmethod
    def _build_groups(
        examples: Sequence[QuestionGenerationExample], *, group_by: str
    ) -> dict[str, list[QuestionGenerationExample]]:
        """Bucket examples by their leakage key under the chosen strategy."""
        groups: dict[str, list[QuestionGenerationExample]] = {}
        for example in examples:
            if group_by == "example":
                key = f"id:{example.id}"
            elif group_by == "topic":
                # Fall back to the context group rather than lumping every unlabelled
                # example into one enormous group, which would force them all into a single
                # split and silently distort the ratios.
                key = example.group_key or example.effective_group_key()
            else:
                key = example.effective_group_key()
            groups.setdefault(key, []).append(example)
        return groups

    @staticmethod
    def _choose_split(
        buckets: dict[SplitName, list[QuestionGenerationExample]],
        targets: dict[SplitName, int],
    ) -> SplitName:
        """Return the split furthest below its target, measured proportionally.

        Proportionally, not absolutely, and the difference matters. With 90/5/5 ratios over
        30 examples the targets are 27/1/2, so an absolute shortfall always favours train and
        validation and test can end up **empty** -- the partition would technically satisfy
        the algorithm while being useless. Dividing the shortfall by the target puts the three
        on a comparable footing, so a small split gets its group early.

        Ties break by split order, so the choice is deterministic. Once every split has met
        its target the remainder goes to train: an oversized validation split costs evaluation
        time and tells you nothing extra.

        The group's size is deliberately not considered. Choosing the split whose deficit best
        *fits* the incoming group would hit the ratios more precisely, but it makes the
        assignment depend on group sizes in a way that is much harder to explain when a
        partition needs justifying. The residual error is bounded by the largest group, which
        is why :attr:`~qa_gen.statistics.DatasetStatistics.distinct_group_keys` is reported:
        a corpus with few large groups cannot honour fine-grained ratios, and that is a
        property of the corpus rather than a bug here.
        """
        ranked: list[tuple[float, int, SplitName]] = []
        for index, name in enumerate(SplitName):
            target = targets[name]
            shortfall = target - len(buckets[name])
            # A split with a zero target was not asked for at all, so it must never win.
            relative = shortfall / target if target > 0 else float("-inf")
            ranked.append((relative, -index, name))

        best = max(ranked)
        if best[0] <= 0:
            return SplitName.TRAIN
        return best[2]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"splitter": self.name}


def _group_digest(seed: int, group_key: str) -> str:
    """Return the SHA-256 digest of ``(seed, group_key)``.

    The seed is part of the hashed payload rather than an offset applied afterwards, so
    changing it produces an entirely different ordering instead of a rotation of the same
    one.
    """
    return hashlib.sha256(f"{seed}\x00{group_key}".encode()).hexdigest()


def _ordered(
    examples: Iterable[QuestionGenerationExample],
) -> list[QuestionGenerationExample]:
    """Return examples in a stable order within a split.

    Sorted by id. Within-split order affects nothing about which examples are in which
    split, but a stable order makes two runs byte-comparable and makes a diff of a written
    split file readable.
    """
    return sorted(examples, key=lambda example: example.id)
