r"""Many corpora in, one reproducible partitioned dataset out.

What this module is for
----------------------
:mod:`qa_gen.adapters` turns one record into one example. :mod:`qa_gen.validation` says which
examples are usable. :mod:`qa_gen.splitting` partitions them without leaking a passage across
splits. What was missing is the thing that runs those in the right order, applies the caps and
the deduplication that :class:`qa_gen.config.QuestionGenerationDatasetConfig` describes, and
records what happened to every example that did not survive.

Four configuration fields existed with nothing reading them -- ``max_examples``,
``max_examples_per_source``, ``drop_duplicates`` and ``shuffle_seed``. This module is where they
take effect, which is why the field docstrings and the behaviour here have to agree exactly.

Order of operations, and why this order
---------------------------------------
1. **Adapt.** Done by the caller, per source, because loading records needs I/O and this package
   has none. The caller passes the per-source rejection counts in as :class:`SourceIngestion`.
2. **Deduplicate by id.** Unconditional, before anything else can trip over it. Adapter ids are
   content digests, so a corpus containing the same record twice produces the same id twice, and
   :meth:`qa_gen.splitting.DeterministicGroupSplitter.split` *raises* on a repeated id. This is
   not the same decision as ``drop_duplicates``: an ambiguous id makes the corpus
   unpartitionable, whereas duplicated content is a judgement call.
3. **Validate, then drop errors.** Warnings are kept and reported; errors are dropped. Both are
   counted. Duplicate *content* is a warning, which is what makes step 4 safe -- were it an
   error, dropping invalid examples would discard both copies rather than one.
4. **Deduplicate by content** when ``drop_duplicates`` is set, keeping the lowest id.
5. **Cap per source**, then **cap overall**. Caps are applied to *usable* examples, after
   validation rather than before, so ``max_examples_per_source=1000`` yields a thousand
   trainable examples rather than a thousand candidates of which some fraction was junk.
6. **Split**, grouped, so a passage cannot straddle two splits.
7. **Order within each split** from ``shuffle_seed``, which is deliberately a different seed
   from the split seed: example order may change without any example changing split.

Determinism, and the trap in the obvious implementation
-------------------------------------------------------
Every selection here is a sort by ``sha256(seed, example_id)``, never an RNG and never input
order. That makes the result independent of the order records arrived in, which matters because
a dataset library may shard or shuffle its own reads.

The trap: capping by sorting on the id itself. Adapter ids are ``{source_id}-{digest}``, so
sorting a mixed corpus by id sorts by *source name* first -- ``edu-mcq-...`` before
``squad-qg-...`` -- and a global cap would then take every MCQ example before a single SQuAD
one. Hashing the id with the seed removes the prefix's influence, so a cap samples across
sources in proportion to their size. :func:`select_examples` is the one place this is done.

What is reported
----------------
:class:`PreparationReport` records the surviving count after each stage, the exact number
removed by each mechanism, and a bounded sample of the ids and reasons. Counts are exact;
samples are truncated, because a full SQuAD ingestion can drop tens of thousands of examples
and a report nobody can open is not evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qa_gen.config import QuestionGenerationDatasetConfig
from qa_gen.examples import QuestionGenerationExample
from qa_gen.metadata import DatasetMetadata
from qa_gen.splitting import (
    DatasetSplits,
    DeterministicGroupSplitter,
    SplitName,
    SplitRatios,
    compute_dataset_fingerprint,
    find_duplicate_examples,
)
from qa_gen.statistics import DatasetStatistics, compute_statistics
from qa_gen.validation import DatasetValidationReport, validate_dataset

__all__ = [
    "PREPARATION_STAGES",
    "PreparationError",
    "PreparationReport",
    "PreparedDataset",
    "SourceIngestion",
    "deduplicate_by_content",
    "deduplicate_by_id",
    "order_examples",
    "prepare_dataset",
    "select_examples",
]

#: The stages a corpus passes through, in order. Recorded as a list of surviving counts so a
#: reader can see where a corpus lost most of its examples without diffing two reports.
PREPARATION_STAGES: tuple[str, ...] = (
    "adapted",
    "unique_ids",
    "valid",
    "deduplicated",
    "capped_per_source",
    "capped_total",
)

#: How many dropped ids and rejection messages a report keeps. Counts stay exact; the samples
#: are for diagnosis, and a hundred thousand of them would make the artifact unusable.
_SAMPLE_LIMIT = 20


class PreparationError(ValueError):
    """Raised when a corpus cannot be prepared at all.

    Distinct from the validation report, which describes examples. This is raised when the
    *request* cannot be honoured -- an unknown source name, or every example filtered out --
    because continuing would produce an empty dataset and a training run that reported a
    plausible loss over nothing.
    """


@dataclass(frozen=True, slots=True)
class SourceIngestion:
    """What one corpus contributed, and what its adapter refused.

    Built by the caller that did the loading, because the record count is only knowable there.
    Carried into the report so a source that silently contributed almost nothing is visible.

    Attributes:
        source_id: The adapter source id.
        dataset_id: Where the records came from, for provenance. A Hub id, a local path, or a
            description.
        records_seen: Records handed to the adapter.
        examples_adapted: Examples the adapter produced.
        rejected: Records the adapter refused.
        rejection_samples: A bounded sample of refusal messages, for diagnosis.
        notes: Anything else worth recording about this source.
    """

    source_id: str
    dataset_id: str = ""
    records_seen: int = 0
    examples_adapted: int = 0
    rejected: int = 0
    rejection_samples: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "rejection_samples", tuple(self.rejection_samples))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def rejection_rate(self) -> float:
        """Share of records the adapter refused, rounded.

        Worth reporting rather than deriving on demand: LearningQ is expected to refuse a large
        share because most of its items have no written answer, and a rate that is *unexpectedly*
        high is the first sign a corpus has been renamed underneath the adapter.
        """
        if not self.records_seen:
            return 0.0
        return round(self.rejected / self.records_seen, 4)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "records_seen": self.records_seen,
            "examples_adapted": self.examples_adapted,
            "rejected": self.rejected,
            "rejection_rate": self.rejection_rate,
            "rejection_samples": list(self.rejection_samples),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PreparationReport:
    """Where every example went, stage by stage.

    Attributes:
        sources: Per-corpus ingestion records.
        stage_counts: Surviving example count after each of :data:`PREPARATION_STAGES`.
        duplicate_ids_dropped: Removed because another example already had that id.
        invalid_dropped: Removed because validation found an error.
        duplicate_content_dropped: Removed by ``drop_duplicates``.
        per_source_capped: Removed by ``max_examples_per_source``.
        total_capped: Removed by ``max_examples``.
        kept: Examples that reached the splitter.
        dropped_samples: A bounded sample of ``example_id`` to reason.
        notes: Anything worth recording about the preparation itself.
    """

    sources: tuple[SourceIngestion, ...] = ()
    stage_counts: dict[str, int] = field(default_factory=dict)
    duplicate_ids_dropped: int = 0
    invalid_dropped: int = 0
    duplicate_content_dropped: int = 0
    per_source_capped: int = 0
    total_capped: int = 0
    kept: int = 0
    dropped_samples: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def records_seen(self) -> int:
        """Total records handed to adapters across every source."""
        return sum(source.records_seen for source in self.sources)

    @property
    def adapter_rejected(self) -> int:
        """Total records refused by adapters across every source."""
        return sum(source.rejected for source in self.sources)

    @property
    def dropped(self) -> int:
        """Examples that were adapted but did not reach the splitter."""
        return (
            self.duplicate_ids_dropped
            + self.invalid_dropped
            + self.duplicate_content_dropped
            + self.per_source_capped
            + self.total_capped
        )

    @property
    def invalid_rate(self) -> float:
        """Share of adapted examples that validation rejected, rounded."""
        adapted = self.stage_counts.get("adapted", 0)
        if not adapted:
            return 0.0
        return round(self.invalid_dropped / adapted, 4)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "sources": [source.as_dict() for source in self.sources],
            "records_seen": self.records_seen,
            "adapter_rejected": self.adapter_rejected,
            "stage_counts": {
                stage: self.stage_counts.get(stage, 0) for stage in PREPARATION_STAGES
            },
            "duplicate_ids_dropped": self.duplicate_ids_dropped,
            "invalid_dropped": self.invalid_dropped,
            "invalid_rate": self.invalid_rate,
            "duplicate_content_dropped": self.duplicate_content_dropped,
            "per_source_capped": self.per_source_capped,
            "total_capped": self.total_capped,
            "dropped": self.dropped,
            "kept": self.kept,
            "dropped_samples": dict(sorted(self.dropped_samples.items())),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """A partitioned, validated corpus with everything needed to reproduce it.

    Attributes:
        splits: The partition.
        metadata: The Phase 17A dataset record, including both fingerprints.
        report: What happened to every example.
        validation: The full validation report, warnings included.
        statistics: Statistics over the kept examples.
        split_statistics: Statistics per split, so a skewed test split is visible.
        duplicate_groups: Content fingerprints shared by two or more *kept* examples. Empty
            when ``drop_duplicates`` is on; non-empty otherwise, and worth seeing either way.
    """

    splits: DatasetSplits
    metadata: DatasetMetadata
    report: PreparationReport
    validation: DatasetValidationReport
    statistics: DatasetStatistics
    split_statistics: dict[str, DatasetStatistics] = field(default_factory=dict)
    duplicate_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def examples(self) -> tuple[QuestionGenerationExample, ...]:
        """Every kept example, train then validation then test."""
        return self.splits.train + self.splits.validation + self.splits.test

    @property
    def fingerprint(self) -> str:
        """Content digest of the kept corpus."""
        return self.splits.dataset_fingerprint

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary.

        The examples are not inlined: a summary belongs in a report, and a full corpus in it
        would make the report unreadable. The split assignments in
        :attr:`~qa_gen.splitting.DatasetSplits.assignments` identify every example by id.
        """
        return {
            "fingerprint": self.fingerprint,
            "splits": self.splits.as_dict(),
            "metadata": self.metadata.as_dict(),
            "preparation": self.report.as_dict(),
            "validation": self.validation.as_dict(),
            "statistics": self.statistics.as_dict(),
            "split_statistics": {
                name: stats.as_dict() for name, stats in sorted(self.split_statistics.items())
            },
            "duplicate_groups": {
                key: list(ids) for key, ids in sorted(self.duplicate_groups.items())
            },
        }


def _selection_digest(seed: int, example_id: str) -> str:
    """Return the digest that orders an example for selection or shuffling.

    The seed is inside the hashed payload rather than applied afterwards, matching
    :func:`qa_gen.splitting._group_digest`, so changing it reorders everything instead of
    rotating one order into another.

    Args:
        seed: The seed to fold in.
        example_id: The example id.

    Returns:
        A hex digest.
    """
    return hashlib.sha256(f"{seed}\x00{example_id}".encode()).hexdigest()


def order_examples(
    examples: Iterable[QuestionGenerationExample], *, seed: int
) -> tuple[QuestionGenerationExample, ...]:
    """Return ``examples`` in a deterministic, seed-dependent order.

    Used for within-split ordering, which is what ``shuffle_seed`` controls. Independent of the
    input order, so two runs that read the same corpus in different orders produce the same
    sequence.

    Args:
        examples: The examples to order.
        seed: Ordering seed.

    Returns:
        The examples, ordered by ``(digest, id)``.
    """
    return tuple(
        sorted(examples, key=lambda item: (_selection_digest(seed, item.id), item.id))
    )


def select_examples(
    examples: Sequence[QuestionGenerationExample], limit: int | None, *, seed: int
) -> tuple[tuple[QuestionGenerationExample, ...], tuple[str, ...]]:
    """Take at most ``limit`` examples, deterministically and without source bias.

    Selection is by ``sha256(seed, id)`` rather than by id, for the reason in the module
    docstring: ids carry a source prefix, so sorting on them would empty one corpus before
    touching another.

    Args:
        examples: Candidates.
        limit: Maximum to keep. ``None`` or a value at or above the input size keeps
            everything and drops nothing.
        seed: Selection seed.

    Returns:
        ``(kept, dropped_ids)``. ``kept`` is in id order so downstream stages do not inherit a
        selection artefact as an ordering; the split's own ordering is applied later.

    Raises:
        PreparationError: If ``limit`` is zero or negative. A cap of zero is almost certainly a
            mistake, and honouring it would produce an empty dataset from a valid corpus.
    """
    if limit is not None and limit <= 0:
        raise PreparationError(
            f"a selection limit must be a positive integer or None, got {limit}. A cap of zero "
            "would discard an entire corpus while looking like a configuration choice."
        )
    if limit is None or len(examples) <= limit:
        return tuple(sorted(examples, key=lambda item: item.id)), ()

    ranked = sorted(examples, key=lambda item: (_selection_digest(seed, item.id), item.id))
    kept = ranked[:limit]
    dropped = ranked[limit:]
    return (
        tuple(sorted(kept, key=lambda item: item.id)),
        tuple(sorted(item.id for item in dropped)),
    )


def deduplicate_by_id(
    examples: Sequence[QuestionGenerationExample],
) -> tuple[tuple[QuestionGenerationExample, ...], tuple[str, ...]]:
    """Keep one example per id.

    Unconditional in :func:`prepare_dataset`, because a repeated id makes a corpus
    unpartitionable rather than merely redundant. Adapter ids are content digests, so this
    removes exact record duplicates within or across corpora.

    Args:
        examples: Candidates.

    Returns:
        ``(kept, dropped_ids)``, kept in id order. Which copy survives does not matter -- they
        have the same id and therefore the same derived content -- but the choice is made
        deterministically anyway, by keeping the first in id-sorted input order.
    """
    seen: dict[str, QuestionGenerationExample] = {}
    dropped: list[str] = []
    for example in sorted(examples, key=lambda item: item.id):
        if example.id in seen:
            dropped.append(example.id)
            continue
        seen[example.id] = example
    return tuple(seen.values()), tuple(dropped)


def deduplicate_by_content(
    examples: Sequence[QuestionGenerationExample],
) -> tuple[tuple[QuestionGenerationExample, ...], tuple[str, ...]]:
    """Keep one example per content fingerprint, the one with the lowest id.

    "Lowest id" is what
    :attr:`qa_gen.config.QuestionGenerationDatasetConfig.drop_duplicates` documents. Since ids
    are digests the choice is arbitrary, which is the point: it must not depend on which corpus
    happened to be read first, or the surviving copy's ``source`` would vary between runs and
    the per-source counts with it.

    Args:
        examples: Candidates.

    Returns:
        ``(kept, dropped_ids)``, kept in id order.
    """
    by_fingerprint: dict[str, QuestionGenerationExample] = {}
    dropped: list[str] = []
    for example in sorted(examples, key=lambda item: item.id):
        fingerprint = example.fingerprint()
        if fingerprint in by_fingerprint:
            dropped.append(example.id)
            continue
        by_fingerprint[fingerprint] = example
    return (
        tuple(sorted(by_fingerprint.values(), key=lambda item: item.id)),
        tuple(sorted(dropped)),
    )


def _cap_per_source(
    examples: Sequence[QuestionGenerationExample], limit: int | None, *, seed: int
) -> tuple[tuple[QuestionGenerationExample, ...], tuple[str, ...]]:
    """Apply a per-corpus cap, independently within each source.

    Args:
        examples: Candidates from every source, mixed.
        limit: Cap per source, or ``None``.
        seed: Selection seed.

    Returns:
        ``(kept, dropped_ids)``, kept in id order.
    """
    if limit is None:
        return tuple(sorted(examples, key=lambda item: item.id)), ()

    grouped: dict[str, list[QuestionGenerationExample]] = {}
    for example in examples:
        grouped.setdefault(example.source, []).append(example)

    kept: list[QuestionGenerationExample] = []
    dropped: list[str] = []
    for source in sorted(grouped):
        survivors, removed = select_examples(grouped[source], limit, seed=seed)
        kept.extend(survivors)
        dropped.extend(removed)
    return tuple(sorted(kept, key=lambda item: item.id)), tuple(sorted(dropped))


def _record_drops(
    samples: dict[str, str], example_ids: Sequence[str], reason: str
) -> None:
    """Add a bounded sample of dropped ids to the report, in place."""
    for example_id in example_ids:
        if len(samples) >= _SAMPLE_LIMIT:
            return
        samples.setdefault(example_id, reason)


def prepare_dataset(
    adapted: Mapping[str, Sequence[QuestionGenerationExample]],
    config: QuestionGenerationDatasetConfig,
    *,
    ingestion: Sequence[SourceIngestion] = (),
    splitter: DeterministicGroupSplitter | None = None,
    config_hash: str = "",
    notes: Sequence[str] = (),
) -> PreparedDataset:
    """Run the deterministic preparation pipeline over already-adapted examples.

    Takes adapted examples rather than raw records because loading them needs I/O and this
    package deliberately has none; :mod:`qa_gen_runtime.sources` is the half that reads.

    Args:
        adapted: Examples grouped by source id. The grouping is used for the per-source cap and
            for the reported ordering; ``example.source`` remains the authority.
        config: The dataset configuration. Every field it documents takes effect here.
        ingestion: Per-source load records, for the report. Optional, because a caller working
            from in-memory examples has no record counts to give.
        splitter: The splitter to use. Defaults to
            :class:`qa_gen.splitting.DeterministicGroupSplitter`.
        config_hash: Hash of the enclosing experiment configuration, recorded in the metadata so
            a prepared dataset can be tied to the run that asked for it.
        notes: Extra notes for the report.

    Returns:
        The :class:`PreparedDataset`.

    Raises:
        PreparationError: If no examples were supplied, or every example was filtered out. Both
            would otherwise produce an empty dataset that trains successfully on nothing.
        qa_gen.config.GenerationConfigError: If the configuration is invalid.
        qa_gen.splitting.SplitError: If the partition cannot be produced.
    """
    config.validate()

    pooled: list[QuestionGenerationExample] = []
    for source in sorted(adapted):
        pooled.extend(adapted[source])

    if not pooled:
        raise PreparationError(
            "no examples were supplied, so there is nothing to prepare. Either every source "
            "returned zero records or every record was refused by its adapter; the per-source "
            "ingestion counts say which."
        )

    stage_counts: dict[str, int] = {"adapted": len(pooled)}
    samples: dict[str, str] = {}
    stage_notes = list(notes)

    unique, duplicate_ids = deduplicate_by_id(pooled)
    stage_counts["unique_ids"] = len(unique)
    _record_drops(samples, duplicate_ids, "duplicate_example_id")

    validation = validate_dataset(unique, config=config, allow_empty=True)
    invalid_ids = validation.invalid_example_ids
    valid = tuple(example for example in unique if example.id not in invalid_ids)
    stage_counts["valid"] = len(valid)
    _record_drops(samples, sorted(invalid_ids), "validation_error")

    if config.drop_duplicates:
        deduplicated, duplicate_content = deduplicate_by_content(valid)
        _record_drops(samples, duplicate_content, "duplicate_content")
    else:
        deduplicated, duplicate_content = valid, ()
        stage_notes.append(
            "drop_duplicates is off: examples sharing a content fingerprint were kept, and "
            "grouped splitting is the only thing preventing the same content reaching two "
            "splits"
        )
    stage_counts["deduplicated"] = len(deduplicated)

    per_source, per_source_dropped = _cap_per_source(
        deduplicated, config.max_examples_per_source, seed=config.seed
    )
    stage_counts["capped_per_source"] = len(per_source)
    _record_drops(samples, per_source_dropped, "max_examples_per_source")

    kept, total_dropped = select_examples(per_source, config.max_examples, seed=config.seed)
    stage_counts["capped_total"] = len(kept)
    _record_drops(samples, total_dropped, "max_examples")

    if not kept:
        raise PreparationError(
            f"every example was filtered out: {len(pooled)} adapted, "
            f"{len(duplicate_ids)} duplicate ids, {len(invalid_ids)} invalid, "
            f"{len(duplicate_content)} duplicate content. Check the validation report's "
            "code counts before changing the configuration; a context-length bound that "
            "excludes the whole corpus is the usual cause."
        )

    splits = (splitter or DeterministicGroupSplitter()).split(
        kept,
        seed=config.seed,
        ratios=SplitRatios(
            train=config.train_ratio,
            validation=config.validation_ratio,
            test=config.test_ratio,
        ),
        group_by=config.group_by,
    )
    splits = _reorder_splits(splits, seed=config.shuffle_seed)

    report = PreparationReport(
        sources=tuple(ingestion),
        stage_counts=stage_counts,
        duplicate_ids_dropped=len(duplicate_ids),
        invalid_dropped=len(invalid_ids),
        duplicate_content_dropped=len(duplicate_content),
        per_source_capped=len(per_source_dropped),
        total_capped=len(total_dropped),
        kept=len(kept),
        dropped_samples=samples,
        notes=tuple(stage_notes),
    )

    statistics = compute_statistics(kept)
    split_statistics = {
        name.value: compute_statistics(splits[name]) for name in SplitName
    }
    metadata = DatasetMetadata(
        fingerprint=compute_dataset_fingerprint(kept),
        config_hash=config_hash,
        sources=tuple(sorted({example.source for example in kept})),
        split_summary=splits.as_dict(),
        statistics=statistics,
        split_statistics=split_statistics,
        validation_summary=validation.as_dict(),
        notes=tuple(stage_notes),
    )

    return PreparedDataset(
        splits=splits,
        metadata=metadata,
        report=report,
        validation=validation,
        statistics=statistics,
        split_statistics=split_statistics,
        duplicate_groups=find_duplicate_examples(kept),
    )


def _reorder_splits(splits: DatasetSplits, *, seed: int) -> DatasetSplits:
    """Return ``splits`` with each split reordered from ``shuffle_seed``.

    The splitter sorts within a split by id, which is stable but groups every example from one
    source together -- ids share a source prefix. Training through that order would show the
    model one corpus at a time, which is a curriculum nobody chose.

    Only the order changes. No example moves between splits, which is exactly why
    ``shuffle_seed`` is a separate field from ``seed``: the partition and its fingerprint are
    untouched.

    Args:
        splits: The partition to reorder.
        seed: Ordering seed.

    Returns:
        A new :class:`~qa_gen.splitting.DatasetSplits` with the same membership.
    """
    return DatasetSplits(
        train=order_examples(splits.train, seed=seed),
        validation=order_examples(splits.validation, seed=seed),
        test=order_examples(splits.test, seed=seed),
        seed=splits.seed,
        group_by=splits.group_by,
        ratios=splits.ratios,
        dataset_fingerprint=splits.dataset_fingerprint,
        assignments=splits.assignments,
        splitter=splits.splitter,
    )
