r"""Prepare the real question-generation dataset, and measure what training it would cost.

What this command is for
-----------------------
::

    python -m qa_gen_runtime.prepare --config ml/configs/qgen/qgen-smoke.yaml --plan
    python -m qa_gen_runtime.prepare --config ... --allow-download
    python -m qa_gen_runtime.prepare --config ... --allow-download --write-dataset

It reads the corpora a configuration names, maps them through the Phase 17A adapters, validates
and deduplicates and caps them, partitions them without leaking a passage, tokenizes the
rendered records with the *real* tokenizer, and writes one machine-readable report under
``artifacts/``. It does not train, and it loads no model weights.

``--plan`` is the free version: it resolves which corpora would be read, from where, and whether
each needs a download or a local file, then stops. Nothing is fetched and nothing is written.
Worth running first, because the answer to "will this pull nine gigabytes" should not itself pull
nine gigabytes.

Nothing downloads by default
----------------------------
``--allow-download`` is required to reach the network. Without it,
:mod:`qa_gen_runtime.sources` forces the Hugging Face libraries offline, so a corpus that is
already cached is read and one that is not produces an error naming the flag. Two of the four
supported corpora have no Hub mirror at all and need ``--source-path``; ``--plan`` says which.

Tokenizer only
--------------
Sizing loads the tokenizer and stops -- a few megabytes rather than eight gigabytes. That is
what makes this runnable on a laptop and alongside a busy GPU, and it is asserted by a test
rather than left as an intention. ``--no-token-stats`` skips even that, for a structural audit
on a machine with no tokenizer cached; the report then says the lengths were not measured rather
than estimating them from character counts.

Exit codes
----------
``0`` prepared. ``1`` a configuration, source or sizing failure. ``2`` argparse usage error.
``3`` prepared, but an audit check did not pass -- a split missing a source that training has,
or a truncation rate above the threshold. Distinct from ``1`` because "it worked and the numbers
are wrong" needs a different response from "it did not work".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_gen.adapters import AdapterError
from qa_gen.config import GenerationConfigError, GenerationExperimentConfig
from qa_gen.preparation import PreparationError, PreparedDataset, prepare_dataset
from qa_gen.splitting import SplitError, SplitName
from qa_gen_runtime.config_io import ConfigIOError, load_experiment_config
from qa_gen_runtime.dataset import build_training_records
from qa_gen_runtime.deps import RuntimeDependencyError
from qa_gen_runtime.sizing import (
    DEFAULT_STEP_PLANS,
    DatasetSizing,
    SizingError,
    SplitSizing,
    build_step_estimates,
    estimate_tokenization_seconds,
    load_sizing_tokenizer,
    measure_record_lengths,
    summarize_token_lengths,
)
from qa_gen_runtime.sources import (
    LoadedSource,
    SourceLoadError,
    adapt_source,
    describe_catalogue,
    describe_requirements,
    load_source_records,
    request_is_readable,
    resolve_requests,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_SUBDIR",
    "AuditFinding",
    "PreparationOutcome",
    "audit_dataset",
    "build_parser",
    "format_outcome",
    "main",
    "run_preparation",
]

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_AUDIT_FAILED = 3

#: Directory under the artifacts root holding prepared datasets and their reports. Separate from
#: ``qgen-runs`` because a dataset outlives the runs that consume it.
DATASET_SUBDIR = "qgen-datasets"

#: Truncation above this share is reported as a finding. Not zero: a handful of over-long
#: passages is normal and filtering them is a configuration change, whereas one example in
#: twenty losing its target is a silent data-quality problem.
_TRUNCATION_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One thing about a prepared dataset that a reader should look at.

    Attributes:
        code: Short machine-readable identifier.
        message: What was found, and why it matters.
        blocking: Whether this should change the exit code. A finding that is merely
            informative must not fail a pipeline.
    """

    code: str
    message: str
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"code": self.code, "message": self.message, "blocking": self.blocking}


@dataclass(frozen=True, slots=True)
class PreparationOutcome:
    """Everything one invocation produced.

    Attributes:
        config_path: The configuration that was read.
        experiment: The experiment name.
        config_hash: Hash of the resolved configuration.
        planned: Whether this was a ``--plan`` invocation, which reads nothing.
        requests: The resolved source read requests.
        catalogue: What is known about every supported corpus.
        loaded: Per-source load summaries.
        prepared: The prepared dataset, absent under ``--plan``.
        sizing: Token-length measurements and step estimates.
        tokenization: How long tokenizing took, or an estimate when it was skipped.
        findings: Audit findings.
        artifacts: Files written.
        notes: Anything else worth recording.
    """

    config_path: str
    experiment: str = ""
    config_hash: str = ""
    planned: bool = False
    requests: tuple[Any, ...] = ()
    catalogue: dict[str, Any] = field(default_factory=dict)
    loaded: tuple[LoadedSource, ...] = ()
    prepared: PreparedDataset | None = None
    sizing: DatasetSizing | None = None
    tokenization: dict[str, Any] = field(default_factory=dict)
    findings: tuple[AuditFinding, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        for name in ("requests", "loaded", "findings", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def ok(self) -> bool:
        """Whether no blocking finding was raised."""
        return not any(finding.blocking for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "config_path": self.config_path,
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "planned": self.planned,
            "ok": self.ok,
            "requests": [request.as_dict() for request in self.requests],
            "catalogue": dict(self.catalogue),
            "sources": [source.as_dict() for source in self.loaded],
            "dataset": self.prepared.as_dict() if self.prepared else None,
            "sizing": self.sizing.as_dict() if self.sizing else None,
            "tokenization": dict(self.tokenization),
            "findings": [finding.as_dict() for finding in self.findings],
            "artifacts": dict(self.artifacts),
            "notes": list(self.notes),
        }


def audit_dataset(
    prepared: PreparedDataset, sizing: DatasetSizing | None
) -> tuple[AuditFinding, ...]:
    """Check a prepared dataset for the problems that make a metric meaningless.

    Not a validation pass -- :mod:`qa_gen.validation` already ran per example. These are
    dataset-level properties that only exist once the partition does, and each one has been the
    cause of a confusing number at some point:

    - A source present in training but absent from validation or test. Its metric would be
      measured over zero examples and read as a model failure.
    - An empty split, which makes checkpoint selection or the final number impossible.
    - Truncation, which removes the end of the target -- the part carrying the answer.
    - Leaked group keys, which would make every metric optimistic. Always empty for a partition
      this pipeline produced, which is why asserting it is worth the two lines.

    Args:
        prepared: The prepared dataset.
        sizing: Token measurements, when they were taken.

    Returns:
        The findings, in a stable order.
    """
    findings: list[AuditFinding] = []
    breakdown = prepared.splits.source_breakdown()
    train_sources = set(breakdown.get(SplitName.TRAIN.value, {}))

    leaked = sorted(prepared.splits.leaked_group_keys())
    if leaked:
        # group_by="example" exists so the effect of grouping can be measured, which means it
        # leaks on purpose. Reporting it is right; failing the run for doing what was asked is
        # not, so the finding is downgraded rather than suppressed.
        deliberate = prepared.splits.group_by == "example"
        findings.append(
            AuditFinding(
                code="group_leakage",
                message=(
                    f"{len(leaked)} leakage group(s) appear in more than one split, e.g. "
                    f"{leaked[:3]}. Every metric would be optimistic."
                    + (
                        " group_by is 'example', so leakage control was switched off "
                        "deliberately; this is the measurement that option exists for, and "
                        "no headline number should come from it."
                        if deliberate
                        else ""
                    )
                ),
                blocking=not deliberate,
            )
        )

    for name in SplitName:
        examples = breakdown.get(name.value, {})
        if not examples:
            findings.append(
                AuditFinding(
                    code=f"empty_{name.value}_split",
                    message=(
                        f"the {name.value} split is empty. "
                        + (
                            "There is nothing to fine-tune on."
                            if name is SplitName.TRAIN
                            else "Checkpoint selection or the final number is impossible; "
                            "with few large leakage groups the splitter cannot honour fine "
                            "ratios, so either widen the ratio or group by 'context'."
                        )
                    ),
                )
            )
            continue
        missing = sorted(train_sources - set(examples))
        if missing and name is not SplitName.TRAIN:
            findings.append(
                AuditFinding(
                    code=f"source_missing_from_{name.value}",
                    message=(
                        f"source(s) {missing} are in train but absent from {name.value}, so "
                        "their metrics would be measured over zero examples. This is usually "
                        "a corpus with few, very large leakage groups: check "
                        "distinct_group_keys per source."
                    ),
                    blocking=False,
                )
            )

    if sizing is not None and sizing.measured and sizing.overall is not None:
        rate = sizing.overall.truncation_rate
        if rate > _TRUNCATION_THRESHOLD:
            findings.append(
                AuditFinding(
                    code="truncation_above_threshold",
                    message=(
                        f"{sizing.overall.truncated} of {sizing.overall.examples} sequences "
                        f"({100 * rate:.2f}%) exceed max_seq_length "
                        f"{sizing.overall.max_seq_length}, above the "
                        f"{100 * _TRUNCATION_THRESHOLD:.0f}% threshold. Truncation removes the "
                        "end of the target, which is where the answer is. Lower "
                        "dataset.max_context_chars or raise model.max_seq_length -- the latter "
                        "is an untested extrapolation from the measured baseline."
                    ),
                )
            )

    return tuple(findings)


def _load_sources(
    requests: Sequence[Any], *, allow_download: bool, strict: bool
) -> tuple[LoadedSource, ...]:
    """Read and adapt every requested corpus.

    Args:
        requests: Resolved read requests.
        allow_download: Permit network fetches.
        strict: Make the first adapter refusal fatal instead of counting it.

    Returns:
        One :class:`~qa_gen_runtime.sources.LoadedSource` per request.

    Raises:
        qa_gen_runtime.sources.SourceLoadError: If a corpus cannot be read.
        qa_gen.adapters.AdapterError: If a record is refused and ``strict`` is set.
    """
    loaded: list[LoadedSource] = []
    for request in requests:
        records = load_source_records(request, allow_download=allow_download)
        logger.info("%s: read %d record(s)", request.source_id, len(records))
        loaded.append(adapt_source(request, records, skip_invalid=not strict))
    return tuple(loaded)


def _measure(
    prepared: PreparedDataset,
    config: GenerationExperimentConfig,
    *,
    plans: Sequence[tuple[int, int, int]],
    sample: int | None,
    seconds_per_step: float | None,
    token_stats: bool,
) -> tuple[DatasetSizing, dict[str, Any]]:
    """Tokenize the prepared splits and assemble the sizing report.

    Args:
        prepared: The prepared dataset.
        config: The experiment configuration.
        plans: Step-estimate configurations.
        sample: Measure at most this many records per split. ``None`` measures everything.
        seconds_per_step: A measured training rate, for wall-clock estimates.
        token_stats: Whether to load a tokenizer at all.

    Returns:
        ``(sizing, tokenization_timing)``.

    Raises:
        qa_gen_runtime.sizing.SizingError: If the tokenizer cannot be loaded or a record is the
            wrong shape.
    """
    train_examples = len(prepared.splits.train)
    estimates = build_step_estimates(
        train_examples,
        plans=plans,
        warmup_ratio=config.training.warmup_ratio,
        seconds_per_step=seconds_per_step,
    )

    if not token_stats:
        return (
            DatasetSizing(
                tokenizer_id=config.model.effective_tokenizer_id,
                max_seq_length=config.model.max_seq_length,
                measured=False,
                step_estimates=estimates,
                notes=(
                    "token statistics were skipped with --no-token-stats. No character-based "
                    "estimate is substituted: an estimate that disagrees with the tokenizer "
                    "would be worse than an absent measurement.",
                ),
            ),
            estimate_tokenization_seconds(len(prepared.examples)),
        )

    tokenizer = load_sizing_tokenizer(config)
    started = time.perf_counter()
    per_split: dict[str, SplitSizing] = {}
    measured_records = 0

    for name in SplitName:
        examples = prepared.splits[name]
        if sample is not None:
            examples = examples[:sample]
        if not examples:
            continue
        records = build_training_records(examples, config, tokenizer=tokenizer)
        per_split[name.value] = measure_record_lengths(
            records, tokenizer, config, split=name.value
        )
        measured_records += len(records)

    elapsed = time.perf_counter() - started
    overall = summarize_token_lengths(per_split)
    notes = [
        "lengths are token counts from the real tokenizer, applying the real chat template, "
        "over the records the trainer receives",
        "the overall p50 and p95 are the widest of the per-split percentiles rather than a "
        "pooled percentile; the counts, means and extremes are exact",
    ]
    if sample is not None:
        notes.append(
            f"each split was sampled to at most {sample} records, so the length "
            "distributions describe a subset"
        )

    return (
        DatasetSizing(
            tokenizer_id=config.model.effective_tokenizer_id,
            max_seq_length=config.model.max_seq_length,
            measured=True,
            splits=per_split,
            overall=overall,
            step_estimates=estimates,
            sampled=sample,
            notes=tuple(notes),
        ),
        {
            "example_count": measured_records,
            "measured": True,
            "elapsed_seconds": round(elapsed, 2),
            "records_per_second": (
                round(measured_records / elapsed, 1) if elapsed > 0 else None
            ),
            "note": (
                "wall clock for tokenizing every measured record twice (prompt, then prompt "
                "plus completion), single process"
            ),
        },
    )


def run_preparation(
    config_path: str,
    *,
    plan_only: bool = False,
    allow_download: bool = False,
    local_paths: dict[str, str] | None = None,
    splits: dict[str, str] | None = None,
    read_limit: int | None = None,
    strict_adapters: bool = False,
    token_stats: bool = True,
    sample: int | None = None,
    plans: Sequence[tuple[int, int, int]] = DEFAULT_STEP_PLANS,
    seconds_per_step: float | None = None,
    output_dir: str | None = None,
    write_dataset: bool = False,
) -> PreparationOutcome:
    """Prepare and size a dataset, or just plan the reads.

    Args:
        config_path: Path to the experiment configuration.
        plan_only: Resolve the reads and stop. Fetches nothing, writes nothing.
        allow_download: Permit network fetches.
        local_paths: Source id to local JSON/JSONL path.
        splits: Source id to upstream split override.
        read_limit: Read at most this many records per source.
        strict_adapters: Make the first adapter refusal fatal.
        token_stats: Load the tokenizer and measure token lengths.
        sample: Measure at most this many records per split.
        plans: ``(batch, accumulation, epochs)`` triples for step estimation.
        seconds_per_step: A measured training rate, for wall-clock estimates.
        output_dir: Where to write. Defaults to ``<artifacts>/qgen-datasets``.
        write_dataset: Also write the split examples, not just the report.

    Returns:
        The :class:`PreparationOutcome`.

    Raises:
        qa_gen_runtime.config_io.ConfigIOError: If the configuration cannot be read.
        qa_gen.config.GenerationConfigError: If it is invalid.
        qa_gen_runtime.sources.SourceLoadError: If a corpus cannot be read.
        qa_gen.preparation.PreparationError: If the corpus cannot be prepared.
        qa_gen_runtime.sizing.SizingError: If sizing fails.
        qa_gen.splitting.SplitError: If the partition cannot be produced.
    """
    config = load_experiment_config(config_path)
    requests = resolve_requests(
        config.dataset.sources,
        local_paths=local_paths,
        splits=splits,
        limit=read_limit,
        require_readable=not plan_only,
    )
    catalogue = describe_catalogue()

    if plan_only:
        unreadable = [
            request.source_id for request in requests if not request_is_readable(request)
        ]
        notes = [
            "--plan resolved the reads and stopped: nothing was downloaded, no tokenizer was "
            "loaded and no file was written",
            f"downloads would be {'allowed' if allow_download else 'refused'} for the real run",
            *describe_requirements(requests),
        ]
        if unreadable:
            notes.append(
                f"source(s) {unreadable} cannot be read yet; supply --source-path for each "
                "before running without --plan"
            )
        return PreparationOutcome(
            config_path=config_path,
            experiment=config.name,
            config_hash=config.config_hash(),
            planned=True,
            requests=requests,
            catalogue=catalogue,
            notes=tuple(notes),
        )

    loaded = _load_sources(requests, allow_download=allow_download, strict=strict_adapters)
    adapted = {source.request.source_id: list(source.examples) for source in loaded}
    prepared = prepare_dataset(
        adapted,
        config.dataset,
        ingestion=[source.ingestion for source in loaded],
        config_hash=config.config_hash(),
        notes=(f"prepared by qa_gen_runtime.prepare from {config_path}",),
    )

    sizing, tokenization = _measure(
        prepared,
        config,
        plans=plans,
        sample=sample,
        seconds_per_step=seconds_per_step,
        token_stats=token_stats,
    )
    findings = audit_dataset(prepared, sizing)

    outcome = PreparationOutcome(
        config_path=config_path,
        experiment=config.name,
        config_hash=config.config_hash(),
        requests=requests,
        catalogue=catalogue,
        loaded=loaded,
        prepared=prepared,
        sizing=sizing,
        tokenization=tokenization,
        findings=findings,
        notes=(
            "no model weights were loaded; sizing uses the tokenizer only",
            f"downloads were {'allowed' if allow_download else 'refused (cache only)'}",
        ),
    )
    artifacts = _write_artifacts(
        outcome, prepared, output_dir=output_dir, write_dataset=write_dataset
    )
    return PreparationOutcome(
        config_path=outcome.config_path,
        experiment=outcome.experiment,
        config_hash=outcome.config_hash,
        requests=outcome.requests,
        catalogue=outcome.catalogue,
        loaded=outcome.loaded,
        prepared=outcome.prepared,
        sizing=outcome.sizing,
        tokenization=outcome.tokenization,
        findings=outcome.findings,
        artifacts=artifacts,
        notes=outcome.notes,
    )


def _dataset_root(output_dir: str | None) -> Path:
    """Return the directory prepared datasets are written to.

    Args:
        output_dir: An explicit location, or ``None`` for ``<artifacts>/qgen-datasets``.

    Returns:
        An absolute path. Not created.
    """
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    from qa_ml.paths import get_paths

    return get_paths().artifacts / DATASET_SUBDIR


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a mapping as an indented JSON document, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _write_artifacts(
    outcome: PreparationOutcome,
    prepared: PreparedDataset,
    *,
    output_dir: str | None,
    write_dataset: bool,
) -> dict[str, str]:
    """Write the report, and optionally the examples, under the dataset root.

    Args:
        outcome: The assembled outcome.
        prepared: The prepared dataset.
        output_dir: Override the dataset root.
        write_dataset: Also write the split examples as JSONL.

    Returns:
        A mapping of document name to the path written.
    """
    root = _dataset_root(output_dir)
    stem = f"{outcome.experiment}-{prepared.fingerprint}"
    directory = root / stem
    written = {"report": _write_json(directory / "sizing.json", outcome.as_dict()).as_posix()}

    if not write_dataset:
        return written

    for name in SplitName:
        examples = prepared.splits[name]
        if not examples:
            continue
        path = directory / f"{name.value}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(example.as_dict(), ensure_ascii=False, default=str) + "\n"
                for example in examples
            ),
            encoding="utf-8",
        )
        written[name.value] = path.as_posix()

    written["metadata"] = _write_json(
        directory / "dataset.json", prepared.metadata.as_dict()
    ).as_posix()
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser. Downloads are opt-in and ``--plan`` is the cheapest useful invocation, so
        the default command reads only what is already cached.
    """
    parser = argparse.ArgumentParser(
        prog="qa_gen_runtime.prepare",
        description=(
            "Prepare the real question-generation dataset and measure what training it would "
            "cost. Loads no model weights. Downloads nothing without --allow-download."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML or JSON experiment configuration."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Resolve which corpora would be read and from where, then stop. Fetches nothing "
            "and writes nothing."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Permit fetching corpora from the Hugging Face Hub. Off by default: without it the "
            "libraries are forced offline and only cached corpora are read."
        ),
    )
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "Read a source from a local JSON or JSONL file, e.g. "
            "--source-path edu-mcq=data/mcq.jsonl. Required for corpora with no Hub mirror. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--source-split",
        action="append",
        default=[],
        metavar="SOURCE=SPLIT",
        help="Override a source's upstream split, e.g. --source-split squad-qg=validation.",
    )
    parser.add_argument(
        "--read-limit",
        type=int,
        default=None,
        help="Read at most this many records per source. For a fast structural check.",
    )
    parser.add_argument(
        "--strict-adapters",
        action="store_true",
        help=(
            "Make the first adapter refusal fatal instead of counting it. Use on a corpus you "
            "believe is clean; leave off for LearningQ, which refuses most records by design."
        ),
    )
    parser.add_argument(
        "--no-token-stats",
        action="store_true",
        help=(
            "Skip loading the tokenizer. The report then states that lengths were not "
            "measured rather than estimating them from character counts."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Measure token lengths on at most this many records per split.",
    )
    parser.add_argument(
        "--step-plan",
        action="append",
        default=[],
        metavar="BATCH,ACCUM,EPOCHS",
        help=(
            "Add a step-estimate configuration, e.g. --step-plan 1,8,3. Repeatable. Defaults "
            "to a spread around the measured L4 configuration."
        ),
    )
    parser.add_argument(
        "--seconds-per-step",
        type=float,
        default=None,
        help=(
            "A measured seconds-per-optimiser-step, to turn step counts into wall clock. "
            "Omit rather than guess; the estimate is then absent instead of wrong."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Where to write. Defaults to <artifacts>/{DATASET_SUBDIR}.",
    )
    parser.add_argument(
        "--write-dataset",
        action="store_true",
        help="Also write the split examples as JSONL beside the report.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON rather than as text."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level. Defaults to INFO.")
    return parser


def _parse_mapping(values: Sequence[str], *, flag: str) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` arguments into a mapping.

    Args:
        values: The raw argument strings.
        flag: The flag name, for the error message.

    Returns:
        The mapping.

    Raises:
        SystemExit: If an entry has no ``=``. Raised rather than reported because it is a usage
            error, and argparse's own convention for those is to exit.
    """
    mapping: dict[str, str] = {}
    for entry in values:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise SystemExit(
                f"{flag} expects SOURCE=VALUE, got {entry!r}."
            )
        mapping[key.strip()] = value.strip()
    return mapping


def _parse_plans(values: Sequence[str]) -> tuple[tuple[int, int, int], ...]:
    """Parse repeated ``BATCH,ACCUM,EPOCHS`` arguments.

    Args:
        values: The raw argument strings.

    Returns:
        The plans, or :data:`~qa_gen_runtime.sizing.DEFAULT_STEP_PLANS` when none were given.

    Raises:
        SystemExit: If an entry is not three positive integers.
    """
    if not values:
        return DEFAULT_STEP_PLANS
    plans: list[tuple[int, int, int]] = []
    for entry in values:
        parts = entry.split(",")
        if len(parts) != 3:
            raise SystemExit(f"--step-plan expects BATCH,ACCUM,EPOCHS, got {entry!r}.")
        try:
            batch, accumulation, epochs = (int(part) for part in parts)
        except ValueError as exc:
            raise SystemExit(
                f"--step-plan expects three integers, got {entry!r}: {exc}"
            ) from exc
        plans.append((batch, accumulation, epochs))
    return tuple(plans)


def format_outcome(outcome: PreparationOutcome) -> str:
    """Render an outcome as readable text.

    Args:
        outcome: The outcome.

    Returns:
        The formatted text.
    """
    lines = [
        "=" * 74,
        "  QUESTION-GENERATION DATASET PREPARATION",
        "=" * 74,
        f"configuration : {outcome.config_path}",
        f"experiment    : {outcome.experiment} ({outcome.config_hash})",
        f"mode          : {'plan only' if outcome.planned else 'prepared'}",
        "",
        "[ SOURCES ]",
    ]
    for request in outcome.requests:
        origin = (
            request.local_path
            or (f"{request.dataset_id} @ {request.revision}" if request.dataset_id else "")
            or "NEEDS --source-path"
        )
        lines.append(f"  {request.source_id:16s} {origin} split={request.split}")
    if not outcome.requests:
        lines.append("  (none resolved)")

    if outcome.planned:
        lines.extend(["", *(f"  note: {note}" for note in outcome.notes), "=" * 74])
        return "\n".join(lines)

    for source in outcome.loaded:
        ingestion = source.ingestion
        lines.append(
            f"  {ingestion.source_id:16s} records={ingestion.records_seen:,} "
            f"adapted={ingestion.examples_adapted:,} "
            f"rejected={ingestion.rejected:,} ({100 * ingestion.rejection_rate:.1f}%)"
        )
        for message in source.rejection_messages[:2]:
            lines.append(f"      ! {message[:110]}")

    prepared = outcome.prepared
    if prepared is not None:
        report = prepared.report
        stats = prepared.statistics
        lines.extend(
            [
                "",
                "[ PREPARATION ]",
                f"  fingerprint         {prepared.fingerprint}",
                "  stage counts        "
                + " -> ".join(
                    f"{stage}={report.stage_counts.get(stage, 0):,}"
                    for stage in report.as_dict()["stage_counts"]
                ),
                f"  duplicate ids       {report.duplicate_ids_dropped:,}",
                f"  invalid dropped     {report.invalid_dropped:,} "
                f"({100 * report.invalid_rate:.2f}%)",
                f"  duplicate content   {report.duplicate_content_dropped:,}",
                f"  per-source capped   {report.per_source_capped:,}",
                f"  total capped        {report.total_capped:,}",
                f"  kept                {report.kept:,}",
                "",
                "[ SPLITS ]",
                "  sizes               " + json.dumps(prepared.splits.sizes),
                "  by source           " + json.dumps(prepared.splits.source_breakdown()),
                f"  leakage groups      {len(prepared.splits.leaked_group_keys())}",
                f"  distinct groups     {stats.distinct_group_keys:,}",
                "",
                "[ CONTENT ]",
                f"  examples            {stats.total_examples:,}",
                f"  targets             {stats.total_targets:,}",
                "  by question type    " + json.dumps(stats.targets_by_question_type),
                "  by difficulty       " + json.dumps(stats.targets_by_difficulty),
                "  by source           " + json.dumps(stats.examples_by_source),
                f"  grounded            {stats.grounded_examples:,} "
                f"({100 * stats.grounded_rate:.1f}%)",
                f"  mean marks          {stats.mean_marks}",
            ]
        )

    sizing = outcome.sizing
    if sizing is not None:
        lines.extend(["", "[ TOKEN LENGTHS ]", f"  tokenizer           {sizing.tokenizer_id}"])
        if not sizing.measured:
            lines.append("  not measured        --no-token-stats was set")
        elif sizing.overall is not None:
            for label, summary in (
                ("prompt", sizing.overall.prompt_tokens),
                ("completion", sizing.overall.completion_tokens),
                ("total", sizing.overall.total_tokens),
            ):
                lines.append(
                    f"  {label:19s} min={summary.minimum} mean={summary.mean} "
                    f"p50={summary.p50} p95={summary.p95} max={summary.maximum}"
                )
            lines.append(
                f"  truncated           {sizing.overall.truncated:,} of "
                f"{sizing.overall.examples:,} "
                f"({100 * sizing.overall.truncation_rate:.2f}%) "
                f"at max_seq_length {sizing.overall.max_seq_length}"
            )

        lines.extend(["", "[ ESTIMATED TRAINING STEPS ]"])
        for estimate in sizing.step_estimates:
            hours = estimate.as_dict()["estimated_hours"]
            lines.append(
                f"  batch={estimate.batch_size} accum={estimate.gradient_accumulation_steps} "
                f"epochs={estimate.epochs}: effective={estimate.effective_batch_size} "
                f"steps/epoch={estimate.steps_per_epoch:,} total={estimate.total_steps:,} "
                f"warmup={estimate.warmup_steps}"
                + (f" ~{hours}h" if hours is not None else "")
            )

    if outcome.tokenization:
        timing = outcome.tokenization
        seconds = timing.get("elapsed_seconds", timing.get("estimated_seconds"))
        rate = timing.get("records_per_second", timing.get("assumed_records_per_second"))
        records = timing.get("example_count")
        lines.extend(
            [
                "",
                "[ TOKENIZATION TIME ]",
                f"  measured            {timing.get('measured')}",
                f"  records             {records:,}" if records is not None else
                "  records             (unknown)",
                f"  seconds             {seconds}",
                f"  records/second      {rate}",
            ]
        )

    if outcome.findings:
        lines.extend(["", "[ FINDINGS ]"])
        lines.extend(
            f"  [{'!' if finding.blocking else '?'}] {finding.code}: {finding.message}"
            for finding in outcome.findings
        )

    if outcome.artifacts:
        lines.extend(["", "[ ARTIFACTS ]"])
        lines.extend(f"  {name}: {path}" for name, path in sorted(outcome.artifacts.items()))

    lines.extend(["", f"overall       : {'ok' if outcome.ok else 'FINDINGS BLOCKING'}", "=" * 74])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` prepared, ``1`` on a failure, ``3`` when a blocking audit finding was raised.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        outcome = run_preparation(
            args.config,
            plan_only=args.plan,
            allow_download=args.allow_download,
            local_paths=_parse_mapping(args.source_path, flag="--source-path"),
            splits=_parse_mapping(args.source_split, flag="--source-split"),
            read_limit=args.read_limit,
            strict_adapters=args.strict_adapters,
            token_stats=not args.no_token_stats,
            sample=args.sample,
            plans=_parse_plans(args.step_plan),
            seconds_per_step=args.seconds_per_step,
            output_dir=args.output_dir,
            write_dataset=args.write_dataset,
        )
    except (
        AdapterError,
        ConfigIOError,
        GenerationConfigError,
        PreparationError,
        RuntimeDependencyError,
        SizingError,
        SourceLoadError,
        SplitError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    if args.json:
        print(json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(format_outcome(outcome))

    if not outcome.ok:
        blocking = [finding.code for finding in outcome.findings if finding.blocking]
        print(
            "\naudit findings block this dataset: " + ", ".join(blocking) + "\n"
            "The dataset was prepared and the report written; the findings above say why the "
            "numbers from it would be misleading.",
            file=sys.stderr,
        )
        return _EXIT_AUDIT_FAILED

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
