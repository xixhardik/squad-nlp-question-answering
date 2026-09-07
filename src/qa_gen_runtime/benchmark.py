r"""Measuring real training speed on the real corpus, for a bounded number of steps.

What this answers
-----------------
Phase 17B.3 proved the QLoRA path runs: two optimiser steps over six hand-written examples,
4.775 GiB peak of 22.034. Phase 17C sized the real corpus: 87,467 usable SQuAD examples, 78,552
in train, mean 445 tokens, p95 585, truncation 0.01%. Neither says how long a real run takes.

The proposed production configuration is batch 1, accumulation 8, two epochs -- 19,638 optimiser
steps. Whether that is four hours or twenty is the difference between "run it tonight" and
"reconsider the plan", and the only honest way to find out is to run some of it. So this module
runs a configurable number of steps -- fifty by default -- over a deterministic subset of the
**real prepared training split**, through the same loader, the same trainer factory and the same
records the production path would use, and reports what it measured plus a projection.

::

    python -m qa_gen_runtime.benchmark --config ml/configs/qgen/qgen-squad.yaml --inspect-only
    python -m qa_gen_runtime.benchmark --config ml/configs/qgen/qgen-squad.yaml --run

Why not the smoke corpus
------------------------
Six hand-written examples cannot measure throughput. Their lengths were chosen by hand, they all
came from one writer, and a step over a 200-token sequence costs a fraction of a step over a
600-token one. Sequence length is the dominant term in attention cost, so a speed figure from an
unrepresentative length distribution is not a speed figure. This module reads
``train.jsonl`` and reports the subset's own mean, p50 and p95 so the numbers can be checked
against the full-corpus figures rather than assumed to match them.

What is held identical to the validated path
--------------------------------------------
Everything that affects cost: 4-bit NF4 with double quantization, bf16 compute, the seven-module
LoRA configuration, ``paged_adamw_8bit``, gradient checkpointing, ``completion_only_loss``,
``max_seq_length`` 1024, the learning rate, the cosine schedule and the warmup *ratio*. The
records go through :func:`qa_gen_runtime.dataset.build_training_records` with the tokenizer, so
they carry the ``chat_template_kwargs`` column that keeps Qwen3's empty ``<think></think>`` block
out of the supervised completion -- the Phase 17B.2 fix. The trainer is built by
:func:`qa_gen_runtime.trainer.build_trainer`, unchanged.

Overridden, and only these: ``max_steps`` (that is the point), ``logging_steps`` to 1 so every
step's loss is recorded, and evaluation and checkpointing off so neither adds its own memory peak
or wall clock to a timing measurement.

What the loss from this does *not* tell you
-------------------------------------------
At fifty steps the cosine schedule has barely moved and warmup is two steps rather than the 589
a production run would have, so the loss trajectory here is not the trajectory a real run would
follow. Timing and memory transfer; the loss curve does not. The report says so, because a
plausible-looking loss is the easiest thing to over-read from a benchmark.

Two modes, and only one trains
------------------------------
``--inspect-only`` does everything except step the optimiser: it reads the subset, builds the
records, loads the model, constructs the trainer and measures the subset's token lengths. Run it
first -- it catches a missing dataset, a bad path or a truncation surprise before any GPU time is
spent. ``--run`` is the only mode that reaches :func:`_call_trainer_train`, the one function here
that calls ``trainer.train()``. Neither is the default.

Exit codes
----------
``0`` measured. ``1`` a configuration, dataset or runtime failure. ``2`` argparse usage error.
``3`` it ran but a check did not pass -- no adapter saved, base weights in the output, or a
subset whose length distribution makes the timing unrepresentative.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch

from qa_gen.config import GenerationConfigError, GenerationExperimentConfig
from qa_gen.examples import QuestionGenerationExample
from qa_gen.preparation import PreparationError, select_examples
from qa_gen.splitting import SplitError, SplitName, compute_dataset_fingerprint
from qa_gen_runtime.config_io import ConfigIOError, load_experiment_config
from qa_gen_runtime.dataset import (
    CHAT_TEMPLATE_KWARGS_COLUMN,
    DatasetBuildError,
    build_hf_dataset,
    build_training_records,
    resolve_record_format,
)
from qa_gen_runtime.deps import RuntimeDependencyError
from qa_gen_runtime.diagnostics import collect_diagnostics, memory_report
from qa_gen_runtime.loader import ModelLoadError, load_trainable_model
from qa_gen_runtime.precision import PrecisionError
from qa_gen_runtime.prepared import (
    PreparedDatasetError,
    read_prepared_split,
    resolve_prepared_directory,
)
from qa_gen_runtime.sizing import (
    SizingError,
    SplitSizing,
    estimate_step_count,
    measure_record_lengths,
)
from qa_gen_runtime.trainer import TrainerBuildError, build_trainer

logger = logging.getLogger(__name__)

__all__ = [
    "BENCHMARK_BATCH_SIZE",
    "BENCHMARK_GRADIENT_ACCUMULATION_STEPS",
    "BENCHMARK_REPORT_FILENAME",
    "BENCHMARK_STEPS",
    "BENCHMARK_SUBSET_SEED",
    "BenchmarkError",
    "BenchmarkFinding",
    "BenchmarkMeasurements",
    "BenchmarkMode",
    "BenchmarkReport",
    "BenchmarkSubset",
    "audit_benchmark_output",
    "benchmark_training_overrides",
    "build_parser",
    "check_subset",
    "estimate_padded_tokens",
    "format_report",
    "is_out_of_memory",
    "main",
    "project_production_run",
    "resolve_mode",
    "run_benchmark",
    "select_benchmark_subset",
    "subset_size_for",
]

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_CHECKS_FAILED = 3

#: A distinct code for running out of memory, so a script sweeping micro-batch sizes can tell
#: "this configuration does not fit" apart from "the benchmark is broken" without parsing text.
_EXIT_OUT_OF_MEMORY = 4

#: Optimiser steps measured by default. Enough that the per-step figure is not dominated by the
#: first step's one-off costs -- CUDA context, kernel autotuning, the allocator warming up -- and
#: small enough to finish in minutes on an L4.
BENCHMARK_STEPS = 50

#: Micro-batch and accumulation, matching the proposed production configuration. Changing either
#: changes what the measurement is *of*, so they are named here rather than inherited silently.
BENCHMARK_BATCH_SIZE = 1
BENCHMARK_GRADIENT_ACCUMULATION_STEPS = 8

#: Seed for the deterministic subset. Separate from the dataset's split seed so re-running the
#: benchmark on the same corpus measures the same examples, and a different subset can be drawn
#: without touching the partition.
BENCHMARK_SUBSET_SEED = 20260907

#: The benchmark's own report, written into the run directory.
BENCHMARK_REPORT_FILENAME = "benchmark.json"

#: Directory under the artifacts root holding benchmark runs. Separate from ``qgen-runs`` so a
#: timing run is never mistaken for a training run whose adapter someone might ship.
BENCHMARK_SUBDIR = "qgen-benchmarks"

#: Filenames that would mean base-model weights were copied into the output.
_BASE_WEIGHT_PATTERNS: tuple[str, ...] = (
    "model*.safetensors",
    "model*.bin",
    "pytorch_model*.bin",
    "pytorch_model*.safetensors",
    "consolidated*.pth",
    "*.gguf",
)

#: Any single file above this size in the output is suspicious. A rank-16 adapter over seven
#: projections of a 4B model is roughly 66 MiB.
_MAX_EXPECTED_FILE_BYTES = 512 * 1024**2

#: Files PEFT writes for an adapter checkpoint.
_EXPECTED_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
)

#: How far the subset's mean sequence length may sit from the full corpus's before the timing is
#: called unrepresentative. Twenty per cent: attention cost grows with length, so a subset that
#: is a quarter shorter than the corpus produces a per-step figure that is optimistic by more
#: than the noise this measurement is trying to resolve.
_LENGTH_TOLERANCE = 0.20


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed, or refuses to.

    Distinct from the runtime's own errors so the CLI can tell "the trainer rejected its
    arguments" apart from "the benchmark was asked to train without ``--run``".
    """


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory failure.

    Both routes are checked. ``torch.cuda.OutOfMemoryError`` is the modern one and subclasses
    ``RuntimeError``; older paths and some kernels raise a bare ``RuntimeError`` whose message
    says "out of memory". Matching on the message alone would be fragile, and matching on the
    type alone would miss the second case, so both are tried.

    This matters for a micro-batch sweep specifically: "batch 4 does not fit" is a *result*, not
    a crash, and it has to be recorded as one so the other configurations' numbers stand.

    Args:
        exc: The exception to classify.

    Returns:
        ``True`` when it is an out-of-memory failure.
    """
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class BenchmarkMode(str, Enum):
    """The two things this benchmark can be asked to do.

    Attributes:
        INSPECT_ONLY: Build everything and measure the subset. Steps nothing.
        RUN: Everything ``INSPECT_ONLY`` does, then the configured number of optimiser steps.
    """

    INSPECT_ONLY = "inspect_only"
    RUN = "run"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def subset_size_for(
    steps: int,
    *,
    batch_size: int = BENCHMARK_BATCH_SIZE,
    gradient_accumulation_steps: int = BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
) -> int:
    """Return how many examples ``steps`` optimiser steps will consume.

    One optimiser step consumes ``batch_size * gradient_accumulation_steps`` examples, so the
    subset is sized to be consumed exactly once. Sizing it larger would leave the tail unread and
    make "examples processed" a guess; sizing it smaller would make the trainer wrap into a
    second epoch and measure some examples twice.

    Args:
        steps: Optimiser steps to measure.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.

    Returns:
        The example count.

    Raises:
        BenchmarkError: If any argument is not positive.
    """
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if value <= 0:
            raise BenchmarkError(f"{name} must be a positive integer, got {value}.")
    return steps * batch_size * gradient_accumulation_steps


def estimate_padded_tokens(lengths: Sequence[int], batch_size: int) -> dict[str, Any]:
    """Estimate the padded tensor volume a micro-batch size implies.

    Why this is reported at all
    --------------------------
    A micro-batch is padded to its longest member, so the work the GPU does is
    ``max(length) * batch_size`` per micro-batch, not the sum of the lengths. At batch 1 the two
    are identical and there is no waste. At batch 4 over sequences ranging from 300 to 900
    tokens the waste is substantial, and it is the whole reason a larger micro-batch can be
    slower per example than the arithmetic suggests.

    Without this figure, "batch 4 processed examples 1.8x faster" invites the conclusion that
    batch 4 is 1.8x more efficient, when part of the gain is better GPU occupancy and part of
    the loss is padding nobody accounted for.

    An estimate, not a measurement
    ------------------------------
    The trainer's sampler shuffles, so which lengths actually share a micro-batch is not knowable
    here. This chunks the subset in record order, which is one plausible grouping of many. The
    expected overhead is close for a large subset and the exact figure will differ. Labelled as
    an estimate throughout, for that reason.

    Args:
        lengths: Every record's total token length.
        batch_size: Micro-batch per device.

    Returns:
        A mapping of padding-free tokens, estimated padded tokens, and the overhead between
        them.

    Raises:
        BenchmarkError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise BenchmarkError(f"batch_size must be a positive integer, got {batch_size}.")
    if not lengths:
        return {
            "batch_size": batch_size,
            "unpadded_tokens": 0,
            "estimated_padded_tokens": 0,
            "estimated_padding_overhead": 0.0,
            "measured": False,
            "note": "no lengths were supplied, so no estimate was made",
        }

    unpadded = sum(lengths)
    padded = 0
    for start in range(0, len(lengths), batch_size):
        chunk = lengths[start : start + batch_size]
        padded += max(chunk) * len(chunk)
    return {
        "batch_size": batch_size,
        "unpadded_tokens": unpadded,
        "estimated_padded_tokens": padded,
        "estimated_padding_overhead": round((padded - unpadded) / unpadded, 4),
        "measured": False,
        "note": (
            "estimated by chunking the subset in record order; the trainer's sampler shuffles, "
            "so the exact grouping differs. At batch 1 the overhead is zero by construction."
        ),
    }


def benchmark_training_overrides(
    *,
    steps: int = BENCHMARK_STEPS,
    batch_size: int = BENCHMARK_BATCH_SIZE,
    gradient_accumulation_steps: int = BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
) -> dict[str, Any]:
    """Return the in-memory overrides that bound the benchmark.

    Deliberately absent, so the measurement describes the production configuration:
    ``learning_rate``, ``lr_scheduler_type``, ``warmup_ratio``, ``optimizer``, ``precision``,
    ``gradient_checkpointing``, ``completion_only_loss``, ``packing``, ``seed`` and everything
    under ``model`` and ``lora``. Those are the settings that determine what a step costs, and
    changing any of them would make the timing describe a configuration nobody intends to run.

    Args:
        steps: Optimiser steps to measure.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.

    Returns:
        A mapping shaped for :func:`qa_gen_runtime.config_io.load_experiment_config`'s
        ``overrides`` argument, which merges one level deep and so leaves the rest of the
        training section intact. The configuration file itself is never written to.
    """
    return {
        "training": {
            "max_steps": steps,
            "num_train_epochs": 1,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # Every step's loss, so the per-step series is available rather than a single mean.
            "logging_steps": 1,
            # An evaluation pass would add its own memory peak and wall clock to a timing run.
            "evaluation_strategy": "no",
            # No intermediate checkpoints: writing 66 MiB mid-run would land in the timing.
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
    }


# ---------------------------------------------------------------------------
# The subset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkSubset:
    """The examples the benchmark will train on, and where they came from.

    Attributes:
        examples: The chosen examples.
        source_directory: The prepared dataset they were read from.
        source_split: Which split.
        available: How many examples that split holds.
        requested: How many were asked for.
        seed: The selection seed.
        fingerprint: Content digest of the subset, so a report identifies exactly what was
            measured and a re-run can be shown to have measured the same thing.
        source_fingerprint: The prepared corpus's recorded fingerprint.
        examples_by_source: Corpus mix within the subset.
        notes: Anything worth recording.
    """

    examples: tuple[QuestionGenerationExample, ...] = ()
    source_directory: str = ""
    source_split: str = SplitName.TRAIN.value
    available: int = 0
    requested: int = 0
    seed: int = BENCHMARK_SUBSET_SEED
    fingerprint: str = ""
    source_fingerprint: str | None = None
    examples_by_source: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "notes", tuple(self.notes))

    def __len__(self) -> int:
        """Return the number of chosen examples."""
        return len(self.examples)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary. The examples are not inlined."""
        return {
            "examples": len(self.examples),
            "source_directory": self.source_directory,
            "source_split": self.source_split,
            "available": self.available,
            "requested": self.requested,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "examples_by_source": dict(sorted(self.examples_by_source.items())),
            "example_ids_head": [item.id for item in self.examples[:5]],
            "notes": list(self.notes),
        }


def select_benchmark_subset(
    dataset_root: Path | str,
    *,
    directory: Path | str | None = None,
    fingerprint: str | None = None,
    split: SplitName | str = SplitName.TRAIN,
    size: int = subset_size_for(BENCHMARK_STEPS),
    seed: int = BENCHMARK_SUBSET_SEED,
) -> BenchmarkSubset:
    """Read the prepared split and choose a deterministic subset of it.

    Selection is :func:`qa_gen.preparation.select_examples`, which orders by
    ``sha256(seed, example_id)`` rather than by id. That matters for the same reason it mattered
    in Phase 17C: adapter ids carry a source prefix, so an id-ordered subset of a mixed corpus
    would take every example from one source before touching the next, and the length
    distribution of one corpus is not the length distribution of the mix.

    Args:
        dataset_root: Where prepared datasets live.
        directory: An explicit prepared dataset directory.
        fingerprint: Select the prepared dataset by corpus fingerprint. Use this for anything
            reported: "the newest" changes when somebody prepares another dataset.
        split: Which split to draw from. Training, normally -- a benchmark should not touch
            test, and validation is smaller.
        size: How many examples to draw.
        seed: Selection seed.

    Returns:
        The :class:`BenchmarkSubset`.

    Raises:
        BenchmarkError: If the split holds fewer examples than requested. Silently measuring a
            shorter run would report a step count that does not match the request.
        qa_gen_runtime.prepared.PreparedDatasetError: If the dataset cannot be read.
    """
    info = resolve_prepared_directory(dataset_root, directory=directory, fingerprint=fingerprint)
    available = read_prepared_split(info.directory, split)

    if len(available) < size:
        raise BenchmarkError(
            f"the {SplitName(split).value} split of {info.directory.name} holds "
            f"{len(available):,} example(s) but {size:,} were requested for the benchmark. "
            "Lower --steps, or prepare a larger dataset; measuring fewer steps than asked for "
            "would make the reported step count disagree with the request."
        )

    chosen, _ = select_examples(available, size, seed=seed)
    counts: dict[str, int] = {}
    for item in chosen:
        counts[item.source] = counts.get(item.source, 0) + 1

    return BenchmarkSubset(
        examples=chosen,
        source_directory=info.directory.as_posix(),
        source_split=SplitName(split).value,
        available=len(available),
        requested=size,
        seed=seed,
        fingerprint=compute_dataset_fingerprint(chosen),
        source_fingerprint=info.recorded_fingerprint or info.fingerprint or None,
        examples_by_source=counts,
        notes=(
            "drawn from the real prepared corpus, not the hand-written smoke examples",
            f"selection is deterministic in (seed={seed}, example id), so a re-run measures "
            "the same examples",
        ),
    )


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurements:
    """Everything measured across the bounded run.

    Attributes:
        optimizer_steps: Steps the trainer reported completing.
        requested_steps: Steps that were asked for. A gap means the trainer stopped early.
        effective_batch_size: Examples per optimiser step.
        examples_processed: ``optimizer_steps * effective_batch_size``.
        tokens_processed: Total tokens in the consumed subset, when lengths were measured.
        wall_clock_seconds: Time inside ``trainer.train()``.
        seconds_per_optimizer_step: Wall clock divided by completed steps.
        examples_per_second: Throughput in examples.
        tokens_per_second: Throughput in tokens, which is the figure that transfers to a corpus
            with a different length distribution.
        losses: One entry per logged step.
        final_loss: The mean training loss the trainer reported.
        train_metrics: The trainer's own metrics mapping.
        memory_before: Device memory before training, after the peak counters were reset.
        memory_after: Device memory after, so its peaks describe the run.
        peak_stats_reset: Whether the peak counters were cleared first.
        trainable_parameters: Counted from the wrapped model.
        total_parameters: Counted from the wrapped model.
        trainable_fraction: The ratio.
        optimizer_requested: The ``optim`` value asked for.
        optimizer_class: The optimiser class actually constructed.
        gradient_checkpointing_requested: The configured value.
        gradient_checkpointing_active: What the model reports about itself.
        use_cache: The model's cache setting; must be off with checkpointing.
        dropped_trainer_arguments: Settings the installed TRL did not accept.
        warmup_steps: Warmup, after conversion from the ratio.
        adapter_bytes: Size of the saved adapter.
        padding: Estimated padded tensor volume for this micro-batch size.
        failed: Whether training raised. ``True`` with ``failure_type`` set to
            ``"out_of_memory"`` is the expected outcome for a micro-batch that does not fit, and
            the timing fields are then ``None`` rather than partial.
        failure_type: ``"out_of_memory"`` or ``None``.
        failure_message: What the exception said, truncated.
        notes: Anything worth recording.
    """

    optimizer_steps: int | None = None
    requested_steps: int = 0
    effective_batch_size: int = 0
    examples_processed: int | None = None
    tokens_processed: int | None = None
    wall_clock_seconds: float | None = None
    seconds_per_optimizer_step: float | None = None
    examples_per_second: float | None = None
    tokens_per_second: float | None = None
    losses: tuple[dict[str, Any], ...] = ()
    final_loss: float | None = None
    train_metrics: dict[str, Any] = field(default_factory=dict)
    memory_before: dict[str, Any] = field(default_factory=dict)
    memory_after: dict[str, Any] = field(default_factory=dict)
    peak_stats_reset: bool = False
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    trainable_fraction: float | None = None
    optimizer_requested: str = ""
    optimizer_class: str | None = None
    gradient_checkpointing_requested: bool = False
    gradient_checkpointing_active: bool | None = None
    use_cache: bool | None = None
    dropped_trainer_arguments: tuple[str, ...] = ()
    warmup_steps: int = 0
    adapter_bytes: int | None = None
    padding: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    failure_type: str | None = None
    failure_message: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "losses", tuple(self.losses))
        object.__setattr__(
            self, "dropped_trainer_arguments", tuple(self.dropped_trainer_arguments)
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "optimizer_steps": self.optimizer_steps,
            "requested_steps": self.requested_steps,
            "effective_batch_size": self.effective_batch_size,
            "examples_processed": self.examples_processed,
            "tokens_processed": self.tokens_processed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "seconds_per_optimizer_step": self.seconds_per_optimizer_step,
            "examples_per_second": self.examples_per_second,
            "tokens_per_second": self.tokens_per_second,
            "losses": [dict(entry) for entry in self.losses],
            "final_loss": self.final_loss,
            "train_metrics": dict(self.train_metrics),
            "memory_before": dict(self.memory_before),
            "memory_after": dict(self.memory_after),
            "peak_stats_reset": self.peak_stats_reset,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
            "optimizer_requested": self.optimizer_requested,
            "optimizer_class": self.optimizer_class,
            "gradient_checkpointing_requested": self.gradient_checkpointing_requested,
            "gradient_checkpointing_active": self.gradient_checkpointing_active,
            "use_cache": self.use_cache,
            "dropped_trainer_arguments": list(self.dropped_trainer_arguments),
            "warmup_steps": self.warmup_steps,
            "adapter_bytes": self.adapter_bytes,
            "padding": dict(self.padding),
            "failed": self.failed,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "notes": list(self.notes),
        }


def project_production_run(
    train_examples: int,
    *,
    seconds_per_step: float | None,
    batch_size: int = BENCHMARK_BATCH_SIZE,
    gradient_accumulation_steps: int = BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
    epochs: int = 2,
    warmup_ratio: float = 0.0,
) -> dict[str, Any]:
    """Extrapolate the measured step time to a full training run.

    Linear in the step count, which is the assumption worth stating rather than burying: it
    holds only while the step cost is constant, and it will drift if the real corpus's length
    distribution differs from the subset's, if a later epoch pages differently, or if the device
    thermally throttles over hours in a way it does not over minutes. Treat the result as an
    order of magnitude that decides whether to proceed, not as a schedule.

    Args:
        train_examples: Size of the real training split.
        seconds_per_step: The measured rate. ``None`` leaves the time absent rather than
            inventing one.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        epochs: Passes over the training split.
        warmup_ratio: Warmup share, for the step conversion.

    Returns:
        A JSON-serializable projection, with its assumption attached.
    """
    estimate = estimate_step_count(
        train_examples,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        seconds_per_step=seconds_per_step,
    )
    payload = estimate.as_dict()
    payload["measured_seconds_per_step"] = seconds_per_step
    payload["assumption"] = (
        "linear in the step count at the measured per-step rate. Valid only while the step cost "
        "is constant; a corpus whose sequences are longer than the benchmark subset's will cost "
        "more per step."
    )
    return payload


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkFinding:
    """One thing about a benchmark that a reader should look at.

    Attributes:
        code: Short machine-readable identifier.
        message: What was found, and why it matters.
        blocking: Whether this should change the exit code.
    """

    code: str
    message: str
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"code": self.code, "message": self.message, "blocking": self.blocking}


def check_subset(
    subset: BenchmarkSubset,
    sizing: SplitSizing | None,
    *,
    corpus_mean_tokens: float | None = None,
) -> tuple[BenchmarkFinding, ...]:
    """Check that the subset can support a meaningful timing measurement.

    Args:
        subset: The chosen examples.
        sizing: Token-length measurements for the subset, when taken.
        corpus_mean_tokens: The full corpus's mean total length, from a Phase 17C sizing report.
            Supplied so the subset can be compared against the thing it is standing in for;
            without it the comparison is simply not made rather than guessed at.

    Returns:
        The findings, in a stable order.
    """
    findings: list[BenchmarkFinding] = []

    if len(subset) != subset.requested:
        findings.append(
            BenchmarkFinding(
                code="subset_size_mismatch",
                message=(
                    f"{len(subset)} examples were selected but {subset.requested} were "
                    "requested, so the run will not consume the subset exactly once."
                ),
            )
        )

    if sizing is None or not sizing.examples:
        return tuple(findings)

    if sizing.completion_tokens.minimum <= 0:
        findings.append(
            BenchmarkFinding(
                code="empty_completion_tokens",
                message=(
                    "at least one record has a completion of zero tokens, so it carries no "
                    "learning signal. Check that the sizing measurement is reading token ids "
                    "rather than a BatchEncoding."
                ),
            )
        )

    if sizing.truncation_rate > 0.05:
        findings.append(
            BenchmarkFinding(
                code="subset_truncation_high",
                message=(
                    f"{sizing.truncated} of {sizing.examples} subset sequences "
                    f"({100 * sizing.truncation_rate:.2f}%) exceed max_seq_length "
                    f"{sizing.max_seq_length}. Truncated targets both distort the loss and "
                    "understate the step cost."
                ),
            )
        )

    if corpus_mean_tokens:
        subset_mean = sizing.total_tokens.mean
        drift = abs(subset_mean - corpus_mean_tokens) / corpus_mean_tokens
        if drift > _LENGTH_TOLERANCE:
            findings.append(
                BenchmarkFinding(
                    code="subset_length_unrepresentative",
                    message=(
                        f"the subset's mean sequence length is {subset_mean} tokens against "
                        f"{corpus_mean_tokens} for the corpus, a {100 * drift:.1f}% difference. "
                        "Attention cost grows with length, so the per-step figure would not "
                        "transfer. Draw a larger subset or a different seed."
                    ),
                )
            )

    return tuple(findings)


def audit_benchmark_output(
    adapter_dir: Path, run_dir: Path, *, expect_adapter: bool
) -> tuple[tuple[BenchmarkFinding, ...], dict[str, Any]]:
    """Check what the benchmark left on disk.

    Args:
        adapter_dir: Where the adapter should have been saved.
        run_dir: The whole run directory.
        expect_adapter: Whether an adapter was supposed to be written. ``False`` under
            ``--inspect-only``, where its absence is correct.

    Returns:
        ``(findings, details)``.
    """
    files = _relative_files(run_dir)
    sizes = {name: (run_dir / name).stat().st_size for name in files}
    base_weights = tuple(
        name
        for name in files
        if any(fnmatch(Path(name).name, pattern) for pattern in _BASE_WEIGHT_PATTERNS)
    )
    oversized = tuple(name for name, size in sizes.items() if size > _MAX_EXPECTED_FILE_BYTES)
    adapter_files = _relative_files(adapter_dir)
    missing = tuple(name for name in _EXPECTED_ADAPTER_FILES if name not in adapter_files)

    findings: list[BenchmarkFinding] = []
    if base_weights:
        findings.append(
            BenchmarkFinding(
                code="base_weights_copied",
                message=(
                    f"file(s) {list(base_weights)} look like base-model weights. Only the "
                    "adapter belongs in a benchmark output."
                ),
            )
        )
    if oversized:
        findings.append(
            BenchmarkFinding(
                code="unexpectedly_large_files",
                message=(
                    f"file(s) {list(oversized)} exceed "
                    f"{_MAX_EXPECTED_FILE_BYTES // 1024**2} MiB, which no adapter should."
                ),
            )
        )
    if expect_adapter and missing:
        findings.append(
            BenchmarkFinding(
                code="adapter_incomplete",
                message=(
                    f"the adapter checkpoint is missing {list(missing)}, so nothing usable was "
                    "saved."
                ),
            )
        )

    details = {
        "run_dir": run_dir.as_posix(),
        "adapter_dir": adapter_dir.as_posix(),
        "adapter_exists": adapter_dir.is_dir(),
        "adapter_files": list(adapter_files),
        "adapter_bytes": sum(
            size for name, size in sizes.items() if name.startswith(f"{adapter_dir.name}/")
        ),
        "run_dir_bytes": sum(sizes.values()),
        "files": list(files),
        "base_weight_files": list(base_weights),
        "oversized_files": list(oversized),
        "missing_adapter_files": list(missing),
    }
    return tuple(findings), details


def _relative_files(root: Path) -> tuple[str, ...]:
    """Return every file under ``root``, as POSIX paths relative to it, sorted."""
    if not root.is_dir():
        return ()
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Everything one benchmark invocation found out.

    Attributes:
        mode: Which mode ran.
        status: ``"inspected"`` or ``"measured"``.
        trained: Whether ``trainer.train()`` was called.
        config_path: The configuration that was read.
        experiment: The experiment name.
        config_hash: Hash of the resolved configuration, overrides included.
        record_format: The TRL record shape used.
        overrides: The in-memory training overrides applied.
        subset: What was measured.
        sizing: Token-length measurements for the subset.
        measurements: What was measured during training.
        projection: The extrapolated production run.
        diagnostics: The runtime diagnostic report.
        trainer_plan: The translated trainer arguments.
        output: What was left on disk.
        findings: Checks that did not pass.
        artifacts: Files written.
        notes: Anything worth recording.
    """

    mode: str
    status: str
    trained: bool = False
    config_path: str = ""
    experiment: str = ""
    config_hash: str = ""
    record_format: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    subset: BenchmarkSubset | None = None
    sizing: SplitSizing | None = None
    measurements: BenchmarkMeasurements | None = None
    projection: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    trainer_plan: dict[str, Any] | None = None
    output: dict[str, Any] = field(default_factory=dict)
    findings: tuple[BenchmarkFinding, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def ok(self) -> bool:
        """Whether no blocking finding was raised."""
        return not any(finding.blocking for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "mode": self.mode,
            "status": self.status,
            "trained": self.trained,
            "ok": self.ok,
            "config_path": self.config_path,
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "record_format": self.record_format,
            "overrides": json.loads(json.dumps(self.overrides)),
            "subset": self.subset.as_dict() if self.subset else None,
            "sizing": self.sizing.as_dict() if self.sizing else None,
            "measurements": self.measurements.as_dict() if self.measurements else None,
            "projection": dict(self.projection) if self.projection else None,
            "diagnostics": dict(self.diagnostics) if self.diagnostics else None,
            "trainer_plan": dict(self.trainer_plan) if self.trainer_plan else None,
            "output": dict(self.output),
            "findings": [finding.as_dict() for finding in self.findings],
            "artifacts": dict(self.artifacts),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The one training call
# ---------------------------------------------------------------------------


def _call_trainer_train(trainer: Any, *, execute: bool) -> Any:
    """Call ``trainer.train()``. The only place in this module that does.

    Kept as a one-line function with a mandatory keyword guard, matching
    :func:`qa_gen_runtime.smoke._call_trainer_train`, so "where does this module train?" has
    exactly one answer and a test can assert the inspect-only path never reaches it.

    Args:
        trainer: A constructed ``SFTTrainer``.
        execute: Must be ``True``. Passed explicitly by the caller that resolved
            :attr:`BenchmarkMode.RUN`; there is no default.

    Returns:
        The trainer's ``TrainOutput``.

    Raises:
        BenchmarkError: If ``execute`` is ``False``.
    """
    if not execute:
        raise BenchmarkError(
            "refusing to call trainer.train(): this benchmark trains only under --run. "
            "--inspect-only reads the subset, builds the records, the model and the trainer, "
            "and measures the subset's token lengths, without stepping the optimiser."
        )
    return trainer.train()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    config_path: str,
    *,
    mode: BenchmarkMode,
    dataset_dir: str | None = None,
    dataset_fingerprint: str | None = None,
    split: str = SplitName.TRAIN.value,
    steps: int = BENCHMARK_STEPS,
    batch_size: int = BENCHMARK_BATCH_SIZE,
    gradient_accumulation_steps: int = BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
    subset_seed: int = BENCHMARK_SUBSET_SEED,
    corpus_mean_tokens: float | None = None,
    production_epochs: int = 2,
    output_dir: str | None = None,
    run_id: str | None = None,
    label: str | None = None,
    expect_effective_batch: int | None = None,
    expect_subset_fingerprint: str | None = None,
) -> BenchmarkReport:
    """Measure the real training path, and train only under :attr:`BenchmarkMode.RUN`.

    Args:
        config_path: Path to the experiment configuration.
        mode: Which of the two things to do.
        dataset_dir: An explicit prepared dataset directory.
        dataset_fingerprint: Select the prepared dataset by corpus fingerprint.
        split: Which prepared split to draw from.
        steps: Optimiser steps to measure.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        subset_seed: Seed for the deterministic subset.
        corpus_mean_tokens: The full corpus's mean sequence length, for the representativeness
            check.
        production_epochs: Epochs to project.
        output_dir: Where to write. Defaults to ``<artifacts>/qgen-benchmarks``.
        run_id: Override the generated run directory name.
        label: Short tag prefixed to the generated run directory name, e.g. ``"A"``. For a
            sweep, so the reports on disk say which configuration each one is.
        expect_effective_batch: Refuse unless ``batch_size * gradient_accumulation_steps``
            equals this. The point of a micro-batch comparison is that the effective batch is
            held constant, and a mistyped accumulation would silently measure a different
            optimisation problem instead of a different memory layout.
        expect_subset_fingerprint: Refuse unless the selected subset has this fingerprint.
            Pass the first configuration's fingerprint into the rest and the comparison is
            proven to be over identical data rather than assumed to be.

    Returns:
        The :class:`BenchmarkReport`.

    Raises:
        BenchmarkError: If the subset cannot be assembled, or either expectation fails.
        qa_gen_runtime.config_io.ConfigIOError: If the configuration cannot be read.
        qa_gen.config.GenerationConfigError: If it is invalid.
        qa_gen_runtime.prepared.PreparedDatasetError: If the prepared dataset cannot be read.
        qa_gen_runtime.loader.ModelLoadError: If the model cannot be loaded or adapted.
        qa_gen_runtime.trainer.TrainerBuildError: If the trainer cannot be built.
        qa_gen_runtime.sizing.SizingError: If the subset cannot be measured.
    """
    effective_batch = batch_size * gradient_accumulation_steps
    if expect_effective_batch is not None and effective_batch != expect_effective_batch:
        raise BenchmarkError(
            f"batch_size {batch_size} x gradient_accumulation_steps "
            f"{gradient_accumulation_steps} is an effective batch of {effective_batch}, but "
            f"{expect_effective_batch} was expected. A micro-batch comparison is only "
            "meaningful with the effective batch held constant; otherwise the configurations "
            "differ in the optimisation problem as well as the memory layout."
        )

    overrides = benchmark_training_overrides(
        steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    config = load_experiment_config(config_path, overrides=overrides)

    size = subset_size_for(
        steps, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps
    )
    root = _benchmark_root(output_dir)
    subset = select_benchmark_subset(
        _dataset_root(),
        directory=dataset_dir,
        fingerprint=dataset_fingerprint,
        split=split,
        size=size,
        seed=subset_seed,
    )

    if (
        expect_subset_fingerprint is not None
        and subset.fingerprint != expect_subset_fingerprint
    ):
        raise BenchmarkError(
            f"the selected subset has fingerprint {subset.fingerprint!r} but "
            f"{expect_subset_fingerprint!r} was expected. The comparison would be over "
            "different data. Check --dataset-fingerprint, --subset-seed and --steps: the "
            "subset size is steps x batch x accumulation, so holding the effective batch "
            "constant is what keeps it identical."
        )

    paths = _benchmark_paths(
        config, root, run_id=run_id, label=label, create=mode is BenchmarkMode.RUN
    )
    notes = [
        f"micro-batch {batch_size} x accumulation {gradient_accumulation_steps} = effective "
        f"batch {effective_batch}",
        f"mode={mode.value}; trainer.train() is reachable only under {BenchmarkMode.RUN.value}",
        "the configuration file was not modified; the overrides above were applied in memory",
        f"subset drawn from {subset.source_directory} ({subset.source_split} split)",
        "loss from a bounded run is not the loss trajectory of a production run: warmup and "
        "the cosine schedule are compressed. Timing and memory transfer; the loss curve does "
        "not.",
    ]

    loaded = load_trainable_model(config)
    records = build_training_records(subset.examples, config, tokenizer=loaded.tokenizer)
    if records and CHAT_TEMPLATE_KWARGS_COLUMN not in records[0]:
        notes.append(
            "records carry no chat_template_kwargs column, so on Qwen3 the empty "
            "<think></think> block sits inside the supervised completion; see "
            "qa_gen_runtime.dataset"
        )
    sizing = measure_record_lengths(records, loaded.tokenizer, config, split=subset.source_split)
    train_dataset = build_hf_dataset(records)

    trainer, plan = build_trainer(
        config,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        train_dataset=train_dataset,
        output_dir=paths["run"],
        train_examples=len(records),
        precision=loaded.precision,
    )

    findings = list(check_subset(subset, sizing, corpus_mean_tokens=corpus_mean_tokens))
    common = {
        "config_path": config_path,
        "experiment": config.name,
        "config_hash": config.config_hash(),
        "record_format": resolve_record_format(config).value,
        "overrides": overrides,
        "subset": subset,
        "sizing": sizing,
        "trainer_plan": plan.as_dict(),
    }

    if mode is BenchmarkMode.INSPECT_ONLY:
        notes.append(
            "no optimiser step was taken and no adapter was saved; run again with --run to "
            "measure the step time"
        )
        return BenchmarkReport(
            mode=mode.value,
            status="inspected",
            trained=False,
            diagnostics=collect_diagnostics(
                config,
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                precision=loaded.precision,
                trainer_plan=plan.as_dict(),
                extra_notes=("inspect-only: no optimiser step was taken",),
            ).as_dict(),
            projection=project_production_run(
                subset.available,
                seconds_per_step=None,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                epochs=production_epochs,
                warmup_ratio=config.training.warmup_ratio,
            ),
            findings=tuple(findings),
            notes=tuple(notes),
            **common,
        )

    measurements, train_notes = _train_and_measure(
        trainer,
        loaded=loaded,
        config=config,
        plan=plan,
        sizing=sizing,
        requested_steps=steps,
        effective_batch=effective_batch,
        accum=gradient_accumulation_steps,
    )
    notes.extend(train_notes)

    adapter_dir = Path(paths["adapter"])
    if measurements.failed:
        # Nothing usable was trained, so nothing is saved. Writing an adapter from a run that
        # died partway would leave a checkpoint that looks like the others and is not.
        findings.append(
            BenchmarkFinding(
                code=measurements.failure_type or "training_failed",
                message=(
                    f"training did not complete: {measurements.failure_message}. No adapter was "
                    "saved. Run the other configurations separately; their numbers are "
                    "unaffected."
                ),
            )
        )
    else:
        loaded.model.save_pretrained(str(adapter_dir))
    output_findings, output_details = audit_benchmark_output(
        adapter_dir, Path(paths["run"]), expect_adapter=not measurements.failed
    )
    findings.extend(output_findings)
    measurements = _with_adapter_bytes(measurements, output_details.get("adapter_bytes"))

    diagnostics = collect_diagnostics(
        config,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        precision=loaded.precision,
        trainer_plan=plan.as_dict(),
        extra_notes=(
            f"benchmark: {measurements.optimizer_steps} optimiser step(s) over "
            f"{len(subset)} real prepared examples",
            "a timing measurement, not a trained adapter",
        ),
    )
    report = BenchmarkReport(
        mode=mode.value,
        status=measurements.failure_type or "measured",
        trained=True,
        measurements=measurements,
        projection=project_production_run(
            subset.available,
            seconds_per_step=measurements.seconds_per_optimizer_step,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=production_epochs,
            warmup_ratio=config.training.warmup_ratio,
        ),
        diagnostics=diagnostics.as_dict(),
        output=output_details,
        findings=tuple(findings),
        artifacts={"adapter": adapter_dir.as_posix()},
        notes=tuple(notes),
        **common,
    )
    written = _write_json(Path(paths["run"]) / BENCHMARK_REPORT_FILENAME, report.as_dict())
    return BenchmarkReport(
        mode=report.mode,
        status=report.status,
        trained=report.trained,
        measurements=report.measurements,
        projection=report.projection,
        diagnostics=report.diagnostics,
        output=report.output,
        findings=report.findings,
        artifacts={**report.artifacts, "report": written.as_posix()},
        notes=report.notes,
        **common,
    )


def _dataset_root() -> Path:
    """Return the directory prepared datasets live in."""
    from qa_ml.paths import get_paths

    return get_paths().artifacts / "qgen-datasets"


def _benchmark_root(output_dir: str | None) -> Path:
    """Return the directory benchmark runs are written to.

    Args:
        output_dir: An explicit location, or ``None`` for ``<artifacts>/qgen-benchmarks``.

    Returns:
        An absolute path. Not created.
    """
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    from qa_ml.paths import get_paths

    return get_paths().artifacts / BENCHMARK_SUBDIR


def _benchmark_paths(
    config: GenerationExperimentConfig,
    root: Path,
    *,
    run_id: str | None,
    label: str | None = None,
    create: bool,
) -> dict[str, str]:
    """Decide where this invocation writes.

    Args:
        config: The experiment configuration, which generates the run id.
        root: The benchmark root.
        run_id: Override the generated name.
        label: Short tag prefixed to a generated name, so a sweep's directories are
            identifiable at a glance.
        create: Create the directory. Only under :attr:`BenchmarkMode.RUN`.

    Returns:
        ``{"run": ..., "adapter": ...}`` as strings.

    Raises:
        BenchmarkError: If the directory exists already and would be written into.
    """
    from qa_gen_runtime.outputs import ADAPTER_DIRNAME, utc_timestamp

    prefix = f"bench-{label}-" if label else "bench-"
    resolved = run_id or f"{prefix}{config.run_id(utc_timestamp())}"
    run_dir = root / resolved
    if create:
        if run_dir.exists():
            raise BenchmarkError(
                f"benchmark directory already exists: {run_dir}\nRefusing to overwrite a "
                "previous measurement. The generated name embeds a UTC timestamp, so let it "
                "regenerate, or pass --run-id."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    return {"run": str(run_dir), "adapter": str(run_dir / ADAPTER_DIRNAME)}


def _train_and_measure(
    trainer: Any,
    *,
    loaded: Any,
    config: GenerationExperimentConfig,
    plan: Any,
    sizing: SplitSizing,
    requested_steps: int,
    effective_batch: int,
    accum: int,
) -> tuple[BenchmarkMeasurements, list[str]]:
    """Run the bounded training and measure it.

    Args:
        trainer: The constructed trainer.
        loaded: The :class:`~qa_gen_runtime.loader.LoadedModel`.
        config: The experiment configuration.
        plan: The :class:`~qa_gen_runtime.trainer.TrainerPlan`.
        sizing: Token-length measurements for the subset.
        requested_steps: Steps that were asked for.
        effective_batch: Examples per optimiser step.
        accum: Gradient accumulation steps, so the micro-batch size can be recovered for the
            padding estimate.

    Returns:
        ``(measurements, notes)``. An out-of-memory failure returns rather than raises, with
        :attr:`BenchmarkMeasurements.failed` set.
    """
    reset = _reset_peak_memory()
    before = memory_report()
    padding = estimate_padded_tokens(sizing.sequence_lengths, effective_batch // max(1, accum))

    started = time.perf_counter()
    try:
        output = _call_trainer_train(trainer, execute=True)
    except BaseException as exc:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_out_of_memory(exc):
            raise
        # A micro-batch that does not fit is a result, not a crash. The peak memory reached
        # before the failure is the useful part, so it is captured rather than lost.
        elapsed = time.perf_counter() - started
        logger.warning("out of memory after %.1fs; recording it as a result", elapsed)
        return (
            BenchmarkMeasurements(
                optimizer_steps=getattr(getattr(trainer, "state", None), "global_step", 0),
                requested_steps=requested_steps,
                effective_batch_size=effective_batch,
                memory_before=before,
                memory_after=memory_report(),
                peak_stats_reset=reset,
                wall_clock_seconds=round(elapsed, 3),
                trainable_parameters=loaded.trainable_parameters,
                total_parameters=loaded.total_parameters,
                trainable_fraction=loaded.trainable_fraction,
                optimizer_requested=config.training.optimizer,
                gradient_checkpointing_requested=config.training.gradient_checkpointing,
                dropped_trainer_arguments=plan.dropped_arguments,
                warmup_steps=plan.warmup_steps,
                padding=padding,
                failed=True,
                failure_type="out_of_memory",
                failure_message=str(exc)[:600],
                notes=(
                    "this micro-batch size does not fit on this device; the timing fields are "
                    "absent rather than partial",
                    "the peak memory figures describe the allocation reached before the "
                    "failure, which is a lower bound on what the configuration needs",
                ),
            ),
            [
                f"out of memory at batch {effective_batch // max(1, accum)} "
                f"(effective {effective_batch}); recorded as a result so the other "
                "configurations' numbers stand"
            ],
        )
    elapsed = time.perf_counter() - started

    after = memory_report()
    state = getattr(trainer, "state", None)
    steps = getattr(state, "global_step", None) or getattr(output, "global_step", None)
    metrics = dict(getattr(output, "metrics", None) or {})
    losses = tuple(
        {
            "step": entry.get("step"),
            "epoch": entry.get("epoch"),
            "loss": entry.get("loss"),
            "learning_rate": entry.get("learning_rate"),
            "grad_norm": entry.get("grad_norm"),
        }
        for entry in (getattr(state, "log_history", None) or [])
        if "loss" in entry
    )

    examples = steps * effective_batch if steps else None
    tokens = sizing.total_tokens.total if sizing.examples else None
    per_step = round(elapsed / steps, 4) if steps else None

    notes: list[str] = []
    if not reset:
        notes.append(
            "CUDA is not available, so every memory figure is null and this ran on the CPU. A "
            "CPU timing measurement says nothing about GPU throughput."
        )
    if steps and steps != requested_steps:
        notes.append(
            f"the trainer completed {steps} step(s) of the {requested_steps} requested; the "
            "per-step figure is still valid but the subset was not consumed as planned"
        )
    if not losses:
        notes.append(
            "no per-step loss was logged despite logging_steps=1; the mean in train_metrics is "
            "still reported"
        )
    notes.append(
        "tokens_processed is the token count of the consumed subset, measured with the real "
        "tokenizer; it counts padding-free sequence lengths, not padded batch volume"
    )

    return (
        BenchmarkMeasurements(
            optimizer_steps=steps,
            requested_steps=requested_steps,
            effective_batch_size=effective_batch,
            examples_processed=examples,
            tokens_processed=tokens,
            wall_clock_seconds=round(elapsed, 3),
            seconds_per_optimizer_step=per_step,
            examples_per_second=(
                round(examples / elapsed, 3) if examples and elapsed > 0 else None
            ),
            tokens_per_second=(round(tokens / elapsed, 1) if tokens and elapsed > 0 else None),
            losses=losses,
            final_loss=metrics.get("train_loss", getattr(output, "training_loss", None)),
            train_metrics=metrics,
            memory_before=before,
            memory_after=after,
            peak_stats_reset=reset,
            trainable_parameters=loaded.trainable_parameters,
            total_parameters=loaded.total_parameters,
            trainable_fraction=loaded.trainable_fraction,
            optimizer_requested=config.training.optimizer,
            optimizer_class=_describe_optimizer(trainer),
            gradient_checkpointing_requested=config.training.gradient_checkpointing,
            gradient_checkpointing_active=getattr(
                loaded.model, "is_gradient_checkpointing", None
            ),
            use_cache=getattr(getattr(loaded.model, "config", None), "use_cache", None),
            dropped_trainer_arguments=plan.dropped_arguments,
            warmup_steps=plan.warmup_steps,
            padding=padding,
            notes=tuple(notes),
        ),
        notes,
    )


def _with_adapter_bytes(
    measurements: BenchmarkMeasurements, adapter_bytes: int | None
) -> BenchmarkMeasurements:
    """Return ``measurements`` with the adapter size filled in.

    The size is only knowable after the adapter is written, which happens after training, so it
    cannot be measured in the same pass. :class:`BenchmarkMeasurements` is frozen, hence a
    rebuild rather than a mutation.

    Args:
        measurements: The measurements to update.
        adapter_bytes: Size of the saved adapter.

    Returns:
        A new :class:`BenchmarkMeasurements`.
    """
    payload = {
        field_name: getattr(measurements, field_name)
        for field_name in measurements.__slots__
        if field_name != "adapter_bytes"
    }
    return BenchmarkMeasurements(**payload, adapter_bytes=adapter_bytes)


def _describe_optimizer(trainer: Any) -> str | None:
    """Return the fully qualified class name of the optimiser the trainer built."""
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return None
    kind = type(optimizer)
    return f"{kind.__module__}.{kind.__qualname__}"


def _reset_peak_memory() -> bool:
    """Clear the CUDA peak-memory counters.

    Returns:
        ``True`` when they were reset, ``False`` when there is no CUDA device. Reported rather
        than assumed, because an unreset peak folds the model load into the step measurement.
    """
    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a mapping as an indented JSON document, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser. The mode flags form a required mutually exclusive group, so there is no
        default mode: a command that does not say which of the two it wants exits ``2`` without
        loading a model.
    """
    parser = argparse.ArgumentParser(
        prog="qa_gen_runtime.benchmark",
        description=(
            "Measure real training speed on the real prepared corpus, for a bounded number of "
            "optimiser steps. Trains only under --run. Saves the adapter and a report, nothing "
            "else."
        ),
        epilog=(
            "Reads the prepared dataset written by qa_gen_runtime.prepare --write-dataset. "
            "Downloads the base model and tokenizer; downloads no dataset."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML or JSON experiment configuration."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "Read the subset, build the records, the model and the trainer, and measure the "
            "subset's token lengths. Never calls trainer.train()."
        ),
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="The only mode that trains. Takes --steps optimiser steps and saves one adapter.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=BENCHMARK_STEPS,
        help=f"Optimiser steps to measure. Defaults to {BENCHMARK_STEPS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BENCHMARK_BATCH_SIZE,
        help=f"Micro-batch per device. Defaults to {BENCHMARK_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Micro-batches per optimiser step. Defaults to "
            f"{BENCHMARK_GRADIENT_ACCUMULATION_STEPS}, matching the proposed production run."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Prepared dataset directory. Defaults to the most recent under "
            "<artifacts>/qgen-datasets."
        ),
    )
    parser.add_argument(
        "--dataset-fingerprint",
        default=None,
        help=(
            "Select the prepared dataset by corpus fingerprint. Use this for a reported "
            "measurement; 'most recent' is not reproducible."
        ),
    )
    parser.add_argument(
        "--split",
        default=SplitName.TRAIN.value,
        choices=[member.value for member in SplitName],
        help="Which prepared split to draw the subset from. Defaults to train.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=BENCHMARK_SUBSET_SEED,
        help=f"Seed for the deterministic subset. Defaults to {BENCHMARK_SUBSET_SEED}.",
    )
    parser.add_argument(
        "--corpus-mean-tokens",
        type=float,
        default=None,
        help=(
            "The full corpus's mean total sequence length, from a Phase 17C sizing report. "
            "Supplied, the subset is checked for representativeness; omitted, the check is "
            "skipped rather than guessed."
        ),
    )
    parser.add_argument(
        "--production-epochs",
        type=int,
        default=2,
        help="Epochs to project the measured step time over. Defaults to 2.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Where to write. Defaults to <artifacts>/{BENCHMARK_SUBDIR}.",
    )
    parser.add_argument("--run-id", default=None, help="Override the run directory name.")
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Short tag prefixed to the run directory, e.g. --label A. For a micro-batch sweep, "
            "so the reports on disk say which configuration each one is."
        ),
    )
    parser.add_argument(
        "--expect-effective-batch",
        type=int,
        default=None,
        help=(
            "Refuse unless batch-size x gradient-accumulation-steps equals this. Use it for a "
            "micro-batch comparison: the whole point is that the effective batch is held "
            "constant, and a mistyped accumulation would measure a different optimisation "
            "problem rather than a different memory layout."
        ),
    )
    parser.add_argument(
        "--expect-subset-fingerprint",
        default=None,
        help=(
            "Refuse unless the selected subset has this fingerprint. Pass the first "
            "configuration's fingerprint into the rest and the comparison is proven to be over "
            "identical data."
        ),
    )
    parser.add_argument(
        "--report", default=None, help="Also write the JSON report to this path."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON rather than as text."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level. Defaults to INFO.")
    return parser


def resolve_mode(args: argparse.Namespace) -> BenchmarkMode:
    """Map parsed arguments onto a mode.

    Args:
        args: Parsed arguments.

    Returns:
        :attr:`BenchmarkMode.RUN` when ``--run`` was given, :attr:`BenchmarkMode.INSPECT_ONLY`
        otherwise. Written so anything other than an explicit ``--run`` resolves to the mode
        that cannot train.
    """
    return BenchmarkMode.RUN if getattr(args, "run", False) else BenchmarkMode.INSPECT_ONLY


def format_report(report: BenchmarkReport) -> str:
    """Render a report as readable text.

    Args:
        report: The report.

    Returns:
        The formatted text.
    """
    lines = [
        "=" * 74,
        "  REAL-DATA QLoRA TRAINING BENCHMARK",
        "=" * 74,
        f"mode          : {report.mode}",
        f"trained       : {report.trained}",
        f"configuration : {report.config_path}",
        f"experiment    : {report.experiment} ({report.config_hash})",
        f"record format : {report.record_format}",
    ]
    overrides = report.overrides.get("training", {})
    lines.append(
        "overrides     : max_steps={steps} batch={batch} accum={accum}".format(
            steps=overrides.get("max_steps"),
            batch=overrides.get("per_device_train_batch_size"),
            accum=overrides.get("gradient_accumulation_steps"),
        )
    )

    subset = report.subset
    if subset is not None:
        lines.extend(
            [
                "",
                "[ SUBSET ]",
                f"  examples            {len(subset):,} of {subset.available:,} available",
                f"  source              {subset.source_directory}",
                f"  split               {subset.source_split}",
                f"  seed                {subset.seed}",
                f"  subset fingerprint  {subset.fingerprint}",
                f"  corpus fingerprint  {subset.source_fingerprint}",
                "  by source           " + json.dumps(subset.examples_by_source),
            ]
        )

    sizing = report.sizing
    if sizing is not None:
        lines.extend(["", "[ SUBSET SEQUENCE LENGTHS ]"])
        for label, summary in (
            ("prompt", sizing.prompt_tokens),
            ("completion", sizing.completion_tokens),
            ("total", sizing.total_tokens),
        ):
            lines.append(
                f"  {label:19s} min={summary.minimum} mean={summary.mean} "
                f"p50={summary.p50} p95={summary.p95} max={summary.maximum}"
            )
        lines.append(
            f"  truncated           {sizing.truncated:,} of {sizing.examples:,} "
            f"({100 * sizing.truncation_rate:.2f}%) at max_seq_length {sizing.max_seq_length}"
        )
        lines.append(f"  tokens in subset    {sizing.total_tokens.total:,}")

    measured = report.measurements
    if measured is not None:
        lines.extend(_format_measurements(measured))

    if report.projection:
        projection = report.projection
        lines.extend(
            [
                "",
                "[ PROJECTED PRODUCTION RUN ]",
                f"  train examples      {projection.get('train_examples'):,}",
                f"  configuration       batch={projection.get('batch_size')} "
                f"accum={projection.get('gradient_accumulation_steps')} "
                f"epochs={projection.get('epochs')}",
                f"  steps/epoch         {projection.get('steps_per_epoch'):,}",
                f"  total steps         {projection.get('total_steps'):,}",
                f"  warmup steps        {projection.get('warmup_steps'):,}",
                f"  seconds/step        {projection.get('measured_seconds_per_step')}",
                f"  estimated hours     {projection.get('estimated_hours')}",
            ]
        )

    if report.output:
        adapter_files = ", ".join(report.output.get("adapter_files") or []) or "none"
        base_weights = ", ".join(report.output.get("base_weight_files") or []) or "none found"
        adapter_bytes = report.output.get("adapter_bytes")
        lines.extend(
            [
                "",
                "[ OUTPUT ]",
                f"  adapter             {report.output.get('adapter_dir')}",
                f"  adapter files       {adapter_files}",
                f"  adapter bytes       {adapter_bytes:,}"
                if adapter_bytes is not None
                else "  adapter bytes       (unknown)",
                f"  base weights        {base_weights}",
            ]
        )

    if report.findings:
        lines.extend(["", "[ FINDINGS ]"])
        lines.extend(
            f"  [{'!' if finding.blocking else '?'}] {finding.code}: {finding.message}"
            for finding in report.findings
        )

    if report.artifacts:
        lines.extend(["", "[ ARTIFACTS ]"])
        lines.extend(f"  {name}: {path}" for name, path in sorted(report.artifacts.items()))

    lines.extend(["", f"overall       : {'ok' if report.ok else 'FINDINGS BLOCKING'}"])
    if report.notes:
        lines.append("notes         :")
        lines.extend(f"  - {note}" for note in report.notes)
    lines.append("=" * 74)
    return "\n".join(lines)


def _format_measurements(measured: BenchmarkMeasurements) -> list[str]:
    """Render the measurement section."""
    after = measured.memory_after
    if measured.failed:
        return [
            "",
            "[ FAILED ]",
            f"  failure             {measured.failure_type}",
            f"  after               {measured.wall_clock_seconds} s",
            f"  effective batch     {measured.effective_batch_size}",
            f"  vram peak alloc     {after.get('max_allocated_gib')} GiB (lower bound)",
            f"  vram peak reserved  {after.get('max_reserved_gib')} GiB (lower bound)",
            f"  vram total          {after.get('total_vram_gib')} GiB",
            f"  message             {(measured.failure_message or '')[:160]}",
            "  no adapter was saved and no timing figure is reported",
        ]

    padding = measured.padding
    lines = [
        "",
        "[ MEASURED ]",
        f"  optimizer steps     {measured.optimizer_steps} of {measured.requested_steps} "
        f"requested (warmup {measured.warmup_steps})",
        f"  effective batch     {measured.effective_batch_size}",
        f"  examples processed  {measured.examples_processed:,}"
        if measured.examples_processed is not None
        else "  examples processed  (unknown)",
        f"  tokens processed    {measured.tokens_processed:,} (padding-free)"
        if measured.tokens_processed is not None
        else "  tokens processed    (unknown)",
        f"  padded tokens ~     {padding.get('estimated_padded_tokens'):,} "
        f"(+{100 * padding.get('estimated_padding_overhead', 0.0):.1f}% estimated padding "
        f"at micro-batch {padding.get('batch_size')})"
        if padding.get("estimated_padded_tokens")
        else "  padded tokens ~     (not estimated)",
        f"  wall clock          {measured.wall_clock_seconds} s",
        f"  seconds/step        {measured.seconds_per_optimizer_step}",
        f"  examples/second     {measured.examples_per_second}",
        f"  tokens/second       {measured.tokens_per_second}",
        f"  final loss          {measured.final_loss}",
        f"  vram peak alloc     {after.get('max_allocated_gib')} GiB",
        f"  vram peak reserved  {after.get('max_reserved_gib')} GiB",
        f"  vram total          {after.get('total_vram_gib')} GiB "
        f"(peak counters reset: {measured.peak_stats_reset})",
        f"  optimizer           requested {measured.optimizer_requested!r} -> "
        f"{measured.optimizer_class}",
        f"  checkpointing       requested {measured.gradient_checkpointing_requested}, "
        f"active {measured.gradient_checkpointing_active}, use_cache {measured.use_cache}",
        f"  trainable params    {measured.trainable_parameters:,} of "
        f"{measured.total_parameters:,} ({measured.trainable_fraction})"
        if measured.trainable_parameters and measured.total_parameters
        else "  trainable params    (unknown)",
        f"  adapter bytes       {measured.adapter_bytes:,}"
        if measured.adapter_bytes is not None
        else "  adapter bytes       (unknown)",
        f"  dropped args        {', '.join(measured.dropped_trainer_arguments) or 'none'}",
    ]
    if measured.losses:
        lines.append("  per-step loss       (first and last 3)")
        selected = (
            measured.losses
            if len(measured.losses) <= 6
            else [*measured.losses[:3], *measured.losses[-3:]]
        )
        lines.extend(
            f"    step {entry.get('step')}: loss={entry.get('loss')} "
            f"lr={entry.get('learning_rate')} grad_norm={entry.get('grad_norm')}"
            for entry in selected
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` measured, ``1`` on a failure, ``3`` when a blocking check did not pass.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = resolve_mode(args)
    logger.info("benchmark mode: %s", mode.value)

    try:
        report = run_benchmark(
            args.config,
            mode=mode,
            dataset_dir=args.dataset_dir,
            dataset_fingerprint=args.dataset_fingerprint,
            split=args.split,
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            subset_seed=args.subset_seed,
            corpus_mean_tokens=args.corpus_mean_tokens,
            production_epochs=args.production_epochs,
            output_dir=args.output_dir,
            run_id=args.run_id,
            label=args.label,
            expect_effective_batch=args.expect_effective_batch,
            expect_subset_fingerprint=args.expect_subset_fingerprint,
        )
    except (
        BenchmarkError,
        ConfigIOError,
        DatasetBuildError,
        GenerationConfigError,
        ModelLoadError,
        PrecisionError,
        PreparationError,
        PreparedDatasetError,
        RuntimeDependencyError,
        SizingError,
        SplitError,
        TrainerBuildError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    if args.report:
        _write_json(Path(args.report), report.as_dict())

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(report))

    if report.measurements is not None and report.measurements.failure_type == "out_of_memory":
        print(
            "\nout of memory: this micro-batch size does not fit on this device. The report was "
            "written and records the peak allocation reached before the failure. The other "
            "configurations are unaffected -- run each one as a separate invocation.",
            file=sys.stderr,
        )
        return _EXIT_OUT_OF_MEMORY

    if not report.ok:
        blocking = [finding.code for finding in report.findings if finding.blocking]
        print(
            "\nchecks did not pass: " + ", ".join(blocking) + "\n"
            "The benchmark ran and the report was written; the findings above say why the "
            "numbers would be misleading.",
            file=sys.stderr,
        )
        return _EXIT_CHECKS_FAILED

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
