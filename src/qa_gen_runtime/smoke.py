"""Phase 17B.2: the tiny real QLoRA smoke harness, and the only place training is called.

What this is for
----------------
Phase 17B.1 built and tested every piece of the runtime and deliberately stopped short of
``trainer.train()``. This module takes the last step, under the tightest conditions that still
constitute a real run: the real Qwen3-4B checkpoint, real 4-bit NF4 weights, real PEFT
adapters, the real TRL ``SFTTrainer``, six hand-written examples and two optimiser steps.

Two modes, and the difference is the point
------------------------------------------
::

    python -m qa_gen_runtime.smoke --config ml/configs/qgen/qgen-smoke.yaml --inspect-only
    python -m qa_gen_runtime.smoke --config ml/configs/qgen/qgen-smoke.yaml --run

``--inspect-only`` does everything except step the optimiser. It builds the corpus, loads the
quantized model, attaches the adapters, constructs the real ``SFTTrainer`` and reads the first
tokenized record back out of it. ``--run`` is the *only* mode that reaches
:func:`_call_trainer_train`, which is the only function in this repository that calls
``trainer.train()``. Neither mode is the default: the mode flags are a required mutually
exclusive group, so the harness cannot be started by a command that forgot to say what it was
for.

What the inspection is actually checking
----------------------------------------
One invariant matters more than everything else here: **the completion mask must cover the
target JSON and must not extend into the prompt.** TRL builds it arithmetically --

.. code-block:: python

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

-- where ``prompt_ids`` comes from ``apply_chat_template(prompt, add_generation_prompt=True)``
and ``prompt_completion_ids`` from ``apply_chat_template(prompt + completion)``. If the first
is not a token-exact prefix of the second, TRL emits a warning and carries on with a mask that
is silently offset. An offset mask trains on a truncated target and reports a plausible loss
while doing it, which is the failure this harness exists to make impossible to miss. So the
inspection decodes the masked and unmasked halves separately, checks that the masked half
starts with ``{"question_type":`` and parses through
:func:`qa_gen.examples.target_from_json`, checks that the source context appears in the
unmasked half and not in the masked one, and captures any warning TRL logged while preparing
the dataset.

Qwen3 thinking markers are reported, never removed
--------------------------------------------------
``reasoning_mode: disabled`` in the configuration does not reach the tokenizer during
training; :mod:`qa_gen_runtime.chat` documents why in detail. The consequence is that whether
``<think>`` appears in a trained sequence is a fact to be measured rather than assumed, so the
inspection counts the markers in the decoded prompt and completion and reports what it found.
It does not strip them. Editing the training record to make the report look clean would
destroy the only evidence that the setting is not doing what its name suggests.

Bounded on purpose, and not by editing the shipped configuration
----------------------------------------------------------------
``ml/configs/qgen/qgen-smoke.yaml`` is the *planning* configuration and stays as it is: 20
steps at gradient accumulation 8, which is what ``--plan`` reports and what the tests pin.
This harness overrides a named handful of training fields in memory, through
:func:`qa_gen_runtime.config_io.load_experiment_config`'s ``overrides`` argument, and records
exactly which ones in its report. The first measured execution is batch 1 with accumulation 1
so its peak VRAM is directly comparable with the 3.943 GiB single-example forward/backward
figure in :data:`qa_gen.config.VERIFIED_QWEN3_4B_L4`; leaving accumulation at 8 would have
measured eight micro-batches and produced a number with nothing to compare it to.

Exit codes
----------
``0`` the harness ran and every invariant held. ``1`` a configuration, dependency or runtime
failure. ``2`` argparse usage error, including no mode selected. ``3`` the harness ran but an
invariant failed or an output audit did not pass -- a distinct code because "it crashed" and
"it worked and the mask is wrong" call for different reactions.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch

from qa_gen.config import GenerationConfigError, GenerationExperimentConfig
from qa_gen.examples import QuestionGenerationTarget, TargetParseError, target_from_json
from qa_gen.metadata import TrainingRunMetadata
from qa_gen_runtime.config_io import (
    ConfigIOError,
    load_experiment_config,
    write_resolved_config,
)
from qa_gen_runtime.dataset import (
    DatasetBuildError,
    build_hf_dataset,
    build_training_records,
    resolve_record_format,
)
from qa_gen_runtime.deps import RuntimeDependencyError
from qa_gen_runtime.diagnostics import attach_to_metadata, collect_diagnostics, memory_report
from qa_gen_runtime.loader import ModelLoadError, load_trainable_model
from qa_gen_runtime.outputs import (
    RunOutputError,
    RunPaths,
    create_run_directory,
    resolve_run_root,
    utc_timestamp,
)
from qa_gen_runtime.precision import PrecisionError
from qa_gen_runtime.smoke_data import (
    build_smoke_examples,
    describe_smoke_corpus,
    validate_smoke_examples,
)
from qa_gen_runtime.trainer import TrainerBuildError, build_trainer
from qa_ml.paths import find_repo_root

logger = logging.getLogger(__name__)

__all__ = [
    "COMPLETION_MASK_COLUMNS",
    "EXPECTED_COMPLETION_PREFIX",
    "SMOKE_BATCH_SIZE",
    "SMOKE_GRADIENT_ACCUMULATION_STEPS",
    "SMOKE_MAX_STEPS",
    "SMOKE_REPORT_FILENAME",
    "THINKING_MARKERS",
    "OutputAudit",
    "RecordInspection",
    "SmokeHarnessError",
    "SmokeMeasurements",
    "SmokeMode",
    "SmokeReport",
    "audit_output",
    "build_parser",
    "first_tokenized_record",
    "format_report",
    "git_status",
    "inspect_tokenized_record",
    "main",
    "resolve_mode",
    "run_smoke",
    "smoke_training_overrides",
]

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_INVARIANT_FAILED = 3

#: The first characters of the canonical target JSON. Derived from
#: :data:`qa_gen.examples.TARGET_JSON_FIELDS` ordering plus the compact separators
#: ``to_json`` uses, and asserted against a rendered target by the tests rather than trusted.
EXPECTED_COMPLETION_PREFIX = '{"question_type":'

#: Qwen3's reasoning-block delimiters. Counted in the decoded training sequence and reported;
#: never removed. See the module docstring.
THINKING_MARKERS: tuple[str, ...] = ("<think>", "</think>")

#: Columns TRL 0.29.1 may use for the loss mask, in the order they are looked for.
#: ``completion_mask`` is what a prompt-completion dataset produces; ``assistant_masks`` only
#: appears when ``assistant_only_loss`` is on, which this runtime never sets.
COMPLETION_MASK_COLUMNS: tuple[str, ...] = ("completion_mask", "assistant_masks")

#: Bounded-experiment defaults for the first measured execution. Batch 1 with accumulation 1
#: so the peak VRAM is comparable with the recorded single-example measurement.
SMOKE_MAX_STEPS = 2
SMOKE_BATCH_SIZE = 1
SMOKE_GRADIENT_ACCUMULATION_STEPS = 1

#: Warmup as a ratio, kept at the configuration's own value. At two total steps
#: :func:`qa_gen_runtime.trainer.resolve_warmup_steps` converts it to exactly one step, which
#: is the "at most one warmup step" this run wants.
SMOKE_WARMUP_RATIO = 0.03

#: The harness's own report, written beside the Phase 17A run record.
SMOKE_REPORT_FILENAME = "smoke.json"

#: Filenames that would mean base-model weights were copied into the run directory. An
#: adapter checkpoint is ``adapter_model.safetensors`` plus ``adapter_config.json``; none of
#: these patterns match those.
_BASE_WEIGHT_PATTERNS: tuple[str, ...] = (
    "model*.safetensors",
    "model*.bin",
    "pytorch_model*.bin",
    "pytorch_model*.safetensors",
    "consolidated*.pth",
    "*.gguf",
)

#: Any single file above this size in the run directory is treated as suspicious. A rank-16
#: adapter over seven projections of a 4B model is roughly 66 MiB, so this leaves two orders
#: of magnitude of headroom before a 4-bit base checkpoint (~2.5 GiB) would slip through.
_MAX_EXPECTED_FILE_BYTES = 512 * 1024**2

#: Files PEFT writes for an adapter checkpoint. Absence of either means nothing usable was
#: saved, whatever else the directory contains.
_EXPECTED_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
)


class SmokeHarnessError(RuntimeError):
    """Raised when the harness cannot proceed, or refuses to.

    Distinct from the runtime's own errors so the CLI can tell "the trainer rejected its
    arguments" apart from "the harness was asked to train without ``--run``".
    """


class SmokeMode(str, Enum):
    """The two things this harness can be asked to do.

    Subclasses ``(str, Enum)`` to match the Phase 17A enums, which target Python 3.10 and so
    cannot use ``StrEnum``.

    Attributes:
        INSPECT_ONLY: Build everything, inspect the first tokenized record, train nothing.
        RUN: Everything ``INSPECT_ONLY`` does, then a bounded number of optimiser steps.
    """

    INSPECT_ONLY = "inspect_only"
    RUN = "run"


# ---------------------------------------------------------------------------
# Configuration overrides
# ---------------------------------------------------------------------------


def smoke_training_overrides(
    *,
    max_steps: int = SMOKE_MAX_STEPS,
    batch_size: int = SMOKE_BATCH_SIZE,
    gradient_accumulation_steps: int = SMOKE_GRADIENT_ACCUMULATION_STEPS,
) -> dict[str, Any]:
    """Return the in-memory overrides that bound the smoke execution.

    Every field is named explicitly, including the ones the shipped configuration already
    sets, so the measured run is defined by this function rather than by whatever the YAML
    happens to say. The file itself is never written to.

    Deliberately absent: ``gradient_checkpointing``, ``completion_only_loss``, ``packing``,
    ``optimizer``, ``precision``, ``learning_rate`` and ``seed``. Those are the settings the
    L4 feasibility measurement covered, and changing them here would make the result
    incomparable to it.

    Args:
        max_steps: Optimiser steps to run.
        batch_size: Micro-batch per device, for both training and evaluation.
        gradient_accumulation_steps: Micro-batches per optimiser step.

    Returns:
        A mapping shaped for :func:`qa_gen_runtime.config_io.load_experiment_config`'s
        ``overrides`` argument, which merges one level deep and so leaves the rest of the
        training section intact.
    """
    return {
        "training": {
            "max_steps": max_steps,
            "num_train_epochs": 1,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": SMOKE_WARMUP_RATIO,
            # No evaluation on the first measured run: an eval pass would add its own memory
            # peak to the number being compared against the baseline.
            "evaluation_strategy": "no",
            # No intermediate checkpoints. The adapter is saved once, explicitly, at the end.
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "logging_steps": 1,
            "save_total_limit": 1,
        }
    }


# ---------------------------------------------------------------------------
# Record inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordInspection:
    """What one tokenized training record actually contains.

    Attributes:
        example_id: The source example, when the record still carries the column.
        columns: Every column present on the tokenized record, sorted.
        input_ids_length: Token count of the trained sequence.
        max_length: The configured truncation limit, for comparison.
        mask_column: Which of :data:`COMPLETION_MASK_COLUMNS` was found, or ``None``.
        completion_mask_length: Length of that mask.
        completion_token_count: Tokens the mask selects for the loss.
        prompt_token_count: Tokens the mask excludes from the loss.
        decoded_sequence: The whole trained sequence, special tokens included.
        decoded_prompt: The unmasked half, decoded.
        decoded_completion: The masked half, decoded.
        extracted_json: The JSON object found inside the decoded completion, or ``None``.
        expected_completion_prefix: What the completion was required to start with.
        parsed_target: The completion re-parsed through the canonical parser.
        parse_error: Why parsing failed, when it did.
        thinking_markers: Reasoning-marker findings. Reported, never acted on.
        checks: Named invariants. ``True`` held, ``False`` failed, ``None`` undetermined.
        notes: Anything else worth recording.
    """

    example_id: str | None = None
    columns: tuple[str, ...] = ()
    input_ids_length: int = 0
    max_length: int = 0
    mask_column: str | None = None
    completion_mask_length: int | None = None
    completion_token_count: int | None = None
    prompt_token_count: int | None = None
    decoded_sequence: str = ""
    decoded_prompt: str = ""
    decoded_completion: str = ""
    extracted_json: str | None = None
    expected_completion_prefix: str = EXPECTED_COMPLETION_PREFIX
    parsed_target: dict[str, Any] | None = None
    parse_error: str | None = None
    thinking_markers: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool | None] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of invariants that did not hold."""
        return tuple(name for name, value in self.checks.items() if value is False)

    @property
    def undetermined_checks(self) -> tuple[str, ...]:
        """Names of invariants that could not be evaluated.

        Treated as failure by :attr:`ok`. An invariant this harness could not check is not
        evidence that it holds, and the whole purpose here is to stop guessing.
        """
        return tuple(name for name, value in self.checks.items() if value is None)

    @property
    def ok(self) -> bool:
        """Whether every invariant was evaluated and held."""
        return not self.failed_checks and not self.undetermined_checks

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "example_id": self.example_id,
            "columns": list(self.columns),
            "input_ids_length": self.input_ids_length,
            "max_length": self.max_length,
            "mask_column": self.mask_column,
            "completion_mask_length": self.completion_mask_length,
            "completion_token_count": self.completion_token_count,
            "prompt_token_count": self.prompt_token_count,
            "decoded_sequence": self.decoded_sequence,
            "decoded_prompt": self.decoded_prompt,
            "decoded_completion": self.decoded_completion,
            "extracted_json": self.extracted_json,
            "expected_completion_prefix": self.expected_completion_prefix,
            "parsed_target": dict(self.parsed_target) if self.parsed_target else None,
            "parse_error": self.parse_error,
            "thinking_markers": dict(self.thinking_markers),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "undetermined_checks": list(self.undetermined_checks),
            "ok": self.ok,
            "notes": list(self.notes),
        }


def first_tokenized_record(trainer: Any) -> dict[str, Any]:
    """Read the first prepared record back out of a constructed trainer.

    TRL tokenizes inside ``SFTTrainer.__init__``, so by the time a trainer exists its
    ``train_dataset`` is the processed one. Reading it there rather than re-tokenizing is the
    whole value of the check: it inspects the tokens the optimiser would actually see, not an
    independent reconstruction of them that could agree while the real path disagrees.

    Args:
        trainer: A constructed ``SFTTrainer``.

    Returns:
        The first record, as a plain mapping.

    Raises:
        SmokeHarnessError: If the trainer has no training dataset, or it is empty, or it does
            not support indexing.
    """
    dataset = getattr(trainer, "train_dataset", None)
    if dataset is None:
        raise SmokeHarnessError(
            "the trainer has no train_dataset, so there is no tokenized record to inspect."
        )
    try:
        row = dataset[0]
    except (IndexError, KeyError, TypeError) as exc:
        raise SmokeHarnessError(
            f"could not read the first record from the prepared dataset: "
            f"{type(exc).__name__}: {exc}. An IterableDataset cannot be indexed; this harness "
            "builds an in-memory datasets.Dataset, so reaching here means something replaced "
            "it."
        ) from exc
    if not isinstance(row, dict):
        return dict(row)
    return row


def inspect_tokenized_record(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    context_probe: str = "",
    source_target: QuestionGenerationTarget | None = None,
    expected_prefix: str = EXPECTED_COMPLETION_PREFIX,
) -> RecordInspection:
    """Decode one tokenized record and check the completion mask against it.

    Args:
        row: A prepared record, from :func:`first_tokenized_record`.
        tokenizer: The tokenizer that produced it, used only to decode.
        max_length: The configured truncation limit. A sequence at the limit may have lost the
            tail of its target, so the comparison is reported.
        context_probe: A distinctive slice of the source passage. Expected in the unmasked
            half and absent from the masked one; that pair of checks is what distinguishes a
            correct mask from one offset in either direction.
        source_target: The target this record was rendered from. When supplied, the re-parsed
            completion is compared against it.
        expected_prefix: What the decoded completion must start with.

    Returns:
        The :class:`RecordInspection`.

    Raises:
        SmokeHarnessError: If the record carries no usable ``input_ids``.
    """
    columns = tuple(sorted(row))
    input_ids = _as_int_list(row.get("input_ids"))
    if not input_ids:
        raise SmokeHarnessError(
            "the tokenized record has no usable 'input_ids' column. Columns present: "
            f"{list(columns)}. TRL prepares the dataset inside SFTTrainer.__init__, so an "
            "absent input_ids means the dataset was not the prompt-completion shape this "
            "runtime emits."
        )

    notes: list[str] = []
    mask_column = next((name for name in COMPLETION_MASK_COLUMNS if name in row), None)
    mask = _as_int_list(row.get(mask_column)) if mask_column else []
    aligned = bool(mask) and len(mask) == len(input_ids)

    if mask_column is None:
        notes.append(
            "no completion mask column found, so the loss would be taken over the whole "
            "sequence including the prompt. Check training.completion_only_loss and the "
            "record format."
        )
    elif not aligned:
        notes.append(
            f"{mask_column} has {len(mask)} entries for {len(input_ids)} tokens; the halves "
            "cannot be separated, so every mask-dependent check is undetermined."
        )

    decoded_sequence = _decode(tokenizer, input_ids)
    completion_ids: list[int] = []
    prompt_ids: list[int] = []
    marked: list[int] = []
    if aligned:
        pairs = list(zip(input_ids, mask, strict=True))
        completion_ids = [token for token, flag in pairs if flag]
        prompt_ids = [token for token, flag in pairs if not flag]
        marked = [index for index, flag in enumerate(mask) if flag]

    decoded_prompt = _decode(tokenizer, prompt_ids)
    decoded_completion = _decode(tokenizer, completion_ids)

    extracted = _extract_json_object(decoded_completion)
    parsed: QuestionGenerationTarget | None = None
    parse_error: str | None = None
    if extracted is None:
        if decoded_completion:
            parse_error = "no JSON object found in the decoded completion"
    else:
        try:
            parsed = target_from_json(extracted)
        except TargetParseError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    checks: dict[str, bool | None] = {
        "mask_column_present": mask_column is not None,
        "mask_length_matches_input_ids": aligned,
        "mask_covers_a_non_empty_region": bool(marked) if aligned else None,
        "mask_region_is_contiguous": (
            (marked[-1] - marked[0] + 1 == len(marked)) if marked else None
        ),
        "mask_excludes_the_first_token": (marked[0] > 0) if marked else None,
        "mask_reaches_the_final_token": (
            (marked[-1] == len(input_ids) - 1) if marked else None
        ),
        "sequence_within_max_length": len(input_ids) < max_length if max_length else None,
        "prompt_contains_the_source_context": (
            (context_probe in decoded_prompt) if context_probe and decoded_prompt else None
        ),
        "completion_excludes_the_source_context": (
            (context_probe not in decoded_completion)
            if context_probe and decoded_completion
            else None
        ),
        "completion_starts_with_expected_json_prefix": (
            decoded_completion.lstrip().startswith(expected_prefix)
            if decoded_completion
            else None
        ),
        "completion_json_parses": parsed is not None if decoded_completion else None,
        "parsed_target_matches_the_source_target": (
            (parsed == source_target) if parsed is not None and source_target else None
        ),
    }

    return RecordInspection(
        example_id=row.get("example_id"),
        columns=columns,
        input_ids_length=len(input_ids),
        max_length=max_length,
        mask_column=mask_column,
        completion_mask_length=len(mask) if mask_column else None,
        completion_token_count=len(completion_ids) if aligned else None,
        prompt_token_count=len(prompt_ids) if aligned else None,
        decoded_sequence=decoded_sequence,
        decoded_prompt=decoded_prompt,
        decoded_completion=decoded_completion,
        extracted_json=extracted,
        expected_completion_prefix=expected_prefix,
        parsed_target=parsed.as_dict() if parsed else None,
        parse_error=parse_error,
        thinking_markers=_describe_thinking_markers(
            decoded_sequence, decoded_prompt, decoded_completion
        ),
        checks=checks,
        notes=tuple(notes),
    )


def _describe_thinking_markers(
    decoded_sequence: str, decoded_prompt: str, decoded_completion: str
) -> dict[str, Any]:
    """Count Qwen3 reasoning markers in a decoded record, without altering anything.

    Args:
        decoded_sequence: The whole trained sequence.
        decoded_prompt: The unmasked half.
        decoded_completion: The masked half.

    Returns:
        A mapping of where each marker appears and how often.
    """
    counts = {marker: decoded_sequence.count(marker) for marker in THINKING_MARKERS}
    return {
        "markers": list(THINKING_MARKERS),
        "counts": counts,
        "present_in_sequence": any(counts.values()),
        "present_in_prompt": any(marker in decoded_prompt for marker in THINKING_MARKERS),
        "present_in_completion": any(
            marker in decoded_completion for marker in THINKING_MARKERS
        ),
        "record_modified_to_remove_them": False,
        "note": (
            "reasoning_mode does not reach the tokenizer during training: TRL applies the "
            "chat template itself and reads template arguments from a per-example "
            "'chat_template_kwargs' column, which this runtime does not emit. So a marker "
            "found here came from the template's own default and is reported as measured. "
            "Markers in the prompt half are the dangerous case, because TRL derives the "
            "completion mask from the prompt-only rendering and an extra block there offsets "
            "the mask into the target."
        ),
    }


def _decode(tokenizer: Any, ids: list[int]) -> str:
    """Decode token ids with special tokens kept.

    Special tokens are kept because the chat markup is exactly what has to be readable: the
    boundary between the prompt and the completion is a special token, and hiding it would
    hide the thing being inspected.

    Args:
        tokenizer: The tokenizer.
        ids: Token ids. An empty list decodes to an empty string without calling anything.

    Returns:
        The decoded text.
    """
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


def _as_int_list(value: Any) -> list[int]:
    """Coerce a column value into a list of ints, returning ``[]`` when it cannot be.

    Args:
        value: A list, a tensor, or anything else a dataset column might hold.

    Returns:
        The integers, or an empty list. Never raises: a missing or oddly-typed column is a
        finding to report, not a crash.
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _extract_json_object(text: str) -> str | None:
    """Return the outermost ``{...}`` span in ``text``, or ``None``.

    The decoded completion carries the chat template's end-of-turn markup around the JSON, so
    the object has to be located rather than assumed to be the whole string.

    Args:
        text: The decoded completion.

    Returns:
        The candidate JSON text, or ``None`` when there is no brace pair.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmokeMeasurements:
    """Everything measured across a bounded execution.

    Attributes:
        trainable_parameters: Counted from the wrapped model.
        total_parameters: Counted from the wrapped model.
        trainable_fraction: The ratio, rounded.
        memory_before: :func:`~qa_gen_runtime.diagnostics.memory_report` before training,
            after peak stats were reset.
        memory_after: The same report after training, so its peaks describe the run.
        peak_stats_reset: Whether the peak counters were cleared first. ``False`` means the
            peaks may include an earlier allocation and are an upper bound only.
        wall_clock_seconds: Time spent inside ``trainer.train()``.
        optimizer_steps: Optimiser steps completed, from the trainer state.
        seconds_per_optimizer_step: Wall clock divided by steps.
        losses: One entry per logged step.
        final_loss: The mean training loss the trainer reported.
        train_metrics: The trainer's own metrics mapping.
        optimizer_requested: The ``optim`` value asked for.
        optimizer_class: The optimiser class actually constructed.
        gradient_checkpointing_requested: The configured value.
        gradient_checkpointing_active: What the model reports about itself.
        use_cache: The model's cache setting; must be off with checkpointing.
        dropped_trainer_arguments: Settings the installed TRL did not accept.
        warmup_steps: Warmup, after conversion from the ratio.
        notes: Anything else worth recording.
    """

    trainable_parameters: int | None = None
    total_parameters: int | None = None
    trainable_fraction: float | None = None
    memory_before: dict[str, Any] = field(default_factory=dict)
    memory_after: dict[str, Any] = field(default_factory=dict)
    peak_stats_reset: bool = False
    wall_clock_seconds: float | None = None
    optimizer_steps: int | None = None
    seconds_per_optimizer_step: float | None = None
    losses: tuple[dict[str, Any], ...] = ()
    final_loss: float | None = None
    train_metrics: dict[str, Any] = field(default_factory=dict)
    optimizer_requested: str = ""
    optimizer_class: str | None = None
    gradient_checkpointing_requested: bool = False
    gradient_checkpointing_active: bool | None = None
    use_cache: bool | None = None
    dropped_trainer_arguments: tuple[str, ...] = ()
    warmup_steps: int = 0
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
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
            "memory_before": dict(self.memory_before),
            "memory_after": dict(self.memory_after),
            "peak_stats_reset": self.peak_stats_reset,
            "wall_clock_seconds": self.wall_clock_seconds,
            "optimizer_steps": self.optimizer_steps,
            "seconds_per_optimizer_step": self.seconds_per_optimizer_step,
            "losses": [dict(entry) for entry in self.losses],
            "final_loss": self.final_loss,
            "train_metrics": dict(self.train_metrics),
            "optimizer_requested": self.optimizer_requested,
            "optimizer_class": self.optimizer_class,
            "gradient_checkpointing_requested": self.gradient_checkpointing_requested,
            "gradient_checkpointing_active": self.gradient_checkpointing_active,
            "use_cache": self.use_cache,
            "dropped_trainer_arguments": list(self.dropped_trainer_arguments),
            "warmup_steps": self.warmup_steps,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Output audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputAudit:
    """What the run actually left on disk, and whether the tree stayed clean.

    Attributes:
        run_root: The directory that holds run directories.
        run_dir: This run's directory.
        adapter_dir: Where the adapter was saved.
        adapter_exists: Whether that directory exists.
        adapter_files: Relative filenames inside it.
        adapter_bytes: Total size of the adapter directory.
        run_dir_bytes: Total size of the whole run directory.
        missing_expected_files: Which of :data:`_EXPECTED_ADAPTER_FILES` are absent.
        base_weight_files: Files whose names say base weights were copied.
        oversized_files: Files larger than an adapter has any reason to be.
        largest_file: The biggest file found, for a human to sanity-check.
        git_before: :func:`git_status` from before training started.
        git_after: :func:`git_status` from after everything was written.
        checks: Named invariants. ``True`` held, ``False`` failed, ``None`` undetermined.
        notes: Anything else worth recording.
    """

    run_root: str = ""
    run_dir: str = ""
    adapter_dir: str = ""
    adapter_exists: bool = False
    adapter_files: tuple[str, ...] = ()
    adapter_bytes: int = 0
    run_dir_bytes: int = 0
    missing_expected_files: tuple[str, ...] = ()
    base_weight_files: tuple[str, ...] = ()
    oversized_files: tuple[str, ...] = ()
    largest_file: dict[str, Any] | None = None
    git_before: dict[str, Any] = field(default_factory=dict)
    git_after: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool | None] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "adapter_files", tuple(self.adapter_files))
        object.__setattr__(self, "missing_expected_files", tuple(self.missing_expected_files))
        object.__setattr__(self, "base_weight_files", tuple(self.base_weight_files))
        object.__setattr__(self, "oversized_files", tuple(self.oversized_files))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of invariants that did not hold."""
        return tuple(name for name, value in self.checks.items() if value is False)

    @property
    def ok(self) -> bool:
        """Whether no invariant failed.

        Undetermined checks are tolerated here, unlike in :class:`RecordInspection`: a git
        query is unavailable outside a checkout, and that is not a reason to call a training
        run bad.
        """
        return not self.failed_checks

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "run_root": self.run_root,
            "run_dir": self.run_dir,
            "adapter_dir": self.adapter_dir,
            "adapter_exists": self.adapter_exists,
            "adapter_files": list(self.adapter_files),
            "adapter_bytes": self.adapter_bytes,
            "run_dir_bytes": self.run_dir_bytes,
            "missing_expected_files": list(self.missing_expected_files),
            "base_weight_files": list(self.base_weight_files),
            "oversized_files": list(self.oversized_files),
            "largest_file": dict(self.largest_file) if self.largest_file else None,
            "git_before": dict(self.git_before),
            "git_after": dict(self.git_after),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "ok": self.ok,
            "notes": list(self.notes),
        }


def git_status(repo_root: Path | None = None) -> dict[str, Any]:
    """Report whether the working tree is clean.

    Read-only: ``git status --porcelain`` and nothing else. Recorded because a training run
    that dirtied the repository has written somewhere it should not have, and the run
    directory living under a git-ignored ``artifacts/`` is the property being checked.

    Args:
        repo_root: The checkout to query. Discovered from this file when omitted.

    Returns:
        ``{"available", "clean", "entries", "entry_count"}``, with ``error`` added when git
        could not be consulted. ``clean`` is ``None`` when unavailable rather than ``True``,
        so an absent git is never mistaken for a clean tree.
    """
    try:
        root = repo_root or find_repo_root()
    except (OSError, RuntimeError) as exc:  # pragma: no cover - depends on the checkout
        return {"available": False, "clean": None, "entries": [], "error": str(exc)}

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "clean": None, "entries": [], "error": str(exc)}

    if completed.returncode != 0:
        return {
            "available": False,
            "clean": None,
            "entries": [],
            "error": completed.stderr.strip() or f"git exited {completed.returncode}",
        }

    entries = [line for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "clean": not entries,
        "entries": entries[:50],
        "entry_count": len(entries),
        "repo_root": root.as_posix(),
    }


def audit_output(
    paths: RunPaths,
    *,
    run_root: Path,
    git_before: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    check_git: bool = True,
) -> OutputAudit:
    """Check what a run wrote, and where.

    The git invariant is a *comparison*, not an absolute. "The tree is clean" is a fact about
    whoever is running the harness; "the run added nothing tracked" is a fact about the run,
    and it is the second one that says the artifacts landed under a git-ignored directory. So
    the porcelain output from before training is compared against the output from after, and
    only a difference counts as a failure. Absolute cleanliness is still reported, for a
    reviewer who cares.

    Args:
        paths: The run's paths.
        run_root: The directory run directories live under.
        git_before: The porcelain snapshot taken before training.
        repo_root: The checkout to query.
        check_git: Consult git at all.

    Returns:
        The :class:`OutputAudit`.
    """
    adapter = paths.adapter
    adapter_files = _relative_files(adapter)
    run_files = _relative_files(paths.root)

    base_weights = tuple(
        name
        for name in run_files
        if any(fnmatch(Path(name).name, pattern) for pattern in _BASE_WEIGHT_PATTERNS)
    )
    sizes = {name: (paths.root / name).stat().st_size for name in run_files}
    oversized = tuple(
        name for name, size in sizes.items() if size > _MAX_EXPECTED_FILE_BYTES
    )
    largest = max(sizes.items(), key=lambda item: item[1], default=None)
    missing = tuple(name for name in _EXPECTED_ADAPTER_FILES if name not in adapter_files)

    unavailable: dict[str, Any] = {"available": False, "clean": None, "entries": []}
    before = git_before if git_before is not None else unavailable
    after = git_status(repo_root) if check_git else unavailable
    comparable = bool(before.get("available")) and bool(after.get("available"))

    checks: dict[str, bool | None] = {
        "adapter_directory_exists": adapter.is_dir(),
        "adapter_checkpoint_is_complete": not missing and adapter.is_dir(),
        "adapter_is_under_the_run_root": adapter.is_relative_to(run_root),
        "run_directory_is_under_the_run_root": paths.root.is_relative_to(run_root),
        "no_base_model_weights_were_copied": not base_weights,
        "no_unexpectedly_large_files": not oversized,
        "the_run_added_no_tracked_changes": (
            sorted(before.get("entries") or []) == sorted(after.get("entries") or [])
            if comparable
            else None
        ),
    }

    notes = [
        "the adapter is saved by PEFT's save_pretrained, which writes adapter weights only; "
        "the base checkpoint stays in the Hugging Face cache",
        "the tokenizer is deliberately not copied into the run directory, so any large file "
        "here is a finding rather than an expected artifact",
        f"the harness's own {SMOKE_REPORT_FILENAME} is written after this audit and is "
        "therefore absent from the file list and the byte totals",
    ]
    if not comparable:
        notes.append(
            "git could not be compared before and after, so 'the run added no tracked "
            "changes' is undetermined rather than passing"
        )
    elif not after.get("clean"):
        notes.append(
            "the working tree is not clean, but it holds the same entries it held before "
            "training, so the run added nothing tracked"
        )

    return OutputAudit(
        run_root=run_root.as_posix(),
        run_dir=paths.root.as_posix(),
        adapter_dir=adapter.as_posix(),
        adapter_exists=adapter.is_dir(),
        adapter_files=adapter_files,
        adapter_bytes=sum(
            size for name, size in sizes.items() if name.startswith(f"{adapter.name}/")
        ),
        run_dir_bytes=sum(sizes.values()),
        missing_expected_files=missing,
        base_weight_files=base_weights,
        oversized_files=oversized,
        largest_file=(
            {"path": largest[0], "bytes": largest[1]} if largest is not None else None
        ),
        git_before=before,
        git_after=after,
        checks=checks,
        notes=tuple(notes),
    )


def _relative_files(root: Path) -> tuple[str, ...]:
    """Return every file under ``root``, as POSIX paths relative to it, sorted.

    Args:
        root: The directory to walk. A missing directory yields an empty tuple.

    Returns:
        The relative filenames.
    """
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        )
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmokeReport:
    """Everything one invocation of the harness found out.

    Attributes:
        mode: Which mode ran.
        status: ``"inspected"``, ``"trained"``, or ``"failed"``.
        trained: Whether ``trainer.train()`` was called. The single unambiguous answer to the
            question a reviewer will actually ask.
        config_path: The configuration file that was read.
        experiment: The experiment name.
        config_hash: Hash of the resolved configuration, overrides included.
        record_format: The TRL record shape used.
        overrides: The in-memory training overrides applied.
        run_root: Where run directories live.
        output_dir: The trainer's output directory.
        output_dir_created: Whether it exists after the harness ran.
        dataset: Smoke-corpus provenance.
        dataset_validation: Result of the Phase 17A validation over the corpus.
        records: The untokenized TRL records, for provenance.
        inspection: The first tokenized record, inspected.
        measurements: What was measured, when training ran.
        output_audit: What was left on disk, when training ran.
        diagnostics: The runtime diagnostic report.
        trainer_plan: The translated trainer arguments.
        trl_warnings: Warnings TRL logged while preparing the dataset.
        artifacts: Files the harness wrote.
        notes: Anything else worth recording.
    """

    mode: str
    status: str
    trained: bool = False
    config_path: str = ""
    experiment: str = ""
    config_hash: str = ""
    record_format: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    run_root: str = ""
    output_dir: str = ""
    output_dir_created: bool = False
    dataset: dict[str, Any] = field(default_factory=dict)
    dataset_validation: dict[str, Any] = field(default_factory=dict)
    records: tuple[dict[str, Any], ...] = ()
    inspection: RecordInspection | None = None
    measurements: SmokeMeasurements | None = None
    output_audit: OutputAudit | None = None
    diagnostics: dict[str, Any] | None = None
    trainer_plan: dict[str, Any] | None = None
    trl_warnings: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "trl_warnings", tuple(self.trl_warnings))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def ok(self) -> bool:
        """Whether every check the harness was able to make held."""
        if self.inspection is not None and not self.inspection.ok:
            return False
        return not (self.output_audit is not None and not self.output_audit.ok)

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
            "run_root": self.run_root,
            "output_dir": self.output_dir,
            "output_dir_created": self.output_dir_created,
            "dataset": dict(self.dataset),
            "dataset_validation": dict(self.dataset_validation),
            "records": [dict(record) for record in self.records],
            "inspection": self.inspection.as_dict() if self.inspection else None,
            "measurements": self.measurements.as_dict() if self.measurements else None,
            "output_audit": self.output_audit.as_dict() if self.output_audit else None,
            "diagnostics": dict(self.diagnostics) if self.diagnostics else None,
            "trainer_plan": dict(self.trainer_plan) if self.trainer_plan else None,
            "trl_warnings": list(self.trl_warnings),
            "artifacts": dict(self.artifacts),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The one training call
# ---------------------------------------------------------------------------


def _call_trainer_train(trainer: Any, *, execute: bool) -> Any:
    """Call ``trainer.train()``. The only place in this repository that does.

    Kept as a one-line function with a mandatory keyword guard so the question "where does
    this project train?" has exactly one answer, and so a test can assert that the
    inspect-only path never reaches it.

    Args:
        trainer: A constructed ``SFTTrainer``.
        execute: Must be ``True``. Passed explicitly by the caller that resolved
            :attr:`SmokeMode.RUN`; there is no default.

    Returns:
        The trainer's ``TrainOutput``.

    Raises:
        SmokeHarnessError: If ``execute`` is ``False``. A defensive check rather than the
            primary one -- the caller already branches on the mode -- but the cost of a second
            guard here is one comparison and the cost of not having it is an accidental
            fourteen-minute GPU run.
    """
    if not execute:
        raise SmokeHarnessError(
            "refusing to call trainer.train(): this harness trains only under --run. "
            "--inspect-only builds the corpus, the model, the adapters and the trainer, and "
            "inspects the first tokenized record, without stepping the optimiser."
        )
    return trainer.train()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_smoke(
    config_path: str,
    *,
    mode: SmokeMode,
    output_dir: str | None = None,
    run_id: str | None = None,
    max_steps: int = SMOKE_MAX_STEPS,
    batch_size: int = SMOKE_BATCH_SIZE,
    gradient_accumulation_steps: int = SMOKE_GRADIENT_ACCUMULATION_STEPS,
    check_git: bool = True,
) -> SmokeReport:
    """Build the harness, inspect it, and train only under :attr:`SmokeMode.RUN`.

    The sequence, identical in both modes up to the last step: apply the bounded overrides,
    load the configuration, build and validate the six-example corpus, render it into TRL
    records, wrap them in an in-memory ``datasets.Dataset``, load and adapt the real model,
    construct the real ``SFTTrainer``, then read the first prepared record back out and check
    it. Under :attr:`SmokeMode.RUN`, and only then, the optimiser steps.

    Args:
        config_path: Path to the experiment configuration.
        mode: Which of the two things to do.
        output_dir: Override the run root. Defaults to ``<artifacts>/qgen-runs``.
        run_id: Override the generated run directory name.
        max_steps: Optimiser steps, under :attr:`SmokeMode.RUN`.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        check_git: Consult git during the output audit.

    Returns:
        The :class:`SmokeReport`.

    Raises:
        SmokeHarnessError: If the corpus does not validate, or the prepared record cannot be
            inspected.
        qa_gen_runtime.config_io.ConfigIOError: If the configuration cannot be read.
        qa_gen.config.GenerationConfigError: If it is invalid.
        qa_gen_runtime.precision.PrecisionError: If the precision is unavailable here.
        qa_gen_runtime.deps.RuntimeDependencyError: If PEFT, TRL or ``datasets`` is missing.
        qa_gen_runtime.loader.ModelLoadError: If the model cannot be loaded or adapted.
        qa_gen_runtime.trainer.TrainerBuildError: If the trainer cannot be built.
        qa_gen_runtime.outputs.RunOutputError: If the run directory cannot be created.
    """
    overrides = smoke_training_overrides(
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    config = load_experiment_config(config_path, overrides=overrides)

    examples = build_smoke_examples()
    validation = validate_smoke_examples(examples)
    if not validation.ok:
        raise SmokeHarnessError(
            "the built-in smoke corpus does not validate, which is a defect in "
            "qa_gen_runtime.smoke_data rather than in the configuration:\n"
            + "\n".join(f"  - {issue.message}" for issue in validation.errors)
        )

    records = build_training_records(examples, config)
    run_root = resolve_run_root(output_dir)
    paths = _resolve_paths(
        config, mode=mode, output_dir=output_dir, run_id=run_id, run_root=run_root
    )
    # Taken before anything is loaded, so the comparison in audit_output measures the run and
    # not the state of somebody's checkout.
    git_before = git_status() if check_git else None

    notes = [
        f"mode={mode.value}; trainer.train() is reachable only under {SmokeMode.RUN.value}",
        "the shipped configuration file was not modified; the overrides above were applied "
        "in memory only",
        f"corpus: {len(examples)} hand-written examples from qa_gen_runtime.smoke_data, "
        "nothing downloaded",
    ]

    train_dataset = build_hf_dataset(records)
    loaded = load_trainable_model(config)

    with _capture_logger_warnings("trl") as trl_warnings:
        trainer, plan = build_trainer(
            config,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            train_dataset=train_dataset,
            output_dir=paths.root,
            train_examples=len(records),
            precision=loaded.precision,
        )

    inspection = inspect_tokenized_record(
        first_tokenized_record(trainer),
        loaded.tokenizer,
        max_length=config.model.max_seq_length,
        context_probe=examples[0].context[:48],
        source_target=examples[0].primary_target,
    )

    if mode is SmokeMode.INSPECT_ONLY:
        notes.append(
            "no optimiser step was taken and no adapter was saved; run again with --run to "
            "measure a bounded execution"
        )
        return SmokeReport(
            mode=mode.value,
            status="inspected",
            trained=False,
            config_path=config_path,
            experiment=config.name,
            config_hash=config.config_hash(),
            record_format=resolve_record_format(config).value,
            overrides=overrides,
            run_root=run_root.as_posix(),
            output_dir=paths.root.as_posix(),
            output_dir_created=paths.root.exists(),
            dataset=describe_smoke_corpus(examples),
            dataset_validation=validation.as_dict(),
            records=tuple(records),
            inspection=inspection,
            diagnostics=collect_diagnostics(
                config,
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                precision=loaded.precision,
                trainer_plan=plan.as_dict(),
                extra_notes=("inspect-only: no optimiser step was taken",),
            ).as_dict(),
            trainer_plan=plan.as_dict(),
            trl_warnings=tuple(trl_warnings),
            notes=tuple(notes),
        )

    measurements, train_notes = _train_and_measure(
        trainer, loaded=loaded, config=config, plan=plan
    )
    notes.extend(train_notes)

    loaded.model.save_pretrained(str(paths.adapter))

    diagnostics = collect_diagnostics(
        config,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        precision=loaded.precision,
        trainer_plan=plan.as_dict(),
        extra_notes=(
            f"bounded smoke execution: {measurements.optimizer_steps} optimiser step(s) over "
            f"{len(records)} hand-written examples",
            "no metric from this run means anything; it measures the plumbing",
        ),
    )
    corpus = describe_smoke_corpus(examples)

    # Order matters: every document except the harness's own report is written first, then the
    # directory is audited, then the report -- which contains the audit -- is written last. The
    # audit therefore does not count smoke.json in its byte total, which is recorded in its
    # notes rather than papered over.
    artifacts = _write_run_documents(
        config,
        paths,
        corpus=corpus,
        validation=validation.as_dict(),
        records=records,
        diagnostics=diagnostics,
        measurements=measurements,
        overrides=overrides,
        inspection=inspection,
    )
    audit = audit_output(
        paths, run_root=run_root, git_before=git_before, check_git=check_git
    )
    artifacts["smoke"] = (paths.root / SMOKE_REPORT_FILENAME).as_posix()

    report = SmokeReport(
        mode=mode.value,
        status="trained",
        trained=True,
        config_path=config_path,
        experiment=config.name,
        config_hash=config.config_hash(),
        record_format=resolve_record_format(config).value,
        overrides=overrides,
        run_root=run_root.as_posix(),
        output_dir=paths.root.as_posix(),
        output_dir_created=paths.root.exists(),
        dataset=corpus,
        dataset_validation=validation.as_dict(),
        records=tuple(records),
        inspection=inspection,
        measurements=measurements,
        output_audit=audit,
        diagnostics=diagnostics.as_dict(),
        trainer_plan=plan.as_dict(),
        trl_warnings=tuple(trl_warnings),
        artifacts=artifacts,
        notes=tuple(notes),
    )
    _write_json(paths.root / SMOKE_REPORT_FILENAME, report.as_dict())
    return report


def _resolve_paths(
    config: GenerationExperimentConfig,
    *,
    mode: SmokeMode,
    output_dir: str | None,
    run_id: str | None,
    run_root: Path,
) -> RunPaths:
    """Decide where this invocation writes, creating a directory only when it will train.

    Args:
        config: The experiment configuration, which generates the run id.
        mode: Which mode is running.
        output_dir: Override for the run root.
        run_id: Override for the directory name.
        run_root: The already-resolved run root.

    Returns:
        The :class:`~qa_gen_runtime.outputs.RunPaths`. Created under :attr:`SmokeMode.RUN`;
        derived but not created under :attr:`SmokeMode.INSPECT_ONLY`, where the only thing
        that might create it is the trainer itself.

    Raises:
        qa_gen_runtime.outputs.RunOutputError: If the directory exists already.
    """
    if mode is SmokeMode.RUN:
        return create_run_directory(config, output_dir=output_dir, run_id=run_id)
    resolved = run_id or f"{config.run_id(utc_timestamp())}-inspect"
    return RunPaths.under(run_root / resolved)


def _train_and_measure(
    trainer: Any,
    *,
    loaded: Any,
    config: GenerationExperimentConfig,
    plan: Any,
) -> tuple[SmokeMeasurements, list[str]]:
    """Run the bounded execution and measure it.

    Args:
        trainer: The constructed trainer.
        loaded: The :class:`~qa_gen_runtime.loader.LoadedModel`.
        config: The experiment configuration.
        plan: The :class:`~qa_gen_runtime.trainer.TrainerPlan`.

    Returns:
        ``(measurements, notes)``.
    """
    reset = _reset_peak_memory()
    before = memory_report()

    started = time.perf_counter()
    output = _call_trainer_train(trainer, execute=True)
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

    notes: list[str] = []
    if not reset:
        notes.append(
            "CUDA is not available, so every memory figure is null and the run happened on "
            "the CPU. A CPU smoke run proves the plumbing and measures nothing about VRAM."
        )
    if not losses:
        notes.append(
            "no per-step loss was logged. logging_steps is 1, so an empty history means the "
            "trainer's callback did not record one; the mean loss in train_metrics is still "
            "reported."
        )

    return (
        SmokeMeasurements(
            trainable_parameters=loaded.trainable_parameters,
            total_parameters=loaded.total_parameters,
            trainable_fraction=loaded.trainable_fraction,
            memory_before=before,
            memory_after=after,
            peak_stats_reset=reset,
            wall_clock_seconds=round(elapsed, 3),
            optimizer_steps=steps,
            seconds_per_optimizer_step=(round(elapsed / steps, 3) if steps else None),
            losses=losses,
            final_loss=metrics.get("train_loss", getattr(output, "training_loss", None)),
            train_metrics=metrics,
            optimizer_requested=config.training.optimizer,
            optimizer_class=_describe_optimizer(trainer),
            gradient_checkpointing_requested=config.training.gradient_checkpointing,
            gradient_checkpointing_active=getattr(
                loaded.model, "is_gradient_checkpointing", None
            ),
            use_cache=getattr(getattr(loaded.model, "config", None), "use_cache", None),
            dropped_trainer_arguments=plan.dropped_arguments,
            warmup_steps=plan.warmup_steps,
            notes=tuple(notes),
        ),
        notes,
    )


def _describe_optimizer(trainer: Any) -> str | None:
    """Return the fully qualified class name of the optimiser the trainer built.

    Reported because ``optim="paged_adamw_8bit"`` is a string that transformers resolves at
    training time, and the project has never benchmarked it. Knowing which class was actually
    constructed is the difference between a claim and a measurement.

    Args:
        trainer: The trainer, after training.

    Returns:
        ``"module.ClassName"``, or ``None`` when no optimiser is attached.
    """
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return None
    kind = type(optimizer)
    return f"{kind.__module__}.{kind.__qualname__}"


def _reset_peak_memory() -> bool:
    """Clear the CUDA peak-memory counters.

    Returns:
        ``True`` when they were reset, ``False`` when there is no CUDA device. Reported rather
        than assumed, because an unreset peak would fold the model load into the step
        measurement and make the two indistinguishable.
    """
    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


@contextlib.contextmanager
def _capture_logger_warnings(name: str) -> Iterator[list[str]]:
    """Collect warnings emitted by a logger tree while the block runs.

    Used around trainer construction to catch TRL's own "mismatch between tokenized prompt
    and the start of tokenized prompt+completion" warning, which is the library telling us the
    completion mask is offset. It is logged and then ignored by TRL, so it has to be captured
    here or it scrolls past.

    Args:
        name: Logger name to attach to, e.g. ``"trl"``.

    Yields:
        A list that fills with formatted messages as they are emitted.
    """
    captured: list[str] = []

    class _Collector(logging.Handler):
        """A handler that appends formatted messages to ``captured``."""

        def emit(self, record: logging.LogRecord) -> None:
            """Record one message."""
            captured.append(f"{record.name}: {record.getMessage()}")

    handler = _Collector(level=logging.WARNING)
    target = logging.getLogger(name)
    target.addHandler(handler)
    try:
        yield captured
    finally:
        target.removeHandler(handler)


def _write_run_documents(
    config: GenerationExperimentConfig,
    paths: RunPaths,
    *,
    corpus: dict[str, Any],
    validation: dict[str, Any],
    records: list[dict[str, Any]],
    diagnostics: Any,
    measurements: SmokeMeasurements,
    overrides: dict[str, Any],
    inspection: RecordInspection,
) -> dict[str, str]:
    """Write the Phase 17A run documents into the run directory.

    The harness's own report is written by the caller afterwards, so this function can be
    followed by an output audit that the report then carries.

    Args:
        config: The resolved configuration.
        paths: The run's paths.
        corpus: Smoke-corpus provenance.
        validation: The dataset validation report.
        records: The untokenized TRL records.
        diagnostics: The :class:`~qa_gen_runtime.diagnostics.RuntimeDiagnostics`.
        measurements: What was measured.
        overrides: The in-memory training overrides applied.
        inspection: The record inspection.

    Returns:
        A mapping of document name to the path written.
    """
    metadata = TrainingRunMetadata(
        run_id=paths.root.name,
        experiment_name=config.name,
        phase=config.phase,
        status="completed",
        config=config.to_dict(),
        config_hash=config.config_hash(),
        adapter_path=paths.adapter.as_posix(),
        dataset=corpus,
        notes=[
            "Phase 17B.2 smoke execution: a bounded number of optimiser steps over a "
            "hand-written six-example corpus",
            "not a trained adapter in any useful sense; it exists to prove the path runs",
        ],
    )
    attach_to_metadata(metadata, diagnostics)
    metadata.training["smoke"] = {
        "overrides": overrides,
        "measurements": measurements.as_dict(),
        "inspection_ok": inspection.ok,
        "failed_checks": list(inspection.failed_checks),
    }

    write_resolved_config(config, paths.config)
    _write_json(paths.diagnostics, diagnostics.as_dict())
    _write_json(
        paths.dataset,
        {"provenance": corpus, "validation": validation, "records": list(records)},
    )
    _write_json(paths.record, metadata.as_dict())

    return {
        "config": paths.config.as_posix(),
        "diagnostics": paths.diagnostics.as_posix(),
        "dataset": paths.dataset.as_posix(),
        "record": paths.record.as_posix(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a mapping as an indented JSON document, creating parents.

    Args:
        path: Destination file.
        payload: The mapping to write.

    Returns:
        The path written.
    """
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
        default mode: a command that does not say which of the two it wants exits ``2``
        without loading anything.
    """
    parser = argparse.ArgumentParser(
        prog="qa_gen_runtime.smoke",
        description=(
            "Phase 17B.2 QLoRA smoke harness. Builds a six-example in-memory corpus, the "
            "real quantized model, the real adapters and the real TRL trainer, and inspects "
            "the first tokenized record. Trains only under --run."
        ),
        epilog=(
            "Both modes download the base model. Neither downloads a dataset. Only --run "
            "calls trainer.train()."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML or JSON experiment configuration.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "Build everything and inspect the first tokenized record. Never calls "
            "trainer.train()."
        ),
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help=(
            "The only mode that trains. Takes a bounded number of optimiser steps and saves "
            "one adapter checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the run output root. Defaults to <artifacts>/qgen-runs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the generated run directory name.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SMOKE_MAX_STEPS,
        help=f"Optimiser steps under --run. Defaults to {SMOKE_MAX_STEPS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SMOKE_BATCH_SIZE,
        help=(
            f"Micro-batch per device. Defaults to {SMOKE_BATCH_SIZE}, which is what the L4 "
            "feasibility measurement used."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=SMOKE_GRADIENT_ACCUMULATION_STEPS,
        help=(
            f"Micro-batches per optimiser step. Defaults to "
            f"{SMOKE_GRADIENT_ACCUMULATION_STEPS} so the peak VRAM is comparable with the "
            "recorded single-example figure, not to the 8 in the planning configuration."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Also write the JSON report to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON rather than as text.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Defaults to INFO.",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> SmokeMode:
    """Map parsed arguments onto a mode.

    Args:
        args: Parsed arguments.

    Returns:
        :attr:`SmokeMode.RUN` when ``--run`` was given, :attr:`SmokeMode.INSPECT_ONLY`
        otherwise. Separated from :func:`main` so the mapping is assertable without running
        anything, and written so that anything other than an explicit ``--run`` resolves to
        the mode that cannot train.
    """
    return SmokeMode.RUN if getattr(args, "run", False) else SmokeMode.INSPECT_ONLY


def format_report(report: SmokeReport) -> str:
    """Render a report as readable text.

    Args:
        report: The report.

    Returns:
        The formatted text.
    """
    lines = [
        f"mode          : {report.mode}",
        f"trained       : {report.trained}",
        f"configuration : {report.config_path}",
        f"experiment    : {report.experiment} ({report.config_hash})",
        f"record format : {report.record_format}",
        f"corpus        : {report.dataset.get('example_count')} in-memory examples "
        f"(downloaded: {report.dataset.get('downloaded')})",
        f"output dir    : {report.output_dir} (exists: {report.output_dir_created})",
    ]
    overrides = report.overrides.get("training", {})
    lines.append(
        "overrides     : max_steps={max_steps} batch={batch} accum={accum} "
        "eval={eval} save={save}".format(
            max_steps=overrides.get("max_steps"),
            batch=overrides.get("per_device_train_batch_size"),
            accum=overrides.get("gradient_accumulation_steps"),
            eval=overrides.get("evaluation_strategy"),
            save=overrides.get("save_strategy"),
        )
    )

    inspection = report.inspection
    if inspection is not None:
        lines.extend(_format_inspection(inspection))

    if report.trl_warnings:
        lines.append("trl warnings  :")
        lines.extend(f"  - {message}" for message in report.trl_warnings)

    if report.measurements is not None:
        lines.extend(_format_measurements(report.measurements))

    if report.output_audit is not None:
        lines.extend(_format_audit(report.output_audit))

    if report.artifacts:
        lines.append("artifacts     :")
        lines.extend(f"  - {name}: {path}" for name, path in sorted(report.artifacts.items()))

    lines.append(f"overall       : {'ok' if report.ok else 'CHECKS FAILED'}")
    if report.notes:
        lines.append("notes         :")
        lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)


def _format_inspection(inspection: RecordInspection) -> list[str]:
    """Render the record inspection section."""
    markers = inspection.thinking_markers
    lines = [
        "--- first tokenized record ---",
        f"example id    : {inspection.example_id}",
        f"columns       : {', '.join(inspection.columns)}",
        f"input_ids     : {inspection.input_ids_length} tokens "
        f"(max_length {inspection.max_length})",
        f"mask column   : {inspection.mask_column} "
        f"(length {inspection.completion_mask_length})",
        f"prompt tokens : {inspection.prompt_token_count}",
        f"completion    : {inspection.completion_token_count} tokens",
        f"expected head : {inspection.expected_completion_prefix}",
        f"decoded head  : {inspection.decoded_completion[:80]!r}",
        f"decoded tail  : {inspection.decoded_completion[-40:]!r}",
        f"parsed target : {'yes' if inspection.parsed_target else 'no'}"
        + (f" ({inspection.parse_error})" if inspection.parse_error else ""),
        f"think markers : counts={markers.get('counts')} "
        f"prompt={markers.get('present_in_prompt')} "
        f"completion={markers.get('present_in_completion')} "
        f"(record left unmodified)",
        "checks        :",
    ]
    lines.extend(
        f"  [{_check_glyph(value)}] {name}" for name, value in inspection.checks.items()
    )
    if inspection.notes:
        lines.extend(f"  ! {note}" for note in inspection.notes)
    return lines


def _format_measurements(measurements: SmokeMeasurements) -> list[str]:
    """Render the measurement section."""
    before = measurements.memory_before
    after = measurements.memory_after
    if measurements.trainable_parameters and measurements.total_parameters:
        parameters = (
            f"parameters    : {measurements.trainable_parameters:,} trainable of "
            f"{measurements.total_parameters:,} ({measurements.trainable_fraction})"
        )
    else:
        parameters = "parameters    : not measured"
    lines = [
        "--- measurements ---",
        parameters,
        f"vram initial  : allocated={before.get('allocated_gib')} GiB "
        f"reserved={before.get('reserved_gib')} GiB",
        f"vram peak     : allocated={after.get('max_allocated_gib')} GiB "
        f"reserved={after.get('max_reserved_gib')} GiB",
        f"vram total    : {after.get('total_vram_gib')} GiB "
        f"(peak counters reset: {measurements.peak_stats_reset})",
        f"steps         : {measurements.optimizer_steps} "
        f"(warmup {measurements.warmup_steps})",
        f"wall clock    : {measurements.wall_clock_seconds} s "
        f"({measurements.seconds_per_optimizer_step} s/step)",
        f"final loss    : {measurements.final_loss}",
        f"optimizer     : requested {measurements.optimizer_requested!r} -> "
        f"{measurements.optimizer_class}",
        f"checkpointing : requested {measurements.gradient_checkpointing_requested}, "
        f"active {measurements.gradient_checkpointing_active}, "
        f"use_cache {measurements.use_cache}",
        f"dropped args  : {', '.join(measurements.dropped_trainer_arguments) or 'none'}",
    ]
    if measurements.losses:
        lines.append("per-step loss :")
        lines.extend(
            f"  step {entry.get('step')}: loss={entry.get('loss')} "
            f"lr={entry.get('learning_rate')} grad_norm={entry.get('grad_norm')}"
            for entry in measurements.losses
        )
    return lines


def _format_audit(audit: OutputAudit) -> list[str]:
    """Render the output audit section."""
    lines = [
        "--- output audit ---",
        f"adapter       : {audit.adapter_dir} (exists: {audit.adapter_exists})",
        f"adapter files : {', '.join(audit.adapter_files) or 'none'}",
        f"adapter bytes : {audit.adapter_bytes:,}",
        f"run dir bytes : {audit.run_dir_bytes:,}",
        f"largest file  : {audit.largest_file}",
        f"base weights  : {', '.join(audit.base_weight_files) or 'none found'}",
        f"git before    : available={audit.git_before.get('available')} "
        f"clean={audit.git_before.get('clean')} "
        f"entries={audit.git_before.get('entry_count')}",
        f"git after     : available={audit.git_after.get('available')} "
        f"clean={audit.git_after.get('clean')} "
        f"entries={audit.git_after.get('entry_count')}",
        "checks        :",
    ]
    lines.extend(f"  [{_check_glyph(value)}] {name}" for name, value in audit.checks.items())
    if audit.notes:
        lines.extend(f"  ! {note}" for note in audit.notes)
    return lines


def _check_glyph(value: bool | None) -> str:
    """Return a single character standing for a check result."""
    if value is True:
        return "x"
    if value is False:
        return "!"
    return "?"


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when the harness ran and every check held, ``1`` on a configuration or runtime
        failure, ``3`` when it ran but a check did not hold.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = resolve_mode(args)
    logger.info("smoke harness mode: %s", mode.value)

    try:
        report = run_smoke(
            args.config,
            mode=mode,
            output_dir=args.output_dir,
            run_id=args.run_id,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    except (
        ConfigIOError,
        DatasetBuildError,
        GenerationConfigError,
        ModelLoadError,
        PrecisionError,
        RunOutputError,
        RuntimeDependencyError,
        SmokeHarnessError,
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

    if not report.ok:
        failures = list(report.inspection.failed_checks if report.inspection else ())
        failures.extend(report.inspection.undetermined_checks if report.inspection else ())
        failures.extend(report.output_audit.failed_checks if report.output_audit else ())
        print(
            "\nchecks did not pass: " + ", ".join(failures) + "\n"
            "The harness ran; the invariants above did not hold. The completion mask checks "
            "are the ones that matter: a failure there means the loss was taken over the "
            "wrong tokens.",
            file=sys.stderr,
        )
        return _EXIT_INVARIANT_FAILED

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
