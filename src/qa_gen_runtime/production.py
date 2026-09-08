"""Phase 18: real QLoRA training over a prepared corpus.

What this module is, and what it deliberately is not
----------------------------------------------------
This is the production training execution path. It reads a corpus that
:mod:`qa_gen_runtime.prepare` already wrote, loads the real quantized model, attaches the
real adapters, builds the real TRL trainer, steps the optimiser over the whole training
split, and saves an adapter-only checkpoint.

It is **not** a second runtime. Every component it uses already existed and was tested
before this phase:

============================  ===================================================
step                          owner
============================  ===================================================
read the prepared corpus      :func:`qa_gen_runtime.prepared.read_prepared_split`
load model, tokenizer, LoRA   :func:`qa_gen_runtime.loader.load_trainable_model`
render TRL records            :func:`qa_gen_runtime.dataset.build_training_records`
wrap them for the trainer     :func:`qa_gen_runtime.dataset.build_hf_dataset`
construct the trainer         :func:`qa_gen_runtime.trainer.build_trainer`
describe the runtime          :func:`qa_gen_runtime.diagnostics.collect_diagnostics`
place the run on disk         :func:`qa_gen_runtime.outputs.create_run_directory`
check what was written        :func:`qa_gen_runtime.benchmark.audit_benchmark_output`
============================  ===================================================

What this module adds is the three things that were genuinely missing: an orchestrator that
runs that sequence over a real corpus, a preflight that refuses to spend hours of GPU time
on a misconfigured run, and a final report that says what happened.

The six-example smoke harness in :mod:`qa_gen_runtime.smoke` is untouched and remains the
right tool for proving the path runs at all. This module is the one that produces an adapter
anybody would use.

Why a preflight exists
----------------------
A 2,248-step run on an L4 is roughly four hours and several credits. Every failure this
preflight catches is one that would otherwise be discovered *after* paying for it, and
several of them are silent: a chat template that emits an empty ``<think></think>`` block
into the supervised target, a completion mask that never got enabled, adapters wrapped
twice. None of those raise. They just produce a worse adapter.

So the checks run after the model, the records and the trainer all exist -- which is the
earliest moment they can be checked against reality rather than against intent -- and
before the first optimiser step. A blocking failure stops the run there.

Failure is reported, not swallowed
----------------------------------
If ``trainer.train()`` raises -- CUDA OOM being the likely one -- the report is still
written, with ``success`` false, the exception type and message, and the number of optimiser
steps that completed. Then the exception is re-raised. A run that failed never reports
success, and it never continues to the save step.

Resume
------
Resuming is plumbed through to ``trainer.train(resume_from_checkpoint=...)`` and the run
directory can be reopened with ``allow_existing=True``. It only works if checkpoints exist,
which requires ``training.save_strategy`` to be ``"steps"`` or ``"epoch"``. The shipped
``qgen-mixed.yaml`` sets ``"no"``, so resuming a run made with it is impossible by
construction; :func:`execute_production_training` refuses with that explanation rather than
starting from scratch while looking like it resumed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_gen.config import VERIFIED_QWEN3_4B_L4, GenerationExperimentConfig
from qa_gen.metadata import TrainingRunMetadata
from qa_gen.splitting import SplitName
from qa_gen_runtime.benchmark import audit_benchmark_output
from qa_gen_runtime.chat import chat_template_kwargs
from qa_gen_runtime.config_io import load_experiment_config, write_resolved_config
from qa_gen_runtime.dataset import (
    CHAT_TEMPLATE_KWARGS_COLUMN,
    build_hf_dataset,
    build_training_records,
    resolve_record_format,
)
from qa_gen_runtime.diagnostics import attach_to_metadata, collect_diagnostics, memory_report
from qa_gen_runtime.loader import load_trainable_model
from qa_gen_runtime.outputs import RunPaths, create_run_directory, resolve_run_root
from qa_gen_runtime.prepared import (
    read_prepared_split,
    resolve_prepared_directory,
    verify_fingerprint,
)
from qa_gen_runtime.trainer import build_trainer

logger = logging.getLogger(__name__)

__all__ = [
    "TRAINING_REPORT_FILENAME",
    "PreflightCheck",
    "PreflightReport",
    "ProductionTrainingError",
    "ProductionTrainingReport",
    "TrainingMeasurements",
    "describe_lora_attachment",
    "describe_loaded_quantization",
    "execute_production_training",
    "run_preflight",
]

#: The machine-readable final report, written into the run directory.
TRAINING_REPORT_FILENAME = "training.json"




class ProductionTrainingError(RuntimeError):
    """Raised when a production training run cannot start, or did not finish.

    Distinct from the component errors it may wrap. ``ModelLoadError`` means the model is the
    problem and ``PreparedDatasetError`` means the corpus is; this means the *run* is, which
    is the case where the answer is usually a flag rather than a fix.
    """


# ---------------------------------------------------------------------------
# Inspecting a loaded model
# ---------------------------------------------------------------------------


def describe_lora_attachment(model: Any) -> dict[str, Any]:
    """Report how adapters are attached to a loaded model.

    Measured from the object rather than assumed from the configuration.
    :attr:`qa_gen_runtime.loader.LoadedModel.adapters_attached` is a hard-coded ``True`` and
    :func:`qa_gen_runtime.loader.attach_adapters` has no double-attachment guard, so "LoRA is
    attached exactly once" is a claim that needs checking against the model.

    A double wrap is the failure worth naming: ``get_peft_model`` on an already-wrapped model
    nests a ``PeftModel`` inside a ``PeftModel``. Nothing raises, the parameter count roughly
    doubles, and the saved adapter is not loadable the way anyone expects.

    Args:
        model: The model returned by :func:`~qa_gen_runtime.loader.load_trainable_model`.

    Returns:
        A JSON-serializable mapping. Fields that cannot be determined are ``None`` rather
        than guessed.
    """
    type_name = type(model).__name__
    peft_config = getattr(model, "peft_config", None)
    adapter_names = sorted(peft_config) if isinstance(peft_config, dict) else None

    # A nested wrap shows up as a PeftModel whose base_model chain contains another one, and
    # as duplicated "base_model.model.base_model" segments in the parameter names.
    nested = None
    inner = getattr(model, "base_model", None)
    if inner is not None:
        nested = "PeftModel" in type(inner).__name__ or "PeftModel" in type(
            getattr(inner, "model", None)
        ).__name__

    duplicated_wrap = None
    lora_modules = None
    if hasattr(model, "named_parameters"):
        names = [name for name, _ in model.named_parameters()]
        duplicated_wrap = any(name.count("base_model.model.base_model") for name in names)
        lora_modules = sum(1 for name in names if ".lora_A." in name)

    active = getattr(model, "active_adapters", None)
    if callable(active):  # some PEFT versions expose it as a method
        try:
            active = active()
        except Exception:  # noqa: BLE001 - reporting only, never fatal
            active = None

    return {
        "model_type": type_name,
        "is_peft_model": "PeftModel" in type_name,
        "adapter_names": adapter_names,
        "adapter_count": len(adapter_names) if adapter_names is not None else None,
        "active_adapters": list(active) if isinstance(active, (list, tuple)) else active,
        "base_model_is_also_peft": nested,
        "parameter_names_show_double_wrap": duplicated_wrap,
        "lora_a_parameter_count": lora_modules,
    }


def describe_loaded_quantization(model: Any) -> dict[str, Any]:
    """Report the quantization actually in effect on a loaded model.

    :func:`qa_gen_runtime.quantization.describe_quantization` reports what the *configuration*
    asked for. This reads what the loaded object got, which is the only version that can
    disagree with intent -- a silently unquantized load is a memory problem discovered at the
    first backward pass.

    Args:
        model: The loaded, possibly PEFT-wrapped model.

    Returns:
        A JSON-serializable mapping, with ``None`` for anything not determinable.
    """
    base = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            base = getter()
        except Exception:  # noqa: BLE001 - reporting only
            base = model

    config = getattr(base, "config", None)
    quantization = getattr(config, "quantization_config", None)
    if quantization is None:
        return {
            "quantization_config_present": False,
            "load_in_4bit": None,
            "quant_type": None,
            "double_quant": None,
            "compute_dtype": None,
        }

    def read(name: str) -> Any:
        value = getattr(quantization, name, None)
        if value is None and isinstance(quantization, dict):
            value = quantization.get(name)
        return value

    compute_dtype = read("bnb_4bit_compute_dtype")
    return {
        "quantization_config_present": True,
        "load_in_4bit": read("load_in_4bit"),
        "quant_type": read("bnb_4bit_quant_type"),
        "double_quant": read("bnb_4bit_use_double_quant"),
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else None,
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One thing that had to be true before the optimiser stepped.

    Attributes:
        name: Short identifier, stable enough to assert on.
        passed: ``True`` held, ``False`` failed, ``None`` could not be determined here.
        detail: What was actually observed, so a failure is diagnosable from the report alone.
        blocking: Whether a failure stops the run. A non-blocking check that fails is worth
            seeing and not worth cancelling four GPU hours over.
    """

    name: str
    passed: bool | None
    detail: str = ""
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Every preflight check, and whether the run may proceed.

    Attributes:
        checks: The checks, in the order they were made.
    """

    checks: tuple[PreflightCheck, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence field to a tuple."""
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def failed(self) -> tuple[PreflightCheck, ...]:
        """Blocking checks that did not hold.

        An undetermined check is not a failure. Several of these cannot be answered without a
        GPU, and refusing to run on a machine that cannot answer them would make the preflight
        the thing that prevents training.
        """
        return tuple(
            check for check in self.checks if check.blocking and check.passed is False
        )

    @property
    def undetermined(self) -> tuple[PreflightCheck, ...]:
        """Checks whose answer could not be established."""
        return tuple(check for check in self.checks if check.passed is None)

    @property
    def ok(self) -> bool:
        """Whether the run may proceed."""
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
            "failed": [check.name for check in self.failed],
            "undetermined": [check.name for check in self.undetermined],
        }


def run_preflight(
    config: GenerationExperimentConfig,
    *,
    loaded: Any,
    records: list[dict[str, Any]],
    plan: Any,
    paths: RunPaths,
    dataset_info: Any,
    fingerprint_check: dict[str, Any],
    expect_train_examples: int | None = None,
    expect_fingerprint: str | None = None,
    expect_source_counts: dict[str, int] | None = None,
    source_counts: dict[str, int] | None = None,
) -> PreflightReport:
    """Check everything that must hold before the first real optimiser step.

    Args:
        config: The resolved experiment configuration.
        loaded: The :class:`~qa_gen_runtime.loader.LoadedModel`.
        records: The rendered TRL records.
        plan: The :class:`~qa_gen_runtime.trainer.TrainerPlan`.
        paths: Where this run writes.
        dataset_info: The :class:`~qa_gen_runtime.prepared.PreparedSplitInfo` in use.
        fingerprint_check: Output of :func:`~qa_gen_runtime.prepared.verify_fingerprint`.
        expect_train_examples: Refuse unless the split holds exactly this many examples.
        expect_fingerprint: Refuse unless the dataset carries this fingerprint.
        expect_source_counts: Refuse unless the split's per-source counts match.
        source_counts: The observed per-source counts.

    Returns:
        The :class:`PreflightReport`.
    """
    checks: list[PreflightCheck] = []
    arguments = dict(getattr(plan, "arguments", {}) or {})
    dropped = set(getattr(plan, "dropped_arguments", ()) or ())

    # --- the corpus -------------------------------------------------------
    directory = getattr(dataset_info, "directory", None)
    checks.append(
        PreflightCheck(
            name="dataset_directory_exists",
            passed=bool(directory) and Path(directory).is_dir(),
            detail=f"directory={Path(directory).as_posix() if directory else None}",
        )
    )

    recorded = getattr(dataset_info, "recorded_fingerprint", None) or getattr(
        dataset_info, "fingerprint", None
    )
    if expect_fingerprint is None:
        checks.append(
            PreflightCheck(
                name="dataset_fingerprint_matches_request",
                passed=None,
                detail=(
                    f"no expected fingerprint was supplied; the dataset reports {recorded!r}. "
                    "Pass --dataset-fingerprint to pin the corpus a reported number came from."
                ),
                blocking=False,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="dataset_fingerprint_matches_request",
                passed=recorded == expect_fingerprint,
                detail=f"recorded={recorded!r} expected={expect_fingerprint!r}",
            )
        )

    # The split's own fingerprint cannot equal the whole corpus's, so this is reported and
    # never blocking. It is here because a *changed* split file is worth seeing.
    checks.append(
        PreflightCheck(
            name="split_fingerprint_recomputed",
            passed=bool(fingerprint_check.get("recomputed_for_this_split")),
            detail=json.dumps(fingerprint_check, default=str),
            blocking=False,
        )
    )

    observed_counts = dict(source_counts or {})
    if expect_source_counts:
        checks.append(
            PreflightCheck(
                name="source_counts_match_request",
                passed=observed_counts == dict(expect_source_counts),
                detail=f"observed={observed_counts} expected={dict(expect_source_counts)}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="expected_sources_are_present",
                passed=bool(observed_counts),
                detail=f"observed={observed_counts}",
            )
        )

    if expect_train_examples is None:
        checks.append(
            PreflightCheck(
                name="train_examples_match_request",
                passed=None,
                detail=(
                    f"no expected count was supplied; the split holds {len(records)}. "
                    "Pass --expect-train-examples to make the size an assertion."
                ),
                blocking=False,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="train_examples_match_request",
                passed=len(records) == expect_train_examples,
                detail=f"records={len(records)} expected={expect_train_examples}",
            )
        )

    # --- the supervision signal -------------------------------------------
    suppressed = config.model.suppresses_reasoning
    template_kwargs = chat_template_kwargs(config.model, loaded.tokenizer)
    carried = sum(1 for record in records if record.get(CHAT_TEMPLATE_KWARGS_COLUMN))
    reasoning_ok = (
        suppressed
        and template_kwargs.get("enable_thinking") is False
        and carried == len(records)
    )
    checks.append(
        PreflightCheck(
            name="reasoning_suppression_is_active",
            passed=reasoning_ok,
            detail=(
                f"reasoning_mode={config.model.reasoning_mode!r} "
                f"template_kwargs={template_kwargs} "
                f"records_carrying_the_column={carried}/{len(records)}. Without the column on "
                "every record, Qwen3's template puts an empty <think></think> block inside the "
                "supervised completion and the model learns to emit one before every answer."
            ),
        )
    )

    completion_only = arguments.get("completion_only_loss")
    checks.append(
        PreflightCheck(
            name="completion_only_masking_is_active",
            passed=completion_only is True and "completion_only_loss" not in dropped,
            detail=(
                f"completion_only_loss={completion_only} "
                f"dropped_by_installed_trl={'completion_only_loss' in dropped}. False here "
                "means loss is taken over the prompt as well, so the model is trained to "
                "reproduce the instructions."
            ),
        )
    )

    checks.append(
        PreflightCheck(
            name="max_sequence_length_is_correct",
            passed=arguments.get("max_length") == config.model.max_seq_length,
            detail=(
                f"trainer max_length={arguments.get('max_length')} "
                f"config max_seq_length={config.model.max_seq_length}"
            ),
        )
    )

    # --- the model --------------------------------------------------------
    attachment = describe_lora_attachment(loaded.model)
    attached_once = (
        attachment["is_peft_model"] is True
        and attachment["adapter_count"] == 1
        and attachment["base_model_is_also_peft"] is not True
        and attachment["parameter_names_show_double_wrap"] is not True
        and bool(loaded.trainable_parameters)
    )
    checks.append(
        PreflightCheck(
            name="lora_is_attached_exactly_once",
            passed=attached_once,
            detail=(
                f"{json.dumps(attachment, default=str)} "
                f"trainable={loaded.trainable_parameters:,} of {loaded.total_parameters:,}"
            ),
        )
    )

    quantization = describe_loaded_quantization(loaded.model)
    if not quantization["quantization_config_present"]:
        checks.append(
            PreflightCheck(
                name="quantization_is_4bit_nf4",
                passed=False if config.model.is_4bit else None,
                detail=(
                    "the loaded model carries no quantization_config, so the base weights are "
                    f"not quantized. The configuration asked for "
                    f"{config.model.quantization_settings()}."
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="quantization_is_4bit_nf4",
                passed=(
                    quantization["load_in_4bit"] is True
                    and quantization["quant_type"] == "nf4"
                    and quantization["double_quant"] is config.model.double_quantization
                ),
                detail=json.dumps(quantization, default=str),
            )
        )

    precision = loaded.precision
    bf16_ok = getattr(precision, "bf16", None) is True and arguments.get("bf16") is True
    checks.append(
        PreflightCheck(
            name="bf16_is_active",
            passed=bf16_ok,
            detail=(
                f"precision={getattr(precision, 'name', None)!r} "
                f"plan_bf16={arguments.get('bf16')} plan_fp16={arguments.get('fp16')} "
                f"compute_dtype={quantization.get('compute_dtype')}"
            ),
        )
    )

    # --- the output -------------------------------------------------------
    # "No base model output path is configured" means: the only thing this run will write as
    # weights is the adapter directory, and nothing already in the run directory looks like a
    # base checkpoint. The save call itself is PEFT's, which writes adapter tensors only.
    findings, output_details = audit_benchmark_output(
        paths.adapter, paths.root, expect_adapter=False
    )
    checks.append(
        PreflightCheck(
            name="no_base_model_output_path_is_configured",
            passed=(
                not output_details["base_weight_files"]
                and paths.adapter.parent == paths.root
                and arguments.get("output_dir") == str(paths.root)
            ),
            detail=(
                f"adapter_target={paths.adapter.as_posix()} "
                f"trainer_output_dir={arguments.get('output_dir')} "
                f"preexisting_base_weight_files={output_details['base_weight_files']}"
            ),
        )
    )
    checks.append(
        PreflightCheck(
            name="save_strategy_will_not_write_base_weights",
            passed=arguments.get("save_strategy") in ("no", "steps", "epoch"),
            detail=(
                f"save_strategy={arguments.get('save_strategy')!r} "
                f"save_total_limit={arguments.get('save_total_limit')}. TRL checkpoints a PEFT "
                "model as adapter tensors, so intermediate checkpoints stay small."
            ),
            blocking=False,
        )
    )
    _ = findings

    # --- the schedule -----------------------------------------------------
    checks.append(
        PreflightCheck(
            name="effective_batch_matches_configuration",
            passed=(
                arguments.get("per_device_train_batch_size")
                * arguments.get("gradient_accumulation_steps")
                == config.training.effective_batch_size
            ),
            detail=(
                f"{arguments.get('per_device_train_batch_size')} x "
                f"{arguments.get('gradient_accumulation_steps')} = "
                f"{config.training.effective_batch_size}"
            ),
        )
    )

    return PreflightReport(checks=tuple(checks))


# ---------------------------------------------------------------------------
# Measurements and the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingMeasurements:
    """What the training call cost and produced.

    Attributes:
        optimizer_steps: Steps the trainer reports completing.
        planned_steps: Steps the plan expected.
        wall_clock_seconds: Elapsed training time.
        steps_per_second: Completed steps divided by elapsed time.
        seconds_per_step: Its reciprocal, which is the figure used for projections.
        final_loss: Mean training loss the trainer reported.
        losses: Per-step log history, when the trainer recorded one.
        train_metrics: The trainer's own metrics mapping.
        memory_before: Device memory before training.
        memory_after: Device memory after training, including the peaks.
        peak_stats_reset: Whether the CUDA peak counters were cleared first. ``False`` means
            every peak figure includes the model load and is not comparable to a baseline.
        optimizer_class: The optimiser class the trainer actually built.
        gradient_checkpointing_active: Measured from the model, not requested.
        use_cache: The model's KV-cache setting, which must be off under checkpointing.
        notes: Anything worth recording.
    """

    optimizer_steps: int | None = None
    planned_steps: int | None = None
    wall_clock_seconds: float | None = None
    steps_per_second: float | None = None
    seconds_per_step: float | None = None
    final_loss: float | None = None
    losses: tuple[dict[str, Any], ...] = ()
    train_metrics: dict[str, Any] = field(default_factory=dict)
    memory_before: dict[str, Any] = field(default_factory=dict)
    memory_after: dict[str, Any] = field(default_factory=dict)
    peak_stats_reset: bool = False
    optimizer_class: str | None = None
    gradient_checkpointing_active: bool | None = None
    use_cache: Any = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "losses", tuple(self.losses))
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "optimizer_steps": self.optimizer_steps,
            "planned_steps": self.planned_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "steps_per_second": self.steps_per_second,
            "seconds_per_step": self.seconds_per_step,
            "final_loss": self.final_loss,
            "losses": [dict(entry) for entry in self.losses],
            "train_metrics": dict(self.train_metrics),
            "memory_before": dict(self.memory_before),
            "memory_after": dict(self.memory_after),
            "peak_stats_reset": self.peak_stats_reset,
            "optimizer_class": self.optimizer_class,
            "gradient_checkpointing_active": self.gradient_checkpointing_active,
            "use_cache": self.use_cache,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ProductionTrainingReport:
    """The machine-readable record of one production training run.

    Attributes:
        run_id: The run directory name.
        success: Whether ``trainer.train()`` completed and the adapter was saved. Never
            ``True`` when the training call raised.
        status: ``"completed"``, ``"failed"`` or ``"preflight_failed"``.
        config_path: The configuration file that was read.
        experiment: The experiment name.
        config_hash: Hash of the resolved configuration.
        model_id: The base model.
        model_revision: Its pinned revision.
        dataset_directory: Where the corpus was read from.
        dataset_fingerprint: The corpus fingerprint.
        dataset_split: Which split was trained on.
        train_examples: Examples in the training split.
        validation_examples: Examples in the validation split, when counted.
        test_examples: Examples in the test split, when counted.
        source_counts: Per-source example counts in the training split.
        max_seq_length: The truncation limit.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        effective_batch_size: The product of the two above.
        total_optimizer_steps: Steps the plan expected.
        completed_optimizer_steps: Steps the trainer reports completing.
        warmup_steps: Warmup steps, converted from the ratio.
        resumed_from: The checkpoint resumed from, or ``None``.
        resumable: Whether this run wrote checkpoints it could later be resumed from.
            Recorded at the time it started, because it is the difference between an
            interruption costing an interval and costing everything.
        save_strategy: When checkpoints were written.
        save_steps: The checkpoint interval, when one was set.
        adapter_path: Where the adapter was saved.
        adapter_bytes: Size of the adapter checkpoint on disk.
        trainable_parameters: Parameters that trained.
        total_parameters: Parameters in the wrapped model.
        preflight: The preflight report.
        measurements: What training cost.
        diagnostics: The runtime diagnostic report.
        trainer_plan: The translated trainer arguments.
        output_audit: What the run left on disk.
        error: Exception type, message and traceback on failure.
        artifacts: Files written.
        notes: Anything worth recording.
    """

    run_id: str
    success: bool
    status: str
    config_path: str = ""
    experiment: str = ""
    config_hash: str = ""
    model_id: str = ""
    model_revision: str = ""
    dataset_directory: str = ""
    dataset_fingerprint: str | None = None
    dataset_split: str = SplitName.TRAIN.value
    train_examples: int = 0
    validation_examples: int | None = None
    test_examples: int | None = None
    source_counts: dict[str, int] = field(default_factory=dict)
    max_seq_length: int = 0
    batch_size: int = 0
    gradient_accumulation_steps: int = 0
    effective_batch_size: int = 0
    total_optimizer_steps: int | None = None
    completed_optimizer_steps: int | None = None
    warmup_steps: int | None = None
    resumed_from: str | None = None
    resumable: bool = False
    save_strategy: str = ""
    save_steps: int | None = None
    adapter_path: str | None = None
    adapter_bytes: int | None = None
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    preflight: PreflightReport = field(default_factory=PreflightReport)
    measurements: TrainingMeasurements | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trainer_plan: dict[str, Any] = field(default_factory=dict)
    output_audit: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence field to a tuple."""
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "run_id": self.run_id,
            "success": self.success,
            "status": self.status,
            "config_path": self.config_path,
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset": {
                "directory": self.dataset_directory,
                "fingerprint": self.dataset_fingerprint,
                "split": self.dataset_split,
                "train_examples": self.train_examples,
                "validation_examples": self.validation_examples,
                "test_examples": self.test_examples,
                "source_counts": dict(sorted(self.source_counts.items())),
            },
            "schedule": {
                "max_seq_length": self.max_seq_length,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
                "total_optimizer_steps": self.total_optimizer_steps,
                "completed_optimizer_steps": self.completed_optimizer_steps,
                "warmup_steps": self.warmup_steps,
                "resumed_from": self.resumed_from,
                "resumable": self.resumable,
                "save_strategy": self.save_strategy,
                "save_steps": self.save_steps,
            },
            "adapter": {
                "path": self.adapter_path,
                "bytes": self.adapter_bytes,
                "trainable_parameters": self.trainable_parameters,
                "total_parameters": self.total_parameters,
            },
            "preflight": self.preflight.as_dict(),
            "measurements": self.measurements.as_dict() if self.measurements else None,
            "diagnostics": dict(self.diagnostics),
            "trainer_plan": dict(self.trainer_plan),
            "output_audit": dict(self.output_audit),
            "error": dict(self.error) if self.error else None,
            "artifacts": dict(self.artifacts),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The one training call
# ---------------------------------------------------------------------------


def _call_trainer_train(
    trainer: Any, *, execute: bool, resume_from_checkpoint: str | None = None
) -> Any:
    """Call ``trainer.train()``. The only place in this module that does.

    Matches :func:`qa_gen_runtime.smoke._call_trainer_train` and
    :func:`qa_gen_runtime.benchmark._call_trainer_train`: a one-line body behind a mandatory
    keyword guard, so "where does this module train?" has exactly one answer and a test can
    assert that no validation path reaches it.

    Args:
        trainer: A constructed ``SFTTrainer``.
        execute: Must be ``True``. There is no default, so reaching this function without
            having decided to train is a programming error rather than an accident.
        resume_from_checkpoint: A checkpoint directory, or ``None`` to start fresh.

    Returns:
        The trainer's ``TrainOutput``.

    Raises:
        ProductionTrainingError: If ``execute`` is ``False``.
    """
    if not execute:
        raise ProductionTrainingError(
            "refusing to call trainer.train(): production training runs only under "
            "--execute-training. Use --plan to translate and inspect the configuration "
            "without stepping the optimiser."
        )
    if resume_from_checkpoint:
        return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer.train()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def execute_production_training(
    config_path: str,
    *,
    dataset_dir: str | None = None,
    dataset_fingerprint: str | None = None,
    split: str = SplitName.TRAIN.value,
    eval_split: str | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: str | None = None,
    expect_train_examples: int | None = None,
    expect_source_counts: dict[str, int] | None = None,
    execute: bool = False,
) -> ProductionTrainingReport:
    """Train a QLoRA adapter over a prepared corpus, and record what happened.

    The sequence, in order: read the corpus, resolve the run directory, load the model with
    adapters, render the records, build the trainer, run the preflight, step the optimiser,
    save the adapter, audit the output, write the report.

    Args:
        config_path: Path to the experiment configuration.
        dataset_dir: An explicit prepared dataset directory. Wins over the fingerprint.
        dataset_fingerprint: Select the prepared dataset by corpus fingerprint. Use this for
            anything whose numbers get reported; "the newest" is not reproducible.
        split: Which prepared split to train on.
        eval_split: An optional split to evaluate on. Read and counted only when the
            configuration actually evaluates.
        output_dir: Override the run root. Defaults to ``<artifacts>/qgen-runs``.
        run_id: Override the generated run directory name.
        resume_from_checkpoint: Resume from this checkpoint directory.
        expect_train_examples: Refuse unless the split holds exactly this many examples.
        expect_source_counts: Refuse unless the split's per-source counts match.
        execute: Must be ``True`` to train. Defaults to ``False`` so that calling this
            function without having decided to train cannot start a four-hour run.

    Returns:
        The :class:`ProductionTrainingReport`.

    Raises:
        ProductionTrainingError: If ``execute`` is ``False``, the preflight fails, resuming is
            impossible, or training raised. The report is written to disk before the raise in
            every case where a run directory exists.
        qa_gen_runtime.config_io.ConfigIOError: If the configuration cannot be read.
        qa_gen.config.GenerationConfigError: If it is invalid.
        qa_gen_runtime.prepared.PreparedDatasetError: If the corpus cannot be read.
        qa_gen_runtime.loader.ModelLoadError: If the model cannot be loaded or adapted.
        qa_gen_runtime.trainer.TrainerBuildError: If the trainer cannot be built.
        qa_gen_runtime.deps.RuntimeDependencyError: If PEFT, TRL or ``datasets`` is missing.
        qa_gen_runtime.outputs.RunOutputError: If the run directory cannot be created.
    """
    if not execute:
        raise ProductionTrainingError(
            "execute=False: refusing to start production training.\n"
            "This function trains over the whole prepared corpus and costs hours of GPU time, "
            "so it does nothing unless told to explicitly. Pass execute=True, or use the CLI's "
            "--execute-training flag."
        )

    config = load_experiment_config(config_path)

    if resume_from_checkpoint and not config.training.is_resumable:
        raise ProductionTrainingError(
            f"cannot resume: training.save_strategy is "
            f"{config.training.save_strategy!r}, so this configuration writes no "
            "checkpoints and there is nothing to resume from.\n"
            "Resuming needs save_strategy to be 'steps' or 'epoch' on the run being resumed. "
            "Starting from scratch here would look like a resume and silently repeat work, so "
            "it is refused instead."
        )

    # --- the corpus, before anything expensive ----------------------------
    dataset_root = _prepared_root(dataset_dir)
    info = resolve_prepared_directory(
        dataset_root, directory=dataset_dir, fingerprint=dataset_fingerprint
    )
    examples = read_prepared_split(info.directory, split)
    recorded_fingerprint = info.recorded_fingerprint or info.fingerprint
    fingerprint_check = verify_fingerprint(examples, recorded_fingerprint)

    source_counts: dict[str, int] = {}
    for item in examples:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    counts = {split: len(examples)}
    for name in SplitName:
        if name.value == split:
            continue
        path = info.directory / f"{name.value}.jsonl"
        if path.is_file():
            try:
                counts[name.value] = len(read_prepared_split(info.directory, name))
            except Exception:  # noqa: BLE001 - a sibling split is informational only
                counts[name.value] = None

    eval_examples = None
    if eval_split and config.training.evaluation_strategy != "no":
        eval_examples = read_prepared_split(info.directory, eval_split)

    # --- where this run writes --------------------------------------------
    run_root = resolve_run_root(output_dir)
    paths = create_run_directory(
        config,
        output_dir=output_dir,
        run_id=run_id,
        allow_existing=bool(resume_from_checkpoint),
    )
    logger.info("Run directory: %s", paths.root)

    notes: list[str] = [
        f"corpus: {len(examples)} example(s) from {info.directory.as_posix()} split {split!r}",
        f"per-source counts: {dict(sorted(source_counts.items()))}",
        "the adapter is saved with PEFT's save_pretrained, which writes adapter tensors only; "
        "the base checkpoint stays in the Hugging Face cache",
    ]

    # The model is loaded before the records are rendered, for the reason
    # qa_gen_runtime.smoke documents: the tokenizer decides whether the chat_template_kwargs
    # column is emitted, and on Qwen3 that column is what keeps the empty <think></think>
    # block out of the supervised completion.
    loaded = load_trainable_model(config)
    records = build_training_records(examples, config, tokenizer=loaded.tokenizer)
    train_dataset = build_hf_dataset(records)

    eval_dataset = None
    if eval_examples:
        eval_dataset = build_hf_dataset(
            build_training_records(eval_examples, config, tokenizer=loaded.tokenizer)
        )

    trainer, plan = build_trainer(
        config,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=paths.root,
        train_examples=len(records),
        precision=loaded.precision,
    )

    preflight = run_preflight(
        config,
        loaded=loaded,
        records=records,
        plan=plan,
        paths=paths,
        dataset_info=info,
        fingerprint_check=fingerprint_check,
        expect_train_examples=expect_train_examples,
        expect_fingerprint=dataset_fingerprint,
        expect_source_counts=expect_source_counts,
        source_counts=source_counts,
    )

    base_report = _base_report(
        config=config,
        config_path=config_path,
        paths=paths,
        info=info,
        split=split,
        recorded_fingerprint=recorded_fingerprint,
        counts=counts,
        source_counts=source_counts,
        loaded=loaded,
        plan=plan,
        preflight=preflight,
        resumed_from=resume_from_checkpoint,
        notes=notes,
    )

    if not preflight.ok:
        failed = ", ".join(check.name for check in preflight.failed)
        report = _replace_report(
            base_report,
            success=False,
            status="preflight_failed",
            notes=(*notes, f"preflight refused the run: {failed}"),
        )
        _write_report(paths, report)
        raise ProductionTrainingError(
            f"preflight failed, so no optimiser step was taken: {failed}\n"
            + "\n".join(
                f"  - {check.name}: {check.detail}" for check in preflight.failed
            )
            + f"\nThe report was written to {(paths.root / TRAINING_REPORT_FILENAME)}."
        )

    # --- train ------------------------------------------------------------
    reset = _reset_peak_memory()
    before = memory_report()
    if not reset:
        notes.append(
            "CUDA is not available, so every memory figure is null and this ran on the CPU."
        )

    started = time.perf_counter()
    try:
        output = _call_trainer_train(
            trainer, execute=True, resume_from_checkpoint=resume_from_checkpoint
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        measurements = _measure(
            trainer,
            output=None,
            loaded=loaded,
            config=config,
            plan=plan,
            elapsed=elapsed,
            before=before,
            reset=reset,
            notes=[
                "trainer.train() raised; the figures below describe the partial run.",
            ],
        )
        report = _replace_report(
            base_report,
            success=False,
            status="failed",
            measurements=measurements,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "completed_optimizer_steps": measurements.optimizer_steps,
            },
            completed_optimizer_steps=measurements.optimizer_steps,
            notes=(
                *notes,
                "no adapter was saved: the training call did not complete, and saving a "
                "partially trained adapter as though it were finished would be worse than "
                "having none.",
                "the trainer's own checkpoints, if the configuration wrote any, are still in "
                "the run directory.",
            ),
        )
        _write_report(paths, report)
        raise ProductionTrainingError(
            f"training failed after {measurements.optimizer_steps} optimiser step(s): "
            f"{type(exc).__name__}: {exc}\n"
            f"The report was written to {(paths.root / TRAINING_REPORT_FILENAME)}."
        ) from exc

    elapsed = time.perf_counter() - started
    measurements = _measure(
        trainer,
        output=output,
        loaded=loaded,
        config=config,
        plan=plan,
        elapsed=elapsed,
        before=before,
        reset=reset,
        notes=[],
    )

    # --- save, audit, record ---------------------------------------------
    loaded.model.save_pretrained(str(paths.adapter))
    findings, output_details = audit_benchmark_output(
        paths.adapter, paths.root, expect_adapter=True
    )
    if findings:
        notes.extend(f"output finding: {finding.code}: {finding.message}" for finding in findings)

    diagnostics = collect_diagnostics(
        config,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        precision=loaded.precision,
        trainer_plan=plan.as_dict(),
        extra_notes=(
            f"production training over {len(records)} prepared example(s)",
            f"completed {measurements.optimizer_steps} of {plan.total_steps} optimiser step(s)",
        ),
    )

    artifacts = _write_run_documents(
        config,
        paths,
        info=info,
        split=split,
        recorded_fingerprint=recorded_fingerprint,
        counts=counts,
        source_counts=source_counts,
        diagnostics=diagnostics,
        measurements=measurements,
        preflight=preflight,
    )

    report = _replace_report(
        base_report,
        success=True,
        status="completed",
        measurements=measurements,
        adapter_bytes=output_details.get("adapter_bytes"),
        completed_optimizer_steps=measurements.optimizer_steps,
        diagnostics=diagnostics.as_dict(),
        output_audit=output_details,
        artifacts=artifacts,
        notes=tuple(notes),
    )
    path = _write_report(paths, report)
    report = _replace_report(report, artifacts={**artifacts, "training": path.as_posix()})
    _write_report(paths, report)
    _ = run_root
    return report


def _prepared_root(dataset_dir: str | None) -> Path:
    """Return the directory prepared datasets are discovered under.

    Args:
        dataset_dir: An explicit dataset directory, in which case discovery is irrelevant and
            its parent is returned so the message in a failure names something real.

    Returns:
        The root path.
    """
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().parent

    from qa_gen_runtime.prepare import DATASET_SUBDIR
    from qa_ml.paths import get_paths

    return get_paths().artifacts / DATASET_SUBDIR


def _reset_peak_memory() -> bool:
    """Clear the CUDA peak-memory counters.

    Returns:
        ``True`` when they were reset, ``False`` when there is no CUDA device. Reported rather
        than assumed: an unreset peak folds the model load into the training measurement.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def _measure(
    trainer: Any,
    *,
    output: Any,
    loaded: Any,
    config: GenerationExperimentConfig,
    plan: Any,
    elapsed: float,
    before: dict[str, Any],
    reset: bool,
    notes: list[str],
) -> TrainingMeasurements:
    """Read everything measurable off a trainer, whether or not training succeeded.

    Args:
        trainer: The trainer, after the call returned or raised.
        output: The ``TrainOutput``, or ``None`` when training raised.
        loaded: The :class:`~qa_gen_runtime.loader.LoadedModel`.
        config: The experiment configuration.
        plan: The :class:`~qa_gen_runtime.trainer.TrainerPlan`.
        elapsed: Wall-clock seconds spent in the training call.
        before: Memory report taken before training.
        reset: Whether the peak counters were cleared.
        notes: Notes to carry into the measurements.

    Returns:
        The :class:`TrainingMeasurements`.
    """
    state = getattr(trainer, "state", None)
    steps = getattr(state, "global_step", None)
    if steps is None and output is not None:
        steps = getattr(output, "global_step", None)
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

    collected = list(notes)
    if not losses:
        collected.append(
            "no per-step loss was logged, so the loss history is empty; the mean loss in "
            "train_metrics is still reported when the trainer produced one."
        )

    optimizer = getattr(trainer, "optimizer", None)
    optimizer_class = (
        f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        if optimizer is not None
        else None
    )

    return TrainingMeasurements(
        optimizer_steps=steps,
        planned_steps=getattr(plan, "total_steps", None),
        wall_clock_seconds=round(elapsed, 3),
        steps_per_second=(round(steps / elapsed, 4) if steps and elapsed else None),
        seconds_per_step=(round(elapsed / steps, 3) if steps else None),
        final_loss=metrics.get("train_loss", getattr(output, "training_loss", None)),
        losses=losses,
        train_metrics=metrics,
        memory_before=before,
        memory_after=memory_report(),
        peak_stats_reset=reset,
        optimizer_class=optimizer_class,
        gradient_checkpointing_active=getattr(
            loaded.model, "is_gradient_checkpointing", None
        ),
        use_cache=getattr(getattr(loaded.model, "config", None), "use_cache", None),
        notes=tuple(collected),
    )


def _base_report(
    *,
    config: GenerationExperimentConfig,
    config_path: str,
    paths: RunPaths,
    info: Any,
    split: str,
    recorded_fingerprint: str | None,
    counts: dict[str, Any],
    source_counts: dict[str, int],
    loaded: Any,
    plan: Any,
    preflight: PreflightReport,
    resumed_from: str | None,
    notes: list[str],
) -> ProductionTrainingReport:
    """Assemble the fields every outcome shares, successful or not."""
    return ProductionTrainingReport(
        run_id=paths.root.name,
        success=False,
        status="unknown",
        config_path=config_path,
        experiment=config.name,
        config_hash=config.config_hash(),
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        dataset_directory=info.directory.as_posix(),
        dataset_fingerprint=recorded_fingerprint,
        dataset_split=split,
        train_examples=counts.get(split) or 0,
        validation_examples=counts.get(SplitName.VALIDATION.value),
        test_examples=counts.get(SplitName.TEST.value),
        source_counts=dict(source_counts),
        max_seq_length=config.model.max_seq_length,
        batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        effective_batch_size=config.training.effective_batch_size,
        total_optimizer_steps=getattr(plan, "total_steps", None),
        warmup_steps=getattr(plan, "warmup_steps", None),
        resumed_from=resumed_from,
        resumable=config.training.is_resumable,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        adapter_path=paths.adapter.as_posix(),
        trainable_parameters=loaded.trainable_parameters,
        total_parameters=loaded.total_parameters,
        preflight=preflight,
        trainer_plan=plan.as_dict(),
        notes=tuple(notes),
    )


def _replace_report(
    report: ProductionTrainingReport, **changes: Any
) -> ProductionTrainingReport:
    """Return a copy of ``report`` with ``changes`` applied.

    ``dataclasses.replace`` would do this, but the report has ``slots=True`` and a
    ``__post_init__`` that re-coerces tuples, and going through the constructor keeps that
    coercion in one place.
    """
    import dataclasses

    return dataclasses.replace(report, **changes)


def _write_report(paths: RunPaths, report: ProductionTrainingReport) -> Path:
    """Write the machine-readable training report into the run directory."""
    return _write_json(paths.root / TRAINING_REPORT_FILENAME, report.as_dict())


def _write_run_documents(
    config: GenerationExperimentConfig,
    paths: RunPaths,
    *,
    info: Any,
    split: str,
    recorded_fingerprint: str | None,
    counts: dict[str, Any],
    source_counts: dict[str, int],
    diagnostics: Any,
    measurements: TrainingMeasurements,
    preflight: PreflightReport,
) -> dict[str, str]:
    """Write the Phase 17A run documents, so the adapter is loadable and traceable.

    The tokenizer identity is recorded rather than copied. The adapter is loaded against the
    base model, whose id and revision are both in the resolved configuration, so a copied
    tokenizer would be a hundred megabytes restating something already written down.

    Args:
        config: The resolved configuration.
        paths: The run's paths.
        info: The prepared dataset in use.
        split: Which split was trained on.
        recorded_fingerprint: The corpus fingerprint.
        counts: Per-split example counts.
        source_counts: Per-source counts in the training split.
        diagnostics: The :class:`~qa_gen_runtime.diagnostics.RuntimeDiagnostics`.
        measurements: What training cost.
        preflight: The preflight report.

    Returns:
        A mapping of document name to the path written.
    """
    corpus = {
        "directory": info.directory.as_posix(),
        "fingerprint": recorded_fingerprint,
        "split": split,
        "counts": dict(counts),
        "source_counts": dict(sorted(source_counts.items())),
        "prepared_dataset": info.as_dict(),
        "tokenizer_id": config.model.effective_tokenizer_id,
        "tokenizer_revision": config.model.revision,
    }

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
            "Phase 18 production training over the prepared corpus",
            f"completed {measurements.optimizer_steps} optimiser step(s)",
            "adapter-only artifact: the base model is not copied into the run directory",
        ],
    )
    attach_to_metadata(metadata, diagnostics)
    metadata.training["production"] = {
        "measurements": measurements.as_dict(),
        "preflight": preflight.as_dict(),
        "baseline_deviations": list(config.baseline_deviations(VERIFIED_QWEN3_4B_L4)),
        "record_format": resolve_record_format(config).value,
    }

    write_resolved_config(config, paths.config)
    _write_json(paths.diagnostics, diagnostics.as_dict())
    _write_json(paths.dataset, corpus)
    _write_json(paths.record, metadata.as_dict())

    return {
        "config": paths.config.as_posix(),
        "diagnostics": paths.diagnostics.as_posix(),
        "dataset": paths.dataset.as_posix(),
        "record": paths.record.as_posix(),
        "adapter": paths.adapter.as_posix(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a mapping as an indented JSON document, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
