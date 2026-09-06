"""What ran, on what, with which settings.

Why a diagnostic report is worth its own module
-----------------------------------------------
A loss curve without the configuration beside it is an anecdote. The measured feasibility run
is only useful because it came with the GPU, the VRAM, the quantization, the adapter shape and
the parameter counts attached -- and a real training run needs the same treatment or its
numbers cannot be compared to anything, including to itself a month later.

Everything here is derived from the objects that will actually be used, never from the
configuration alone where a measurement is available. Parameter counts come from the wrapped
model; VRAM comes from the device; the trainer arguments come from the translated plan rather
than from the fields that fed it. Reporting intent as though it were fact is how a run comes to
be described by settings it did not use.

Safe without a GPU
------------------
Every CUDA field degrades to ``None`` rather than being omitted, so the shape of the report does
not depend on the hardware and a laptop dry run diffs cleanly against a Studio run. That is also
what lets the CLI print a full report during configuration validation.

The output drops straight into :class:`qa_gen.metadata.TrainingRunMetadata`.
:meth:`RuntimeDiagnostics.as_dict` is JSON-serializable throughout, and
:func:`attach_to_metadata` puts it where a reader will look for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from qa_gen.config import VERIFIED_QWEN3_4B_L4, GenerationExperimentConfig
from qa_gen.metadata import TrainingRunMetadata
from qa_gen_runtime.chat import describe_chat_handling
from qa_gen_runtime.deps import dependency_report
from qa_gen_runtime.precision import PrecisionPlan, describe_device
from qa_gen_runtime.quantization import describe_quantization

__all__ = [
    "RuntimeDiagnostics",
    "attach_to_metadata",
    "collect_diagnostics",
    "memory_report",
]


def memory_report() -> dict[str, Any]:
    """Return current and peak device memory, in GiB.

    Both allocated and reserved are reported. They differ, sometimes considerably: allocated is
    what tensors hold, reserved is what the caching allocator has taken from the driver, and it
    is the reserved figure that determines whether another process fits on the card. The
    measured 3.943 GiB peak is a reserved-style number, so reporting only allocated would make
    a run look cheaper than the baseline it is being compared against.

    Returns:
        A JSON-serializable mapping. Every field is ``None`` on a machine with no CUDA device.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gib": None,
            "reserved_gib": None,
            "max_allocated_gib": None,
            "max_reserved_gib": None,
            "total_vram_gib": None,
        }

    def gib(value: int) -> float:
        return round(value / 1024**3, 3)

    try:
        properties = torch.cuda.get_device_properties(0)
        total = gib(properties.total_memory)
    except (AssertionError, RuntimeError):  # pragma: no cover - driver-specific
        total = None

    return {
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "max_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "max_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "total_vram_gib": total,
    }


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    """A complete description of one configured run.

    Attributes:
        model_id: The model that will be or was loaded.
        model_revision: The pinned revision.
        device: Output of :func:`~qa_gen_runtime.precision.describe_device`.
        memory: Output of :func:`memory_report`.
        precision: The resolved precision decision, or ``None`` before resolution.
        quantization: The translated quantization settings.
        lora: The adapter settings.
        chat: How prompts will be assembled, including reasoning mode.
        trainable_parameters: Measured count, or ``None`` before the model is loaded.
        total_parameters: Measured count, or ``None``.
        sequence_length: Maximum tokenized length.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        effective_batch_size: The product of the two above.
        gradient_checkpointing: Whether activation checkpointing is enabled.
        optimizer: Optimiser name passed to the trainer.
        dependencies: Availability and versions of the whole stack.
        baseline_deviations: How this configuration departs from
            :data:`qa_gen.config.VERIFIED_QWEN3_4B_L4`.
        trainer_plan: The translated trainer arguments, when built.
        notes: Anything else worth recording.
    """

    model_id: str
    model_revision: str
    device: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    precision: dict[str, Any] | None = None
    quantization: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] = field(default_factory=dict)
    chat: dict[str, Any] = field(default_factory=dict)
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    sequence_length: int = 0
    batch_size: int = 0
    gradient_accumulation_steps: int = 0
    effective_batch_size: int = 0
    gradient_checkpointing: bool = False
    optimizer: str = ""
    dependencies: dict[str, dict[str, object]] = field(default_factory=dict)
    baseline_deviations: tuple[str, ...] = ()
    trainer_plan: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "baseline_deviations", tuple(self.baseline_deviations))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def trainable_fraction(self) -> float | None:
        """Share of parameters that train, or ``None`` before the model is loaded."""
        if not self.trainable_parameters or not self.total_parameters:
            return None
        return round(self.trainable_parameters / self.total_parameters, 6)

    @property
    def is_verified_configuration(self) -> bool:
        """Whether this configuration matches the measured baseline exactly."""
        return not self.baseline_deviations

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": dict(self.device),
            "memory": dict(self.memory),
            "precision": dict(self.precision) if self.precision else None,
            "quantization": dict(self.quantization),
            "lora": dict(self.lora),
            "chat": dict(self.chat),
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimizer": self.optimizer,
            "dependencies": {
                name: dict(detail) for name, detail in sorted(self.dependencies.items())
            },
            "baseline_deviations": list(self.baseline_deviations),
            "is_verified_configuration": self.is_verified_configuration,
            "trainer_plan": dict(self.trainer_plan) if self.trainer_plan else None,
            "notes": list(self.notes),
        }


def collect_diagnostics(
    config: GenerationExperimentConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
    precision: PrecisionPlan | None = None,
    trainer_plan: dict[str, Any] | None = None,
    extra_notes: tuple[str, ...] = (),
) -> RuntimeDiagnostics:
    """Assemble the diagnostic report for a configured run.

    Callable at three points, and useful at all three: before anything is loaded, for a dry
    run; after the model is prepared, to record the real parameter counts; and after training,
    to record peak memory.

    Args:
        config: The experiment configuration.
        model: The prepared model. Parameter counts are measured from it when supplied and
            reported as ``None`` otherwise -- never estimated from the configuration, because a
            plausible wrong number is worse than an honest absent one.
        tokenizer: The loaded tokenizer, for the chat and reasoning report.
        precision: The resolved precision plan.
        trainer_plan: Output of :meth:`~qa_gen_runtime.trainer.TrainerPlan.as_dict`.
        extra_notes: Additional notes to record.

    Returns:
        The :class:`RuntimeDiagnostics`.
    """
    trainable: int | None = None
    total: int | None = None
    if model is not None:
        from qa_gen_runtime.loader import count_parameters

        trainable, total = count_parameters(model)

    training = config.training
    notes = list(extra_notes)
    notes.append(
        "batch size and gradient checkpointing are runtime configuration; the feasibility "
        "measurement used batch 1 and did not isolate checkpointing"
    )
    notes.append(
        f"optimizer {training.optimizer!r} has not been benchmarked by this project"
    )

    return RuntimeDiagnostics(
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        device=describe_device(),
        memory=memory_report(),
        precision=precision.as_dict() if precision else None,
        quantization=describe_quantization(config.model),
        lora={
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "scaling": config.lora.scaling,
            "dropout": config.lora.dropout,
            "bias": config.lora.bias,
            "task_type": config.lora.task_type,
            "target_modules": list(config.lora.target_modules),
            "use_rslora": config.lora.use_rslora,
            "modules_to_save": list(config.lora.modules_to_save),
        },
        chat=describe_chat_handling(config.model, tokenizer),
        trainable_parameters=trainable,
        total_parameters=total,
        sequence_length=config.model.max_seq_length,
        batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        effective_batch_size=training.effective_batch_size,
        gradient_checkpointing=training.gradient_checkpointing,
        optimizer=training.optimizer,
        dependencies=dependency_report(),
        baseline_deviations=config.baseline_deviations(VERIFIED_QWEN3_4B_L4),
        trainer_plan=trainer_plan,
        notes=tuple(notes),
    )


def attach_to_metadata(
    metadata: TrainingRunMetadata, diagnostics: RuntimeDiagnostics
) -> TrainingRunMetadata:
    """Record a diagnostic report on a Phase 17A run record, in place.

    The measured parameter counts are also promoted to the record's own fields, so
    :attr:`qa_gen.metadata.TrainingRunMetadata.trainable_fraction` reports a measured value
    rather than staying ``None`` beside a diagnostics block that has it.

    Args:
        metadata: The run record to fill in.
        diagnostics: The report to attach.

    Returns:
        The same record, mutated. Returned for convenience in a call chain; it is not a copy.
    """
    metadata.training["diagnostics"] = diagnostics.as_dict()
    metadata.training["measured_baseline"] = VERIFIED_QWEN3_4B_L4.as_dict()
    metadata.training["baseline_deviations"] = list(diagnostics.baseline_deviations)
    if diagnostics.trainable_parameters is not None:
        metadata.trainable_parameters = diagnostics.trainable_parameters
    if diagnostics.total_parameters is not None:
        metadata.total_parameters = diagnostics.total_parameters
    if not metadata.base_model:
        metadata.base_model = diagnostics.model_id
    if not metadata.base_model_revision:
        metadata.base_model_revision = diagnostics.model_revision
    return metadata
