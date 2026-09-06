"""Building a TRL ``SFTConfig`` and ``SFTTrainer`` from the Phase 17A configuration.

Verified against TRL v0.29.1, not assumed
-----------------------------------------
The signatures below were read from the v0.29.1 sources rather than recalled. Three of them
have changed inside the last year and getting any wrong is a ``TypeError`` after a model
download:

- ``SFTConfig.max_length``, **not** ``max_seq_length``. Renamed around 0.18.
- ``SFTTrainer(processing_class=...)``, **not** ``tokenizer=``.
- ``SFTConfig.completion_only_loss``, which TRL supports **only** for prompt-completion
  datasets -- which is why :mod:`qa_gen_runtime.dataset` emits that shape.

``SFTConfig`` subclasses ``BaseConfig`` which subclasses ``transformers.TrainingArguments``, so
the optimiser, precision and checkpointing arguments come from there. Two transformers 5.16.1
names bite here, both confirmed by introspecting the installed package:

- ``eval_strategy``, not ``evaluation_strategy``.
- **``warmup_ratio`` no longer exists.** Only ``warmup_steps`` does, which means a ratio has
  to be converted into a step count, which means the total number of steps has to be known.
  See :func:`build_sft_config`.

Belt and braces: every argument is filtered
-------------------------------------------
Having read the signatures, the kwargs are still filtered against them at runtime and anything
dropped is recorded in :attr:`TrainerPlan.dropped_arguments`. Two libraries in this stack rename
things between minor versions; a run should degrade with a note in its metadata rather than
crash, and "which of my settings did not apply?" should be answerable from the artifact.

The trainer is never asked to attach adapters
---------------------------------------------
``peft_config`` is deliberately not passed. :func:`qa_gen_runtime.loader.attach_adapters` owns
the k-bit preparation and the LoRA attachment, and TRL would wrap the model a second time if
also handed a config. One path, asserted by a test.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_gen.config import GenerationExperimentConfig
from qa_gen_runtime.deps import require_trl
from qa_gen_runtime.precision import PrecisionPlan, resolve_precision

logger = logging.getLogger(__name__)

__all__ = [
    "TrainerBuildError",
    "TrainerPlan",
    "build_sft_config",
    "build_trainer",
    "plan_trainer_arguments",
    "resolve_warmup_steps",
]


class TrainerBuildError(RuntimeError):
    """Raised when trainer arguments cannot be built from the configuration."""


@dataclass(frozen=True, slots=True)
class TrainerPlan:
    """The arguments that will be handed to ``SFTConfig``, and what was left out.

    Separated from the ``SFTConfig`` object so the translation is inspectable without TRL
    installed. Every mapping decision this module makes is visible in :attr:`arguments`, which
    is what the tests assert against.

    Attributes:
        arguments: Keyword arguments for ``SFTConfig``.
        precision: The resolved precision decision.
        total_steps: Estimated optimiser steps, when derivable. Needed for warmup.
        warmup_steps: The converted warmup, in steps.
        dropped_arguments: Names the installed ``SFTConfig`` does not accept. Empty on the
            verified version; non-empty means a rename and a setting that did not apply.
        notes: Translation decisions worth recording.
    """

    arguments: dict[str, Any] = field(default_factory=dict)
    precision: PrecisionPlan | None = None
    total_steps: int | None = None
    warmup_steps: int = 0
    dropped_arguments: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "dropped_arguments", tuple(self.dropped_arguments))
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Argument values are stringified when they are not natively serializable, so a plan
        containing a ``torch.dtype`` or a ``Path`` still writes to a record.
        """
        serializable: dict[str, Any] = {}
        for key, value in sorted(self.arguments.items()):
            serializable[key] = (
                value if isinstance(value, str | int | float | bool | type(None)) else str(value)
            )
        return {
            "arguments": serializable,
            "precision": self.precision.as_dict() if self.precision else None,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "dropped_arguments": list(self.dropped_arguments),
            "notes": list(self.notes),
        }


def resolve_warmup_steps(
    config: GenerationExperimentConfig, *, train_examples: int | None = None
) -> tuple[int, int | None, list[str]]:
    """Convert the configured warmup ratio into a step count.

    transformers 5.16.1 removed ``warmup_ratio`` and kept only ``warmup_steps``, so the ratio
    has to be multiplied by the total number of optimiser steps -- which means knowing how many
    there will be. ``max_steps`` gives it directly; otherwise it comes from the corpus size,
    the effective batch size and the epoch count.

    Args:
        config: The experiment configuration.
        train_examples: Size of the training split. Required when the warmup ratio is above
            zero and ``max_steps`` is unset.

    Returns:
        ``(warmup_steps, total_steps, notes)``. ``total_steps`` is ``None`` when it could not
        be derived and no warmup was requested.

    Raises:
        TrainerBuildError: If a warmup ratio is requested but the total step count cannot be
            derived. Silently dropping the warmup would change the optimisation schedule
            without saying so, and on a freshly initialised adapter at 2e-4 that is not a
            harmless difference.
    """
    training = config.training
    notes: list[str] = []

    total_steps: int | None = None
    if training.max_steps is not None:
        total_steps = training.max_steps
        notes.append(f"total_steps taken from max_steps={training.max_steps}")
    elif train_examples is not None:
        effective = training.effective_batch_size
        steps_per_epoch = max(1, math.ceil(train_examples / effective))
        total_steps = max(1, steps_per_epoch * training.num_train_epochs)
        notes.append(
            f"total_steps={total_steps} derived from {train_examples} examples / "
            f"effective batch {effective} x {training.num_train_epochs} epoch(s)"
        )

    if training.warmup_ratio <= 0:
        return 0, total_steps, notes

    if total_steps is None:
        raise TrainerBuildError(
            f"training.warmup_ratio is {training.warmup_ratio} but the total step count cannot "
            "be derived: transformers 5.x removed warmup_ratio and accepts only warmup_steps, "
            "so a ratio has to be multiplied by a known number of steps.\n"
            "Supply train_examples, or set training.max_steps, or set warmup_ratio to 0."
        )

    warmup_steps = max(1, int(round(total_steps * training.warmup_ratio)))
    notes.append(
        f"warmup_ratio {training.warmup_ratio} converted to warmup_steps={warmup_steps} "
        "(transformers 5.x removed warmup_ratio)"
    )
    return warmup_steps, total_steps, notes


def plan_trainer_arguments(
    config: GenerationExperimentConfig,
    output_dir: str | Path,
    *,
    train_examples: int | None = None,
    precision: PrecisionPlan | None = None,
) -> TrainerPlan:
    """Build the ``SFTConfig`` keyword arguments without needing TRL installed.

    The whole translation, in one inspectable place. Every naming decision documented in this
    module's docstring is applied here.

    Args:
        config: The experiment configuration.
        output_dir: Where the trainer writes checkpoints.
        train_examples: Size of the training split, for warmup conversion.
        precision: A resolved plan. Resolved from the configuration when omitted.

    Returns:
        The :class:`TrainerPlan`. Not yet filtered against ``SFTConfig``; that happens in
        :func:`build_sft_config`, which needs TRL.

    Raises:
        TrainerBuildError: If the warmup ratio cannot be converted.
        qa_gen_runtime.precision.PrecisionError: If the requested precision is unavailable.
    """
    plan_precision = precision or resolve_precision(config.training.precision)
    warmup_steps, total_steps, notes = resolve_warmup_steps(
        config, train_examples=train_examples
    )
    training = config.training

    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        # --- schedule -----------------------------------------------------------------
        "learning_rate": training.learning_rate,
        "num_train_epochs": training.num_train_epochs,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "weight_decay": training.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": training.lr_scheduler_type,
        "max_grad_norm": training.max_grad_norm,
        "optim": training.optimizer,
        "seed": training.seed,
        # --- precision ----------------------------------------------------------------
        "bf16": plan_precision.bf16,
        "fp16": plan_precision.fp16,
        # --- memory -------------------------------------------------------------------
        "gradient_checkpointing": training.gradient_checkpointing,
        # --- evaluation and checkpointing ---------------------------------------------
        # transformers 5.x: eval_strategy, not evaluation_strategy.
        "eval_strategy": training.evaluation_strategy,
        "save_strategy": training.save_strategy,
        "save_total_limit": training.save_total_limit,
        "load_best_model_at_end": training.load_best_model_at_end,
        "metric_for_best_model": training.metric_for_best_model,
        "greater_is_better": training.greater_is_better,
        "logging_steps": training.logging_steps,
        "dataloader_num_workers": training.dataloader_num_workers,
        # --- SFT-specific -------------------------------------------------------------
        # TRL: max_length, not max_seq_length. Renamed around 0.18.
        "max_length": config.model.max_seq_length,
        "packing": training.packing,
        "completion_only_loss": training.completion_only_loss,
        # Nothing is reported anywhere by default. An unconfigured tracker that silently
        # tries to reach the network is not something a training run should do on its own.
        "report_to": [],
    }

    if training.max_steps is not None:
        arguments["max_steps"] = training.max_steps

    if training.gradient_checkpointing:
        # Non-reentrant checkpointing is what works with PEFT adapters; the reentrant
        # implementation drops the adapter gradients on some architectures.
        arguments["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        notes.append("gradient_checkpointing_kwargs use_reentrant=False set for PEFT")

    notes.append(f"precision: {plan_precision.reason}")
    notes.append(
        f"optimizer {training.optimizer!r} is runtime configuration and has not been "
        "benchmarked by this project"
    )

    return TrainerPlan(
        arguments=arguments,
        precision=plan_precision,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        notes=tuple(notes),
    )


def build_sft_config(
    config: GenerationExperimentConfig,
    output_dir: str | Path,
    *,
    train_examples: int | None = None,
    precision: PrecisionPlan | None = None,
) -> tuple[Any, TrainerPlan]:
    """Build a TRL ``SFTConfig``, dropping anything the installed version does not accept.

    Args:
        config: The experiment configuration.
        output_dir: Where the trainer writes checkpoints.
        train_examples: Size of the training split, for warmup conversion.
        precision: A resolved precision plan.

    Returns:
        ``(sft_config, plan)``. The plan records what was dropped, so the caller can put it in
        the run metadata.

    Raises:
        TrainerBuildError: If ``SFTConfig`` rejects the filtered arguments anyway -- which
            means a value is wrong rather than a name, and the message carries TRL's own
            complaint.
        qa_gen_runtime.deps.RuntimeDependencyError: If TRL is not installed.
    """
    trl = require_trl()
    plan = plan_trainer_arguments(
        config, output_dir, train_examples=train_examples, precision=precision
    )

    supported, dropped = _filter_supported(trl.SFTConfig, plan.arguments)
    if dropped:
        logger.warning(
            "trl.SFTConfig does not accept %s in this version; those settings were not "
            "applied.",
            sorted(dropped),
        )

    try:
        sft_config = trl.SFTConfig(**supported)
    except (TypeError, ValueError) as exc:
        raise TrainerBuildError(
            f"trl.SFTConfig rejected the translated arguments: {type(exc).__name__}: {exc}\n"
            f"Arguments passed: {sorted(supported)}"
        ) from exc

    resolved = TrainerPlan(
        arguments=plan.arguments,
        precision=plan.precision,
        total_steps=plan.total_steps,
        warmup_steps=plan.warmup_steps,
        dropped_arguments=tuple(sorted(dropped)),
        notes=plan.notes,
    )
    return sft_config, resolved


def build_trainer(
    config: GenerationExperimentConfig,
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    output_dir: str | Path,
    eval_dataset: Any = None,
    train_examples: int | None = None,
    precision: PrecisionPlan | None = None,
) -> tuple[Any, TrainerPlan]:
    """Construct a TRL ``SFTTrainer`` for an already-prepared model.

    ``peft_config`` is not passed. The model arriving here has already been through
    :func:`qa_gen_runtime.loader.attach_adapters`, and handing TRL a ``peft_config`` as well
    would wrap it twice.

    Args:
        config: The experiment configuration.
        model: A prepared, adapter-wrapped model.
        tokenizer: The tokenizer, passed as ``processing_class`` -- TRL renamed ``tokenizer``.
        train_dataset: The training dataset. Required by TRL.
        output_dir: Where the trainer writes checkpoints.
        eval_dataset: Optional evaluation dataset.
        train_examples: Size of the training split, for warmup conversion. Derived from
            ``train_dataset`` when it supports ``len()``.
        precision: A resolved precision plan.

    Returns:
        ``(trainer, plan)``.

    Raises:
        TrainerBuildError: If the trainer cannot be constructed.
        qa_gen_runtime.deps.RuntimeDependencyError: If TRL is not installed.
    """
    trl = require_trl()

    if train_dataset is None:
        raise TrainerBuildError(
            "train_dataset is required: TRL raises without one, and a trainer built around an "
            "absent dataset would fail only once training started."
        )

    resolved_examples = train_examples
    if resolved_examples is None:
        try:
            resolved_examples = len(train_dataset)
        except TypeError:
            resolved_examples = None

    sft_config, plan = build_sft_config(
        config, output_dir, train_examples=resolved_examples, precision=precision
    )

    wanted: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        # TRL renamed `tokenizer` to `processing_class`.
        "processing_class": tokenizer,
    }
    if eval_dataset is not None:
        wanted["eval_dataset"] = eval_dataset

    supported, dropped = _filter_supported(trl.SFTTrainer.__init__, wanted)
    if dropped:
        raise TrainerBuildError(
            f"trl.SFTTrainer does not accept {sorted(dropped)} in this version. The installed "
            "TRL is incompatible with this runtime; the verified version is 0.29.1."
        )

    try:
        trainer = trl.SFTTrainer(**supported)
    except (TypeError, ValueError) as exc:
        raise TrainerBuildError(
            f"trl.SFTTrainer rejected its arguments: {type(exc).__name__}: {exc}"
        ) from exc

    return trainer, plan


def _filter_supported(
    target: Any, wanted: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    """Split ``wanted`` into keywords ``target`` accepts and keywords it does not.

    A callable accepting ``**kwargs``, or one whose signature cannot be read, is treated as
    accepting everything: refusing to pass arguments to something introspection cannot
    describe would break more than it protects.

    Args:
        target: The callable or dataclass to inspect.
        wanted: Candidate keyword arguments.

    Returns:
        ``(supported, dropped_names)``.
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
        return dict(wanted), set()

    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return dict(wanted), set()

    supported = {key: value for key, value in wanted.items() if key in parameters}
    return supported, set(wanted) - set(supported)
