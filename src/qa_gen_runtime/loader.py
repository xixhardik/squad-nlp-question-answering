"""Loading the tokenizer, the quantized base model, and the LoRA adapters.

The one k-bit preparation path
------------------------------
``load quantized model -> prepare_model_for_kbit_training -> attach LoRA`` happens in exactly
one function, :func:`attach_adapters`, and nowhere else. That matters more than it sounds.
The order is not decorative: ``prepare_model_for_kbit_training`` upcasts the layer norms and
the LM head to fp32, enables input gradients and makes the frozen quantized weights safe to
backprop through. Doing it *after* attaching adapters silently leaves the adapters
mis-prepared, and the symptom is a loss that does not move rather than an error.

Because there is one path, :func:`build_trainer` deliberately passes ``peft_config=None`` to
TRL. TRL will wrap a model itself when given a ``peft_config``, and doing both would either
double-wrap or apply the adapter to an unprepared model depending on version. The rule is:
this module owns preparation, the trainer factory never asks for it.

transformers 5.x naming, verified rather than assumed
-----------------------------------------------------
Introspected against the installed transformers 5.16.1:

- ``from_pretrained`` takes **``dtype``**, not ``torch_dtype``. The v4 name is gone from the
  documented signature.
- ``TrainingArguments`` has **``eval_strategy``**, not ``evaluation_strategy``, and has **no
  ``warmup_ratio``** at all -- only ``warmup_steps``. That second one is handled in
  :mod:`qa_gen_runtime.trainer`.

Guessing either of these would have produced a ``TypeError`` on the Studio after a model
download, which is the most expensive place to discover a keyword rename.

Nothing here is called at import time
-------------------------------------
Every function in this module downloads or allocates only when invoked. Importing it is free,
which is what lets the CLI validate a configuration without touching the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from qa_gen.config import GenerationExperimentConfig, GeneratorModelConfig, LoRAConfig
from qa_gen_runtime.deps import require_peft
from qa_gen_runtime.precision import PrecisionPlan, resolve_precision
from qa_gen_runtime.quantization import build_quantization_config

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedModel",
    "ModelLoadError",
    "attach_adapters",
    "build_lora_config",
    "build_model_kwargs",
    "count_parameters",
    "load_base_model",
    "load_tokenizer",
    "load_trainable_model",
]


class ModelLoadError(RuntimeError):
    """Raised when the tokenizer or model cannot be loaded or prepared."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A prepared model with its tokenizer and the facts about how it was built.

    Attributes:
        model: The PEFT-wrapped, k-bit-prepared model, ready for a trainer.
        tokenizer: The tokenizer, with a pad token guaranteed.
        precision: The resolved precision decision.
        trainable_parameters: Parameters requiring gradients, counted after wrapping.
        total_parameters: All parameters, counted after wrapping.
        adapters_attached: Whether LoRA was applied. ``False`` only for a full fine-tune,
            which this project does not currently do.
        notes: Anything worth recording about the load, e.g. a pad token being synthesised.
    """

    model: Any
    tokenizer: Any
    precision: PrecisionPlan
    trainable_parameters: int
    total_parameters: int
    adapters_attached: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce ``notes`` to a tuple."""
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def trainable_fraction(self) -> float:
        """Share of parameters that train, from the measured counts."""
        if not self.total_parameters:
            return 0.0
        return round(self.trainable_parameters / self.total_parameters, 6)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary. The model itself is not included."""
        return {
            "precision": self.precision.as_dict(),
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_fraction,
            "adapters_attached": self.adapters_attached,
            "notes": list(self.notes),
        }


def count_parameters(model: Any) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for any module.

    Counted directly from ``model.parameters()`` rather than through PEFT's
    ``get_nb_trainable_parameters``. Two reasons: it works identically on a plain module, a
    prepared module and a wrapped one, so the number is comparable across all three; and it
    does not depend on a PEFT method name surviving the next release. On the measured L4 this
    should report 33,030,144 of 4,055,498,240.

    Args:
        model: Any object exposing ``parameters()``.

    Returns:
        ``(trainable, total)``.

    Raises:
        ModelLoadError: If the object does not expose ``parameters()``.
    """
    if not hasattr(model, "parameters"):
        raise ModelLoadError(
            f"cannot count parameters of {type(model).__name__}: it has no parameters()."
        )
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def load_tokenizer(config: GeneratorModelConfig) -> Any:
    """Load the tokenizer for a model configuration.

    A pad token is guaranteed. Many causal decoders ship without one, and a missing pad token
    surfaces during collation as an opaque error about ``None`` rather than as anything
    pointing at the tokenizer. Falling back to the EOS token is the standard remedy and is
    safe here because loss is masked on the prompt anyway.

    Args:
        config: The Phase 17A model configuration.

    Returns:
        The tokenizer.

    Raises:
        ModelLoadError: If the tokenizer cannot be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.effective_tokenizer_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"could not load tokenizer {config.effective_tokenizer_id!r} at revision "
            f"{config.revision!r}: {type(exc).__name__}: {exc}"
        ) from exc

    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise ModelLoadError(
                f"tokenizer {config.effective_tokenizer_id!r} has neither a pad token nor an "
                "eos token, so batches cannot be padded. Configure a pad token on the "
                "tokenizer before training."
            )
        tokenizer.pad_token = eos
        logger.info("Tokenizer had no pad token; using eos token %r.", eos)

    return tokenizer


def build_model_kwargs(
    config: GeneratorModelConfig, precision: PrecisionPlan
) -> dict[str, Any]:
    """Build the keyword arguments for ``AutoModelForCausalLM.from_pretrained``.

    Separated from the call so the arguments can be asserted without loading anything. This is
    where the transformers 5.x ``dtype`` naming is applied, and where the quantization
    translation is spliced in.

    Args:
        config: The Phase 17A model configuration.
        precision: The resolved precision plan.

    Returns:
        The keyword arguments. ``quantization_config`` is present only when quantization was
        requested -- omitted rather than passed as ``None``, because the two are not always
        equivalent across versions.
    """
    kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        # transformers 5.x: `dtype`, not `torch_dtype`. Verified against 5.16.1.
        "dtype": precision.dtype,
    }
    if config.attn_implementation != "auto":
        kwargs["attn_implementation"] = config.attn_implementation

    quantization_config = build_quantization_config(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        # Let accelerate place the quantized shards. Without this, a 4-bit load lands on CPU
        # and the first forward pass fails on a device mismatch.
        kwargs["device_map"] = "auto"

    return kwargs


def load_base_model(
    config: GeneratorModelConfig, precision: PrecisionPlan | None = None
) -> Any:
    """Load the quantized base model. No adapters, no preparation.

    Args:
        config: The Phase 17A model configuration.
        precision: A resolved plan. Resolved from ``config.precision`` when omitted.

    Returns:
        The loaded model.

    Raises:
        ModelLoadError: If loading fails. The message names the model and revision, because
            the usual causes are a typo, a gated repo and no network, and the id is what
            distinguishes them.
    """
    plan = precision or resolve_precision(config.precision)
    kwargs = build_model_kwargs(config, plan)
    logger.info(
        "Loading %s (revision %s) with %s",
        config.model_id,
        config.revision,
        config.quantization_settings(),
    )
    try:
        return AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    except Exception as exc:
        raise ModelLoadError(
            f"could not load model {config.model_id!r} at revision {config.revision!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def build_lora_config(lora: LoRAConfig) -> Any:
    """Translate the generic adapter configuration into a ``peft.LoraConfig``.

    Unsupported keyword arguments are dropped rather than passed. ``use_rslora`` and
    ``modules_to_save`` have both arrived and moved during PEFT's history, and a keyword the
    installed version does not know is a ``TypeError`` at the worst moment. Anything dropped is
    logged, so a setting that quietly did not apply is discoverable.

    Args:
        lora: The Phase 17A adapter configuration.

    Returns:
        A ``peft.LoraConfig``.

    Raises:
        qa_gen_runtime.deps.RuntimeDependencyError: If PEFT is not installed.
    """
    peft = require_peft()
    wanted: dict[str, Any] = {
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": lora.bias,
        "task_type": lora.task_type,
        "target_modules": list(lora.target_modules),
        "use_rslora": lora.use_rslora,
    }
    if lora.modules_to_save:
        wanted["modules_to_save"] = list(lora.modules_to_save)

    supported, dropped = _filter_supported(peft.LoraConfig, wanted)
    if dropped:
        logger.warning(
            "peft.LoraConfig does not accept %s in this version; those settings were not "
            "applied.",
            sorted(dropped),
        )
    return peft.LoraConfig(**supported)


def attach_adapters(
    model: Any,
    lora: LoRAConfig,
    *,
    gradient_checkpointing: bool = True,
) -> Any:
    """Prepare a quantized model for k-bit training and attach LoRA adapters.

    **The only place this sequence exists.** In order:

    1. ``prepare_model_for_kbit_training`` -- upcasts norms and the head to fp32, enables
       input gradients, makes the frozen quantized weights backprop-safe.
    2. ``get_peft_model`` -- attaches the adapters to the prepared model.

    Reversing these leaves the adapters attached to an unprepared model, and the symptom is a
    loss that does not move rather than an exception.

    ``use_cache`` is set to ``False`` when checkpointing is enabled. The KV cache and
    activation checkpointing are mutually exclusive; transformers warns and overrides it, but
    doing it here means the model's own config states the truth rather than being silently
    corrected later.

    Args:
        model: A freshly loaded, quantized base model.
        lora: The adapter configuration.
        gradient_checkpointing: Whether the trainer will use activation checkpointing. Passed
            through to the preparation call, which needs to know in order to enable input
            gradients.

    Returns:
        The PEFT-wrapped model.

    Raises:
        qa_gen_runtime.deps.RuntimeDependencyError: If PEFT is not installed.
        ModelLoadError: If preparation or wrapping fails.
    """
    peft = require_peft()

    if gradient_checkpointing:
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = False
            logger.info("Disabled use_cache: incompatible with gradient checkpointing.")

    try:
        prepared = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    except Exception as exc:
        raise ModelLoadError(
            f"prepare_model_for_kbit_training failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        wrapped = peft.get_peft_model(prepared, build_lora_config(lora))
    except Exception as exc:
        raise ModelLoadError(f"get_peft_model failed: {type(exc).__name__}: {exc}") from exc

    trainable, total = count_parameters(wrapped)
    logger.info(
        "Adapters attached: %s trainable of %s parameters (%.4f%%).",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total else 0.0,
    )
    return wrapped


def load_trainable_model(config: GenerationExperimentConfig) -> LoadedModel:
    """Load, prepare and adapt a model from a complete experiment configuration.

    The single entry point a training run should use. Everything below it is exposed for
    testing and for callers with unusual needs.

    Args:
        config: The Phase 17A experiment configuration.

    Returns:
        The :class:`LoadedModel`.

    Raises:
        ModelLoadError: If any stage fails.
        qa_gen_runtime.precision.PrecisionError: If the requested precision is unavailable.
        qa_gen_runtime.deps.RuntimeDependencyError: If PEFT is not installed.
    """
    precision = resolve_precision(config.model.precision)
    notes: list[str] = [precision.reason]

    tokenizer = load_tokenizer(config.model)
    model = load_base_model(config.model, precision)
    wrapped = attach_adapters(
        model,
        config.lora,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    trainable, total = count_parameters(wrapped)

    if not trainable:
        raise ModelLoadError(
            "no parameter requires gradients after attaching adapters, so training would be a "
            "no-op. Check lora.target_modules against this architecture's module names."
        )

    return LoadedModel(
        model=wrapped,
        tokenizer=tokenizer,
        precision=precision,
        trainable_parameters=trainable,
        total_parameters=total,
        adapters_attached=True,
        notes=tuple(notes),
    )


def _filter_supported(
    target: Any, wanted: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    """Split ``wanted`` into keywords ``target`` accepts and keywords it does not.

    Shared by the adapter and trainer translations. A callable whose signature cannot be read,
    or which accepts ``**kwargs``, is treated as accepting everything -- refusing to pass
    arguments to something introspection cannot describe would break more than it protects.

    Args:
        target: The callable or dataclass whose ``__init__`` is inspected.
        wanted: Candidate keyword arguments.

    Returns:
        ``(supported, dropped_names)``.
    """
    import inspect

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
    dropped = set(wanted) - set(supported)
    return supported, dropped
