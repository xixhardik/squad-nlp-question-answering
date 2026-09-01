"""Loading tokenizers and question answering models.

Thin, explicit wrappers over the ``Auto*`` classes. They exist to centralise three
things that are easy to get wrong and expensive to discover late:

1. **Fast tokenizers are mandatory.** Offset mappings come from the Rust
   ``tokenizers`` backend. A slow tokenizer would silently omit them and the
   character/token alignment would collapse, so this is checked at load time.
2. **A freshly initialised QA head is expected, once.** ``AutoModelForQuestionAnswering``
   adds an untrained ``qa_outputs`` layer to a pretrained encoder. That is correct
   before fine-tuning and a bug after it, so the distinction is surfaced.
3. **Failures get actionable messages.** A typo in a model id or an offline machine
   otherwise produces a long Hub traceback that buries the cause.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from qa_torch.features import TokenizerNotFastError

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointIntegrityError",
    "ModelBundle",
    "ModelLoadError",
    "count_parameters",
    "describe_model",
    "load_qa_model",
    "load_tokenizer",
    "verify_checkpoint_integrity",
]


class ModelLoadError(RuntimeError):
    """Raised when a tokenizer or model cannot be loaded."""


class CheckpointIntegrityError(RuntimeError):
    """Raised when a saved checkpoint does not reload to identical parameters."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A tokenizer and question answering model loaded together.

    Attributes:
        tokenizer: Fast tokenizer used for both training and inference.
        model: Model with a span-prediction head emitting start/end logits.
        model_name: The identifier or path it was loaded from.
        num_parameters: Total parameter count, measured rather than quoted.
    """

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    model_name: str
    num_parameters: int


def load_tokenizer(model_name: str, **kwargs: object) -> PreTrainedTokenizerBase:
    """Load a fast tokenizer.

    Args:
        model_name: Hugging Face model id or a local directory.
        **kwargs: Forwarded to ``AutoTokenizer.from_pretrained``.

    Returns:
        The loaded tokenizer.

    Raises:
        ModelLoadError: If the tokenizer cannot be loaded at all.
        TokenizerNotFastError: If the loaded tokenizer is not backed by
            ``tokenizers`` and therefore cannot emit offset mappings.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load a tokenizer for {model_name!r}.\n"
            "Check that:\n"
            "  - the model id is spelled correctly (e.g. 'bert-base-uncased')\n"
            "  - or the path exists, if you meant a local checkpoint directory\n"
            "  - this machine can reach huggingface.co (no token is needed for "
            "public models)\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    if not getattr(tokenizer, "is_fast", False):
        raise TokenizerNotFastError(
            f"{model_name!r} resolved to {type(tokenizer).__name__}, which is not a "
            "fast tokenizer. This pipeline requires offset mappings, which only the "
            "Rust `tokenizers` backend provides. There is no workaround: pick a model "
            "with a fast tokenizer."
        )

    logger.info(
        "Loaded tokenizer %s (%s) for %s",
        type(tokenizer).__name__,
        "fast",
        model_name,
    )
    return tokenizer


def load_qa_model(
    model_name: str,
    *,
    expect_trained_head: bool = False,
    **kwargs: object,
) -> PreTrainedModel:
    """Load a model with a question answering span-prediction head.

    Master weights are always fp32
    ------------------------------
    ``from_pretrained`` preserves whatever dtype the published checkpoint stores, and
    not every Hub repository publishes fp32. ``microsoft/deberta-v3-base`` publishes
    **fp16** tensors, while ``bert-base-uncased``, ``distilbert-base-uncased`` and
    ``roberta-base`` publish fp32.

    Loading fp16 and then training is silently destructive, because mixed precision
    via ``TrainingArguments(bf16=True)`` is autocast: it expects fp32 *master* weights
    and casts down per operation. Given fp16 masters, the optimizer updates fp16
    directly, and AdamW does not survive that. Measured with this project's
    hyperparameters (``lr=3e-5``, ``eps=1e-8``, ``betas=(0.9, 0.999)``):

    - ``exp_avg_sq`` holds ``(1 - beta2) * grad**2``, about ``1e-9`` for a realistic
      ``1e-3`` gradient. fp16's smallest subnormal is ``2**-24`` (``5.96e-8``), so it
      underflows to exactly zero.
    - ``eps=1e-8`` is *also* below that floor and rounds to zero in fp16.
    - ``denom = sqrt(0) + 0 = 0``, so the update divides by zero. On a 4096-element
      tensor this produced 4094 ``Inf`` and 2 ``NaN`` on optimizer **step 1**, and
      global gradient clipping then propagated the ``NaN`` to every parameter.

    So the dtype is pinned here rather than left to the checkpoint. This is the single
    choke point every caller goes through, which keeps the invariant in one place.
    The fp16 to fp32 upcast is exact, so no published value is altered. Pass
    ``dtype=`` explicitly to override, for example to load fp16 for inference where no
    optimizer is involved.

    Args:
        model_name: Hugging Face model id or a local checkpoint directory.
        expect_trained_head: Set ``True`` when loading a fine-tuned checkpoint. A
            warning is emitted if the ``qa_outputs`` head turns out to be randomly
            initialised, which would mean the checkpoint is not actually fine-tuned
            and every prediction would be noise.
        **kwargs: Forwarded to ``from_pretrained``. ``dtype`` defaults to
            ``torch.float32``; see above for why that is not left to the checkpoint.

    Returns:
        The loaded model in ``eval`` mode with gradients enabled, ready for either
        training or inference.

    Raises:
        ModelLoadError: If the model or its config cannot be loaded.
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load a model config for {model_name!r}.\n"
            "Check the model id spelling, or that the local checkpoint directory "
            "contains a config.json.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    # Never inherit the checkpoint's dtype for a model that may be trained; see the
    # "Master weights are always fp32" note above.
    kwargs.setdefault("dtype", torch.float32)

    try:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, config=config, **kwargs
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load a question answering model for {model_name!r}.\n"
            "The architecture may not support AutoModelForQuestionAnswering. This "
            "pipeline needs an encoder that emits start and end logits, such as "
            "bert-base-uncased, distilbert-base-uncased, roberta-base or "
            "microsoft/deberta-v3-base.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    if expect_trained_head and not _has_trained_qa_head(model):
        logger.warning(
            "Loaded %r with expect_trained_head=True, but the qa_outputs head looks "
            "randomly initialised. Predictions from this checkpoint would be noise. "
            "Confirm you pointed at a fine-tuned checkpoint rather than a base model.",
            model_name,
        )

    logger.info(
        "Loaded %s for %s (%s parameters)",
        type(model).__name__,
        model_name,
        f"{count_parameters(model):,}",
    )
    return model


def _has_trained_qa_head(model: PreTrainedModel) -> bool:
    """Heuristically detect whether the QA head carries trained weights.

    ``from_pretrained`` initialises a missing ``qa_outputs`` layer from a normal
    distribution with a zero bias. A trained head almost never has an exactly zero
    bias, so that is a cheap and reliable tell.

    Args:
        model: The loaded model.

    Returns:
        ``False`` when the head appears freshly initialised, ``True`` otherwise
        (including when no ``qa_outputs`` layer can be found to inspect).
    """
    head = getattr(model, "qa_outputs", None)
    if head is None or not hasattr(head, "bias") or head.bias is None:
        return True
    return bool(torch.any(head.bias != 0).item())


def count_parameters(model: PreTrainedModel, *, trainable_only: bool = False) -> int:
    """Count model parameters.

    Measured rather than quoted from a model card. ``microsoft/deberta-v3-base``
    publishes no safetensors metadata on the Hub, so this is the only honest way to
    report its size.

    Args:
        model: The model to measure.
        trainable_only: Count only parameters with ``requires_grad``.

    Returns:
        The parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def describe_model(model: PreTrainedModel, model_name: str) -> dict[str, object]:
    """Collect model facts for the experiment record.

    Args:
        model: The loaded model.
        model_name: The identifier it was loaded from.

    Returns:
        A JSON-serializable mapping of architecture and size information.
    """
    config = model.config
    num_parameters = count_parameters(model)
    return {
        "model_name": model_name,
        "architecture": type(model).__name__,
        "model_type": getattr(config, "model_type", None),
        "num_parameters": num_parameters,
        "num_trainable_parameters": count_parameters(model, trainable_only=True),
        # fp32 bytes; the on-disk size of a saved checkpoint tracks this closely.
        "size_fp32_mb": round(num_parameters * 4 / 1e6, 2),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def load_model_bundle(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    expect_trained_head: bool = False,
) -> ModelBundle:
    """Load a tokenizer and model together.

    Args:
        model_name: Model id or local checkpoint directory.
        tokenizer_name: Tokenizer id. Defaults to ``model_name``.
        expect_trained_head: Whether the checkpoint should already be fine-tuned.

    Returns:
        The populated :class:`ModelBundle`.
    """
    tokenizer = load_tokenizer(tokenizer_name or model_name)
    model = load_qa_model(model_name, expect_trained_head=expect_trained_head)
    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        num_parameters=count_parameters(model),
    )


def verify_checkpoint_integrity(
    model: PreTrainedModel,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Reload a just-saved checkpoint and confirm every parameter survived.

    Why this exists
    ---------------
    Some architectures store parameters under legacy names. ``bert-base-uncased``
    ships its LayerNorm parameters as ``LayerNorm.gamma`` / ``LayerNorm.beta``
    rather than ``.weight`` / ``.bias``, a holdover from the original TensorFlow
    release. ``transformers`` maps those names in **both** directions, so
    ``save_pretrained`` writes ``.gamma``/``.beta`` back out and
    ``from_pretrained`` maps them in again. That round trip is lossless, and this
    project's loader always uses ``from_pretrained``.

    A load path that bypasses that mapping is not lossless. Measured on
    ``transformers`` 5.16.1 with a saved BERT checkpoint, a raw
    ``load_state_dict(strict=False)`` reports **50 missing** ``LayerNorm.weight``/
    ``.bias`` keys and **50 unexpected** ``.gamma``/``.beta`` keys, and silently
    leaves those 50 tensors at whatever the model already held. Nothing crashes;
    the model is simply not the one that was saved.

    This check turns that class of failure from silent into loud. It costs one
    extra model load and is worth it: a corrupted checkpoint reported as a
    successful experiment is the worst possible outcome for a results-driven
    project.

    Args:
        model: The in-memory model whose weights were just saved.
        checkpoint_path: Directory the checkpoint was written to.
        strict: Raise on any drift. When ``False``, the drift is reported in the
            return value and logged as an error but no exception is raised.

    Returns:
        A JSON-serializable report with the parameter count, the number of drifted
        tensors, the largest absolute difference, how many LayerNorm parameters sit
        at their default values, how many parameter tensors are non-finite in memory
        and on disk, and an ``ok`` flag. ``ok`` requires finite parameters as well as
        an exact round trip, because ``Inf == Inf`` compares equal and would otherwise
        pass unnoticed.

    Raises:
        CheckpointIntegrityError: If ``strict`` and any parameter differs, or if the
            reloaded key set does not match.
    """
    checkpoint_path = Path(checkpoint_path)
    reference = {name: tensor.detach().cpu() for name, tensor in model.named_parameters()}

    # Checked before any comparison, because a model that already holds NaN cannot
    # round-trip by definition: torch.equal treats NaN as unequal to itself, so every
    # affected tensor reports as "drifted" and the real cause (the training run
    # diverged) is nowhere in the message. Inf matters for the opposite reason --
    # Inf == Inf compares equal, so an Inf-poisoned checkpoint would otherwise pass
    # this guard silently.
    non_finite_reference = [
        name for name, tensor in reference.items() if not torch.isfinite(tensor).all()
    ]

    # Load on CPU so this never competes with training for device memory.
    reloaded = AutoModelForQuestionAnswering.from_pretrained(str(checkpoint_path))
    actual = {name: tensor.detach().cpu() for name, tensor in reloaded.named_parameters()}

    non_finite_reloaded = [
        name for name, tensor in actual.items() if not torch.isfinite(tensor).all()
    ]

    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))

    drifted: list[str] = []
    max_delta = 0.0
    for name, expected in reference.items():
        found = actual.get(name)
        if found is None or found.shape != expected.shape:
            drifted.append(name)
            continue
        if not torch.equal(found, expected):
            drifted.append(name)
            delta = (found.float() - expected.float()).abs().max().item()
            # NaN must not be swallowed here. Python's max(0.0, nan) returns 0.0,
            # because nan > 0.0 is False, which reported a checkpoint of 183,833,090
            # NaN values as "max abs delta 0.000e+00" and sent a real investigation
            # after the checkpoint writer instead of the training run. Once NaN is
            # seen it dominates, since it is the more urgent signal.
            if math.isnan(delta) or delta > max_delta:
                max_delta = delta

    # A LayerNorm weight of exactly all-ones (or bias of all-zeros) is the signature
    # of a parameter that was reinitialised rather than loaded.
    default_layernorm = [
        name
        for name, tensor in actual.items()
        if "LayerNorm" in name
        and (
            (name.endswith(".weight") and torch.allclose(tensor, torch.ones_like(tensor)))
            or (name.endswith(".bias") and torch.allclose(tensor, torch.zeros_like(tensor)))
        )
    ]

    report: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "num_parameters_checked": len(reference),
        "num_drifted": len(drifted),
        "max_abs_delta": max_delta,
        "missing_after_reload": missing,
        "unexpected_after_reload": extra,
        "layernorm_at_default_values": len(default_layernorm),
        "num_non_finite_reference": len(non_finite_reference),
        "num_non_finite_reloaded": len(non_finite_reloaded),
        "ok": (
            not drifted
            and not missing
            and not extra
            and not non_finite_reference
            and not non_finite_reloaded
        ),
    }

    del reloaded

    if report["ok"]:
        logger.info(
            "Checkpoint integrity verified: %d parameters reload bit-identically from %s",
            len(reference),
            checkpoint_path,
        )
        return report

    detail = (
        f"{len(drifted)} parameter(s) differ, {len(missing)} missing, {len(extra)} unexpected"
    )
    if non_finite_reference:
        # Lead with the actual diagnosis. The drift count is a consequence here, not
        # the cause, and pointing at the checkpoint writer would waste the reader's
        # time exactly when they are already debugging a failed run.
        cause = (
            f"The trained model in memory already holds {len(non_finite_reference)} "
            f"non-finite parameter tensor(s) (NaN or Inf), so TRAINING diverged and the "
            f"checkpoint faithfully recorded a broken model. Investigate the training "
            f"run, not the save path: check optimizer state underflow (fp16 master "
            f"weights cannot support AdamW at eps=1e-8), exploding gradients, and "
            f"whether the loss went non-finite. Note that logging_nan_inf_filter is on "
            f"by default and substitutes an average for non-finite losses, so the loss "
            f"log may look finite while grad_norm is already NaN.\n"
            f"First non-finite: {non_finite_reference[:8]}"
        )
    else:
        cause = (
            "Likely causes: a save/load path that bypasses from_pretrained's "
            "key-conversion mapping, or a partially written file."
        )
    message = (
        f"Checkpoint at {checkpoint_path} does NOT reload to the model that was saved: "
        f"{detail} (max abs delta {max_delta:.3e}, non-finite tensors: "
        f"{len(non_finite_reference)} in memory / {len(non_finite_reloaded)} on disk).\n"
        f"First drifted: {drifted[:8]}\n"
        "This means the saved weights are not the trained weights. Do not report "
        f"metrics from this checkpoint. {cause}"
    )
    if strict:
        raise CheckpointIntegrityError(message)
    logger.error(message)
    return report
