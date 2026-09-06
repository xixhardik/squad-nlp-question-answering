"""Translating the generic quantization decision into a ``BitsAndBytesConfig``.

The boundary this module *is*
-----------------------------
:class:`qa_gen.config.GeneratorModelConfig` records four plain values -- ``quantization``,
``quantization_type``, ``double_quantization``, ``compute_dtype`` -- and knows nothing about
``bitsandbytes``. This function is the only place those become a library object. Phase 17A
therefore stays importable with no ML stack, and there is exactly one line of code to change
if the quantization library is ever swapped.

Testable for real, which was worth checking
-------------------------------------------
``transformers.BitsAndBytesConfig`` constructs successfully **without bitsandbytes
installed** -- verified on the development machine, where bitsandbytes is absent. It is a
plain configuration object; the library is only needed when weights are actually loaded. So
the translation asserted by the tests is the real one rather than a mock of it, which is
unusual for this kind of boundary and worth taking advantage of.

The defaults are the measured ones
----------------------------------
NF4, double quantization, bf16 compute: the configuration that loaded Qwen3-4B at 2.53 GiB
peak on the L4. See :data:`qa_gen.config.VERIFIED_QWEN3_4B_L4`.
"""

from __future__ import annotations

from typing import Any

from transformers import BitsAndBytesConfig

from qa_gen.config import GeneratorModelConfig
from qa_gen_runtime.deps import is_available
from qa_gen_runtime.precision import resolve_dtype, resolve_precision

__all__ = [
    "QuantizationError",
    "build_quantization_config",
    "describe_quantization",
]


class QuantizationError(RuntimeError):
    """Raised when a quantization request cannot be translated or satisfied."""


def _resolve_compute_dtype(config: GeneratorModelConfig) -> Any:
    """Resolve the compute dtype, consulting the device only when asked to.

    ``compute_dtype="auto"`` defers to :func:`~qa_gen_runtime.precision.resolve_precision`,
    which will refuse rather than silently pick fp16. An explicit name is mapped directly, so
    a bf16 request on a CPU-only machine still *translates* -- the configuration is legal and
    the failure belongs at load time, not here. Refusing to build the config would make the
    translation untestable anywhere without a GPU.
    """
    if config.compute_dtype == "auto":
        return resolve_precision("auto").dtype
    return resolve_dtype(config.compute_dtype)


def build_quantization_config(config: GeneratorModelConfig) -> BitsAndBytesConfig | None:
    """Translate the generic model configuration into a ``BitsAndBytesConfig``.

    Args:
        config: The Phase 17A model configuration.

    Returns:
        The quantization configuration to pass as ``quantization_config=`` to
        ``from_pretrained``, or ``None`` when ``quantization`` is ``"none"`` -- in which case
        the caller must not pass the keyword at all rather than pass ``None`` explicitly,
        which some versions treat differently from omitting it.

    Raises:
        QuantizationError: If the quantization mode is unrecognised. Value validation lives
            on the config's own ``validate()``; this catches a mode that validated under an
            older vocabulary and has no translation here, which would otherwise silently load
            unquantized weights and blow the memory budget.
    """
    if config.quantization == "none":
        return None

    compute_dtype = _resolve_compute_dtype(config)

    if config.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_use_double_quant=config.double_quantization,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if config.quantization == "8bit":
        # The 4-bit-only keys are deliberately not forwarded. bitsandbytes ignores them in
        # 8-bit mode, and passing them would put values into the run record that had no
        # effect on the run.
        return BitsAndBytesConfig(load_in_8bit=True)

    raise QuantizationError(
        f"no translation exists for quantization mode {config.quantization!r}. "
        "Expected 'none', '4bit' or '8bit'."
    )


def describe_quantization(config: GeneratorModelConfig) -> dict[str, Any]:
    """Return a JSON-serializable description of the quantization actually applied.

    Built from the translated object rather than from the source configuration, so the record
    describes what the library was told rather than what the caller intended. The two agree
    today; recording the former is what would reveal it if they stopped.

    Args:
        config: The Phase 17A model configuration.

    Returns:
        A mapping including whether ``bitsandbytes`` is importable. Quantized weights cannot
        load without it, so a run that reports ``requested`` 4-bit and
        ``bitsandbytes_available`` false has an explanation for its failure in its own
        metadata.
    """
    quantization_config = build_quantization_config(config)
    record: dict[str, Any] = {
        "requested": config.quantization,
        "bitsandbytes_available": is_available("bitsandbytes"),
        "applied": quantization_config is not None,
    }
    if quantization_config is None:
        return record

    record["load_in_4bit"] = bool(getattr(quantization_config, "load_in_4bit", False))
    record["load_in_8bit"] = bool(getattr(quantization_config, "load_in_8bit", False))
    if record["load_in_4bit"]:
        record["quant_type"] = quantization_config.bnb_4bit_quant_type
        record["double_quant"] = bool(quantization_config.bnb_4bit_use_double_quant)
        record["compute_dtype"] = str(quantization_config.bnb_4bit_compute_dtype)
    return record
