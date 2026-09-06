"""Resolving a precision name into a concrete dtype and trainer flags.

The rule this module exists to enforce
--------------------------------------
**Never silently substitute fp16 for bf16.** They are not interchangeable. bf16 has the
same exponent range as fp32 and fp16 does not, so a loss that is stable in bf16 can diverge
or produce NaNs in fp16 -- and a run that quietly swapped one for the other looks like a
model problem for as long as it takes someone to check the flags. So ``"bf16"`` requested on
a device that cannot do it is an **error**, not a downgrade.

``"auto"`` is the only setting permitted to choose, and when it does it records why in
:attr:`PrecisionPlan.reason`. That string goes into the run metadata, so the question "what
precision did this actually train in?" is answered by the artifact.

Device capability, not device presence
--------------------------------------
bf16 support is a property of the *architecture*, not of CUDA being available: a pre-Ampere
GPU has CUDA and no usable bf16. The check therefore asks ``torch.cuda.is_bf16_supported()``
rather than inferring from ``is_available()``. On the measured L4 that returns ``True``, which
is what makes the bf16 default correct there and why the default is not simply hard-coded.

CPU is treated as fp32 only. Autocast bf16 on CPU exists and is slow enough to be useless for
a 4B model, and pretending otherwise would let a smoke test "succeed" on a laptop while
exercising a completely different numerical path from the run it is meant to rehearse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "DTYPE_NAMES",
    "PrecisionError",
    "PrecisionPlan",
    "describe_device",
    "resolve_dtype",
    "resolve_precision",
]

#: Project precision names mapped to torch dtypes. ``"auto"`` is absent on purpose: it is a
#: request to decide, not a dtype, and :func:`resolve_precision` is the only thing allowed to
#: resolve it.
DTYPE_NAMES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class PrecisionError(RuntimeError):
    """Raised when a requested precision cannot be honoured on this device."""


@dataclass(frozen=True, slots=True)
class PrecisionPlan:
    """The resolved precision decision, and why it was made.

    Attributes:
        name: Resolved project precision name -- never ``"auto"``.
        dtype: The corresponding torch dtype, for the model load.
        bf16: Whether to set ``bf16=True`` on the trainer arguments.
        fp16: Whether to set ``fp16=True``. Never both this and :attr:`bf16`.
        requested: What the configuration asked for, kept so an ``"auto"`` resolution is
            distinguishable from an explicit choice in the run record.
        reason: Human-readable explanation of the decision.
        device_type: ``"cuda"``, ``"mps"`` or ``"cpu"``.
        bf16_supported: What the device reported.
    """

    name: str
    dtype: torch.dtype
    bf16: bool
    fp16: bool
    requested: str
    reason: str
    device_type: str
    bf16_supported: bool

    def __post_init__(self) -> None:
        """Guard the one invariant that matters.

        Raises:
            PrecisionError: If both mixed-precision flags are set. ``transformers`` would
                reject this too, but by then the message is about argument validation rather
                than about the bug, which is here.
        """
        if self.bf16 and self.fp16:
            raise PrecisionError(
                "a precision plan cannot request both bf16 and fp16; "
                f"got name={self.name!r}."
            )

    @property
    def is_mixed(self) -> bool:
        """Whether a reduced-precision compute path was selected."""
        return self.bf16 or self.fp16

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        ``dtype`` is stringified because ``torch.bfloat16`` is not JSON-serializable and a
        run record has to survive being written to a file.
        """
        return {
            "name": self.name,
            "dtype": str(self.dtype),
            "bf16": self.bf16,
            "fp16": self.fp16,
            "requested": self.requested,
            "reason": self.reason,
            "device_type": self.device_type,
            "bf16_supported": self.bf16_supported,
        }


def _cuda_available() -> bool:
    """Whether a CUDA device is usable. Indirected so tests can substitute it."""
    return bool(torch.cuda.is_available())


def _bf16_supported() -> bool:
    """Whether this device supports bfloat16 arithmetic.

    Asks the architecture, not merely whether CUDA exists. Returns ``False`` on CPU, and
    tolerates the query itself failing on an odd driver rather than crashing precision
    resolution.
    """
    if not _cuda_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except (AssertionError, RuntimeError):  # pragma: no cover - driver-specific
        return False


def resolve_dtype(name: str) -> torch.dtype:
    """Map a project precision name to a torch dtype.

    Args:
        name: One of ``"fp32"``, ``"fp16"``, ``"bf16"``.

    Returns:
        The torch dtype.

    Raises:
        PrecisionError: If the name is unknown, or is ``"auto"`` -- which has no dtype until
            a device has been consulted, and asking for one is a sign the caller wanted
            :func:`resolve_precision`.
    """
    if name == "auto":
        raise PrecisionError(
            "'auto' is not a dtype. Call resolve_precision() to consult the device first."
        )
    dtype = DTYPE_NAMES.get(name)
    if dtype is None:
        raise PrecisionError(
            f"unknown precision {name!r}; expected one of {sorted(DTYPE_NAMES)} or 'auto'."
        )
    return dtype


def resolve_precision(requested: str) -> PrecisionPlan:
    """Resolve a requested precision against this device's capabilities.

    Args:
        requested: ``"auto"``, ``"fp32"``, ``"fp16"`` or ``"bf16"``.

    Returns:
        The :class:`PrecisionPlan`.

    Raises:
        PrecisionError: If ``"bf16"`` is requested on a device that does not support it, or
            if a reduced precision is requested with no CUDA device. Both fail loudly rather
            than degrading: the whole point of naming a precision explicitly is to get that
            one, and a run that got something else must not look like a run that got what it
            asked for.
    """
    device_type = "cuda" if _cuda_available() else "cpu"
    bf16_ok = _bf16_supported()

    if requested == "auto":
        if bf16_ok:
            return PrecisionPlan(
                name="bf16",
                dtype=torch.bfloat16,
                bf16=True,
                fp16=False,
                requested=requested,
                reason="auto resolved to bf16: the device reports bfloat16 support",
                device_type=device_type,
                bf16_supported=bf16_ok,
            )
        if device_type == "cuda":
            # Deliberately fp32 rather than fp16. fp16 needs loss scaling to be stable and
            # this project has not established that it is; choosing it here would be a
            # numerical decision disguised as a fallback.
            return PrecisionPlan(
                name="fp32",
                dtype=torch.float32,
                bf16=False,
                fp16=False,
                requested=requested,
                reason=(
                    "auto resolved to fp32: CUDA is present but reports no bfloat16 support, "
                    "and fp16 is not selected automatically because it needs loss scaling "
                    "this project has not validated. Request fp16 explicitly to use it."
                ),
                device_type=device_type,
                bf16_supported=bf16_ok,
            )
        return PrecisionPlan(
            name="fp32",
            dtype=torch.float32,
            bf16=False,
            fp16=False,
            requested=requested,
            reason="auto resolved to fp32: no CUDA device, so there is no mixed-precision path",
            device_type=device_type,
            bf16_supported=bf16_ok,
        )

    dtype = resolve_dtype(requested)

    if requested == "bf16" and not bf16_ok:
        raise PrecisionError(
            "bf16 was requested explicitly but this device does not support it "
            f"(device={device_type}, torch.cuda.is_bf16_supported()={bf16_ok}).\n"
            "Refusing to substitute fp16: it has a narrower exponent range, so a run that "
            "silently swapped them could diverge and would look like a model problem.\n"
            "Either run on a bf16-capable GPU -- the measured baseline is an NVIDIA L4 -- or "
            "set precision to 'fp32', or set it to 'auto' to let the device decide."
        )

    if requested == "fp16" and device_type != "cuda":
        raise PrecisionError(
            "fp16 was requested but no CUDA device is available. fp16 training on CPU is not "
            "a supported path. Set precision to 'fp32' or 'auto'."
        )

    return PrecisionPlan(
        name=requested,
        dtype=dtype,
        bf16=requested == "bf16",
        fp16=requested == "fp16",
        requested=requested,
        reason=f"{requested} was requested explicitly and is supported on this device",
        device_type=device_type,
        bf16_supported=bf16_ok,
    )


def describe_device() -> dict[str, Any]:
    """Return what is known about the compute device.

    Safe on a machine with no GPU: every CUDA-specific field is reported as ``None`` rather
    than omitted, so the shape of the record does not depend on the hardware and a diff
    between a laptop run and a Studio run lines up.

    Returns:
        A JSON-serializable mapping.
    """
    available = _cuda_available()
    record: dict[str, Any] = {
        "cuda_available": available,
        "device_type": "cuda" if available else "cpu",
        "torch_version": torch.__version__,
        "bf16_supported": _bf16_supported(),
        "device_count": torch.cuda.device_count() if available else 0,
        "device_name": None,
        "total_vram_gib": None,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if not available:
        return record
    try:
        properties = torch.cuda.get_device_properties(0)
        record["device_name"] = properties.name
        record["total_vram_gib"] = round(properties.total_memory / 1024**3, 3)
    except (AssertionError, RuntimeError):  # pragma: no cover - driver-specific
        pass
    return record
