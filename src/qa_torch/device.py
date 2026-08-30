"""Device resolution and CUDA diagnostics.

Design rules
------------
- No ``"cuda:0"`` anywhere. A bare ``"cuda"`` lets torch use its current device,
  which keeps the code correct on multi-GPU hosts and under launchers that pin a
  device via ``CUDA_VISIBLE_DEVICES``.
- Resolution order is: explicit argument, then ``QAS_DEVICE``, then automatic
  (cuda, mps, cpu).
- Diagnostics never raise merely because CUDA is absent. The local development
  machine is CPU-only and must be able to run every diagnostic.
- A CPU result is never disguised as a GPU result. Callers that require a GPU
  ask for it explicitly via :func:`require_cuda`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import torch

__all__ = [
    "DeviceInfo",
    "DeviceUnavailableError",
    "collect_cuda_diagnostics",
    "describe_device",
    "require_cuda",
    "resolve_device",
]

logger = logging.getLogger(__name__)

_VALID_DEVICE_TYPES = ("cuda", "mps", "cpu")
_DEVICE_ENV_VAR = "QAS_DEVICE"


class DeviceUnavailableError(RuntimeError):
    """Raised when a specifically requested device is not usable."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Description of the compute device actually selected.

    Attributes:
        type: Resolved device type: ``"cuda"``, ``"mps"`` or ``"cpu"``.
        name: Human-readable device name, e.g. ``"NVIDIA L4"``.
        index: CUDA device index, or ``None`` for non-CUDA devices.
        total_memory_bytes: Total device memory, when discoverable.
        capability: CUDA compute capability as ``(major, minor)``.
        supports_bf16: Whether bfloat16 is reported as supported.
        is_cuda: Convenience flag; ``True`` only for a real CUDA device.
    """

    type: str
    name: str
    index: int | None = None
    total_memory_bytes: int | None = None
    capability: tuple[int, int] | None = None
    supports_bf16: bool | None = None
    is_cuda: bool = False

    @property
    def total_memory_gib(self) -> float | None:
        """Total device memory in GiB, or ``None`` when unknown."""
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / (1024**3)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for experiment records."""
        payload = asdict(self)
        payload["total_memory_gib"] = (
            round(self.total_memory_gib, 2) if self.total_memory_gib is not None else None
        )
        if self.capability is not None:
            payload["capability"] = f"{self.capability[0]}.{self.capability[1]}"
        return payload


def _mps_available() -> bool:
    """Return whether an Apple Metal device is present and usable."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - platform-specific defensive path
        return False


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the compute device to use.

    Args:
        requested: Explicit device type (``"cuda"``, ``"mps"`` or ``"cpu"``).
            When ``None``, the ``QAS_DEVICE`` environment variable is consulted,
            and if that is unset or empty the best available device is chosen
            automatically.

    Returns:
        The selected :class:`torch.device`.

    Raises:
        ValueError: If an unrecognized device type is requested.
        DeviceUnavailableError: If a specific device is requested but not
            available. Requesting CUDA on a machine without it is an error
            rather than a silent downgrade to CPU, because a silent downgrade
            would turn a GPU training run into a multi-day CPU run.

    Examples:
        >>> resolve_device("cpu").type
        'cpu'
    """
    if requested is None:
        env_value = os.environ.get(_DEVICE_ENV_VAR, "").strip()
        requested = env_value or None

    if requested is not None:
        choice = requested.strip().lower()
        if choice not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"Unknown device type {requested!r}. "
                f"Expected one of: {', '.join(_VALID_DEVICE_TYPES)}."
            )
        if choice == "cuda" and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Refusing to fall back to CPU silently. Run "
                "`python ml/scripts/check_environment.py` to diagnose."
            )
        if choice == "mps" and not _mps_available():
            raise DeviceUnavailableError(
                "MPS was requested but is not available on this machine."
            )
        logger.debug("Using explicitly requested device: %s", choice)
        return torch.device(choice)

    # Automatic selection. Note the bare "cuda" - no hard-coded index.
    if torch.cuda.is_available():
        logger.debug("Auto-selected CUDA device.")
        return torch.device("cuda")
    if _mps_available():
        logger.debug("Auto-selected MPS device.")
        return torch.device("mps")
    logger.debug("No accelerator found; auto-selected CPU.")
    return torch.device("cpu")


def describe_device(device: torch.device | str | None = None) -> DeviceInfo:
    """Collect descriptive facts about a device.

    Args:
        device: Device to describe. When ``None``, :func:`resolve_device` is
            called first.

    Returns:
        A populated :class:`DeviceInfo`. Fields that cannot be determined on the
        current platform are left as ``None`` rather than guessed.
    """
    if device is None:
        device = resolve_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return DeviceInfo(type=device.type, name=device.type.upper(), is_cuda=False)

    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    try:
        supports_bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - depends on torch build
        supports_bf16 = None

    return DeviceInfo(
        type="cuda",
        name=properties.name,
        index=index,
        total_memory_bytes=properties.total_memory,
        capability=(properties.major, properties.minor),
        supports_bf16=supports_bf16,
        is_cuda=True,
    )


def collect_cuda_diagnostics() -> dict[str, object]:
    """Gather CUDA facts for the environment report.

    Safe to call on a CPU-only machine: every CUDA-specific field is reported as
    ``None`` or an empty list rather than raising.

    Returns:
        Mapping with ``cuda_available``, ``cuda_version``, ``cudnn_version``,
        ``device_count`` and a ``devices`` list of per-GPU descriptions.
    """
    available = torch.cuda.is_available()
    diagnostics: dict[str, object] = {
        "cuda_available": available,
        "cuda_compiled_version": torch.version.cuda,
        "cudnn_version": None,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": [],
    }

    if not available:
        return diagnostics

    try:
        diagnostics["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover - depends on torch build
        diagnostics["cudnn_version"] = None

    devices = []
    for index in range(torch.cuda.device_count()):
        devices.append(describe_device(torch.device("cuda", index)).as_dict())
    diagnostics["devices"] = devices
    return diagnostics


def require_cuda() -> torch.device:
    """Return a CUDA device or fail loudly.

    Used as a hard gate before expensive training runs, so that a missing GPU
    stops the job immediately instead of starting an unusably slow CPU run.

    Returns:
        A CUDA :class:`torch.device`.

    Raises:
        DeviceUnavailableError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "CUDA is required for this operation but is not available. "
            "Expected environment: Lightning AI Studio with an NVIDIA L4. "
            "Run `python ml/scripts/check_environment.py --require-cuda` first."
        )
    return torch.device("cuda")
