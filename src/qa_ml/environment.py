"""Environment and provenance capture for reproducible experiments.

The canonical implementation of "what machine, what versions, what commit".
``ml/scripts/check_environment.py`` is a thin CLI over this module, and the training
pipeline embeds the same output in every experiment record, so the diagnostics a
human reads and the metadata a run stores can never disagree.

Nothing here raises merely because CUDA is absent. The local development machine is
CPU-only and must be able to produce a complete report; a missing GPU is a fact to
record, not an error. Callers that *require* a GPU ask for it explicitly.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "OPTIONAL_PACKAGES",
    "REQUIRED_PACKAGES",
    "collect_environment",
    "collect_git_info",
    "collect_nvidia_smi",
    "collect_package_versions",
    "collect_system_info",
    "collect_torch_info",
    "missing_required_packages",
    "run_tensor_test",
]

#: Packages whose absence makes the ML pipeline unrunnable.
REQUIRED_PACKAGES = ("torch", "transformers", "tokenizers", "datasets")

#: Packages that are recorded for provenance but are not fatal if absent.
OPTIONAL_PACKAGES = (
    "evaluate",
    "accelerate",
    "numpy",
    "yaml",
    "fastapi",
    "pydantic",
    "sentencepiece",
)

# Import name -> distribution name, where they differ.
_PACKAGE_TO_DISTRIBUTION = {"yaml": "PyYAML"}


def _distribution_version(module_name: str) -> str | None:
    """Return an installed package version without importing the package."""
    distribution = _PACKAGE_TO_DISTRIBUTION.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_package_versions() -> dict[str, str | None]:
    """Collect versions of every relevant package.

    Reads distribution metadata rather than importing each package, which keeps the
    call fast and stops an unrelated import error from masking the whole report.

    Returns:
        Mapping of module name to version string, or ``None`` when not installed.
    """
    return {
        name: _distribution_version(name)
        for name in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES)
    }


def missing_required_packages(
    packages: dict[str, str | None] | None = None,
) -> list[str]:
    """Return the required packages that are not installed.

    Args:
        packages: Result of :func:`collect_package_versions`. Collected if omitted.

    Returns:
        Names of missing required packages, empty when all are present.
    """
    packages = packages if packages is not None else collect_package_versions()
    return [name for name in REQUIRED_PACKAGES if packages.get(name) is None]


def _cpu_count() -> int | None:
    """Return usable CPU count, preferring the scheduler-affinity view.

    On Linux this respects cgroup CPU pinning, which matters inside a Lightning
    Studio container where ``os.cpu_count()`` reports the host's cores.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def collect_system_info() -> dict[str, Any]:
    """Collect interpreter and operating system facts."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": _cpu_count(),
    }


def collect_torch_info() -> dict[str, Any]:
    """Collect PyTorch build and CUDA information.

    Safe on a machine with no GPU: CUDA-specific fields come back as ``None`` or
    empty rather than raising.

    Returns:
        Mapping of torch/CUDA facts, or ``{"available": False, ...}`` when torch
        cannot be imported at all.
    """
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "import_error": str(exc)}

    from qa_torch.device import collect_cuda_diagnostics, describe_device, resolve_device

    info: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "torch_cuda_compiled_version": torch.version.cuda,
        "torch_git_version": getattr(torch.version, "git_version", None),
    }
    info.update(collect_cuda_diagnostics())

    device = resolve_device()
    info["resolved_device"] = str(device)
    info["resolved_device_info"] = describe_device(device).as_dict()
    return info


def collect_nvidia_smi() -> dict[str, Any]:
    """Query the NVIDIA driver via ``nvidia-smi``, if present.

    An independent cross-check on torch. Torch may report CUDA unavailable while a
    driver is in fact installed (or the reverse), and knowing which of the two is
    wrong saves considerable debugging time.

    Returns:
        Mapping with ``present`` and, when available, per-GPU details.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return {"present": False, "reason": "nvidia-smi not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"present": False, "reason": "nvidia-smi timed out"}

    if completed.returncode != 0:
        return {
            "present": False,
            "reason": f"nvidia-smi exited {completed.returncode}",
            "stderr": completed.stderr.strip()[:400] or None,
        }

    gpus = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total": parts[2],
                    "memory_used": parts[3],
                    "compute_capability": parts[4],
                }
            )
    return {"present": True, "gpus": gpus}


def collect_git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Capture the git commit a run was launched from.

    ``dirty`` is the important field. A run started from a modified working tree is
    **not reproducible**: the recorded commit does not describe the code that
    actually executed. Recording the flag lets such a run be labelled honestly
    rather than quietly trusted.

    Args:
        repo_root: Repository directory. Discovered from the package location when
            omitted.

    Returns:
        Mapping with ``commit``, ``branch``, ``dirty`` and ``available``.
    """
    if repo_root is None:
        try:
            from qa_ml.paths import find_repo_root

            repo_root = find_repo_root()
        except Exception:  # pragma: no cover - only outside a checkout
            return {"available": False, "reason": "repository root not found"}

    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "reason": "not a git repository or git unavailable"}

    status = _git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_file_count": len(status.splitlines()) if status else 0,
    }


def run_tensor_test(matrix_size: int = 512, tolerance: float = 1e-3) -> dict[str, Any]:
    """Run a matrix-multiplication sanity check on the resolved device.

    Multiplies two matrices on the resolved device and compares against the CPU
    result. On CUDA this exercises the driver, the runtime and kernel launch, which
    catches a broken CUDA installation that merely *reports* as available.

    Args:
        matrix_size: Side length of the square matrices.
        tolerance: Maximum acceptable absolute deviation. float32 matmul
            reassociates differently across backends, so an exact comparison would
            produce false failures; this detects a genuinely broken backend without
            flagging benign reordering.

    Returns:
        Mapping with ``passed``, the device used, the maximum deviation and the
        tolerance. On failure, includes ``error``.
    """
    try:
        import torch

        from qa_torch.device import resolve_device
    except ImportError as exc:
        return {"passed": False, "error": f"torch unavailable: {exc}"}

    try:
        device = resolve_device()
        generator = torch.Generator(device="cpu").manual_seed(0)
        left = torch.randn(matrix_size, matrix_size, generator=generator)
        right = torch.randn(matrix_size, matrix_size, generator=generator)

        expected = left @ right
        actual = (left.to(device) @ right.to(device)).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()

        max_deviation = (expected - actual).abs().max().item()
        return {
            "passed": bool(max_deviation < tolerance),
            "device": str(device),
            "matrix_shape": [matrix_size, matrix_size],
            "max_abs_deviation": float(max_deviation),
            "tolerance": tolerance,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must report, not crash
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_environment(
    *, tensor_test: bool = False, include_git: bool = True
) -> dict[str, Any]:
    """Assemble the full environment report.

    Args:
        tensor_test: Include the numerical sanity check.
        include_git: Include commit provenance.

    Returns:
        Nested mapping suitable for JSON serialization into an experiment record.
    """
    packages = collect_package_versions()
    report: dict[str, Any] = {
        "system": collect_system_info(),
        "packages": packages,
        "missing_required_packages": missing_required_packages(packages),
        "torch": collect_torch_info(),
        "nvidia_smi": collect_nvidia_smi(),
    }
    if include_git:
        report["git"] = collect_git_info()
    if tensor_test:
        report["tensor_test"] = run_tensor_test()
    return report
