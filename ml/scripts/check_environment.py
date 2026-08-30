#!/usr/bin/env python
"""Environment diagnostics for local development and the Lightning AI GPU.

Reports the interpreter, operating system, key library versions and the full
CUDA picture, then optionally runs a numerical sanity check on the resolved
device.

Runs correctly in both target environments:

- **Local Windows dev machine (CPU only)** - reports ``cuda_available: false``
  and exits 0. A missing GPU is a fact to report, not an error.
- **Lightning AI Studio (NVIDIA L4)** - reports the GPU name, compute
  capability, memory and bfloat16 support.

It never claims a GPU that is not there. Pass ``--require-cuda`` to turn a
missing GPU into a non-zero exit; that form is the hard gate before expensive
training, so a job cannot silently start a multi-day CPU run.

Usage:
    python ml/scripts/check_environment.py
    python ml/scripts/check_environment.py --json
    python ml/scripts/check_environment.py --tensor-test
    python ml/scripts/check_environment.py --require-cuda --tensor-test
    python ml/scripts/check_environment.py --json --output artifacts/env.json

Exit codes:
    0  diagnostics completed (and CUDA present when ``--require-cuda`` given)
    1  ``--require-cuda`` was given but CUDA is unavailable
    2  a required package is missing, or the tensor test failed
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `src/` importable when this script is run directly from a fresh checkout,
# before `pip install -e .` has been done.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Packages whose absence is fatal for the ML pipeline, and those that are not.
_REQUIRED_PACKAGES = ("torch", "transformers", "tokenizers", "datasets")
_OPTIONAL_PACKAGES = ("evaluate", "accelerate", "numpy", "yaml", "fastapi", "pydantic")

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

    Uses distribution metadata rather than importing each package, which keeps
    the diagnostic fast and prevents an unrelated import error from masking the
    report.

    Returns:
        Mapping of module name to version string, or ``None`` when not installed.
    """
    return {
        name: _distribution_version(name)
        for name in (*_REQUIRED_PACKAGES, *_OPTIONAL_PACKAGES)
    }


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


def _cpu_count() -> int | None:
    """Return usable CPU count, preferring the scheduler-affinity view."""
    import os

    if hasattr(os, "sched_getaffinity"):  # Linux: respects cgroup/CPU pinning
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def collect_torch_info() -> dict[str, Any]:
    """Collect PyTorch build and CUDA information.

    Safe on a machine with no GPU: CUDA-specific fields come back as ``None`` or
    empty rather than raising.

    Returns:
        Mapping of torch/CUDA facts, or ``{"available": False, ...}`` if torch
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

    This is an independent cross-check on torch. Torch may report CUDA as
    unavailable while a driver is in fact installed (or the reverse), and knowing
    which of the two is wrong saves considerable debugging time.

    Returns:
        Mapping with ``present`` and, when available, ``driver_version`` and a
        list of GPUs.
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


def run_tensor_test() -> dict[str, Any]:
    """Run a matrix-multiplication sanity check on the resolved device.

    Multiplies two 512x512 matrices on the resolved device and compares against
    the CPU result. On CUDA this exercises the driver, the runtime and kernel
    launch, which catches a broken CUDA installation that merely *reports* as
    available.

    Returns:
        Mapping with ``passed``, the device used, the maximum absolute deviation
        and the tolerance applied. On failure, includes ``error``.
    """
    try:
        import torch

        from qa_torch.device import resolve_device
    except ImportError as exc:
        return {"passed": False, "error": f"torch unavailable: {exc}"}

    try:
        device = resolve_device()
        generator = torch.Generator(device="cpu").manual_seed(0)
        left = torch.randn(512, 512, generator=generator)
        right = torch.randn(512, 512, generator=generator)

        expected = left @ right
        actual = (left.to(device) @ right.to(device)).cpu()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # float32 matmul reassociates differently across backends, so an exact
        # comparison would produce false failures. This tolerance detects a
        # genuinely broken backend without flagging benign reordering.
        tolerance = 1e-3
        max_deviation = (expected - actual).abs().max().item()
        return {
            "passed": bool(max_deviation < tolerance),
            "device": str(device),
            "matrix_shape": [512, 512],
            "max_abs_deviation": float(max_deviation),
            "tolerance": tolerance,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must report, not crash
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build_report(*, tensor_test: bool) -> dict[str, Any]:
    """Assemble the complete diagnostics report.

    Args:
        tensor_test: Whether to include the numerical sanity check.

    Returns:
        Nested mapping suitable for JSON serialization into an experiment record.
    """
    packages = collect_package_versions()
    report: dict[str, Any] = {
        "system": collect_system_info(),
        "packages": packages,
        "missing_required_packages": [
            name for name in _REQUIRED_PACKAGES if packages.get(name) is None
        ],
        "torch": collect_torch_info(),
        "nvidia_smi": collect_nvidia_smi(),
    }
    if tensor_test:
        report["tensor_test"] = run_tensor_test()
    return report


def _format_row(label: str, value: object) -> str:
    """Format one aligned ``label: value`` line for the human-readable report."""
    display = "not installed" if value is None else str(value)
    return f"  {label:<28} {display}"


def print_report(report: dict[str, Any]) -> None:
    """Print the report as a readable table.

    Args:
        report: Report produced by :func:`build_report`.
    """
    system = report["system"]
    packages = report["packages"]
    torch_info = report["torch"]
    smi = report["nvidia_smi"]

    print("=" * 74)
    print("  QAS-NLP ENVIRONMENT DIAGNOSTICS")
    print("=" * 74)

    print("\n[ SYSTEM ]")
    print(_format_row("timestamp (UTC)", system["timestamp_utc"]))
    print(_format_row("python", f"{system['python_version']} ({system['python_implementation']})"))
    print(_format_row("executable", system["python_executable"]))
    print(_format_row("os", f"{system['os']} {system['os_release']}"))
    print(_format_row("platform", system["platform"]))
    print(_format_row("machine", system["machine"]))
    print(_format_row("cpu count", system["cpu_count"]))

    print("\n[ PACKAGES ]")
    for name in (*_REQUIRED_PACKAGES, *_OPTIONAL_PACKAGES):
        required = name in _REQUIRED_PACKAGES
        marker = "*" if required else " "
        print(_format_row(f"{marker} {name}", packages.get(name)))
    print("\n  (* = required for the ML pipeline)")

    print("\n[ TORCH / CUDA ]")
    if not torch_info.get("available"):
        print(_format_row("torch", "NOT IMPORTABLE"))
        print(_format_row("import error", torch_info.get("import_error")))
    else:
        print(_format_row("torch version", torch_info["torch_version"]))
        print(_format_row("built against CUDA", torch_info["torch_cuda_compiled_version"]))
        available = torch_info["cuda_available"]
        print(_format_row("cuda available", "YES" if available else "NO"))
        print(_format_row("cudnn version", torch_info.get("cudnn_version")))
        print(_format_row("device count", torch_info["device_count"]))
        print(_format_row("resolved device", torch_info["resolved_device"]))

        devices = torch_info.get("devices") or []
        if devices:
            for index, device in enumerate(devices):
                print(f"\n  GPU {index}:")
                print(_format_row("    name", device.get("name")))
                print(_format_row("    compute capability", device.get("capability")))
                print(_format_row("    total memory (GiB)", device.get("total_memory_gib")))
                print(_format_row("    bf16 supported", device.get("supports_bf16")))
        else:
            print("\n  No CUDA devices detected. This is expected on the local")
            print("  development machine; training runs on Lightning AI.")

    print("\n[ NVIDIA DRIVER (nvidia-smi cross-check) ]")
    if not smi.get("present"):
        print(_format_row("nvidia-smi", f"unavailable ({smi.get('reason')})"))
    else:
        for index, gpu in enumerate(smi.get("gpus", [])):
            print(f"  GPU {index}: {gpu['name']}")
            print(_format_row("    driver", gpu["driver_version"]))
            print(_format_row("    memory total", gpu["memory_total"]))
            print(_format_row("    memory used", gpu["memory_used"]))
            print(_format_row("    compute capability", gpu["compute_capability"]))

    if "tensor_test" in report:
        test = report["tensor_test"]
        print("\n[ TENSOR SANITY TEST ]")
        print(_format_row("result", "PASSED" if test.get("passed") else "FAILED"))
        print(_format_row("device", test.get("device")))
        if "max_abs_deviation" in test:
            print(_format_row("max abs deviation", f"{test['max_abs_deviation']:.3e}"))
            print(_format_row("tolerance", f"{test['tolerance']:.3e}"))
        if test.get("error"):
            print(_format_row("error", test["error"]))

    missing = report["missing_required_packages"]
    print("\n" + "=" * 74)
    if missing:
        print(f"  RESULT: MISSING REQUIRED PACKAGES -> {', '.join(missing)}")
    elif torch_info.get("available") and torch_info.get("cuda_available"):
        print("  RESULT: CUDA environment ready (GPU training possible)")
    else:
        print("  RESULT: CPU-only environment (expected locally; NOT for training)")
    print("=" * 74)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report the Python/PyTorch/CUDA environment for qas-nlp.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the JSON report to PATH (parent directories are created).",
    )
    parser.add_argument(
        "--tensor-test",
        action="store_true",
        help="Run a matmul sanity check on the resolved device.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if CUDA is unavailable. Use as the pre-training gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    report = build_report(tensor_test=args.tensor_test)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if not args.json:
            print(f"\nJSON report written to: {args.output}")

    if report["missing_required_packages"]:
        print(
            "\nERROR: required package(s) missing: "
            f"{', '.join(report['missing_required_packages'])}",
            file=sys.stderr,
        )
        return 2

    tensor_test = report.get("tensor_test")
    if tensor_test is not None and not tensor_test.get("passed"):
        print("\nERROR: tensor sanity test FAILED.", file=sys.stderr)
        return 2

    if args.require_cuda and not report["torch"].get("cuda_available"):
        print(
            "\nERROR: --require-cuda was specified but CUDA is unavailable.\n"
            "Refusing to continue. Training must run on the Lightning AI L4, not on CPU.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
