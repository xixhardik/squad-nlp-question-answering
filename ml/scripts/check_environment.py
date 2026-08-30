#!/usr/bin/env python
"""Environment diagnostics for local development and the Lightning AI GPU.

A thin command-line front end over :mod:`qa_ml.environment`, which is also what the
training pipeline embeds in every experiment record. Sharing one implementation
means the report a human reads and the metadata a run stores cannot disagree.

Runs correctly in both target environments:

- **Local Windows dev machine (CPU only)** - reports ``cuda_available: false`` and
  exits 0. A missing GPU is a fact to report, not an error.
- **Lightning AI Studio (NVIDIA L4)** - reports the GPU name, compute capability,
  memory and bfloat16 support.

It never claims a GPU that is not there. Pass ``--require-cuda`` to turn a missing
GPU into a non-zero exit; that form is the hard gate before expensive training, so a
job cannot silently start a multi-day CPU run.

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
import json
import sys
from pathlib import Path
from typing import Any

# Make `src/` importable when this script is run directly from a fresh checkout,
# before `pip install -e .` has been done.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qa_ml.environment import (  # noqa: E402
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    collect_environment,
)


def _format_row(label: str, value: object) -> str:
    """Format one aligned ``label: value`` line for the human-readable report."""
    display = "not installed" if value is None else str(value)
    return f"  {label:<28} {display}"


def print_report(report: dict[str, Any]) -> None:
    """Print the report as a readable table.

    Args:
        report: Report produced by :func:`qa_ml.environment.collect_environment`.
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

    git = report.get("git")
    if git:
        print("\n[ GIT PROVENANCE ]")
        if not git.get("available"):
            print(_format_row("git", f"unavailable ({git.get('reason')})"))
        else:
            print(_format_row("commit", git.get("commit_short")))
            print(_format_row("branch", git.get("branch")))
            dirty = git.get("dirty")
            suffix = f" ({git.get('dirty_file_count')} modified file(s))" if dirty else ""
            print(_format_row("working tree", ("DIRTY" if dirty else "clean") + suffix))
            if dirty:
                print("\n  WARNING: the working tree is modified, so a run started now")
                print("  would NOT be reproducible from the recorded commit.")

    print("\n[ PACKAGES ]")
    for name in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES):
        marker = "*" if name in REQUIRED_PACKAGES else " "
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
    report = collect_environment(tensor_test=args.tensor_test)

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
