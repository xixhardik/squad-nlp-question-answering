"""The training entry point, which does not train by default.

Safe by default, deliberately
-----------------------------
::

    python -m qa_gen_runtime.train --config ml/configs/qgen/qgen-smoke.yaml
    python -m qa_gen_runtime.train --config ... --plan
    python -m qa_gen_runtime.train --config ... --execute-training

The bare command **validates only**. It parses the configuration, resolves precision against
this device, translates the quantization and trainer arguments, and prints a diagnostic report.
It downloads nothing, allocates nothing on the GPU, creates no directory and writes no file.
That is what makes it usable as a pre-flight check on a laptop, and as the first thing to run
after editing a configuration on the Studio.

Nothing happens by accident because the expensive path is opt-in. ``--execute-training`` is
required to reach it, and it is spelled out rather than abbreviated so it cannot be typed by
mistake. In this phase it refuses with a clear message: the trainer *factory* is implemented and
tested, and wiring ``trainer.train()`` to it is Phase 17B.2. A flag that half-trained would be
worse than one that says so.

Exit codes
----------
``0`` valid, ``1`` invalid configuration or a runtime problem, ``2`` argparse usage error.
A CI check can therefore gate on configuration validity without a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from qa_gen.config import VERIFIED_QWEN3_4B_L4, GenerationConfigError
from qa_gen_runtime.config_io import ConfigIOError, load_experiment_config
from qa_gen_runtime.dataset import resolve_record_format
from qa_gen_runtime.diagnostics import collect_diagnostics
from qa_gen_runtime.outputs import resolve_run_root
from qa_gen_runtime.precision import PrecisionError, resolve_precision
from qa_gen_runtime.trainer import TrainerBuildError, plan_trainer_arguments

logger = logging.getLogger(__name__)

__all__ = ["build_parser", "main", "run_validation"]

_EXIT_OK = 0
_EXIT_ERROR = 1

#: Message for ``--execute-training`` in this phase. Stated as a phase boundary rather than as
#: a failure, because the runtime being complete and the loop being unwired is the intended
#: state of the repository right now.
_TRAINING_NOT_WIRED = (
    "Training execution is not wired up in Phase 17B.1.\n"
    "This phase implements and tests the runtime: the model loader, the k-bit preparation and "
    "LoRA attachment, the dataset construction, the trainer factory and the diagnostics. "
    "Calling trainer.train() on a real corpus is a later phase.\n"
    "Everything except the training call can be exercised now with --plan.\n"
    "To take a bounded number of real optimiser steps over six hand-written examples, use the "
    "Phase 17B.2 smoke harness instead:\n"
    "  python -m qa_gen_runtime.smoke --config <config> --inspect-only\n"
    "  python -m qa_gen_runtime.smoke --config <config> --run"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser. Separated from :func:`main` so the tests can assert the defaults without
        running anything -- in particular that validation, not training, is the default.
    """
    parser = argparse.ArgumentParser(
        prog="qa_gen_runtime.train",
        description=(
            "Validate or run a QLoRA question-generation fine-tune. Validates only unless "
            "--execute-training is given."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML or JSON experiment configuration.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print the resolved diagnostic report, including the translated trainer arguments. "
            "Still downloads nothing."
        ),
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help=(
            "Actually train. Required to leave validation mode; not implemented in Phase "
            "17B.1."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the run output root. Defaults to <artifacts>/qgen-runs.",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=None,
        help=(
            "Size of the training split, used to convert warmup_ratio into warmup_steps "
            "(transformers 5.x removed warmup_ratio)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON rather than as text.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Defaults to INFO.",
    )
    return parser


def run_validation(
    config_path: str,
    *,
    output_dir: str | None = None,
    train_examples: int | None = None,
    include_plan: bool = False,
) -> dict[str, Any]:
    """Validate a configuration and assemble its report. No model, no network, no writes.

    Args:
        config_path: Path to the configuration file.
        output_dir: Override the run output root. Resolved but not created.
        train_examples: Training split size, for the warmup conversion.
        include_plan: Also translate the trainer arguments. Kept optional because the
            translation can raise when a warmup ratio has no derivable step count, and a plain
            validity check should not fail for that reason.

    Returns:
        A JSON-serializable report.

    Raises:
        qa_gen_runtime.config_io.ConfigIOError: If the file cannot be read.
        qa_gen.config.GenerationConfigError: If the configuration is invalid.
        qa_gen_runtime.precision.PrecisionError: If the requested precision is unavailable here.
        qa_gen_runtime.trainer.TrainerBuildError: If the trainer arguments cannot be translated.
    """
    config = load_experiment_config(config_path)

    precision = resolve_precision(config.training.precision)
    run_root = resolve_run_root(output_dir)

    trainer_plan: dict[str, Any] | None = None
    if include_plan:
        plan = plan_trainer_arguments(
            config,
            run_root / "<run-id>",
            train_examples=train_examples,
            precision=precision,
        )
        trainer_plan = plan.as_dict()

    diagnostics = collect_diagnostics(
        config,
        precision=precision,
        trainer_plan=trainer_plan,
        extra_notes=(
            "configuration validated without loading a model or a dataset",
            f"record format: {resolve_record_format(config).value}",
        ),
    )

    return {
        "status": "valid",
        "config_path": config_path,
        "experiment": config.name,
        "config_hash": config.config_hash(),
        "run_root": run_root.as_posix(),
        "run_root_created": False,
        "record_format": resolve_record_format(config).value,
        "is_verified_configuration": config.is_verified_configuration(VERIFIED_QWEN3_4B_L4),
        "baseline_deviations": list(config.baseline_deviations(VERIFIED_QWEN3_4B_L4)),
        "diagnostics": diagnostics.as_dict(),
    }


def _format_text(report: dict[str, Any]) -> str:
    """Render a report as readable text."""
    diagnostics = report["diagnostics"]
    device = diagnostics["device"]
    lines = [
        f"configuration : {report['config_path']}",
        f"experiment    : {report['experiment']} ({report['config_hash']})",
        f"model         : {diagnostics['model_id']} @ {diagnostics['model_revision']}",
        f"quantization  : {diagnostics['quantization']}",
        f"precision     : {(diagnostics['precision'] or {}).get('name')} "
        f"({(diagnostics['precision'] or {}).get('reason')})",
        f"lora          : r={diagnostics['lora']['rank']} "
        f"alpha={diagnostics['lora']['alpha']} "
        f"modules={len(diagnostics['lora']['target_modules'])}",
        f"sequence      : {diagnostics['sequence_length']} tokens",
        f"batch         : {diagnostics['batch_size']} x "
        f"{diagnostics['gradient_accumulation_steps']} accum "
        f"= {diagnostics['effective_batch_size']} effective",
        f"checkpointing : {diagnostics['gradient_checkpointing']}",
        f"optimizer     : {diagnostics['optimizer']} (not benchmarked)",
        f"record format : {report['record_format']}",
        f"device        : {device.get('device_name') or device.get('device_type')} "
        f"(bf16 supported: {device.get('bf16_supported')})",
        f"run root      : {report['run_root']} (not created)",
        f"verified      : {report['is_verified_configuration']}",
    ]
    if report["baseline_deviations"]:
        lines.append("deviations    :")
        lines.extend(f"  - {item}" for item in report["baseline_deviations"])
    missing = [
        name
        for name, detail in diagnostics["dependencies"].items()
        if not detail["available"]
    ]
    if missing:
        lines.append(f"missing deps  : {', '.join(missing)} (needed to train, not to validate)")
    if diagnostics.get("trainer_plan"):
        plan = diagnostics["trainer_plan"]
        lines.append(f"total steps   : {plan.get('total_steps')}")
        lines.append(f"warmup steps  : {plan.get('warmup_steps')}")
        if plan.get("dropped_arguments"):
            lines.append(f"dropped args  : {', '.join(plan['dropped_arguments'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a configuration or runtime problem.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        report = run_validation(
            args.config,
            output_dir=args.output_dir,
            train_examples=args.train_examples,
            include_plan=args.plan or args.execute_training,
        )
    except (
        ConfigIOError,
        GenerationConfigError,
        PrecisionError,
        TrainerBuildError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(_format_text(report))

    if args.execute_training:
        print(f"\n{_TRAINING_NOT_WIRED}", file=sys.stderr)
        return _EXIT_ERROR

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
