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
mistake. It is also *mutually exclusive* with ``--plan``, so the two cannot be combined and a
plan command is never one typo away from a four-hour GPU run.

As of Phase 18 that flag really trains. It reads a corpus
:mod:`qa_gen_runtime.prepare` already wrote, runs the preflight in
:mod:`qa_gen_runtime.production`, steps the optimiser over the whole training split and saves
an adapter-only checkpoint under ``<artifacts>/qgen-runs/<run-id>/``. The orchestration lives
in :mod:`qa_gen_runtime.production`; this module is the command line in front of it.

The six-example harness in :mod:`qa_gen_runtime.smoke` is unchanged and is still the right way
to prove the path runs before spending real time on it::

    python -m qa_gen_runtime.smoke --config <config> --inspect-only
    python -m qa_gen_runtime.smoke --config <config> --run

Exit codes
----------
``0`` valid or trained, ``1`` invalid configuration or a runtime problem, ``2`` argparse usage
error, ``3`` a training run the preflight refused or that failed after starting. A CI check can
therefore gate on configuration validity without a GPU, and a training wrapper can tell "this
configuration is wrong" apart from "this run died".
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

#: Exit code for a valid configuration, or a training run that completed.

_EXIT_OK = 0
_EXIT_ERROR = 1

#: Exit code for a run the preflight refused, or that failed after starting. Distinct from a
#: configuration error: the configuration was readable and the corpus was found, and what went
#: wrong is recorded in the run's own report rather than only on stderr.
_EXIT_TRAINING_FAILED = 3


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
        epilog=(
            "The bare command and --plan load no model and write nothing. Only "
            "--execute-training reads a corpus, loads weights and steps the optimiser."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML or JSON experiment configuration.",
    )
    # Mutually exclusive, so "--plan --execute-training" is a usage error that argparse
    # rejects with exit 2 rather than an ambiguity this module has to resolve. A plan command
    # must never be one typo away from a four-hour GPU run.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print the resolved diagnostic report, including the translated trainer arguments. "
            "Still downloads nothing and never trains."
        ),
    )
    mode.add_argument(
        "--execute-training",
        action="store_true",
        help=(
            "Actually train over the prepared corpus. The only flag that loads weights, reads "
            "a dataset or calls trainer.train()."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the run output root. Defaults to <artifacts>/qgen-runs.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "An explicit prepared dataset directory. Wins over --dataset-fingerprint. "
            "Execution only."
        ),
    )
    parser.add_argument(
        "--dataset-fingerprint",
        default=None,
        help=(
            "Select the prepared dataset by corpus fingerprint, and refuse if it does not "
            "match. Use this for any run whose numbers get reported: 'the newest prepared "
            "dataset' changes the moment somebody prepares another one."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which prepared split to train on. Defaults to train.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help=(
            "An optional prepared split to evaluate on. Read only when the configuration's "
            "evaluation_strategy is not 'no'."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the generated run directory name.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Resume from a checkpoint directory. Requires the configuration's save_strategy to "
            "be 'steps' or 'epoch'; with 'no' there are no checkpoints and the run is refused "
            "rather than restarted from zero."
        ),
    )
    parser.add_argument(
        "--expect-train-examples",
        type=int,
        default=None,
        help=(
            "Refuse unless the training split holds exactly this many examples. Turns the "
            "corpus size from an assumption into an assertion."
        ),
    )
    parser.add_argument(
        "--expect-source-count",
        action="append",
        default=[],
        metavar="SOURCE=N",
        help=(
            "Refuse unless the split holds exactly N examples from SOURCE, e.g. "
            "--expect-source-count squad-qg=10000. Repeatable; when given, every source must "
            "be listed."
        ),
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
        A process exit code: ``0`` on success, ``1`` on a configuration or runtime problem,
        ``3`` when a training run was refused by its preflight or failed after starting.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.execute_training:
        return _execute(args)

    try:
        report = run_validation(
            args.config,
            output_dir=args.output_dir,
            train_examples=args.train_examples,
            include_plan=args.plan,
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

    return _EXIT_OK


def _execute(args: argparse.Namespace) -> int:
    """Run production training and print its report.

    Imported lazily so the validation path -- the one that runs on a laptop -- does not pull in
    the training orchestrator and everything under it.

    Args:
        args: The parsed arguments.

    Returns:
        The process exit code.
    """
    from qa_gen_runtime.deps import RuntimeDependencyError
    from qa_gen_runtime.loader import ModelLoadError
    from qa_gen_runtime.outputs import RunOutputError
    from qa_gen_runtime.prepared import PreparedDatasetError
    from qa_gen_runtime.production import (
        ProductionTrainingError,
        execute_production_training,
    )

    try:
        expected_sources = _parse_source_counts(args.expect_source_count)
        report = execute_production_training(
            args.config,
            dataset_dir=args.dataset_dir,
            dataset_fingerprint=args.dataset_fingerprint,
            split=args.split,
            eval_split=args.eval_split,
            output_dir=args.output_dir,
            run_id=args.run_id,
            resume_from_checkpoint=args.resume_from_checkpoint,
            expect_train_examples=args.expect_train_examples,
            expect_source_counts=expected_sources,
            execute=True,
        )
    except ProductionTrainingError as exc:
        # A refused or failed run. The run's own report holds the detail; this is the summary.
        print(f"ProductionTrainingError: {exc}", file=sys.stderr)
        return _EXIT_TRAINING_FAILED
    except (
        ConfigIOError,
        GenerationConfigError,
        ModelLoadError,
        PrecisionError,
        PreparedDatasetError,
        RunOutputError,
        RuntimeDependencyError,
        TrainerBuildError,
    ) as exc:
        # Everything that means "this run could not start": a bad configuration, an absent
        # corpus, a missing dependency, an occupied run directory. Reported as a message rather
        # than a traceback, because none of them is a defect in this code.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(_format_training_text(report))

    return _EXIT_OK if report.success else _EXIT_TRAINING_FAILED


def _parse_source_counts(values: list[str]) -> dict[str, int] | None:
    """Parse repeated ``SOURCE=N`` arguments into a mapping.

    Args:
        values: The raw argument strings.

    Returns:
        The mapping, or ``None`` when nothing was given.

    Raises:
        SystemExit: If an entry is not ``SOURCE=<integer>``. A usage error, and argparse's own
            convention for those is to exit rather than to report.
    """
    if not values:
        return None
    counts: dict[str, int] = {}
    for entry in values:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"--expect-source-count expects SOURCE=N, got {entry!r}.")
        try:
            counts[key.strip()] = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"--expect-source-count expects an integer count, got {entry!r}: {exc}"
            ) from exc
    return counts


def _format_training_text(report: Any) -> str:
    """Render a production training report as readable text.

    Args:
        report: The :class:`~qa_gen_runtime.production.ProductionTrainingReport`.

    Returns:
        The formatted text.
    """
    measurements = report.measurements
    lines = [
        "=" * 74,
        "  QLORA PRODUCTION TRAINING",
        "=" * 74,
        f"run id        : {report.run_id}",
        f"status        : {report.status} (success={report.success})",
        f"configuration : {report.config_path}",
        f"experiment    : {report.experiment} ({report.config_hash})",
        f"model         : {report.model_id} @ {report.model_revision}",
        "",
        "[ DATASET ]",
        f"  directory         {report.dataset_directory}",
        f"  fingerprint       {report.dataset_fingerprint}",
        f"  split             {report.dataset_split}",
        f"  train examples    {report.train_examples:,}",
        f"  validation        {report.validation_examples}",
        f"  test              {report.test_examples}",
        f"  by source         {json.dumps(dict(sorted(report.source_counts.items())))}",
        "",
        "[ SCHEDULE ]",
        f"  sequence length   {report.max_seq_length}",
        f"  batch x accum     {report.batch_size} x {report.gradient_accumulation_steps} "
        f"= {report.effective_batch_size} effective",
        f"  planned steps     {report.total_optimizer_steps}",
        f"  completed steps   {report.completed_optimizer_steps}",
        f"  warmup steps      {report.warmup_steps}",
        f"  checkpointing     save_strategy={report.save_strategy} "
        f"save_steps={report.save_steps} (resumable={report.resumable})",
        f"  resumed from      {report.resumed_from}",
        "",
        "[ PREFLIGHT ]",
        f"  ok                {report.preflight.ok}",
    ]
    for check in report.preflight.checks:
        mark = {True: "pass", False: "FAIL", None: "n/a "}[check.passed]
        lines.append(f"  [{mark}] {check.name}")
    if measurements is not None:
        lines.extend(
            [
                "",
                "[ TRAINING ]",
                f"  wall clock        {measurements.wall_clock_seconds} s",
                f"  seconds/step      {measurements.seconds_per_step}",
                f"  steps/second      {measurements.steps_per_second}",
                f"  final loss        {measurements.final_loss}",
                f"  logged losses     {len(measurements.losses)}",
                f"  optimizer         {measurements.optimizer_class}",
                f"  checkpointing     {measurements.gradient_checkpointing_active}",
                f"  peak allocated    "
                f"{measurements.memory_after.get('max_allocated_gib')} GiB",
                f"  peak reserved     "
                f"{measurements.memory_after.get('max_reserved_gib')} GiB",
            ]
        )
    lines.extend(
        [
            "",
            "[ ADAPTER ]",
            f"  path              {report.adapter_path}",
            f"  bytes             {report.adapter_bytes}",
            f"  trainable params  {report.trainable_parameters:,}"
            if report.trainable_parameters
            else "  trainable params  (unknown)",
        ]
    )
    if report.error:
        lines.extend(
            [
                "",
                "[ ERROR ]",
                f"  type              {report.error.get('type')}",
                f"  message           {report.error.get('message')}",
                f"  completed steps   {report.error.get('completed_optimizer_steps')}",
            ]
        )
    if report.artifacts:
        lines.extend(["", "[ ARTIFACTS ]"])
        lines.extend(f"  {name}: {path}" for name, path in sorted(report.artifacts.items()))
    lines.extend(["", f"overall       : {'ok' if report.success else 'FAILED'}", "=" * 74])
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
