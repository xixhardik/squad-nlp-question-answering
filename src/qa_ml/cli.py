"""Command-line interface for the SQuAD question answering pipeline.

Four subcommands, run as ``python -m qa_ml <command>``:

``prepare``
    Download SQuAD 1.1, validate its schema and answer offsets, build features and
    report statistics. Does not train. Run it first: it turns dataset and
    preprocessing problems into a fast, cheap failure instead of a wasted GPU hour.

``train``
    Fine-tune a model and write a complete experiment record.

``evaluate``
    Score a saved checkpoint on a split with Exact Match and F1.

``predict``
    Answer a single question about a passage.

Every command reads an experiment YAML, so training length, model and preprocessing
can be changed without editing source. ``--set`` applies ad-hoc overrides for
one-offs.

Errors are raised as :class:`SystemExit` with an explanatory message rather than a
traceback, because the common failures here are operational (no GPU, no network, a run
directory that already exists) and the fix belongs in the message.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from qa_ml.config import ConfigError, ExperimentConfig, load_experiment_config
from qa_ml.logging_utils import configure_logging

logger = logging.getLogger("qa_ml.cli")

__all__ = ["build_parser", "main"]


def _parse_override(item: str) -> tuple[list[str], Any]:
    """Parse a ``dotted.key=value`` override into a path and a typed value.

    Values are parsed as JSON when possible, so ``training.learning_rate=5e-5`` becomes
    a float and ``output.allow_existing=true`` becomes a bool, while bare words stay
    strings.

    Args:
        item: The ``key=value`` string.

    Returns:
        A ``(key_path, value)`` pair.

    Raises:
        ValueError: If ``item`` has no ``=``.
    """
    if "=" not in item:
        raise ValueError(
            f"Override {item!r} must be of the form dotted.key=value, "
            "e.g. training.num_train_epochs=1"
        )
    key, raw_value = item.split("=", 1)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key.split("."), value


def _apply_overrides(items: list[str] | None) -> dict[str, Any]:
    """Build a nested override mapping from ``--set`` arguments."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        path, value = _parse_override(item)
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return overrides


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load and validate the experiment config named on the command line."""
    try:
        return load_experiment_config(args.config, overrides=_apply_overrides(args.set))
    except (ConfigError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def command_prepare(args: argparse.Namespace) -> int:
    """Download and validate SQuAD, then build and summarise features."""
    from qa_ml.data import (
        assert_squad_v1,
        load_squad_splits,
        summarize_split,
        verify_answer_offsets,
    )
    from qa_ml.experiment import write_json
    from qa_ml.paths import get_paths
    from qa_ml.preprocess import build_train_features, build_validation_features
    from qa_torch.features import SquadFeatureBuilder
    from qa_torch.loader import load_tokenizer

    config = _load_config(args)
    capped = (
        config.data.max_train_samples is not None
        or config.data.max_eval_samples is not None
    )
    report: dict[str, Any] = {
        "config": args.config,
        "dataset_name": config.data.dataset_name,
        "dataset_version": config.data.dataset_version,
        "model_name": config.model_name,
        "max_seq_length": config.preprocessing.max_seq_length,
        "doc_stride": config.preprocessing.doc_stride,
        # Recorded so a sample-capped report is self-identifying. Without this a
        # smoke report showing 256 examples looks like a characterisation of the
        # full 87,599-example split.
        "sample_capped": capped,
        "max_train_samples": config.data.max_train_samples,
        "max_eval_samples": config.data.max_eval_samples,
    }

    try:
        splits = load_squad_splits(config.data, seed=config.seed)
    except Exception as exc:
        raise SystemExit(f"Dataset error: {exc}") from exc

    print("=" * 74)
    print("  SQuAD 1.1 PREPARATION")
    print("=" * 74)
    if capped:
        print(
            f"\n  NOTE: sample caps are active "
            f"(train={config.data.max_train_samples}, "
            f"eval={config.data.max_eval_samples}).\n"
            "  These figures describe a SUBSET, not the full SQuAD 1.1 splits."
        )

    report["splits"] = {}
    for name, dataset in splits.items():
        assert_squad_v1(dataset, name)
        summary = summarize_split(dataset, name)
        offsets = verify_answer_offsets(
            dataset, sample_size=None if args.verify_all else 2000
        )
        report["splits"][name] = {
            "summary": summary.as_dict(),
            "offset_verification": offsets.as_dict(),
        }

        print(f"\n[ {name.upper()} ]")
        print(f"  examples            {summary.num_examples:,}")
        print(f"  distinct titles     {summary.num_titles:,}")
        print(f"  context chars       min={summary.context_chars[0]} "
              f"mean={summary.context_chars[1]:.1f} max={summary.context_chars[2]}")
        print(f"  question chars      min={summary.question_chars[0]} "
              f"mean={summary.question_chars[1]:.1f} max={summary.question_chars[2]}")
        print(f"  answers/example     min={summary.answers_per_example[0]} "
              f"mean={summary.answers_per_example[1]:.2f} "
              f"max={summary.answers_per_example[2]}")
        print(f"  offsets checked     {offsets.checked:,}")
        print(f"  offsets exact       {offsets.exact_matches:,} "
              f"({100 * offsets.exact_match_rate:.3f}%)")
        print(f"  whitespace-only     {offsets.whitespace_only_mismatches:,}")
        print(f"  real mismatches     {offsets.mismatches:,}")
        if offsets.mismatch_samples:
            print("  sample mismatches:")
            for sample in offsets.mismatch_samples[:3]:
                print(f"    id={sample['id']} answer={sample['answer_text']!r} "
                      f"slice={sample['context_slice']!r}")

    if args.build_features:
        tokenizer = load_tokenizer(config.effective_tokenizer_name)
        builder = SquadFeatureBuilder(
            tokenizer,
            max_seq_length=config.preprocessing.max_seq_length,
            doc_stride=config.preprocessing.doc_stride,
            max_question_length=config.preprocessing.max_question_length,
            padding=config.preprocessing.padding,
        )
        train_bundle = build_train_features(splits["train"], builder, num_proc=args.num_proc)
        validation_bundle = build_validation_features(
            splits["validation"], builder, num_proc=args.num_proc
        )
        report["features"] = {
            "train": train_bundle.as_dict(),
            "validation": validation_bundle.as_dict(),
        }

        print("\n[ FEATURES ]")
        for label, bundle in (
            ("train", train_bundle.as_dict()),
            ("validation", validation_bundle.as_dict()),
        ):
            alignment = bundle["alignment"]
            print(f"  {label}:")
            print(f"    examples          {bundle['num_examples']:,}")
            print(f"    features          {bundle['num_features']:,}")
            print(f"    features/example  {bundle['features_per_example']}")
            print(f"    aligned           {alignment['aligned']:,} "
                  f"({100 * alignment['aligned_fraction']:.2f}%)")
            print(f"    outside window    {alignment['answer_outside_window']:,}")
            print(f"    NO aligned window {alignment['examples_with_no_aligned_feature']:,}")

    output = args.output or (get_paths().reports / "dataset_preparation.json")
    write_json(Path(output), report)
    print(f"\nReport written to: {output}")
    print("=" * 74)
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def command_train(args: argparse.Namespace) -> int:
    """Fine-tune a model according to the experiment config."""
    from qa_ml.experiment import ExperimentExistsError
    from qa_ml.train import run_training
    from qa_torch.device import DeviceUnavailableError

    config = _load_config(args)
    try:
        outcome = run_training(
            config,
            allow_cpu=args.allow_cpu,
            resume=args.resume,
            run_id=args.run_id,
            num_proc=args.num_proc,
            cross_check=args.cross_check,
        )
    except DeviceUnavailableError as exc:
        raise SystemExit(f"GPU error: {exc}") from exc
    except ExperimentExistsError as exc:
        raise SystemExit(f"Experiment error: {exc}") from exc

    print("\n" + "=" * 74)
    print(f"  RUN {outcome.run_id}")
    print("=" * 74)
    print(f"  run directory   {outcome.run_dir}")
    print(f"  checkpoint      {outcome.checkpoint_dir}")
    print(f"  reproducible    {outcome.record.is_reproducible}")
    if outcome.evaluation is not None:
        print(f"  {outcome.evaluation.summary_line()}")
    print("=" * 74)
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def command_evaluate(args: argparse.Namespace) -> int:
    """Score a saved checkpoint on a SQuAD split."""
    from qa_ml.evaluate import run_evaluation

    config = _load_config(args)
    try:
        result = run_evaluation(
            config,
            args.model,
            split=args.split,
            batch_size=args.batch_size,
            cross_check=not args.no_cross_check,
            num_proc=args.num_proc,
            save_predictions_to=Path(args.predictions) if args.predictions else None,
        )
    except Exception as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    print("\n" + "=" * 74)
    print(f"  EVALUATION: {args.model} on {args.split}")
    print("=" * 74)
    print(f"  Exact Match            {result.exact_match:.4f}")
    print(f"  F1                     {result.f1:.4f}")
    print(f"  examples               {result.total_examples:,}")
    print(f"  features               {result.total_features:,}")
    print(f"  examples w/o answer    {result.examples_without_answer:,}")
    if result.validation_loss is not None:
        print(f"  validation loss        {result.validation_loss:.6f}")
    if result.cross_check:
        if result.cross_check.get("available"):
            print(f"  evaluate cross-check   EM={result.cross_check['exact_match']:.4f} "
                  f"F1={result.cross_check['f1']:.4f} "
                  f"agrees={result.cross_check['agrees']}")
        else:
            print(f"  evaluate cross-check   unavailable "
                  f"({result.cross_check.get('reason')})")
    print("=" * 74)
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def command_predict(args: argparse.Namespace) -> int:
    """Answer a single question about a passage."""
    from qa_torch.inference import ExtractiveQAEngine

    config = _load_config(args)

    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    if not context or not context.strip():
        raise SystemExit(
            "A context is required. Pass --context 'text' or --context-file path.txt"
        )
    if not args.question or not args.question.strip():
        raise SystemExit("A question is required. Pass --question 'text'")

    try:
        engine = ExtractiveQAEngine(
            args.model,
            max_seq_length=config.preprocessing.max_seq_length,
            doc_stride=config.preprocessing.doc_stride,
            max_question_length=config.preprocessing.max_question_length,
            n_best_size=config.decoding.n_best_size,
            max_answer_length=config.decoding.max_answer_length,
            max_n_best=config.decoding.max_n_best,
            score_type=config.decoding.score_type,
        )
        result = engine.answer(args.question, context)
    except Exception as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print("\n" + "=" * 74)
    print("  EXTRACTIVE QUESTION ANSWERING")
    print("=" * 74)
    print(f"\n  question   {args.question}")
    print(f"\n  answer     {result.answer!r}")
    print(f"  span       characters [{result.char_start}, {result.char_end})")
    print(f"  score      {result.score:.4f}  ({result.score_type})")
    print(f"  latency    {result.latency_ms:.1f} ms")
    print(f"  windows    {result.num_windows}")
    print(f"  model      {result.model_id}")

    if result.has_answer:
        # Show the answer in context, which is the point of extractive QA: the answer
        # is a location in the passage, not generated text.
        before = context[max(0, result.char_start - 120) : result.char_start]
        after = context[result.char_end : result.char_end + 120]
        print("\n  in context:")
        print(f"    ...{before}[[{result.answer}]]{after}...")
    else:
        print("\n  No valid answer span was found in the supplied context.")

    if len(result.n_best) > 1:
        print("\n  alternatives:")
        for span in result.n_best[1:4]:
            print(f"    {span['score']:.4f}  {span['answer']!r}")
    print("=" * 74)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, *, default_config: str) -> None:
    """Add arguments shared by every subcommand."""
    parser.add_argument(
        "--config",
        default=default_config,
        help=f"Experiment YAML in ml/configs, or a path (default: {default_config}).",
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.num_train_epochs=1. Repeatable.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Worker processes for dataset preprocessing. Leave unset on Windows.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m qa_ml",
        description="SQuAD 1.1 extractive question answering pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Download and validate SQuAD 1.1, and summarise features."
    )
    _add_common(prepare, default_config="experiment_a_distilbert.yaml")
    prepare.add_argument(
        "--build-features",
        action="store_true",
        help="Also tokenize both splits and report feature/alignment statistics.",
    )
    prepare.add_argument(
        "--verify-all",
        action="store_true",
        help="Verify answer offsets on every example instead of a 2,000-row sample.",
    )
    prepare.add_argument("--output", default=None, help="Where to write the JSON report.")
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="Fine-tune a QA model.")
    _add_common(train, default_config="experiment_a_distilbert.yaml")
    train.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit training without CUDA. Only sensible for a tiny smoke config.",
    )
    train.add_argument(
        "--resume", action="store_true", help="Resume from the newest checkpoint."
    )
    train.add_argument("--run-id", default=None, help="Override the generated run id.")
    train.add_argument(
        "--cross-check",
        action="store_true",
        help="Also score the final evaluation with the `evaluate` library.",
    )
    train.set_defaults(func=command_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score a checkpoint with Exact Match and F1."
    )
    _add_common(evaluate_parser, default_config="experiment_a_distilbert.yaml")
    evaluate_parser.add_argument(
        "--model", required=True, help="Checkpoint directory or Hugging Face model id."
    )
    evaluate_parser.add_argument(
        "--split", default="validation", choices=["train", "validation"]
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=None)
    evaluate_parser.add_argument(
        "--no-cross-check", action="store_true", help="Skip the `evaluate` comparison."
    )
    evaluate_parser.add_argument(
        "--predictions", default=None, help="Write a per-example prediction dump here."
    )
    evaluate_parser.set_defaults(func=command_evaluate)

    predict = subparsers.add_parser("predict", help="Answer one question.")
    _add_common(predict, default_config="experiment_a_distilbert.yaml")
    predict.add_argument(
        "--model", required=True, help="Checkpoint directory or Hugging Face model id."
    )
    predict.add_argument("--question", required=True)
    predict.add_argument("--context", default=None, help="The passage text.")
    predict.add_argument(
        "--context-file", default=None, help="Read the passage from this file."
    )
    predict.add_argument("--json", action="store_true", help="Emit JSON.")
    predict.set_defaults(func=command_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
