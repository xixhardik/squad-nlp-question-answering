"""Training a SQuAD 1.1 extractive question answering model.

Built on ``transformers.Trainer`` (which delegates to Accelerate) rather than a
hand-rolled loop, because checkpointing, resumption, mixed precision, gradient
accumulation and LR scheduling are exactly the parts where a bespoke loop tends to be
subtly wrong. The parts this project is meant to demonstrate -- character/token
alignment, sliding windows and span decoding -- are hand-written and live in
:mod:`qa_core` and :mod:`qa_torch`.

Exact Match and F1 during training
----------------------------------
``Trainer`` computes eval loss on its own, but loss is a poor model-selection signal
for extractive QA: it scores token positions, while the task is judged on decoded
text. So a ``compute_metrics`` closure over the validation bundle decodes real answers
and returns EM and F1, which is what ``metric_for_best_model`` then selects on.

This relies on the evaluation loop yielding predictions in dataset order so they line
up with the stored offsets. That holds for single-device evaluation, and
:func:`qa_ml.evaluate.decode_all_examples` asserts the count matches rather than
trusting it.

Config-to-TrainingArguments translation
---------------------------------------
The project's config vocabulary is deliberately its own, and two mappings are not
one-to-one on transformers 5.16.1:

- ``evaluation_strategy`` -> ``eval_strategy``. The old name was removed in v5.
- ``warmup_ratio`` -> ``warmup_steps`` **as a float**. ``warmup_ratio`` was removed;
  ``warmup_steps`` is typed ``float`` and a value below 1 is treated as a fraction of
  total steps. Verified: ``TrainingArguments(warmup_steps=0.1).get_warmup_steps(1000)``
  returns ``100``.

Keeping our names stable means a library rename cannot invalidate committed
experiment records.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, default_data_collator

from qa_ml.config import ExperimentConfig
from qa_ml.data import assert_squad_v1, load_squad_splits, summarize_split, verify_answer_offsets
from qa_ml.evaluate import EvaluationResult, build_prediction_dump, evaluate_from_logits
from qa_ml.experiment import (
    CONFIG_FILENAME,
    PREDICTIONS_FILENAME,
    ExperimentRecord,
    create_run_directory,
    resolve_run_root,
    utc_timestamp,
    write_json,
)
from qa_ml.preprocess import build_train_features, build_validation_features
from qa_ml.seeding import set_global_seed
from qa_torch.device import (
    DeviceUnavailableError,
    describe_device,
    resolve_device,
    resolve_precision,
)
from qa_torch.features import SquadFeatureBuilder
from qa_torch.loader import describe_model, load_qa_model, load_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["TrainingOutcome", "build_training_arguments", "run_training"]


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """What a completed training run produced.

    Attributes:
        run_id: Unique run identifier.
        run_dir: Directory holding the checkpoint, record and predictions.
        checkpoint_dir: Directory of the saved best model.
        record: The full experiment record.
        evaluation: Final evaluation result, when evaluation was enabled.
    """

    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    record: ExperimentRecord
    evaluation: EvaluationResult | None


def build_training_arguments(
    config: ExperimentConfig,
    run_dir: Path,
    *,
    bf16: bool,
    fp16: bool,
    use_cpu: bool,
) -> TrainingArguments:
    """Translate the experiment config into ``TrainingArguments``.

    Args:
        config: The experiment configuration.
        run_dir: Directory for checkpoints and trainer state.
        bf16: Enable bfloat16 mixed precision.
        fp16: Enable float16 mixed precision.
        use_cpu: Force CPU execution.

    Returns:
        The populated :class:`~transformers.TrainingArguments`.
    """
    training = config.training
    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy=training.evaluation_strategy,
        save_strategy=training.save_strategy,
        logging_strategy=training.logging_strategy,
        logging_steps=training.logging_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        # A float below 1 is interpreted as a fraction of total steps; see module docs.
        warmup_steps=float(training.warmup_ratio),
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_grad_norm=training.max_grad_norm,
        lr_scheduler_type=training.lr_scheduler_type,
        bf16=bf16,
        fp16=fp16,
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=training.full_determinism,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=training.metric_for_best_model,
        greater_is_better=training.greater_is_better,
        save_total_limit=training.save_total_limit,
        dataloader_num_workers=training.dataloader_num_workers,
        gradient_checkpointing=training.gradient_checkpointing,
        use_cpu=use_cpu,
        # Our datasets contain exactly the model inputs, so nothing should be dropped;
        # an unexpected column is a bug worth surfacing rather than silently removing.
        remove_unused_columns=False,
        label_names=["start_positions", "end_positions"],
        report_to=[],
        disable_tqdm=False,
    )


def run_training(
    config: ExperimentConfig,
    *,
    allow_cpu: bool = False,
    resume: bool = False,
    run_id: str | None = None,
    num_proc: int | None = None,
    cross_check: bool = False,
) -> TrainingOutcome:
    """Fine-tune a question answering model end to end.

    Args:
        config: The experiment configuration.
        allow_cpu: Permit training without CUDA. Off by default so a missing GPU stops
            the job instead of starting an unusably slow CPU run.
        resume: Continue from the newest checkpoint in the run directory.
        run_id: Override the generated run identifier.
        num_proc: Worker processes for dataset preprocessing.
        cross_check: Also score with the ``evaluate`` library and report agreement.

    Returns:
        The :class:`TrainingOutcome`.

    Raises:
        DeviceUnavailableError: If CUDA is unavailable and ``allow_cpu`` is ``False``.
    """
    device = resolve_device()
    if device.type != "cuda" and not allow_cpu:
        raise DeviceUnavailableError(
            "CUDA is not available and --allow-cpu was not passed.\n"
            "Training on CPU would take days rather than hours, so it is refused by "
            "default.\n"
            "Options:\n"
            "  - run this on the Lightning AI L4 Studio (the intended environment)\n"
            "  - verify the GPU first: python ml/scripts/check_environment.py "
            "--require-cuda\n"
            "  - pass --allow-cpu to run a tiny smoke config locally anyway"
        )

    precision = resolve_precision(config.training.precision, device)
    logger.info(
        "Device=%s precision=%s (%s)", device, precision.resolved, precision.reason
    )

    seeding = set_global_seed(config.seed, full_determinism=config.training.full_determinism)

    run_identifier = run_id or config.run_id(utc_timestamp())
    run_root = resolve_run_root(config)
    run_dir = create_run_directory(
        run_root, run_identifier, allow_existing=resume or config.output.allow_existing
    )
    logger.info("Run directory: %s", run_dir)

    record = ExperimentRecord.create(config, run_identifier)
    record.seeding = seeding
    record.precision = precision.as_dict()
    if not record.is_reproducible:
        logger.warning(
            "The git working tree is dirty or unavailable, so this run is marked "
            "non-reproducible in its experiment record."
        )

    # Persist the resolved config immediately: if the run crashes later, the directory
    # still explains exactly what was attempted.
    write_json(run_dir / CONFIG_FILENAME.replace(".yaml", ".json"), config.to_dict())

    try:
        tokenizer = load_tokenizer(config.effective_tokenizer_name)
        model = load_qa_model(config.model_name)
        record.model = describe_model(model, config.model_name)
        record.environment.setdefault("device", describe_device(device).as_dict())

        splits = load_squad_splits(config.data, seed=config.seed)
        for name, split_dataset in splits.items():
            assert_squad_v1(split_dataset, name)

        offsets_report = verify_answer_offsets(splits["train"], sample_size=2000)
        record.dataset = {
            "dataset_name": config.data.dataset_name,
            "dataset_version": config.data.dataset_version,
            "splits": {
                name: summarize_split(split_dataset, name).as_dict()
                for name, split_dataset in splits.items()
            },
            "train_offset_verification": offsets_report.as_dict(),
        }
        logger.info(
            "Answer offsets: %.4f exact, %.4f usable after whitespace tightening",
            offsets_report.exact_match_rate,
            offsets_report.usable_rate,
        )

        builder = SquadFeatureBuilder(
            tokenizer,
            max_seq_length=config.preprocessing.max_seq_length,
            doc_stride=config.preprocessing.doc_stride,
            max_question_length=config.preprocessing.max_question_length,
            padding=config.preprocessing.padding,
            pad_to_multiple_of=config.preprocessing.pad_to_multiple_of,
        )

        train_bundle = build_train_features(splits["train"], builder, num_proc=num_proc)
        validation_bundle = build_validation_features(
            splits["validation"], builder, num_proc=num_proc
        )
        record.preprocessing = {
            "train": train_bundle.as_dict(),
            "validation": validation_bundle.as_dict(),
            "max_seq_length": config.preprocessing.max_seq_length,
            "doc_stride": config.preprocessing.doc_stride,
        }

        def compute_metrics(eval_prediction: Any) -> dict[str, float]:
            """Decode predicted text and score Exact Match and F1.

            Closes over ``validation_bundle`` so the offsets used for decoding are the
            ones produced with these exact features.
            """
            start_logits, end_logits = eval_prediction.predictions[:2]
            result = evaluate_from_logits(
                validation_bundle.eval_bundle,
                start_logits.tolist(),
                end_logits.tolist(),
                config.decoding,
                cross_check=False,
            )
            return {"exact_match": result.exact_match, "f1": result.f1}

        args = build_training_arguments(
            config,
            run_dir,
            bf16=precision.bf16,
            fp16=precision.fp16,
            use_cpu=device.type == "cpu",
        )

        callbacks = []
        if config.training.early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience
                )
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_bundle.dataset,
            eval_dataset=validation_bundle.labelled_dataset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks or None,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        train_result = trainer.train(
            resume_from_checkpoint=config.training.resume_from_checkpoint or resume or None
        )
        train_seconds = time.perf_counter() - started

        num_features = len(train_bundle.dataset)
        record.training = {
            "train_runtime_seconds": round(train_seconds, 3),
            "train_samples_per_second": round(
                num_features * config.training.num_train_epochs / max(train_seconds, 1e-9), 3
            ),
            "num_train_features": num_features,
            "num_train_examples": train_bundle.num_examples,
            "effective_batch_size": config.training.effective_batch_size,
            "epochs": config.training.num_train_epochs,
            "final_train_loss": round(float(train_result.training_loss), 6)
            if train_result.training_loss is not None
            else None,
            "log_history": trainer.state.log_history,
        }
        if device.type == "cuda":
            record.training["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            record.training["peak_gpu_memory_gib"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )

        checkpoint_dir = run_dir / "model"
        trainer.save_model(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        record.checkpoint_path = str(checkpoint_dir)
        logger.info("Saved model and tokenizer to %s", checkpoint_dir)

        # Final evaluation, with the cross-check if requested. Run explicitly rather
        # than reusing the last in-training eval so the recorded numbers unambiguously
        # belong to the saved checkpoint.
        eval_started = time.perf_counter()
        prediction_output = trainer.predict(
            validation_bundle.labelled_dataset, metric_key_prefix="final"
        )
        inference_seconds = time.perf_counter() - eval_started
        start_logits, end_logits = prediction_output.predictions[:2]
        evaluation = evaluate_from_logits(
            validation_bundle.eval_bundle,
            start_logits.tolist(),
            end_logits.tolist(),
            config.decoding,
            validation_loss=prediction_output.metrics.get("final_loss"),
            cross_check=cross_check,
        )

        eval_features = validation_bundle.eval_bundle.num_features
        record.evaluation = {
            **evaluation.as_dict(),
            "inference_seconds": round(inference_seconds, 3),
            "inference_ms_per_example": round(
                1000.0 * inference_seconds / max(evaluation.total_examples, 1), 3
            ),
            "inference_features_per_second": round(
                eval_features / max(inference_seconds, 1e-9), 2
            ),
        }

        if config.output.save_predictions:
            write_json(
                run_dir / PREDICTIONS_FILENAME,
                build_prediction_dump(evaluation, validation_bundle.eval_bundle),
            )

        record.mark_completed()
        record.save(run_dir)
        logger.info("Training complete. %s", evaluation.summary_line())

        return TrainingOutcome(
            run_id=run_identifier,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            record=record,
            evaluation=evaluation,
        )

    except BaseException as exc:
        # A failed experiment is still a result. Record it before re-raising so the
        # run directory never becomes an unexplained gap in the evidence trail.
        record.mark_failed(exc)
        record.save(run_dir)
        logger.exception("Training failed; record written to %s", run_dir)
        raise


def estimate_total_steps(
    num_features: int,
    config: ExperimentConfig,
    *,
    world_size: int = 1,
) -> int:
    """Estimate optimizer steps for a run, for budget forecasting.

    Args:
        num_features: Training features (not examples: windowing expands the count).
        config: The experiment configuration.
        world_size: Number of devices.

    Returns:
        Estimated total optimizer steps.
    """
    training = config.training
    per_step = (
        training.per_device_train_batch_size
        * training.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(num_features / max(per_step, 1))
    return steps_per_epoch * training.num_train_epochs
