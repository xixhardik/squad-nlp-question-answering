"""Tests for the config-to-TrainingArguments adapter and training guards.

No training happens here. The adapter is where a library rename silently changes
behaviour, so it is tested directly against the installed ``transformers``.
"""

from __future__ import annotations

import pytest
from transformers import TrainingArguments

from qa_ml.config import load_experiment_config
from qa_ml.train import build_training_arguments, estimate_total_steps
from qa_torch.device import DeviceUnavailableError


@pytest.fixture
def config():
    return load_experiment_config("smoke.yaml")


@pytest.fixture
def args(config, tmp_path) -> TrainingArguments:
    return build_training_arguments(
        config, tmp_path, bf16=False, fp16=False, use_cpu=True
    )


class TestTrainingArgumentsAdapter:
    def test_builds_without_error(self, args):
        assert isinstance(args, TrainingArguments)

    def test_output_dir_is_inside_the_run_directory(self, config, tmp_path):
        args = build_training_arguments(
            config, tmp_path, bf16=False, fp16=False, use_cpu=True
        )
        assert str(tmp_path) in args.output_dir
        assert args.output_dir.endswith("checkpoints")

    def test_maps_evaluation_strategy_to_eval_strategy(self, config, args):
        """`evaluation_strategy` was removed in transformers v5; ours is translated.

        The attribute is an enum, so compare by value rather than by ``str()``.
        """
        actual = getattr(args.eval_strategy, "value", args.eval_strategy)
        assert str(actual).lower() == config.training.evaluation_strategy

    def test_save_and_logging_strategies_are_carried_through(self, config, args):
        save = getattr(args.save_strategy, "value", args.save_strategy)
        logging_strategy = getattr(args.logging_strategy, "value", args.logging_strategy)
        assert str(save).lower() == config.training.save_strategy
        assert str(logging_strategy).lower() == config.training.logging_strategy
        assert args.logging_steps == config.training.logging_steps

    def test_maps_warmup_ratio_to_float_warmup_steps(self, config, tmp_path):
        """`warmup_ratio` was removed; `warmup_steps` accepts a fraction as a float.

        Verified against the installed library rather than assumed: with
        ``warmup_steps=0.1``, ``get_warmup_steps(1000)`` must return 100.
        """
        overridden = load_experiment_config(
            "smoke.yaml", overrides={"training": {"warmup_ratio": 0.1}}
        )
        args = build_training_arguments(
            overridden, tmp_path, bf16=False, fp16=False, use_cpu=True
        )
        assert args.warmup_steps == pytest.approx(0.1)
        assert args.get_warmup_steps(1000) == 100
        assert args.get_warmup_steps(5000) == 500

    def test_zero_warmup_yields_no_warmup_steps(self, config, tmp_path):
        args = build_training_arguments(
            load_experiment_config("smoke.yaml", overrides={"training": {"warmup_ratio": 0.0}}),
            tmp_path,
            bf16=False,
            fp16=False,
            use_cpu=True,
        )
        assert args.get_warmup_steps(1000) == 0

    def test_label_names_identify_the_qa_labels(self, args):
        """Trainer needs these to compute loss and to populate EvalPrediction."""
        assert args.label_names == ["start_positions", "end_positions"]

    def test_unused_columns_are_not_removed(self, args):
        """Our datasets hold exactly the model inputs; a stray column is a bug."""
        assert args.remove_unused_columns is False

    def test_batch_sizes_are_carried_through(self, config, args):
        assert args.per_device_train_batch_size == config.training.per_device_train_batch_size
        assert args.per_device_eval_batch_size == config.training.per_device_eval_batch_size

    def test_optimizer_settings_are_carried_through(self, config, args):
        assert args.learning_rate == config.training.learning_rate
        assert args.weight_decay == config.training.weight_decay
        assert args.max_grad_norm == config.training.max_grad_norm
        assert args.num_train_epochs == config.training.num_train_epochs
        assert args.gradient_accumulation_steps == config.training.gradient_accumulation_steps

    def test_seed_is_carried_through(self, config, args):
        assert args.seed == config.seed
        assert args.data_seed == config.seed

    def test_best_model_selection_uses_f1(self, config, args):
        assert args.metric_for_best_model == config.training.metric_for_best_model
        assert args.greater_is_better is config.training.greater_is_better
        assert args.load_best_model_at_end is config.training.load_best_model_at_end

    def test_checkpoint_retention_is_bounded(self, config, args):
        assert args.save_total_limit == config.training.save_total_limit

    def test_no_external_reporting_by_default(self, args):
        """JSON records are the source of truth; no tracker is required."""
        assert args.report_to == []

    def test_fp32_flags_are_off(self, config, tmp_path):
        args = build_training_arguments(
            config, tmp_path, bf16=False, fp16=False, use_cpu=True
        )
        assert args.bf16 is False
        assert args.fp16 is False

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="Requires CUDA."
    )
    @pytest.mark.parametrize(("bf16", "fp16"), [(True, False), (False, True)])
    def test_mixed_precision_flags_are_passed_through(self, config, tmp_path, bf16, fp16):
        """TrainingArguments validates mixed precision against the real device.

        It rejects ``bf16=True`` with ``use_cpu=False`` on hardware without bf16
        support, so this can only be checked on a GPU.
        """
        import torch

        if bf16 and not torch.cuda.is_bf16_supported():
            pytest.skip("bf16 not supported on this GPU.")
        args = build_training_arguments(
            config, tmp_path, bf16=bf16, fp16=fp16, use_cpu=False
        )
        assert args.bf16 is bf16
        assert args.fp16 is fp16

    def test_use_cpu_is_honoured(self, config, tmp_path):
        args = build_training_arguments(
            config, tmp_path, bf16=False, fp16=False, use_cpu=True
        )
        assert args.use_cpu is True
        assert args.device.type == "cpu"


class TestEstimateTotalSteps:
    def test_single_epoch_single_device(self, config):
        overridden = load_experiment_config(
            "smoke.yaml",
            overrides={
                "training": {
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 1,
                }
            },
        )
        assert estimate_total_steps(80, overridden) == 10

    def test_rounds_a_partial_batch_up(self, config):
        overridden = load_experiment_config(
            "smoke.yaml",
            overrides={
                "training": {
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 1,
                }
            },
        )
        assert estimate_total_steps(81, overridden) == 11

    def test_gradient_accumulation_reduces_optimizer_steps(self):
        base = load_experiment_config(
            "smoke.yaml",
            overrides={
                "training": {
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 1,
                }
            },
        )
        accumulated = load_experiment_config(
            "smoke.yaml",
            overrides={
                "training": {
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "num_train_epochs": 1,
                }
            },
        )
        # Same effective batch size, so the same number of optimizer steps.
        assert estimate_total_steps(80, base) == estimate_total_steps(80, accumulated)

    def test_epochs_multiply(self):
        overridden = load_experiment_config(
            "smoke.yaml",
            overrides={
                "training": {
                    "per_device_train_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 3,
                }
            },
        )
        assert estimate_total_steps(80, overridden) == 30


class TestCudaGate:
    @pytest.mark.skipif(
        __import__("torch").cuda.is_available(), reason="Requires a machine without CUDA."
    )
    def test_training_refuses_cpu_by_default(self, config):
        """A silent CPU fallback would turn a GPU run into a multi-day CPU run."""
        from qa_ml.train import run_training

        with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
            run_training(config, allow_cpu=False)

    @pytest.mark.skipif(
        __import__("torch").cuda.is_available(), reason="Requires a machine without CUDA."
    )
    def test_error_explains_how_to_proceed(self, config):
        from qa_ml.train import run_training

        with pytest.raises(DeviceUnavailableError) as excinfo:
            run_training(config, allow_cpu=False)
        message = str(excinfo.value)
        assert "--allow-cpu" in message
        assert "check_environment.py" in message
