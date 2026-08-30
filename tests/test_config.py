"""Tests for the YAML experiment configuration system."""

from __future__ import annotations

import pytest

from qa_ml.config import (
    ConfigError,
    ExperimentConfig,
    PreprocessingConfig,
    TrainingConfig,
    load_experiment_config,
    resolve_config_mapping,
)
from qa_ml.paths import get_paths

# Every shipped experiment config, discovered rather than hard-coded so a new
# config file cannot be added without also being validated here.
EXPERIMENT_CONFIG_NAMES = [
    "experiment_a_distilbert.yaml",
    "experiment_b_bert_base.yaml",
    "experiment_c_roberta_base.yaml",
    "experiment_d_deberta_v3.yaml",
]


class TestShippedConfigs:
    """Every configuration file in the repository must load and validate."""

    def test_all_yaml_files_are_covered_by_tests(self):
        """Guard against an untested config being added later."""
        on_disk = {path.name for path in get_paths().configs.glob("*.yaml")}
        expected = {"base.yaml", "smoke.yaml", *EXPERIMENT_CONFIG_NAMES}
        assert on_disk == expected, (
            "ml/configs contents changed. Update EXPERIMENT_CONFIG_NAMES so the "
            "new config is validated by the test suite."
        )

    @pytest.mark.parametrize("filename", EXPERIMENT_CONFIG_NAMES)
    def test_experiment_config_loads(self, filename):
        config = load_experiment_config(filename)
        assert isinstance(config, ExperimentConfig)
        assert config.name
        assert config.model_name

    def test_smoke_config_loads_and_is_capped(self):
        config = load_experiment_config("smoke.yaml")
        assert config.data.max_train_samples is not None
        assert config.data.max_eval_samples is not None
        assert config.training.num_train_epochs == 1

    def test_bare_filename_resolves_from_any_working_directory(self):
        """Config lookup must not depend on the caller's cwd."""
        assert load_experiment_config("base.yaml").name == "base"

    @pytest.mark.parametrize("filename", EXPERIMENT_CONFIG_NAMES)
    def test_experiments_share_controlled_variables(self, filename):
        """The comparison is only valid if the controlled variables really match."""
        base = load_experiment_config("base.yaml")
        config = load_experiment_config(filename)

        assert config.seed == base.seed
        assert config.preprocessing.max_seq_length == base.preprocessing.max_seq_length
        assert config.preprocessing.doc_stride == base.preprocessing.doc_stride
        assert config.decoding.n_best_size == base.decoding.n_best_size
        assert config.decoding.max_answer_length == base.decoding.max_answer_length
        assert config.training.learning_rate == base.training.learning_rate
        assert config.training.num_train_epochs == base.training.num_train_epochs
        assert config.data.dataset_name == base.data.dataset_name

    @pytest.mark.parametrize("filename", EXPERIMENT_CONFIG_NAMES)
    def test_effective_batch_size_is_equal_across_experiments(self, filename):
        """DeBERTa halves batch size but doubles accumulation; totals must match."""
        base = load_experiment_config("base.yaml")
        config = load_experiment_config(filename)
        assert (
            config.training.effective_batch_size == base.training.effective_batch_size
        )

    def test_experiments_use_distinct_models(self):
        models = {
            load_experiment_config(name).model_name for name in EXPERIMENT_CONFIG_NAMES
        }
        assert len(models) == len(EXPERIMENT_CONFIG_NAMES)

    def test_default_precision_is_fp32(self):
        """Mixed precision must be benchmarked on the real GPU before being enabled."""
        assert load_experiment_config("base.yaml").training.precision == "fp32"


class TestInheritance:
    """`extends` composition behaviour."""

    def test_child_overrides_parent(self):
        config = load_experiment_config("smoke.yaml")
        assert config.training.num_train_epochs == 1  # overridden
        assert config.preprocessing.max_seq_length == 384  # inherited

    def test_partial_section_override_keeps_sibling_keys(self):
        """Overriding one key in a section must not wipe the rest of it."""
        config = load_experiment_config("experiment_d_deberta_v3.yaml")
        assert config.training.batch_size == 8  # overridden
        assert config.training.weight_decay == 0.01  # inherited
        assert config.training.lr_scheduler_type == "linear"  # inherited

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_experiment_config("does_not_exist.yaml")


class TestConfigHash:
    """Deterministic config hashing."""

    def test_hash_is_stable_across_calls(self):
        config = load_experiment_config("experiment_a_distilbert.yaml")
        assert config.config_hash() == config.config_hash()

    def test_identical_configs_hash_identically(self):
        first = load_experiment_config("experiment_a_distilbert.yaml")
        second = load_experiment_config("experiment_a_distilbert.yaml")
        assert first.config_hash() == second.config_hash()

    def test_different_configs_hash_differently(self):
        a = load_experiment_config("experiment_a_distilbert.yaml")
        b = load_experiment_config("experiment_b_bert_base.yaml")
        assert a.config_hash() != b.config_hash()

    def test_any_value_change_changes_the_hash(self):
        base = load_experiment_config("experiment_a_distilbert.yaml")
        tweaked = load_experiment_config(
            "experiment_a_distilbert.yaml",
            overrides={"training": {"learning_rate": 5.0e-5}},
        )
        assert base.config_hash() != tweaked.config_hash()

    def test_hash_length_is_configurable(self):
        config = load_experiment_config("base.yaml")
        assert len(config.config_hash(8)) == 8
        assert len(config.config_hash(64)) == 64

    def test_run_id_embeds_name_model_and_hash(self):
        config = load_experiment_config("experiment_a_distilbert.yaml")
        run_id = config.run_id("20260829T120000Z")
        assert config.name in run_id
        assert config.config_hash() in run_id
        assert "20260829T120000Z" in run_id

    def test_run_id_is_filesystem_safe(self):
        """Model ids contain '/', which must not create nested directories."""
        config = load_experiment_config("experiment_d_deberta_v3.yaml")
        assert "/" in config.model_name
        assert "/" not in config.run_id("20260829T120000Z")


class TestOverrides:
    """Programmatic override merging."""

    def test_override_applies(self):
        config = load_experiment_config(
            "experiment_a_distilbert.yaml",
            overrides={"training": {"num_train_epochs": 5}},
        )
        assert config.training.num_train_epochs == 5

    def test_override_is_validated(self):
        with pytest.raises(ConfigError, match="must be positive"):
            load_experiment_config(
                "experiment_a_distilbert.yaml",
                overrides={"training": {"num_train_epochs": 0}},
            )


class TestStrictKeyRejection:
    """Typos must fail loudly, never be silently ignored."""

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ConfigError, match="Unknown top-level"):
            resolve_config_mapping({"name": "x", "modle_name": "distilbert-base-uncased"})

    def test_unknown_nested_key_rejected(self):
        with pytest.raises(ConfigError, match="Unknown key"):
            resolve_config_mapping({"name": "x", "training": {"learing_rate": 3e-5}})

    def test_missing_name_rejected(self):
        with pytest.raises(ConfigError, match="`name`"):
            resolve_config_mapping({"model_name": "distilbert-base-uncased"})

    def test_non_mapping_section_rejected(self):
        with pytest.raises(ConfigError, match="must be a mapping"):
            resolve_config_mapping({"name": "x", "training": "fast"})


class TestPreprocessingValidation:
    """Semantic validation of :class:`PreprocessingConfig`."""

    def test_stride_must_be_smaller_than_max_seq_length(self):
        with pytest.raises(ConfigError, match="must be smaller than"):
            PreprocessingConfig(max_seq_length=384, doc_stride=384).validate()

    def test_question_length_must_leave_room_for_context(self):
        # doc_stride is lowered as well, because the stride check runs first and
        # would otherwise be the error that fires.
        with pytest.raises(ConfigError, match="no room is left"):
            PreprocessingConfig(
                max_seq_length=128, doc_stride=32, max_question_length=128
            ).validate()

    def test_stride_check_precedes_question_length_check(self):
        """Documents validation ordering, so the messages above stay predictable."""
        with pytest.raises(ConfigError, match="must be smaller than"):
            PreprocessingConfig(
                max_seq_length=128, doc_stride=128, max_question_length=128
            ).validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_seq_length": 0},
            {"doc_stride": 0},
            {"max_question_length": -1},
            {"padding": "sometimes"},
            {"pad_to_multiple_of": 0},
        ],
    )
    def test_rejects_out_of_range_values(self, kwargs):
        with pytest.raises(ConfigError):
            PreprocessingConfig(**kwargs).validate()

    def test_valid_defaults_pass(self):
        PreprocessingConfig().validate()


class TestTrainingValidation:
    """Semantic validation of :class:`TrainingConfig`."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rate": 0},
            {"learning_rate": -1e-5},
            {"batch_size": 0},
            {"eval_batch_size": 0},
            {"gradient_accumulation_steps": 0},
            {"num_train_epochs": 0},
            {"weight_decay": -0.1},
            {"warmup_ratio": 1.5},
            {"warmup_ratio": -0.1},
            {"precision": "int8"},
            {"lr_scheduler_type": "magic"},
            {"evaluation_strategy": "sometimes"},
            {"save_strategy": "sometimes"},
            {"logging_strategy": "sometimes"},
            {"save_total_limit": 0},
            {"dataloader_num_workers": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ConfigError):
            TrainingConfig(**kwargs).validate()

    def test_best_model_requires_matching_strategies(self):
        """Otherwise the 'best' checkpoint may never have been written to disk."""
        with pytest.raises(ConfigError, match="match evaluation_strategy"):
            TrainingConfig(
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                save_strategy="steps",
            ).validate()

    def test_best_model_requires_evaluation_enabled(self):
        with pytest.raises(ConfigError, match="'steps' or 'epoch'"):
            TrainingConfig(
                load_best_model_at_end=True,
                evaluation_strategy="no",
                save_strategy="no",
            ).validate()

    def test_effective_batch_size_multiplies_accumulation(self):
        config = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 32

    @pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
    def test_accepts_supported_precisions(self, precision):
        TrainingConfig(precision=precision).validate()


class TestDataValidation:
    """Semantic validation of dataset settings."""

    def test_rejects_non_positive_sample_cap(self):
        with pytest.raises(ConfigError, match="positive integer or null"):
            load_experiment_config(
                "smoke.yaml", overrides={"data": {"max_train_samples": 0}}
            )

    def test_null_sample_cap_means_full_split(self):
        config = load_experiment_config("experiment_a_distilbert.yaml")
        assert config.data.max_train_samples is None
        assert config.data.max_eval_samples is None


class TestTokenizerFallback:
    """`tokenizer_name` defaults to `model_name`."""

    def test_defaults_to_model_name(self):
        config = load_experiment_config("experiment_a_distilbert.yaml")
        assert config.tokenizer_name is None
        assert config.effective_tokenizer_name == config.model_name

    def test_explicit_tokenizer_is_respected(self):
        config = load_experiment_config(
            "experiment_a_distilbert.yaml",
            overrides={"tokenizer_name": "bert-base-uncased"},
        )
        assert config.effective_tokenizer_name == "bert-base-uncased"
