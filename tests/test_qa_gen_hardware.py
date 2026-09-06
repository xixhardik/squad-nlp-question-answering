"""Tests binding the generation configuration to the measured hardware baseline.

Why this file exists separately
------------------------------
The quantization, adapter and sequence-length defaults in :mod:`qa_gen.config` are only
defensible because a feasibility run demonstrated them on an NVIDIA L4. That evidence is
recorded in :data:`qa_gen.config.VERIFIED_QWEN3_4B_L4`, and these tests hold the defaults to
it -- so a default cannot drift away from the measurement without a failure, and the
measurement cannot be edited to match a changed default without someone noticing that is what
they did.

The measurement's *limits* are asserted as carefully as its results. The most important test
here is the one stating that 2048 is not verified, because that is the assumption most likely
to be made silently.

Nothing here needs a GPU, a checkpoint or a network. The measurement is data; this file checks
that the configuration agrees with it.
"""

from __future__ import annotations

import json

import pytest

from qa_gen import (
    DEFAULT_BASE_MODEL,
    VERIFIED_MAX_SEQ_LENGTH,
    VERIFIED_QWEN3_4B_L4,
    GenerationConfigError,
    GeneratorModelConfig,
    LoRAConfig,
    MeasuredBaseline,
    TrainingConfig,
    TrainingRunMetadata,
    experiment_config_from_dict,
)

BASELINE = VERIFIED_QWEN3_4B_L4

#: The seven projections the feasibility run adapted.
MEASURED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TestMeasuredBaselineRecord:
    """The measurement itself, as recorded."""

    def test_the_hardware_is_recorded(self):
        assert BASELINE.gpu == "NVIDIA L4"
        assert BASELINE.total_vram_gib == 22.034
        assert BASELINE.cuda_version == "13.0"
        assert BASELINE.torch_version == "2.13.0+cu130"
        assert BASELINE.bf16_supported is True

    def test_the_model_and_quantization_are_recorded(self):
        assert BASELINE.model_id == "Qwen/Qwen3-4B"
        assert BASELINE.quantization == "4bit"
        assert BASELINE.quantization_type == "nf4"
        assert BASELINE.double_quantization is True
        assert BASELINE.compute_dtype == "bf16"

    def test_the_adapter_configuration_is_recorded(self):
        assert BASELINE.lora_rank == 16
        assert BASELINE.lora_alpha == 32
        assert BASELINE.lora_dropout == 0.05
        assert BASELINE.target_modules == MEASURED_TARGET_MODULES

    def test_the_parameter_counts_are_recorded(self):
        assert BASELINE.total_parameters == 4_055_498_240
        assert BASELINE.trainable_parameters == 33_030_144

    def test_the_trainable_fraction_matches_the_reported_percentage(self):
        """0.8145% as reported, derived from the counts rather than copied."""
        assert BASELINE.trainable_fraction == pytest.approx(0.008145, abs=1e-6)

    def test_the_memory_figures_are_recorded(self):
        assert BASELINE.load_peak_vram_gib == 2.53
        assert BASELINE.step_peak_vram_gib == 3.943
        assert BASELINE.step_seconds == 4.107

    def test_the_headroom_is_derived_not_asserted(self):
        """"Substantial headroom" is a claim; this is the number behind it."""
        assert BASELINE.vram_headroom_gib == pytest.approx(18.091, abs=1e-3)
        assert BASELINE.vram_headroom_gib > BASELINE.step_peak_vram_gib * 4

    def test_the_batch_size_is_recorded_as_one(self):
        """So nobody reads the memory figures as applying to a larger batch."""
        assert BASELINE.batch_size == 1

    def test_the_record_states_that_it_is_not_a_training_run(self):
        notes = " ".join(BASELINE.notes)
        assert "feasibility only" in notes
        assert "not a training run" in notes

    def test_the_record_states_that_2048_was_not_tested(self):
        notes = " ".join(BASELINE.notes)
        assert "2048" in notes
        assert "not verified" in notes

    def test_the_record_admits_the_gradient_checkpointing_gap(self):
        """The measurement did not record it, so the peak cannot be attributed."""
        assert any("checkpointing" in note for note in BASELINE.notes)

    def test_the_record_is_frozen(self):
        with pytest.raises(AttributeError):
            BASELINE.step_seconds = 1.0  # type: ignore[misc]

    def test_the_record_is_json_serializable(self):
        payload = json.loads(json.dumps(BASELINE.as_dict()))
        assert payload["gpu"] == "NVIDIA L4"
        assert payload["trainable_fraction"] == pytest.approx(0.008145, abs=1e-6)

    def test_an_empty_baseline_does_not_divide_by_zero(self):
        empty = MeasuredBaseline(
            label="x",
            gpu="x",
            total_vram_gib=0.0,
            cuda_version="",
            torch_version="",
            bf16_supported=False,
            model_id="",
            quantization="none",
            quantization_type="nf4",
            double_quantization=False,
            compute_dtype="fp32",
            max_seq_length=1,
            batch_size=1,
            lora_rank=1,
            lora_alpha=1,
            lora_dropout=0.0,
            target_modules=(),
            total_parameters=0,
            trainable_parameters=0,
            load_peak_vram_gib=0.0,
            step_peak_vram_gib=0.0,
            step_seconds=0.0,
            loss=0.0,
        )
        assert empty.trainable_fraction == 0.0


class TestDefaultsMatchTheMeasurement:
    """The shipped configuration is the one that was demonstrated to run."""

    def test_the_default_model_is_the_measured_model(self):
        assert GeneratorModelConfig().model_id == BASELINE.model_id
        assert BASELINE.model_id == DEFAULT_BASE_MODEL

    def test_the_default_quantization_is_the_measured_quantization(self):
        config = GeneratorModelConfig()
        assert config.quantization == BASELINE.quantization
        assert config.quantization_type == BASELINE.quantization_type
        assert config.double_quantization is BASELINE.double_quantization
        assert config.compute_dtype == BASELINE.compute_dtype

    def test_the_default_sequence_length_is_the_measured_one(self):
        assert GeneratorModelConfig().max_seq_length == BASELINE.max_seq_length
        assert VERIFIED_MAX_SEQ_LENGTH == 1024

    def test_the_default_adapter_is_the_measured_adapter(self):
        lora = LoRAConfig()
        assert lora.rank == BASELINE.lora_rank
        assert lora.alpha == BASELINE.lora_alpha
        assert lora.dropout == BASELINE.lora_dropout
        assert lora.target_modules == BASELINE.target_modules

    def test_all_seven_measured_projections_are_adapted_by_default(self):
        assert set(LoRAConfig().target_modules) == set(MEASURED_TARGET_MODULES)
        assert len(LoRAConfig().target_modules) == 7

    def test_the_default_model_config_reports_matching_the_baseline(self):
        assert GeneratorModelConfig().matches_baseline(BASELINE) is True

    def test_the_default_experiment_config_has_no_deviations(self):
        config = experiment_config_from_dict({"name": "baseline"})
        assert config.baseline_deviations(BASELINE) == ()
        assert config.is_verified_configuration(BASELINE) is True

    def test_the_default_experiment_config_validates(self):
        experiment_config_from_dict({"name": "baseline"}).validate()


class TestUnverifiedConfigurationsAreVisible:
    """Deviating is allowed. Deviating silently is not."""

    def test_raising_the_sequence_length_is_reported_as_a_deviation(self):
        """2048 may work given the headroom, but it was never measured."""
        config = experiment_config_from_dict({"name": "long", "model": {"max_seq_length": 2048}})
        deviations = config.baseline_deviations(BASELINE)
        assert any("max_seq_length" in item for item in deviations)
        assert config.is_verified_configuration(BASELINE) is False

    def test_a_longer_sequence_length_is_still_permitted(self):
        """Configurability is preserved; only the claim of verification is withdrawn."""
        config = experiment_config_from_dict({"name": "long", "model": {"max_seq_length": 4096}})
        config.validate()
        assert config.model.max_seq_length == 4096

    def test_a_rank_sweep_is_reported_as_a_deviation(self):
        config = experiment_config_from_dict(
            {"name": "r64", "lora": {"rank": 64, "alpha": 128}}
        )
        deviations = config.baseline_deviations(BASELINE)
        assert any("lora.rank" in item for item in deviations)
        assert any("lora.alpha" in item for item in deviations)

    def test_switching_to_fp4_is_reported_as_a_deviation(self):
        config = experiment_config_from_dict(
            {"name": "fp4", "model": {"quantization_type": "fp4"}}
        )
        assert any("quantization_type" in item for item in config.baseline_deviations(BASELINE))

    def test_disabling_double_quantization_is_reported_as_a_deviation(self):
        config = experiment_config_from_dict(
            {"name": "single", "model": {"double_quantization": False}}
        )
        assert any(
            "double_quantization" in item for item in config.baseline_deviations(BASELINE)
        )

    def test_a_different_base_model_is_reported_as_a_deviation(self):
        config = experiment_config_from_dict(
            {"name": "other", "model": {"model_id": "mistralai/Mistral-7B-v0.3"}}
        )
        assert any("model_id" in item for item in config.baseline_deviations(BASELINE))

    def test_a_deviation_message_names_both_values(self):
        config = experiment_config_from_dict({"name": "long", "model": {"max_seq_length": 2048}})
        message = next(
            item for item in config.baseline_deviations(BASELINE) if "max_seq_length" in item
        )
        assert "2048" in message
        assert "1024" in message

    def test_deviations_are_sorted_for_stable_recording(self):
        config = experiment_config_from_dict(
            {"name": "d", "lora": {"rank": 8}, "model": {"max_seq_length": 512}}
        )
        deviations = config.baseline_deviations(BASELINE)
        assert list(deviations) == sorted(deviations)

    def test_the_deviation_list_only_covers_measured_fields(self):
        """The feasibility run had no opinion about the learning rate or the schedule."""
        config = experiment_config_from_dict(
            {"name": "lr", "training": {"learning_rate": 1e-3, "num_train_epochs": 9}}
        )
        assert config.baseline_deviations(BASELINE) == ()


class TestQuantizationConfiguration:
    """The three 4-bit decisions the measurement pinned down."""

    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    def test_both_four_bit_variants_are_accepted(self, quant_type):
        GeneratorModelConfig(quantization_type=quant_type).validate()

    def test_an_unknown_four_bit_variant_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="quantization_type"):
            GeneratorModelConfig(quantization_type="int4").validate()

    def test_an_unknown_compute_dtype_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="compute_dtype"):
            GeneratorModelConfig(compute_dtype="int8").validate()

    def test_is_4bit_reflects_the_quantization_choice(self):
        assert GeneratorModelConfig().is_4bit is True
        assert GeneratorModelConfig(quantization="8bit").is_4bit is False
        assert GeneratorModelConfig(quantization="none").is_4bit is False

    def test_is_quantized_covers_both_integer_modes(self):
        assert GeneratorModelConfig(quantization="8bit").is_quantized is True
        assert GeneratorModelConfig(quantization="none").is_quantized is False

    def test_the_quantization_summary_reports_the_measured_load(self):
        settings = GeneratorModelConfig().quantization_settings()
        assert settings == {
            "quantization": "4bit",
            "quantization_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "bf16",
        }

    @pytest.mark.parametrize("quantization", ["none", "8bit"])
    def test_four_bit_only_keys_are_omitted_when_not_four_bit(self, quantization):
        """A caller must not be able to read a value that does not apply."""
        settings = GeneratorModelConfig(quantization=quantization).quantization_settings()
        assert "quantization_type" not in settings
        assert "double_quantization" not in settings
        assert settings["quantization"] == quantization

    def test_the_summary_is_json_serializable(self):
        payload = json.loads(json.dumps(GeneratorModelConfig().quantization_settings()))
        assert payload["quantization_type"] == "nf4"

    def test_no_bitsandbytes_object_is_constructed(self):
        """This package reports decisions; the runtime translates them in Phase 17B."""
        settings = GeneratorModelConfig().quantization_settings()
        assert all(isinstance(value, str | bool) for value in settings.values())


class TestReasoningModeConfiguration:
    """Qwen3's thinking mode is configuration, not prompt behaviour."""

    def test_reasoning_is_disabled_by_default(self):
        """A reasoning preamble would make the output invalid JSON."""
        config = GeneratorModelConfig()
        assert config.reasoning_mode == "disabled"
        assert config.suppresses_reasoning is True

    @pytest.mark.parametrize("mode", ["disabled", "enabled", "inherit"])
    def test_every_declared_mode_is_accepted(self, mode):
        GeneratorModelConfig(reasoning_mode=mode).validate()

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="reasoning_mode"):
            GeneratorModelConfig(reasoning_mode="thinking").validate()

    def test_suppresses_reasoning_is_false_when_enabled(self):
        assert GeneratorModelConfig(reasoning_mode="enabled").suppresses_reasoning is False

    def test_requesting_a_mode_with_plain_concatenation_is_rejected(self):
        """There would be no chat template to pass the request to, so it would be ignored."""
        with pytest.raises(GenerationConfigError, match="reasoning_mode"):
            GeneratorModelConfig(reasoning_mode="enabled", chat_template="plain").validate()

    def test_inherit_is_compatible_with_plain_concatenation(self):
        GeneratorModelConfig(reasoning_mode="inherit", chat_template="plain").validate()

    def test_the_mode_is_part_of_the_config_hash(self):
        """It changes what the model emits, so it changes the run's identity."""
        base = experiment_config_from_dict({"name": "a"})
        thinking = experiment_config_from_dict(
            {"name": "a", "model": {"reasoning_mode": "enabled"}}
        )
        assert base.config_hash() != thinking.config_hash()

    def test_the_prompt_layer_declares_no_reasoning_identifier(self):
        """The generic template must stay free of any model's behaviour."""
        from test_qa_gen_isolation import code_identifiers

        referenced = code_identifiers("prompts")
        forbidden = {"reasoning_mode", "enable_thinking", "thinking", "reasoning"}
        leaked = sorted(referenced & forbidden)
        assert not leaked, f"the prompt layer references {leaked}"

    def test_the_prompt_layer_emits_no_provider_or_reasoning_text(self):
        """Checked over code string literals, not docstrings.

        The module docstring explains at length that it contains no Qwen-specific markup; a
        naive text search would flag that explanation as a violation of itself.
        """
        from test_qa_gen_isolation import code_string_literals

        for literal in code_string_literals("prompts"):
            lowered = literal.lower()
            for marker in ("qwen", "<think>", "enable_thinking", "im_start", "[inst]"):
                assert marker not in lowered, f"prompt literal {literal!r} contains {marker!r}"

    def test_the_reasoning_switch_lives_on_the_model_config(self):
        """The right home: it is a property of a model's chat template, not of the task."""
        from test_qa_gen_isolation import code_identifiers

        assert "reasoning_mode" in code_identifiers("config")
        assert hasattr(GeneratorModelConfig(), "reasoning_mode")

    def test_the_prompt_fields_do_not_include_a_reasoning_switch(self):
        from qa_gen import PROMPT_FIELDS

        assert not any("think" in name or "reason" in name for name in PROMPT_FIELDS)


class TestSingleSourceOfTruthForGradientCheckpointing:
    """One decision, one field."""

    def test_the_training_config_owns_it(self):
        assert TrainingConfig().gradient_checkpointing is True

    def test_the_model_config_no_longer_duplicates_it(self):
        """Two fields for one setting could disagree, and one would silently lose."""
        assert not hasattr(GeneratorModelConfig(), "gradient_checkpointing")

    def test_it_appears_exactly_once_in_the_serialized_config(self):
        payload = experiment_config_from_dict({"name": "a"}).to_dict()
        assert "gradient_checkpointing" in payload["training"]
        assert "gradient_checkpointing" not in payload["model"]


class TestRunMetadataCarriesTheMeasurement:
    """A run record should be comparable against the demonstrated configuration."""

    def test_the_measured_counts_reproduce_the_reported_percentage(self):
        """Same arithmetic the feasibility run reported, through the run record."""
        record = TrainingRunMetadata(
            run_id="r1",
            experiment_name="baseline",
            trainable_parameters=BASELINE.trainable_parameters,
            total_parameters=BASELINE.total_parameters,
        )
        assert record.trainable_fraction == pytest.approx(0.008145, abs=1e-6)
        assert record.trainable_fraction == BASELINE.trainable_fraction

    def test_a_baseline_can_be_embedded_in_a_run_record(self):
        config = experiment_config_from_dict({"name": "baseline"})
        record = TrainingRunMetadata(
            run_id=config.run_id("20260906T120000Z"),
            experiment_name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash(),
            base_model=config.model.model_id,
            base_model_revision=config.model.revision,
            training={
                "measured_baseline": BASELINE.as_dict(),
                "baseline_deviations": list(config.baseline_deviations(BASELINE)),
            },
        )
        payload = json.loads(record.to_json())
        assert payload["training"]["measured_baseline"]["gpu"] == "NVIDIA L4"
        assert payload["training"]["baseline_deviations"] == []

    def test_an_unverified_run_records_its_deviations(self):
        config = experiment_config_from_dict({"name": "long", "model": {"max_seq_length": 2048}})
        record = TrainingRunMetadata(
            run_id="r2",
            experiment_name=config.name,
            training={"baseline_deviations": list(config.baseline_deviations(BASELINE))},
        )
        payload = json.loads(record.to_json())
        assert payload["training"]["baseline_deviations"]


class TestNoRuntimeDependencyWasIntroduced:
    """The hardware evidence is data. It did not bring the stack with it."""

    def test_the_config_module_imports_nothing_heavy(self):
        from test_qa_gen_isolation import code_identifiers

        referenced = code_identifiers("config")
        forbidden = {
            "torch",
            "transformers",
            "bitsandbytes",
            "BitsAndBytesConfig",
            "peft",
            "trl",
            "datasets",
            "accelerate",
            "cuda",
        }
        leaked = sorted(referenced & forbidden)
        assert not leaked, f"qa_gen.config references {leaked}"

    def test_the_baseline_holds_only_plain_data(self):
        """Strings, numbers and booleans, so it serializes and needs no runtime."""
        for value in BASELINE.as_dict().values():
            assert isinstance(value, str | int | float | bool | list), type(value)
