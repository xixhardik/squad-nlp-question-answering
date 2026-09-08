"""Tests for the QLoRA training runtime.

How these run without a GPU or a checkpoint
-------------------------------------------
Three different techniques, chosen per boundary rather than uniformly:

**Real libraries where they exist.** ``torch`` and ``transformers`` are installed, and
``transformers.BitsAndBytesConfig`` constructs without ``bitsandbytes`` present -- verified. So
the dtype mapping and the quantization translation are asserted against the actual objects.
That is unusual for this kind of boundary and worth using: those assertions cannot drift from
the library the way a mock can.

**Stub modules where they do not.** ``peft`` and ``trl`` are absent on the development machine,
so :mod:`qa_gen_runtime.deps` resolves them through functions that the fixtures below replace
with recording stubs. The stubs capture the keyword arguments, which is what lets the tests
assert that NF4 reaches ``bnb_4bit_quant_type``, that the tokenizer reaches
``processing_class`` and that k-bit preparation happens *before* the adapter is attached.

**No network, ever.** No test loads a real tokenizer or model. The loader's arguments are
asserted through :func:`qa_gen_runtime.loader.build_model_kwargs`, which builds them without
calling anything.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from qa_gen import (
    VERIFIED_QWEN3_4B_L4,
    QuestionGenerationExample,
    QuestionGenerationTarget,
    TrainingRunMetadata,
    experiment_config_from_dict,
    target_from_json,
)
from qa_gen_runtime import (
    CHAT_TEMPLATE_KWARGS_COLUMN,
    DTYPE_NAMES,
    REASONING_TEMPLATE_FLAG,
    ConfigIOError,
    DatasetBuildError,
    ModelLoadError,
    PrecisionError,
    RecordFormat,
    RunOutputError,
    RunPaths,
    RuntimeDependencyError,
    TrainerBuildError,
    TrainingRecordBuilder,
    apply_chat_template,
    attach_adapters,
    attach_to_metadata,
    build_lora_config,
    build_model_kwargs,
    build_quantization_config,
    build_sft_config,
    build_trainer,
    build_training_records,
    chat_template_kwargs,
    collect_diagnostics,
    count_parameters,
    create_run_directory,
    dependency_report,
    describe_chat_handling,
    describe_quantization,
    load_experiment_config,
    load_mapping,
    plan_trainer_arguments,
    resolve_dtype,
    resolve_precision,
    resolve_record_format,
    resolve_run_root,
    resolve_warmup_steps,
    template_supports_reasoning_flag,
    write_resolved_config,
)
from qa_gen_runtime import deps as deps_module
from qa_gen_runtime import loader as loader_module
from qa_gen_runtime import precision as precision_module
from qa_gen_runtime.train import build_parser, main, run_validation
from qa_paper import Difficulty, QuestionType

CONTEXT = (
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. "
    "It generates most of the cell's supply of adenosine triphosphate."
)


def config(**overrides) -> Any:
    """Build a validated experiment configuration, overriding any section."""
    payload: dict[str, Any] = {"name": "runtime-test"}
    payload.update(overrides)
    return experiment_config_from_dict(payload)


def target(**overrides) -> QuestionGenerationTarget:
    """Build a valid short-answer target."""
    defaults = {
        "question_type": QuestionType.SHORT_ANSWER,
        "question": "What does the mitochondrion generate?",
        "answer": "adenosine triphosphate",
        "difficulty": Difficulty.EASY,
        "marks": 1,
    }
    return QuestionGenerationTarget(**{**defaults, **overrides})


def example(index: int = 0, **overrides) -> QuestionGenerationExample:
    """Build a valid single-target example."""
    defaults = {
        "id": f"ex-{index}",
        "context": f"{CONTEXT} Passage {index}.",
        "targets": (target(question=f"Question {index}?"),),
        "source": "squad-qg",
        "topic": "Cell Biology",
    }
    return QuestionGenerationExample(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Stubs for the libraries that are not installed on this machine
# ---------------------------------------------------------------------------


@dataclass
class FakeParameter:
    """A parameter with a size and a gradient flag. Enough for counting."""

    count: int
    requires_grad: bool = False

    def numel(self) -> int:
        """Return the parameter's element count."""
        return self.count


class FakeModel:
    """A model-shaped object that records what was done to it."""

    def __init__(self, parameters: list[FakeParameter] | None = None) -> None:
        """Build a model with the given parameters, or a plausible default split."""
        self._parameters = parameters or [
            FakeParameter(4_022_468_096, requires_grad=False),
            FakeParameter(33_030_144, requires_grad=True),
        ]
        self.config = SimpleNamespace(use_cache=True)
        self.prepared = False
        self.prepare_kwargs: dict[str, Any] = {}
        self.peft_config: Any = None

    def parameters(self):
        """Yield the model's parameters."""
        return iter(self._parameters)


@dataclass
class RecordingPeft:
    """A stand-in for the ``peft`` module that records the calls it receives."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Expose ``LoraConfig`` as a plain recorder."""
        peft_self = self

        class LoraConfig:
            """Records the keyword arguments it was constructed with."""

            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                peft_self.calls.append(("LoraConfig", dict(kwargs)))

        self.LoraConfig = LoraConfig  # noqa: N803 - mirrors the real attribute name

    def prepare_model_for_kbit_training(self, model: Any, **kwargs: Any) -> Any:
        """Record preparation and mark the model."""
        self.calls.append(("prepare_model_for_kbit_training", dict(kwargs)))
        model.prepared = True
        model.prepare_kwargs = dict(kwargs)
        return model

    def get_peft_model(self, model: Any, peft_config: Any) -> Any:
        """Record wrapping and attach the config."""
        self.calls.append(("get_peft_model", {"config": peft_config}))
        if not getattr(model, "prepared", False):
            raise AssertionError(
                "get_peft_model was called before prepare_model_for_kbit_training"
            )
        model.peft_config = peft_config
        return model

    @property
    def call_names(self) -> list[str]:
        """The names of the calls received, in order."""
        return [name for name, _ in self.calls]

    def kwargs_for(self, name: str) -> dict[str, Any]:
        """Return the keyword arguments of the first call to ``name``."""
        for call_name, kwargs in self.calls:
            if call_name == name:
                return kwargs
        raise AssertionError(f"{name} was never called; got {self.call_names}")


class FakeSFTConfig:
    """A stand-in for ``trl.SFTConfig`` accepting only the v0.29.1 argument names."""

    #: Exactly the arguments TRL v0.29.1 accepts that this runtime passes, read from the
    #: released sources. A stub that accepted anything would not catch a rename, which is the
    #: whole reason these tests exist.
    ACCEPTED = frozenset(
        {
            "output_dir",
            "learning_rate",
            "num_train_epochs",
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_steps",
            "lr_scheduler_type",
            "max_grad_norm",
            "optim",
            "seed",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "gradient_checkpointing_kwargs",
            "eval_strategy",
            "save_strategy",
            "save_total_limit",
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
            "logging_steps",
            "dataloader_num_workers",
            "report_to",
            "max_length",
            "packing",
            "completion_only_loss",
            "assistant_only_loss",
            "dataset_text_field",
        }
    )

    def __init__(self, **kwargs: Any) -> None:
        """Reject any keyword the real v0.29.1 class would reject."""
        unknown = sorted(set(kwargs) - self.ACCEPTED)
        if unknown:
            raise TypeError(f"SFTConfig got unexpected keyword argument(s) {unknown}")
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeSFTTrainer:
    """A stand-in for ``trl.SFTTrainer`` with the v0.29.1 signature."""

    def __init__(
        self,
        model: Any = None,
        args: Any = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        compute_loss_func: Any = None,
        compute_metrics: Any = None,
        callbacks: Any = None,
        optimizers: Any = (None, None),
        optimizer_cls_and_kwargs: Any = None,
        preprocess_logits_for_metrics: Any = None,
        peft_config: Any = None,
        formatting_func: Any = None,
    ) -> None:
        """Record everything, and refuse a missing dataset the way TRL does."""
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.peft_config = peft_config
        self.trained = False

    def train(self) -> None:  # pragma: no cover - never called by these tests
        """Would train. No test calls this."""
        self.trained = True


def _fake_trl() -> ModuleType:
    """Build a module-shaped stand-in for ``trl``."""
    module = ModuleType("trl")
    module.SFTConfig = FakeSFTConfig
    module.SFTTrainer = FakeSFTTrainer
    return module


@pytest.fixture
def fake_peft(monkeypatch) -> RecordingPeft:
    """Replace PEFT resolution with a recording stub, in both places it is looked up."""
    peft = RecordingPeft()
    monkeypatch.setattr(deps_module, "require_peft", lambda: peft)
    monkeypatch.setattr(loader_module, "require_peft", lambda: peft)
    return peft


@pytest.fixture
def fake_trl(monkeypatch) -> ModuleType:
    """Replace TRL resolution with a stub enforcing the v0.29.1 signatures."""
    module = _fake_trl()
    monkeypatch.setattr(deps_module, "require_trl", lambda: module)
    from qa_gen_runtime import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "require_trl", lambda: module)
    return module


@pytest.fixture
def cuda_bf16(monkeypatch):
    """Pretend this machine has a bf16-capable CUDA device."""
    monkeypatch.setattr(precision_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(precision_module, "_bf16_supported", lambda: True)


@pytest.fixture
def cuda_no_bf16(monkeypatch):
    """Pretend this machine has CUDA without bf16 support, as on a pre-Ampere card."""
    monkeypatch.setattr(precision_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(precision_module, "_bf16_supported", lambda: False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhase17AStaysDependencyLight:
    """The boundary this whole package exists to protect."""

    def test_qa_gen_still_imports_with_only_the_standard_library(self):
        """Re-asserted here because qa_gen_runtime is the thing most likely to break it."""
        import subprocess
        import textwrap

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import qa_gen  # noqa: F401

                    forbidden = (
                        "torch", "transformers", "peft", "trl", "bitsandbytes",
                        "datasets", "accelerate", "yaml",
                    )
                    leaked = sorted(m for m in forbidden if m in sys.modules)
                    if leaked:
                        print("LEAKED:" + ",".join(leaked))
                        raise SystemExit(1)
                    print("CLEAN")
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, (
            f"importing qa_gen pulled in a heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "CLEAN" in result.stdout

    def test_importing_qa_gen_does_not_import_the_runtime(self):
        import subprocess
        import textwrap

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import qa_gen  # noqa: F401
                    print("RUNTIME_LEAKED" if "qa_gen_runtime" in sys.modules else "SEPARATE")
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert "SEPARATE" in result.stdout, result.stderr

    def test_no_qa_gen_module_imports_the_runtime(self):
        from test_qa_gen_isolation import SUBMODULES, code_identifiers

        for submodule in SUBMODULES:
            assert "qa_gen_runtime" not in code_identifiers(submodule)

    def test_the_runtime_is_importable_without_peft_or_trl(self):
        """Which is the situation on this machine, so this test is not hypothetical."""
        import qa_gen_runtime

        assert qa_gen_runtime.__version__
        assert deps_module.is_available("peft") is False
        assert deps_module.is_available("trl") is False


class TestDependencyResolution:
    """Missing training dependencies must fail usefully, not obscurely."""

    def test_a_missing_dependency_raises_with_an_install_command(self):
        with pytest.raises(RuntimeDependencyError, match="pip install peft==0.20.0"):
            deps_module.require_peft()

    def test_the_error_explains_why_the_package_is_needed(self):
        with pytest.raises(RuntimeDependencyError, match="low-rank adapters"):
            deps_module.require_peft()

    def test_a_missing_dependency_is_an_import_error_subclass(self):
        assert issubclass(RuntimeDependencyError, ImportError)

    def test_availability_is_checked_without_importing(self):
        assert deps_module.is_available("json") is True
        assert deps_module.is_available("definitely_not_a_module") is False

    def test_the_report_covers_the_whole_stack(self):
        report = dependency_report()
        for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"):
            assert name in report
            assert "available" in report[name]
            assert "purpose" in report[name]

    def test_the_report_records_versions_of_installed_packages(self):
        report = dependency_report()
        assert report["torch"]["available"] is True
        assert report["torch"]["version"]

    def test_the_report_is_json_serializable(self):
        assert json.loads(json.dumps(dependency_report()))["torch"]["available"] is True


class TestPrecisionResolution:
    """bf16 is never silently swapped for fp16."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
    )
    def test_names_map_to_torch_dtypes(self, name, expected):
        assert resolve_dtype(name) is expected
        assert DTYPE_NAMES[name] is expected

    def test_auto_is_not_a_dtype(self):
        with pytest.raises(PrecisionError, match="not a dtype"):
            resolve_dtype("auto")

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(PrecisionError, match="unknown precision"):
            resolve_dtype("int4")

    def test_auto_selects_bf16_on_a_capable_device(self, cuda_bf16):
        plan = resolve_precision("auto")
        assert plan.name == "bf16"
        assert plan.bf16 is True
        assert plan.fp16 is False
        assert plan.dtype is torch.bfloat16

    def test_auto_selects_fp32_not_fp16_when_bf16_is_unavailable(self, cuda_no_bf16):
        """fp16 needs loss scaling this project has not validated, so it is never automatic."""
        plan = resolve_precision("auto")
        assert plan.name == "fp32"
        assert plan.fp16 is False
        assert "fp16 is not selected automatically" in plan.reason

    def test_auto_selects_fp32_on_cpu(self):
        plan = resolve_precision("auto")
        assert plan.name == "fp32"
        assert plan.device_type == "cpu"

    def test_explicit_bf16_fails_clearly_when_unsupported(self, cuda_no_bf16):
        with pytest.raises(PrecisionError, match="Refusing to substitute fp16"):
            resolve_precision("bf16")

    def test_the_bf16_failure_names_the_measured_hardware(self):
        with pytest.raises(PrecisionError, match="NVIDIA L4"):
            resolve_precision("bf16")

    def test_explicit_bf16_succeeds_when_supported(self, cuda_bf16):
        plan = resolve_precision("bf16")
        assert plan.bf16 is True
        assert "requested explicitly" in plan.reason

    def test_explicit_fp16_on_cpu_fails(self):
        with pytest.raises(PrecisionError, match="no CUDA device"):
            resolve_precision("fp16")

    def test_explicit_fp32_always_works(self):
        plan = resolve_precision("fp32")
        assert plan.name == "fp32"
        assert plan.is_mixed is False

    def test_both_flags_cannot_be_set(self):
        with pytest.raises(PrecisionError, match="cannot request both"):
            precision_module.PrecisionPlan(
                name="bf16",
                dtype=torch.bfloat16,
                bf16=True,
                fp16=True,
                requested="bf16",
                reason="",
                device_type="cuda",
                bf16_supported=True,
            )

    def test_the_plan_is_json_serializable(self):
        payload = json.loads(json.dumps(resolve_precision("fp32").as_dict()))
        assert payload["dtype"] == "torch.float32"

    def test_the_requested_value_is_kept_alongside_the_resolution(self):
        """So an auto resolution is distinguishable from an explicit choice in the record."""
        plan = resolve_precision("auto")
        assert plan.requested == "auto"
        assert plan.name != "auto"


class TestQuantizationTranslation:
    """The measured NF4 settings must reach BitsAndBytesConfig, on the right keywords."""

    def test_the_default_config_translates_to_the_measured_load(self, cuda_bf16):
        quantization = build_quantization_config(config().model)
        assert quantization is not None
        assert quantization.load_in_4bit is True
        assert quantization.bnb_4bit_quant_type == "nf4"
        assert quantization.bnb_4bit_use_double_quant is True
        assert quantization.bnb_4bit_compute_dtype is torch.bfloat16

    def test_translation_works_without_bitsandbytes_installed(self):
        """Verified: BitsAndBytesConfig is a plain config object, so this is a real assertion."""
        assert deps_module.is_available("bitsandbytes") is False
        assert build_quantization_config(config().model) is not None

    def test_fp4_is_forwarded_when_requested(self):
        quantization = build_quantization_config(
            config(model={"quantization_type": "fp4"}).model
        )
        assert quantization.bnb_4bit_quant_type == "fp4"

    def test_double_quantization_can_be_disabled(self):
        quantization = build_quantization_config(
            config(model={"double_quantization": False}).model
        )
        assert quantization.bnb_4bit_use_double_quant is False

    def test_no_quantization_yields_none(self):
        """So the caller omits the keyword rather than passing None, which differs by version."""
        assert build_quantization_config(config(model={"quantization": "none"}).model) is None

    def test_eight_bit_does_not_forward_four_bit_keys(self):
        """Values that had no effect must not appear in the run record."""
        quantization = build_quantization_config(config(model={"quantization": "8bit"}).model)
        assert quantization.load_in_8bit is True
        assert quantization.load_in_4bit is False

    def test_compute_dtype_is_independent_of_training_precision(self):
        """bf16 compute against quantized weights is set separately from the trainer's mode."""
        model_config = config(
            model={"compute_dtype": "bf16"}, training={"precision": "fp32"}
        ).model
        assert build_quantization_config(model_config).bnb_4bit_compute_dtype is torch.bfloat16

    def test_the_description_reports_bitsandbytes_availability(self):
        """A 4-bit run without bitsandbytes has its explanation in its own metadata."""
        record = describe_quantization(config().model)
        assert record["requested"] == "4bit"
        assert record["applied"] is True
        assert record["bitsandbytes_available"] is False
        assert record["quant_type"] == "nf4"

    def test_the_description_is_json_serializable(self):
        payload = json.loads(json.dumps(describe_quantization(config().model)))
        assert payload["compute_dtype"] == "torch.bfloat16"

    def test_the_description_of_an_unquantized_load(self):
        record = describe_quantization(config(model={"quantization": "none"}).model)
        assert record["applied"] is False
        assert "quant_type" not in record


class TestModelLoaderArguments:
    """What from_pretrained would be called with, asserted without calling it."""

    def test_transformers_five_dtype_keyword_is_used(self):
        """5.16.1 documents `dtype`; `torch_dtype` is the v4 name and would be ignored."""
        kwargs = build_model_kwargs(config().model, resolve_precision("fp32"))
        assert "dtype" in kwargs
        assert "torch_dtype" not in kwargs

    def test_the_precision_dtype_is_forwarded(self, cuda_bf16):
        kwargs = build_model_kwargs(config().model, resolve_precision("bf16"))
        assert kwargs["dtype"] is torch.bfloat16

    def test_the_revision_is_pinned_on_the_call(self):
        kwargs = build_model_kwargs(config(model={"revision": "abc123"}).model,
                                    resolve_precision("fp32"))
        assert kwargs["revision"] == "abc123"

    def test_remote_code_execution_is_off(self):
        kwargs = build_model_kwargs(config().model, resolve_precision("fp32"))
        assert kwargs["trust_remote_code"] is False

    def test_the_quantization_config_is_attached(self):
        kwargs = build_model_kwargs(config().model, resolve_precision("fp32"))
        assert kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"

    def test_a_device_map_is_requested_for_a_quantized_load(self):
        """Without it a 4-bit load lands on CPU and the first forward pass fails."""
        kwargs = build_model_kwargs(config().model, resolve_precision("fp32"))
        assert kwargs["device_map"] == "auto"

    def test_no_quantization_means_no_quantization_keyword_at_all(self):
        kwargs = build_model_kwargs(
            config(model={"quantization": "none"}).model, resolve_precision("fp32")
        )
        assert "quantization_config" not in kwargs
        assert "device_map" not in kwargs

    def test_auto_attention_is_left_to_the_library(self):
        kwargs = build_model_kwargs(config().model, resolve_precision("fp32"))
        assert "attn_implementation" not in kwargs

    def test_an_explicit_attention_implementation_is_forwarded(self):
        kwargs = build_model_kwargs(
            config(model={"attn_implementation": "sdpa"}).model, resolve_precision("fp32")
        )
        assert kwargs["attn_implementation"] == "sdpa"


class TestParameterCounting:
    """Counted from the module, not from a PEFT helper."""

    def test_counts_match_the_measured_baseline_shape(self):
        trainable, total = count_parameters(FakeModel())
        assert trainable == VERIFIED_QWEN3_4B_L4.trainable_parameters
        assert total == VERIFIED_QWEN3_4B_L4.total_parameters

    def test_the_fraction_matches_the_reported_percentage(self):
        trainable, total = count_parameters(FakeModel())
        assert round(trainable / total, 6) == pytest.approx(0.008145, abs=1e-6)

    def test_an_object_without_parameters_is_rejected(self):
        with pytest.raises(ModelLoadError, match="no parameters"):
            count_parameters(object())

    def test_a_frozen_model_reports_zero_trainable(self):
        model = FakeModel([FakeParameter(100, requires_grad=False)])
        assert count_parameters(model) == (0, 100)


class TestKbitPreparationSequence:
    """The order is load, prepare, then attach. Reversing it breaks training silently."""

    def test_preparation_happens_before_the_adapter_is_attached(self, fake_peft):
        attach_adapters(FakeModel(), config().lora)
        assert fake_peft.call_names == [
            "prepare_model_for_kbit_training",
            "LoraConfig",
            "get_peft_model",
        ]

    def test_the_stub_would_catch_the_reversed_order(self, fake_peft):
        """Guards the guard: get_peft_model asserts the model was prepared."""
        with pytest.raises(AssertionError, match="before prepare_model_for_kbit_training"):
            fake_peft.get_peft_model(FakeModel(), object())

    def test_checkpointing_is_passed_to_the_preparation_call(self, fake_peft):
        attach_adapters(FakeModel(), config().lora, gradient_checkpointing=True)
        assert fake_peft.kwargs_for("prepare_model_for_kbit_training") == {
            "use_gradient_checkpointing": True
        }

    def test_use_cache_is_disabled_when_checkpointing(self, fake_peft):
        """The KV cache and activation checkpointing are mutually exclusive."""
        model = FakeModel()
        attach_adapters(model, config().lora, gradient_checkpointing=True)
        assert model.config.use_cache is False

    def test_use_cache_is_left_alone_without_checkpointing(self, fake_peft):
        model = FakeModel()
        attach_adapters(model, config().lora, gradient_checkpointing=False)
        assert model.config.use_cache is True

    def test_a_model_without_a_config_does_not_break_preparation(self, fake_peft):
        model = FakeModel()
        del model.config
        attach_adapters(model, config().lora, gradient_checkpointing=True)
        assert model.prepared is True

    def test_a_preparation_failure_is_wrapped(self, fake_peft, monkeypatch):
        def boom(model, **kwargs):
            raise RuntimeError("no quantized weights")

        monkeypatch.setattr(fake_peft, "prepare_model_for_kbit_training", boom)
        with pytest.raises(ModelLoadError, match="prepare_model_for_kbit_training failed"):
            attach_adapters(FakeModel(), config().lora)

    def test_preparation_needs_peft_and_says_so(self):
        with pytest.raises(RuntimeDependencyError, match="peft"):
            attach_adapters(FakeModel(), config().lora)


class TestLoraTranslation:
    """The measured adapter settings must reach peft.LoraConfig."""

    def test_the_measured_settings_are_forwarded(self, fake_peft):
        build_lora_config(config().lora)
        kwargs = fake_peft.kwargs_for("LoraConfig")
        assert kwargs["r"] == 16
        assert kwargs["lora_alpha"] == 32
        assert kwargs["lora_dropout"] == 0.05
        assert kwargs["bias"] == "none"
        assert kwargs["task_type"] == "CAUSAL_LM"

    def test_all_seven_measured_projections_are_forwarded(self, fake_peft):
        build_lora_config(config().lora)
        assert fake_peft.kwargs_for("LoraConfig")["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_the_project_field_names_are_translated(self, fake_peft):
        """Rank -> r, alpha -> lora_alpha, dropout -> lora_dropout."""
        build_lora_config(config(lora={"rank": 8, "alpha": 16, "dropout": 0.1}).lora)
        kwargs = fake_peft.kwargs_for("LoraConfig")
        assert kwargs == {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(config().lora.target_modules),
            "use_rslora": False,
        }

    def test_modules_to_save_is_omitted_when_empty(self, fake_peft):
        build_lora_config(config().lora)
        assert "modules_to_save" not in fake_peft.kwargs_for("LoraConfig")

    def test_modules_to_save_is_forwarded_when_set(self, fake_peft):
        build_lora_config(config(lora={"modules_to_save": ["embed_tokens"]}).lora)
        assert fake_peft.kwargs_for("LoraConfig")["modules_to_save"] == ["embed_tokens"]

    def test_unsupported_keywords_are_dropped_rather_than_raising(self, monkeypatch):
        """A PEFT version without use_rslora must not turn into a TypeError mid-run."""
        recorded: dict[str, Any] = {}

        class NarrowLoraConfig:
            """Accepts only the arguments every PEFT version has had."""

            def __init__(  # noqa: ANN204 - a stub signature, deliberately narrow
                self, r, lora_alpha, lora_dropout, bias, task_type, target_modules
            ):
                recorded.update(
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=bias,
                    task_type=task_type,
                    target_modules=target_modules,
                )

        narrow = SimpleNamespace(LoraConfig=NarrowLoraConfig)
        monkeypatch.setattr(loader_module, "require_peft", lambda: narrow)
        build_lora_config(config().lora)
        assert recorded["r"] == 16


class TestRecordConstruction:
    """Input and target are structurally separate, and the target is JSON."""

    def test_the_default_shape_is_conversational(self):
        assert resolve_record_format(config()) is RecordFormat.CONVERSATIONAL

    def test_a_plain_chat_template_yields_prompt_completion(self):
        cfg = config(model={"chat_template": "plain", "reasoning_mode": "inherit"})
        assert resolve_record_format(cfg) is RecordFormat.PROMPT_COMPLETION

    def test_packing_yields_language_modeling(self):
        """A single text field cannot express where the prompt ends."""
        cfg = config(training={"packing": True, "completion_only_loss": False})
        assert resolve_record_format(cfg) is RecordFormat.LANGUAGE_MODELING

    def test_a_conversational_record_separates_prompt_from_completion(self):
        record = build_training_records([example()], config())[0]
        assert [message["role"] for message in record["prompt"]] == ["system", "user"]
        assert [message["role"] for message in record["completion"]] == ["assistant"]

    def test_the_completion_is_parseable_canonical_json(self):
        record = build_training_records([example()], config())[0]
        parsed = target_from_json(record["completion"][0]["content"])
        assert parsed == example().primary_target

    def test_the_completion_is_not_a_python_repr(self):
        content = build_training_records([example()], config())[0]["completion"][0]["content"]
        assert content.startswith("{")
        assert "'" not in content.split('"question"')[0]
        json.loads(content)

    def test_the_prompt_comes_from_the_phase_17a_template(self):
        """No second prompt format exists in the runtime."""
        record = build_training_records([example()], config())[0]
        rendered = config().prompt.render(example())
        assert record["prompt"][0]["content"] == rendered.system
        assert record["prompt"][1]["content"] == rendered.user

    def test_the_prompt_carries_every_conditioning_field(self):
        record = build_training_records([example()], config())[0]
        user = record["prompt"][1]["content"]
        assert CONTEXT in user
        assert "short_answer" in user
        assert "easy" in user
        assert "Cell Biology" in user
        assert "Marks per question: 1" in user
        assert "Questions to write: 1" in user
        assert "JSON" in user

    def test_no_provider_chat_markup_reaches_a_record(self):
        record = build_training_records([example()], config())[0]
        serialized = json.dumps(record)
        for marker in ("<|im_start|>", "<|im_end|>", "[INST]", "<s>", "<think>"):
            assert marker not in serialized

    def test_a_prompt_completion_record_uses_plain_strings(self):
        cfg = config(model={"chat_template": "plain", "reasoning_mode": "inherit"})
        record = build_training_records([example()], cfg)[0]
        assert isinstance(record["prompt"], str)
        assert isinstance(record["completion"], str)
        assert CONTEXT in record["prompt"]
        assert record["completion"] not in record["prompt"]

    def test_a_language_modeling_record_has_one_text_field(self):
        cfg = config(training={"packing": True, "completion_only_loss": False})
        record = build_training_records([example()], cfg)[0]
        assert set(record) == {"text", "example_id"}
        assert CONTEXT in record["text"]

    def test_the_example_id_travels_with_the_record(self):
        assert build_training_records([example(7)], config())[0]["example_id"] == "ex-7"

    def test_a_multi_target_example_serializes_as_a_json_array(self):
        multi = example(targets=(target(), target(question="Second?")))
        record = build_training_records([multi], config())[0]
        decoded = json.loads(record["completion"][0]["content"])
        assert isinstance(decoded, list)
        assert len(decoded) == 2

    def test_an_example_without_targets_is_refused(self):
        """Training on an empty completion teaches the model to emit nothing."""
        with pytest.raises(DatasetBuildError, match="no targets"):
            build_training_records([example(targets=())], config())

    def test_records_are_built_in_input_order(self):
        records = build_training_records([example(0), example(1), example(2)], config())
        assert [record["example_id"] for record in records] == ["ex-0", "ex-1", "ex-2"]

    def test_an_empty_corpus_cannot_become_a_dataset(self):
        from qa_gen_runtime import build_hf_dataset

        with pytest.raises(DatasetBuildError, match="zero records"):
            build_hf_dataset([])

    def test_records_convert_to_an_in_memory_dataset(self):
        """Datasets is installed, so this is a real conversion. Nothing is downloaded."""
        from qa_gen_runtime import build_hf_dataset

        records = build_training_records([example(0), example(1)], config())
        dataset = build_hf_dataset(records)
        assert len(dataset) == 2
        assert "prompt" in dataset.column_names

    def test_the_builder_can_emit_a_non_compact_target(self):
        builder = TrainingRecordBuilder(template=config().prompt, compact_target=False)
        content = builder.build(example())["completion"][0]["content"]
        assert "options" in json.loads(content)


class TestReasoningModeHandling:
    """Qwen3's thinking mode is respected at the runtime boundary and nowhere else."""

    def test_the_flag_is_detected_from_the_template_source(self):
        qwen_like = SimpleNamespace(
            chat_template="{% if enable_thinking %}<think>{% endif %}"
        )
        plain = SimpleNamespace(chat_template="{{ messages }}")
        assert template_supports_reasoning_flag(qwen_like) is True
        assert template_supports_reasoning_flag(plain) is False

    def test_a_tokenizer_without_a_template_supports_nothing(self):
        assert template_supports_reasoning_flag(SimpleNamespace(chat_template=None)) is False

    def test_disabled_passes_the_flag_as_false(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        assert chat_template_kwargs(config().model, tokenizer) == {
            REASONING_TEMPLATE_FLAG: False
        }

    def test_enabled_passes_the_flag_as_true(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        cfg = config(model={"reasoning_mode": "enabled"})
        assert chat_template_kwargs(cfg.model, tokenizer) == {REASONING_TEMPLATE_FLAG: True}

    def test_inherit_passes_nothing(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        cfg = config(model={"reasoning_mode": "inherit"})
        assert chat_template_kwargs(cfg.model, tokenizer) == {}

    def test_the_flag_is_not_passed_to_a_template_that_ignores_it(self):
        """Passing an unknown keyword is at best ignored and at worst a Jinja error."""
        tokenizer = SimpleNamespace(chat_template="{{ messages }}")
        assert chat_template_kwargs(config().model, tokenizer) == {}

    def test_the_default_is_disabled(self):
        assert config().model.reasoning_mode == "disabled"
        assert config().model.suppresses_reasoning is True

    def test_applying_a_template_forwards_the_flag(self):
        captured: dict[str, Any] = {}

        class Tokenizer:
            chat_template = "{{ enable_thinking }}"

            def apply_chat_template(self, messages, **kwargs):
                captured.update(messages=messages, **kwargs)
                return "RENDERED"

        rendered = config().prompt.render(example())
        result = apply_chat_template(rendered.as_messages(), config().model, Tokenizer())
        assert result == "RENDERED"
        assert captured[REASONING_TEMPLATE_FLAG] is False
        assert captured["tokenize"] is False
        assert captured["add_generation_prompt"] is False

    def test_a_missing_template_is_refused_rather_than_concatenated(self):
        """Concatenating would produce a format the model was never trained on."""
        tokenizer = SimpleNamespace(chat_template=None)
        with pytest.raises(ValueError, match="no chat template"):
            apply_chat_template([{"role": "user", "content": "x"}], config().model, tokenizer)

    def test_the_description_records_whether_the_request_took_effect_at_inference(self):
        inert = describe_chat_handling(
            config().model,
            SimpleNamespace(chat_template="{{ messages }}"),
            stage="inference",
        )
        assert inert["effective"].startswith("inert")

        applied = describe_chat_handling(
            config().model,
            SimpleNamespace(chat_template="{{ enable_thinking }}"),
            stage="inference",
        )
        assert applied["effective"].startswith("applied")

    def test_the_training_stage_reports_the_flag_as_applied_through_the_column(self):
        """TRL renders the template, but it reads the mode from the column we emit.

        This assertion was the other way round until a Phase 17B.2 inspection on the Studio
        showed the flag does change trained tokens: Qwen3 puts an empty <think></think> block
        in front of the last assistant turn unconditionally, so without the column that block
        lands inside the supervised completion.
        """
        record = describe_chat_handling(
            config().model, SimpleNamespace(chat_template="{{ enable_thinking }}")
        )
        assert record["stage"] == "training"
        assert record["applied_by"] == "trl.SFTTrainer"
        assert record["effective"].startswith("applied")
        assert record["template_kwargs"] == {REASONING_TEMPLATE_FLAG: False}

    def test_the_training_description_names_the_mechanism_and_the_effect(self):
        record = describe_chat_handling(
            config().model, SimpleNamespace(chat_template="{{ enable_thinking }}")
        )
        assert "chat_template_kwargs" in record["effective"]
        assert "<think></think>" in record["effective"]

    def test_a_template_without_the_flag_is_reported_as_inert_at_either_stage(self):
        for stage in ("training", "inference"):
            record = describe_chat_handling(
                config().model,
                SimpleNamespace(chat_template="{{ messages }}"),
                stage=stage,
            )
            assert record["effective"].startswith("inert"), stage
            assert record["template_kwargs"] == {}

    def test_the_description_works_before_a_tokenizer_exists(self):
        for stage in ("training", "inference"):
            record = describe_chat_handling(config().model, None, stage=stage)
            assert record["reasoning_mode"] == "disabled"
            assert record["effective"] is None, stage

    def test_no_chat_template_kwargs_column_without_a_tokenizer(self):
        """Whether the template understands the flag is settled by reading the template.

        A caller with no tokenizer cannot know, so nothing is emitted rather than guessed.
        """
        record = build_training_records([example()], config())[0]
        assert CHAT_TEMPLATE_KWARGS_COLUMN not in record

    def test_the_column_is_emitted_when_the_template_understands_the_flag(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        record = build_training_records([example()], config(), tokenizer=tokenizer)[0]
        assert record[CHAT_TEMPLATE_KWARGS_COLUMN] == {REASONING_TEMPLATE_FLAG: False}

    def test_the_column_is_absent_when_the_template_ignores_the_flag(self):
        """Passing an unknown keyword is at best ignored and at worst a Jinja error."""
        tokenizer = SimpleNamespace(chat_template="{{ messages }}")
        record = build_training_records([example()], config(), tokenizer=tokenizer)[0]
        assert CHAT_TEMPLATE_KWARGS_COLUMN not in record

    def test_the_column_is_absent_under_inherit(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        cfg = config(model={"reasoning_mode": "inherit"})
        record = build_training_records([example()], cfg, tokenizer=tokenizer)[0]
        assert CHAT_TEMPLATE_KWARGS_COLUMN not in record

    def test_the_column_carries_true_when_reasoning_is_enabled(self):
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")
        cfg = config(model={"reasoning_mode": "enabled"})
        record = build_training_records([example()], cfg, tokenizer=tokenizer)[0]
        assert record[CHAT_TEMPLATE_KWARGS_COLUMN] == {REASONING_TEMPLATE_FLAG: True}

    def test_the_column_is_only_emitted_for_the_conversational_shape(self):
        """TRL reads it inside the branch that applies a chat template; the others never do."""
        tokenizer = SimpleNamespace(chat_template="{{ enable_thinking }}")

        # Packing forces language modeling. The mode is still 'disabled' here, so the shape is
        # the only thing suppressing the column.
        packed = config(training={"packing": True, "completion_only_loss": False})
        record = build_training_records([example()], packed, tokenizer=tokenizer)[0]
        assert set(record) == {"text", "example_id"}

        # Phase 17A already refuses 'plain' together with an explicit reasoning mode, so the
        # prompt-completion shape can only ever be reached under 'inherit'.
        from qa_gen import GenerationConfigError

        plain = config(model={"chat_template": "plain", "reasoning_mode": "inherit"})
        record = build_training_records([example()], plain, tokenizer=tokenizer)[0]
        assert CHAT_TEMPLATE_KWARGS_COLUMN not in record
        with pytest.raises(GenerationConfigError, match="reasoning mode is a chat-template"):
            config(model={"chat_template": "plain"})


class TestTrainerArgumentTranslation:
    """Verified against TRL v0.29.1 and transformers 5.16.1 naming."""

    def test_trl_uses_max_length_not_max_seq_length(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["max_length"] == 1024
        assert "max_seq_length" not in plan.arguments

    def test_transformers_five_uses_eval_strategy(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert "eval_strategy" in plan.arguments
        assert "evaluation_strategy" not in plan.arguments

    def test_warmup_ratio_is_converted_to_steps(self):
        """Transformers 5.x removed warmup_ratio entirely."""
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=1000)
        assert "warmup_ratio" not in plan.arguments
        assert plan.arguments["warmup_steps"] == plan.warmup_steps
        assert plan.warmup_steps > 0

    def test_total_steps_are_derived_from_the_corpus_size(self):
        cfg = config(
            training={
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 3,
            }
        )
        steps, total, _ = resolve_warmup_steps(cfg, train_examples=80)
        assert total == 30
        assert steps == max(1, round(30 * cfg.training.warmup_ratio))

    def test_max_steps_takes_precedence_over_the_corpus_size(self):
        cfg = config(training={"max_steps": 50})
        _, total, notes = resolve_warmup_steps(cfg, train_examples=100_000)
        assert total == 50
        assert any("max_steps" in note for note in notes)

    def test_a_warmup_ratio_without_a_step_count_is_refused(self):
        """Silently dropping warmup would change the schedule without saying so."""
        with pytest.raises(TrainerBuildError, match="warmup_ratio"):
            resolve_warmup_steps(config(), train_examples=None)

    def test_zero_warmup_needs_no_step_count(self):
        steps, total, _ = resolve_warmup_steps(
            config(training={"warmup_ratio": 0.0}), train_examples=None
        )
        assert steps == 0
        assert total is None

    def test_the_optimizer_is_forwarded_as_optim(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["optim"] == "paged_adamw_8bit"

    def test_the_optimizer_is_marked_unbenchmarked(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert any("not been benchmarked" in note for note in plan.notes)

    def test_the_optimizer_is_configurable(self):
        plan = plan_trainer_arguments(
            config(training={"optimizer": "adamw_torch"}), "/tmp/out", train_examples=100
        )
        assert plan.arguments["optim"] == "adamw_torch"

    def test_precision_flags_come_from_the_resolved_plan(self, cuda_bf16):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["bf16"] is True
        assert plan.arguments["fp16"] is False

    def test_checkpointing_uses_non_reentrant_for_peft(self):
        """The reentrant implementation drops adapter gradients on some architectures."""
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    def test_checkpointing_kwargs_are_absent_when_disabled(self):
        plan = plan_trainer_arguments(
            config(training={"gradient_checkpointing": False}), "/tmp/out", train_examples=10
        )
        assert "gradient_checkpointing_kwargs" not in plan.arguments

    def test_nothing_is_reported_to_an_external_tracker_by_default(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["report_to"] == []

    def test_completion_only_loss_is_forwarded(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert plan.arguments["completion_only_loss"] is True
        assert plan.arguments["packing"] is False

    def test_the_output_directory_is_forwarded_as_a_string(self):
        plan = plan_trainer_arguments(config(), "/runs/abc", train_examples=100)
        assert plan.arguments["output_dir"] == "/runs/abc"

    def test_max_steps_is_omitted_when_unset(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        assert "max_steps" not in plan.arguments

    def test_the_plan_is_json_serializable(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        payload = json.loads(json.dumps(plan.as_dict()))
        assert payload["arguments"]["max_length"] == 1024

    def test_the_translation_needs_no_trl_installed(self):
        """Which is why it is a separate function from build_sft_config."""
        assert deps_module.is_available("trl") is False
        assert plan_trainer_arguments(config(), "/tmp/out", train_examples=10).arguments


class TestSftConfigConstruction:
    """Against a stub that accepts exactly the v0.29.1 argument names."""

    def test_the_config_is_built_from_the_translated_plan(self, fake_trl):
        sft_config, plan = build_sft_config(config(), "/tmp/out", train_examples=100)
        assert sft_config.max_length == 1024
        assert sft_config.optim == "paged_adamw_8bit"
        assert plan.dropped_arguments == ()

    def test_every_translated_argument_is_accepted_by_the_verified_api(self, fake_trl):
        """The stub rejects unknown keywords, so this catches a rename against v0.29.1."""
        _, plan = build_sft_config(config(), "/tmp/out", train_examples=100)
        assert plan.dropped_arguments == (), (
            f"arguments not accepted by TRL v0.29.1: {plan.dropped_arguments}"
        )

    def test_an_unsupported_argument_is_dropped_and_recorded(self, monkeypatch):
        class NarrowSFTConfig:
            """A TRL version that predates max_length."""

            def __init__(  # noqa: ANN204 - a stub of an older signature
                self, output_dir, learning_rate, max_seq_length=None
            ):
                self.output_dir = output_dir

        narrow = SimpleNamespace(SFTConfig=NarrowSFTConfig, SFTTrainer=FakeSFTTrainer)
        from qa_gen_runtime import trainer as trainer_module

        monkeypatch.setattr(trainer_module, "require_trl", lambda: narrow)
        _, plan = build_sft_config(config(), "/tmp/out", train_examples=100)
        assert "max_length" in plan.dropped_arguments
        assert "optim" in plan.dropped_arguments

    def test_a_rejected_value_is_reported_with_trls_own_complaint(self, monkeypatch):
        class AngrySFTConfig:
            """A config class that refuses every value."""

            def __init__(self, **kwargs) -> None:
                raise ValueError("learning_rate must be positive")

        angry = SimpleNamespace(SFTConfig=AngrySFTConfig, SFTTrainer=FakeSFTTrainer)
        from qa_gen_runtime import trainer as trainer_module

        monkeypatch.setattr(trainer_module, "require_trl", lambda: angry)
        with pytest.raises(TrainerBuildError, match="learning_rate must be positive"):
            build_sft_config(config(), "/tmp/out", train_examples=100)

    def test_building_needs_trl_and_says_so(self):
        with pytest.raises(RuntimeDependencyError, match="trl"):
            build_sft_config(config(), "/tmp/out", train_examples=100)


class TestTrainerFactory:
    """The trainer is constructed with the v0.29.1 signature and no peft_config."""

    def dataset(self) -> list[dict[str, Any]]:
        """A tiny record list that supports len()."""
        return build_training_records([example(0), example(1)], config())

    def test_the_tokenizer_is_passed_as_processing_class(self, fake_trl):
        """TRL renamed `tokenizer` to `processing_class`."""
        tokenizer = SimpleNamespace(name="tok")
        trainer, _ = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=tokenizer,
            train_dataset=self.dataset(),
            output_dir="/tmp/out",
        )
        assert trainer.processing_class is tokenizer

    def test_peft_config_is_never_passed(self, fake_trl):
        """The loader owns adapter attachment; TRL would wrap the model a second time."""
        trainer, _ = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(),
            train_dataset=self.dataset(),
            output_dir="/tmp/out",
        )
        assert trainer.peft_config is None

    def test_the_dataset_is_forwarded(self, fake_trl):
        records = self.dataset()
        trainer, _ = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(),
            train_dataset=records,
            output_dir="/tmp/out",
        )
        assert trainer.train_dataset is records

    def test_the_eval_dataset_is_forwarded_when_given(self, fake_trl):
        trainer, _ = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(),
            train_dataset=self.dataset(),
            eval_dataset=self.dataset(),
            output_dir="/tmp/out",
        )
        assert trainer.eval_dataset is not None

    def test_the_corpus_size_is_derived_for_the_warmup_conversion(self, fake_trl):
        _, plan = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(),
            train_dataset=self.dataset(),
            output_dir="/tmp/out",
        )
        assert plan.total_steps is not None

    def test_a_missing_dataset_is_refused_before_construction(self, fake_trl):
        with pytest.raises(TrainerBuildError, match="train_dataset is required"):
            build_trainer(
                config(),
                model=FakeModel(),
                tokenizer=SimpleNamespace(),
                train_dataset=None,
                output_dir="/tmp/out",
            )

    def test_an_incompatible_trl_signature_is_reported(self, monkeypatch):
        class OldTrainer:
            """A pre-0.20 TRL that still called the tokenizer `tokenizer`."""

            def __init__(  # noqa: ANN204 - a stub of an older signature
                self, model=None, args=None, train_dataset=None, tokenizer=None
            ):
                self.tokenizer = tokenizer

        old = SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=OldTrainer)
        from qa_gen_runtime import trainer as trainer_module

        monkeypatch.setattr(trainer_module, "require_trl", lambda: old)
        with pytest.raises(TrainerBuildError, match="processing_class"):
            build_trainer(
                config(),
                model=FakeModel(),
                tokenizer=SimpleNamespace(),
                train_dataset=self.dataset(),
                output_dir="/tmp/out",
            )

    def test_the_trainer_is_not_trained_by_building_it(self, fake_trl):
        trainer, _ = build_trainer(
            config(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(),
            train_dataset=self.dataset(),
            output_dir="/tmp/out",
        )
        assert trainer.trained is False


class TestDiagnostics:
    """Everything required, and honest about what has not been measured."""

    def report(self, **kwargs) -> Any:
        """Collect diagnostics for the default configuration."""
        return collect_diagnostics(config(), **kwargs)

    @pytest.mark.parametrize(
        "field_name",
        [
            "model_id",
            "device",
            "memory",
            "quantization",
            "lora",
            "sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "gradient_checkpointing",
            "optimizer",
            "dependencies",
        ],
    )
    def test_every_required_field_is_present(self, field_name):
        assert field_name in self.report().as_dict()

    def test_gpu_name_and_vram_are_reported_when_present(self):
        device = self.report().as_dict()["device"]
        assert "device_name" in device
        assert "total_vram_gib" in device

    def test_allocated_and_reserved_memory_are_both_reported(self):
        """They differ, and it is the reserved figure the baseline is comparable to."""
        memory = self.report().as_dict()["memory"]
        for key in ("allocated_gib", "reserved_gib", "max_allocated_gib", "max_reserved_gib"):
            assert key in memory

    def test_memory_fields_are_none_without_cuda(self):
        assert self.report().as_dict()["memory"]["allocated_gib"] is None

    def test_parameter_counts_are_measured_not_estimated(self):
        report = self.report(model=FakeModel())
        assert report.trainable_parameters == VERIFIED_QWEN3_4B_L4.trainable_parameters
        assert report.trainable_fraction == pytest.approx(0.008145, abs=1e-6)

    def test_parameter_counts_are_absent_before_the_model_loads(self):
        """An honest absent number beats a plausible wrong one."""
        report = self.report()
        assert report.trainable_parameters is None
        assert report.trainable_fraction is None

    def test_the_lora_settings_are_reported_in_full(self):
        lora = self.report().as_dict()["lora"]
        assert lora["rank"] == 16
        assert lora["alpha"] == 32
        assert lora["scaling"] == 2.0
        assert len(lora["target_modules"]) == 7

    def test_the_quantization_settings_are_reported(self):
        quantization = self.report().as_dict()["quantization"]
        assert quantization["quant_type"] == "nf4"
        assert quantization["double_quant"] is True

    def test_the_batch_arithmetic_is_reported(self):
        report = self.report().as_dict()
        assert report["effective_batch_size"] == (
            report["batch_size"] * report["gradient_accumulation_steps"]
        )

    def test_unbenchmarked_settings_are_flagged(self):
        notes = " ".join(self.report().notes)
        assert "batch 1" in notes
        assert "not been benchmarked" in notes

    def test_the_baseline_comparison_is_included(self):
        assert self.report().is_verified_configuration is True
        deviant = collect_diagnostics(config(model={"max_seq_length": 2048}))
        assert deviant.is_verified_configuration is False
        assert deviant.baseline_deviations

    def test_dependency_versions_are_recorded(self):
        dependencies = self.report().as_dict()["dependencies"]
        assert dependencies["transformers"]["version"]
        assert dependencies["peft"]["available"] is False

    def test_the_report_is_json_serializable(self):
        payload = json.loads(json.dumps(self.report(model=FakeModel()).as_dict()))
        assert payload["trainable_parameters"] == VERIFIED_QWEN3_4B_L4.trainable_parameters

    def test_the_trainer_plan_can_be_embedded(self):
        plan = plan_trainer_arguments(config(), "/tmp/out", train_examples=100)
        report = self.report(trainer_plan=plan.as_dict())
        assert report.as_dict()["trainer_plan"]["arguments"]["max_length"] == 1024

    def test_diagnostics_attach_to_phase_17a_run_metadata(self):
        metadata = TrainingRunMetadata(run_id="r1", experiment_name="runtime-test")
        attach_to_metadata(metadata, self.report(model=FakeModel()))
        payload = json.loads(metadata.to_json())
        assert payload["training"]["diagnostics"]["lora"]["rank"] == 16
        assert payload["training"]["measured_baseline"]["gpu"] == "NVIDIA L4"

    def test_attaching_promotes_the_measured_counts_to_the_record(self):
        metadata = TrainingRunMetadata(run_id="r1", experiment_name="runtime-test")
        attach_to_metadata(metadata, self.report(model=FakeModel()))
        assert metadata.trainable_fraction == pytest.approx(0.008145, abs=1e-6)
        assert metadata.base_model == "Qwen/Qwen3-4B"

    def test_the_chat_and_reasoning_handling_is_reported(self):
        chat = self.report().as_dict()["chat"]
        assert chat["reasoning_mode"] == "disabled"
        assert chat["reasoning_suppressed"] is True


class TestOutputPaths:
    """Run output is configurable and already ignored by git."""

    def test_the_default_root_is_under_artifacts(self):
        assert resolve_run_root().name == "qgen-runs"
        assert "artifacts" in resolve_run_root().parts

    def test_the_default_root_is_git_ignored(self):
        """No .gitignore change was needed: artifacts/ has been ignored since Phase 1."""
        import subprocess

        from qa_ml.paths import find_repo_root

        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                "--no-index",
                "artifacts/qgen-runs/r1/adapter/x.safetensors",
            ],
            cwd=find_repo_root(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, "run output is not git-ignored"

    def test_the_root_is_configurable(self, tmp_path):
        assert resolve_run_root(tmp_path / "elsewhere").name == "elsewhere"

    def test_a_run_directory_is_created_with_every_path(self, tmp_path):
        paths = create_run_directory(
            config(), output_dir=tmp_path, timestamp="20260906T120000Z"
        )
        assert paths.root.is_dir()
        assert paths.adapter.name == "adapter"
        assert paths.record.name == "run.json"
        assert paths.config.name == "config.resolved.json"
        assert paths.diagnostics.name == "diagnostics.json"

    def test_the_run_id_is_embedded_in_the_directory_name(self, tmp_path):
        paths = create_run_directory(
            config(), output_dir=tmp_path, timestamp="20260906T120000Z"
        )
        assert "Qwen--Qwen3-4B" in paths.root.name
        assert paths.root.name.endswith("20260906T120000Z")

    def test_an_existing_directory_is_refused(self, tmp_path):
        """A finished run is evidence and must not be overwritten by re-running a command."""
        create_run_directory(config(), output_dir=tmp_path, run_id="fixed")
        with pytest.raises(RunOutputError, match="already exists"):
            create_run_directory(config(), output_dir=tmp_path, run_id="fixed")

    def test_reuse_can_be_requested_explicitly(self, tmp_path):
        create_run_directory(config(), output_dir=tmp_path, run_id="fixed")
        paths = create_run_directory(
            config(), output_dir=tmp_path, run_id="fixed", allow_existing=True
        )
        assert paths.root.is_dir()

    def test_paths_serialize_with_forward_slashes(self, tmp_path):
        paths = create_run_directory(config(), output_dir=tmp_path, run_id="fixed")
        payload = json.loads(json.dumps(paths.as_dict()))
        assert "\\" not in payload["adapter"]

    def test_no_directory_is_created_by_resolving_a_root(self, tmp_path):
        target_dir = tmp_path / "not-created"
        resolve_run_root(target_dir)
        assert not target_dir.exists()

    def test_paths_derive_consistently_from_a_root(self, tmp_path):
        paths = RunPaths.under(tmp_path / "run")
        assert paths.adapter.parent == paths.root
        assert paths.record.parent == paths.root


class TestShippedGenerativeConfigs:
    """Every generative config must load and validate.

    The counterpart to ``tests/test_config.py::TestShippedConfigs`` for this config family.
    That test asserts every ``ml/configs/*.yaml`` is a ``qa_ml.ExperimentConfig``, which is a
    real contract and the reason the generative configs live in ``ml/configs/qgen/`` -- they
    have a different schema and a different loader. This guard gives the new directory the same
    protection: a config added there cannot escape validation.
    """

    def config_dir(self):
        """Path to the generative configuration directory."""
        from qa_ml.paths import find_repo_root

        return find_repo_root() / "ml" / "configs" / "qgen"

    def test_the_directory_exists(self):
        assert self.config_dir().is_dir()

    def test_every_shipped_generative_config_loads_and_validates(self):
        paths = sorted(self.config_dir().glob("*.yaml"))
        assert paths, "no generative configs found; this guard would pass vacuously"
        for path in paths:
            loaded = load_experiment_config(path)
            loaded.validate()
            assert loaded.name

    def test_generative_configs_are_not_in_the_extractive_directory(self):
        """The parent directory's contract is that every *.yaml there is an ExperimentConfig."""
        from qa_ml.paths import find_repo_root

        parent = find_repo_root() / "ml" / "configs"
        top_level = {path.name for path in parent.glob("*.yaml")}
        assert not any(name.startswith("qgen") for name in top_level)


class TestConfigIO:
    """YAML and JSON, using the PyYAML the project already pins."""

    def test_the_shipped_smoke_config_loads_and_validates(self):
        from qa_ml.paths import find_repo_root

        loaded = load_experiment_config(find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml")
        assert loaded.name == "qgen-smoke"
        assert loaded.model.max_seq_length == 1024

    def test_the_shipped_smoke_config_matches_the_measured_baseline(self):
        from qa_ml.paths import find_repo_root

        loaded = load_experiment_config(find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml")
        assert loaded.baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()

    def test_yaml_round_trips_through_a_file(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("name: from-yaml\nlora:\n  rank: 8\n  alpha: 16\n", encoding="utf-8")
        loaded = load_experiment_config(path)
        assert loaded.name == "from-yaml"
        assert loaded.lora.rank == 8

    def test_json_is_accepted(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"name": "from-json"}), encoding="utf-8")
        assert load_experiment_config(path).name == "from-json"

    def test_a_resolved_config_round_trips_through_json(self, tmp_path):
        original = config(description="d", lora={"rank": 32, "alpha": 64})
        written = write_resolved_config(original, tmp_path / "resolved.json")
        assert load_experiment_config(written) == original

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ConfigIOError, match="not found"):
            load_mapping(tmp_path / "absent.yaml")

    def test_an_unsupported_extension_is_reported(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("name = 'x'", encoding="utf-8")
        with pytest.raises(ConfigIOError, match="unsupported configuration format"):
            load_mapping(path)

    def test_an_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigIOError, match="empty"):
            load_mapping(path)

    def test_a_non_mapping_document_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigIOError, match="mapping at the top level"):
            load_mapping(path)

    def test_malformed_yaml_is_reported(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigIOError, match="could not parse YAML"):
            load_mapping(path)

    def test_a_config_error_propagates_unwrapped(self, tmp_path):
        """The message naming the offending field must reach the caller intact."""
        from qa_gen import GenerationConfigError

        path = tmp_path / "c.yaml"
        path.write_text("name: bad\nlora:\n  rank: 0\n", encoding="utf-8")
        with pytest.raises(GenerationConfigError, match="rank"):
            load_experiment_config(path)

    def test_overrides_merge_per_section(self, tmp_path):
        """Overriding one training field must not discard the rest of the section."""
        path = tmp_path / "c.yaml"
        path.write_text(
            "name: o\ntraining:\n  num_train_epochs: 5\n  logging_steps: 3\n", encoding="utf-8"
        )
        loaded = load_experiment_config(path, overrides={"training": {"max_steps": 7}})
        assert loaded.training.max_steps == 7
        assert loaded.training.num_train_epochs == 5
        assert loaded.training.logging_steps == 3


class TestCli:
    """Safe by default: validation downloads nothing and writes nothing."""

    def smoke_config_path(self) -> str:
        """Path to the shipped smoke configuration."""
        from qa_ml.paths import find_repo_root

        return str(find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml")

    def test_training_is_not_the_default(self):
        args = build_parser().parse_args(["--config", "c.yaml"])
        assert args.execute_training is False
        assert args.plan is False

    def test_a_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_validation_succeeds_and_reports(self, capsys):
        code = main(["--config", self.smoke_config_path(), "--log-level", "WARNING"])
        assert code == 0
        output = capsys.readouterr().out
        assert "Qwen/Qwen3-4B" in output
        assert "verified" in output

    def test_validation_creates_nothing(self, tmp_path):
        target_dir = tmp_path / "runs"
        code = main(
            [
                "--config",
                self.smoke_config_path(),
                "--output-dir",
                str(target_dir),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        assert not target_dir.exists()

    def test_validation_does_not_import_peft_or_trl(self):
        """It could not, since they are absent; asserted so it stays true when they are not."""
        report = run_validation(self.smoke_config_path())
        assert report["status"] == "valid"
        assert report["diagnostics"]["dependencies"]["peft"]["available"] is False

    def test_the_plan_flag_adds_the_translated_arguments(self):
        report = run_validation(self.smoke_config_path(), include_plan=True)
        assert report["diagnostics"]["trainer_plan"]["arguments"]["max_length"] == 1024

    def test_json_output_is_parseable(self, capsys):
        code = main(
            ["--config", self.smoke_config_path(), "--json", "--log-level", "WARNING"]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["experiment"] == "qgen-smoke"

    def test_an_invalid_config_exits_non_zero(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text("name: bad\nlora:\n  rank: 0\n", encoding="utf-8")
        code = main(["--config", str(path), "--log-level", "WARNING"])
        assert code == 1
        assert "rank" in capsys.readouterr().err

    def test_a_missing_config_exits_non_zero(self, tmp_path, capsys):
        code = main(
            ["--config", str(tmp_path / "absent.yaml"), "--log-level", "WARNING"]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_plan_and_execute_training_cannot_be_combined(self):
        """Argparse rejects the pair, so a plan command is never one typo from a real run."""
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(
                ["--config", "c.yaml", "--plan", "--execute-training"]
            )
        assert excinfo.value.code == 2

    def test_execute_training_does_not_fall_back_to_validation(self, capsys):
        """It either trains or fails. Printing a validation report would imply it had run."""
        code = main(
            [
                "--config",
                self.smoke_config_path(),
                "--execute-training",
                "--dataset-dir",
                "definitely-not-a-directory",
                "--log-level",
                "WARNING",
            ]
        )
        assert code != 0
        assert "Qwen/Qwen3-4B" not in capsys.readouterr().out

    def test_the_execution_flags_default_to_nothing_selected(self):
        """Every execution-only flag is absent by default, so validation is unaffected."""
        args = build_parser().parse_args(["--config", "c.yaml"])
        assert args.dataset_dir is None
        assert args.dataset_fingerprint is None
        assert args.eval_split is None
        assert args.run_id is None
        assert args.resume_from_checkpoint is None
        assert args.expect_train_examples is None
        assert args.expect_source_count == []
        assert args.split == "train"

    def test_the_report_names_the_record_format(self):
        report = run_validation(self.smoke_config_path())
        assert report["record_format"] == "conversational"

    def test_the_report_states_whether_the_configuration_is_verified(self):
        report = run_validation(self.smoke_config_path())
        assert report["is_verified_configuration"] is True
        assert report["baseline_deviations"] == []


class TestNothingDownloadsOrTrains:
    """The constraints of this phase, asserted rather than assumed."""

    def test_no_runtime_module_calls_from_pretrained_at_import_time(self):
        """Importing the package must be free of network and allocation."""
        import subprocess
        import textwrap

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import qa_gen_runtime  # noqa: F401
                    print("IMPORTED")
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORTED" in result.stdout

    def test_no_runtime_module_calls_train(self):
        import ast
        import importlib
        import inspect

        modules = (
            "deps",
            "precision",
            "quantization",
            "loader",
            "chat",
            "dataset",
            "trainer",
            "diagnostics",
            "outputs",
            "config_io",
            "train",
        )
        for name in modules:
            module = importlib.import_module(f"qa_gen_runtime.{name}")
            tree = ast.parse(inspect.getsource(module))
            called = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert "train" not in called, f"qa_gen_runtime.{name} calls .train()"
            assert "backward" not in called, f"qa_gen_runtime.{name} calls .backward()"

    def test_no_dataset_is_downloaded(self):
        import ast
        import importlib
        import inspect

        for name in ("dataset", "config_io", "train"):
            module = importlib.import_module(f"qa_gen_runtime.{name}")
            tree = ast.parse(inspect.getsource(module))
            names = {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            assert "load_dataset" not in names, f"qa_gen_runtime.{name} loads a dataset"

    def test_the_extractive_predict_route_is_untouched(self):
        from app.main import app

        paths = {getattr(route, "path", None) for route in app.routes}
        assert {"/health", "/predict"} <= paths
