"""Tests for model loading and checkpoint integrity.

The checkpoint tests here exist because of a real, measured hazard rather than a
hypothetical one. ``bert-base-uncased`` stores its LayerNorm parameters as
``LayerNorm.gamma`` / ``LayerNorm.beta``, a holdover from the original TensorFlow
release, and ``transformers`` maps those names in **both** directions:
``save_pretrained`` writes ``.gamma``/``.beta`` back out and ``from_pretrained`` maps
them in again.

That round trip is lossless. A load path that skips the mapping is not: measured on
``transformers`` 5.16.1, a raw ``load_state_dict(strict=False)`` over a saved BERT
checkpoint reports 50 missing ``LayerNorm.weight``/``.bias`` keys and 50 unexpected
``.gamma``/``.beta`` keys, and leaves those 50 tensors silently unloaded.

These tests pin the property that matters: a checkpoint written by this project
reloads, through this project's loader, to bit-identical parameters.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import BertConfig, BertForQuestionAnswering

from qa_torch.loader import (
    CheckpointIntegrityError,
    ModelLoadError,
    count_parameters,
    describe_model,
    load_qa_model,
    load_tokenizer,
    verify_checkpoint_integrity,
)

# A small BERT built from config: no download, but the same architecture and
# therefore the same legacy LayerNorm key mapping as bert-base-uncased.
TINY_BERT = {
    "vocab_size": 256,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
}


@pytest.fixture
def tiny_bert_qa() -> BertForQuestionAnswering:
    """A small BertForQuestionAnswering with every parameter perturbed.

    Perturbing everything is what makes the round-trip assertions meaningful: any
    parameter silently reset to a default (LayerNorm weight 1.0 / bias 0.0, or a
    freshly initialised head) becomes detectable.
    """
    torch.manual_seed(0)
    model = BertForQuestionAnswering(BertConfig(**TINY_BERT))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return model


def _to_legacy_layernorm_names(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename LayerNorm ``.weight``/``.bias`` to the legacy ``.gamma``/``.beta``.

    Reproduces the on-disk convention of the published ``bert-base-uncased``
    checkpoint offline, so the hazard can be tested without a download.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if "LayerNorm" in key and key.endswith(".weight"):
            renamed[key[: -len(".weight")] + ".gamma"] = tensor
        elif "LayerNorm" in key and key.endswith(".bias"):
            renamed[key[: -len(".bias")] + ".beta"] = tensor
        else:
            renamed[key] = tensor
    return renamed


class TestLegacyLayerNormKeyNames:
    """Document and pin the naming behaviour behind the reload warning.

    Note on scope: the legacy naming follows the **checkpoint's history**, not the
    architecture. A model built from ``BertConfig`` saves modern ``.weight``/``.bias``;
    a model loaded from ``bert-base-uncased`` preserves ``.gamma``/``.beta`` on save.
    The offline tests here therefore construct the legacy convention explicitly, and
    the real-checkpoint behaviour is asserted in :class:`TestRealBertCheckpoint`.
    """

    def test_a_raw_state_dict_load_loses_legacy_named_keys(self, tiny_bert_qa):
        """The failure mode being guarded against, demonstrated rather than asserted."""
        legacy = _to_legacy_layernorm_names(tiny_bert_qa.state_dict())
        assert any(key.endswith((".gamma", ".beta")) for key in legacy)

        fresh = BertForQuestionAnswering(BertConfig(**TINY_BERT))
        result = fresh.load_state_dict(legacy, strict=False)

        missing_layernorm = [k for k in result.missing_keys if "LayerNorm" in k]
        unexpected_legacy = [
            k for k in result.unexpected_keys if k.endswith((".gamma", ".beta"))
        ]
        assert missing_layernorm, (
            "a raw load_state_dict was expected to report missing LayerNorm.weight/bias"
        )
        assert unexpected_legacy, (
            "a raw load_state_dict was expected to report unexpected .gamma/.beta"
        )
        # 3 layers x 2 LayerNorms + 1 embedding LayerNorm = 7 modules x 2 params.
        assert len(missing_layernorm) == len(unexpected_legacy) == 14

    def test_a_raw_load_leaves_layernorm_at_the_wrong_values(self, tiny_bert_qa):
        """Nothing crashes -- the tensors are just silently not the saved ones."""
        legacy = _to_legacy_layernorm_names(tiny_bert_qa.state_dict())
        fresh = BertForQuestionAnswering(BertConfig(**TINY_BERT))
        fresh.load_state_dict(legacy, strict=False)

        expected = tiny_bert_qa.bert.embeddings.LayerNorm.weight.detach()
        actual = fresh.bert.embeddings.LayerNorm.weight.detach()
        assert not torch.equal(actual, expected), (
            "the raw load unexpectedly restored LayerNorm; the hazard this guard exists "
            "for may no longer apply and the rationale should be revisited"
        )
        # A fresh nn.LayerNorm starts at exactly all-ones: the reinitialisation tell.
        assert torch.allclose(actual, torch.ones_like(actual))

    def test_from_pretrained_maps_legacy_names_correctly(self, tiny_bert_qa, tmp_path):
        """The path this project uses. Same legacy file, nothing lost."""
        from safetensors.torch import save_file

        tiny_bert_qa.config.save_pretrained(tmp_path)
        save_file(
            _to_legacy_layernorm_names(tiny_bert_qa.state_dict()),
            str(tmp_path / "model.safetensors"),
            metadata={"format": "pt"},
        )

        reloaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        before = dict(tiny_bert_qa.named_parameters())
        after = dict(reloaded.named_parameters())
        for name, expected in before.items():
            if "LayerNorm" not in name:
                continue
            assert torch.equal(after[name].detach(), expected.detach()), (
                f"{name} was not restored from its legacy .gamma/.beta name"
            )


class TestCheckpointRoundTrip:
    """The property that actually matters for reported metrics."""

    def test_every_parameter_survives_save_and_reload(self, tiny_bert_qa, tmp_path):
        tiny_bert_qa.save_pretrained(tmp_path)
        reloaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        before = {n: t.detach().clone() for n, t in tiny_bert_qa.named_parameters()}
        after = {n: t.detach().clone() for n, t in reloaded.named_parameters()}

        assert set(before) == set(after), "the parameter key set changed across reload"
        drifted = [n for n in before if not torch.equal(before[n], after[n])]
        assert not drifted, f"{len(drifted)} parameter(s) changed across reload: {drifted[:5]}"

    def test_no_layernorm_is_left_at_its_default_value(self, tiny_bert_qa, tmp_path):
        """All-ones weight or all-zeros bias is the signature of a reinitialised param."""
        tiny_bert_qa.save_pretrained(tmp_path)
        reloaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        at_default = []
        for name, tensor in reloaded.named_parameters():
            if "LayerNorm" not in name:
                continue
            detached = tensor.detach()
            if name.endswith(".weight") and torch.allclose(detached, torch.ones_like(detached)):
                at_default.append(name)
            if name.endswith(".bias") and torch.allclose(detached, torch.zeros_like(detached)):
                at_default.append(name)
        assert not at_default, f"LayerNorm parameters left at defaults: {at_default}"

    def test_logits_are_identical_after_reload(self, tiny_bert_qa, tmp_path):
        """The end-to-end consequence: identical weights must give identical outputs."""
        tiny_bert_qa.eval()
        probe = {
            "input_ids": torch.randint(0, 256, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        with torch.inference_mode():
            before = tiny_bert_qa(**probe)

        tiny_bert_qa.save_pretrained(tmp_path)
        reloaded = load_qa_model(str(tmp_path), expect_trained_head=True)
        reloaded.eval()
        with torch.inference_mode():
            after = reloaded(**probe)

        assert torch.equal(before.start_logits, after.start_logits)
        assert torch.equal(before.end_logits, after.end_logits)

    def test_trained_qa_head_is_preserved(self, tiny_bert_qa, tmp_path):
        """The head is freshly initialised on a base checkpoint; it must persist here."""
        expected = tiny_bert_qa.qa_outputs.weight.detach().clone()
        tiny_bert_qa.save_pretrained(tmp_path)
        reloaded = load_qa_model(str(tmp_path), expect_trained_head=True)
        assert torch.equal(reloaded.qa_outputs.weight.detach(), expected)


class TestVerifyCheckpointIntegrity:
    """The guard itself must both pass on good checkpoints and fail on bad ones."""

    def test_passes_for_a_faithfully_saved_checkpoint(self, tiny_bert_qa, tmp_path):
        tiny_bert_qa.save_pretrained(tmp_path)
        report = verify_checkpoint_integrity(tiny_bert_qa, tmp_path)

        assert report["ok"] is True
        assert report["num_drifted"] == 0
        assert report["max_abs_delta"] == 0.0
        assert report["missing_after_reload"] == []
        assert report["unexpected_after_reload"] == []
        assert report["layernorm_at_default_values"] == 0
        assert report["num_parameters_checked"] > 0

    def test_raises_when_the_checkpoint_is_not_the_model(self, tiny_bert_qa, tmp_path):
        """Save, then change the model, so disk and memory genuinely disagree."""
        tiny_bert_qa.save_pretrained(tmp_path)
        with torch.no_grad():
            tiny_bert_qa.qa_outputs.weight.add_(1.0)

        with pytest.raises(CheckpointIntegrityError, match="does NOT reload"):
            verify_checkpoint_integrity(tiny_bert_qa, tmp_path)

    def test_non_strict_mode_reports_without_raising(self, tiny_bert_qa, tmp_path):
        tiny_bert_qa.save_pretrained(tmp_path)
        with torch.no_grad():
            tiny_bert_qa.qa_outputs.weight.add_(1.0)

        report = verify_checkpoint_integrity(tiny_bert_qa, tmp_path, strict=False)
        assert report["ok"] is False
        assert report["num_drifted"] >= 1
        assert report["max_abs_delta"] > 0.0

    def test_error_message_says_not_to_trust_the_metrics(self, tiny_bert_qa, tmp_path):
        tiny_bert_qa.save_pretrained(tmp_path)
        with torch.no_grad():
            tiny_bert_qa.qa_outputs.bias.add_(2.0)

        with pytest.raises(CheckpointIntegrityError) as excinfo:
            verify_checkpoint_integrity(tiny_bert_qa, tmp_path)
        assert "Do not report" in str(excinfo.value)

    def test_layernorm_drift_is_detected(self, tiny_bert_qa, tmp_path):
        """The specific hazard: LayerNorm silently differing from the checkpoint."""
        tiny_bert_qa.save_pretrained(tmp_path)
        with torch.no_grad():
            tiny_bert_qa.bert.embeddings.LayerNorm.weight.fill_(1.0)

        report = verify_checkpoint_integrity(tiny_bert_qa, tmp_path, strict=False)
        assert report["ok"] is False
        assert report["num_drifted"] >= 1


class TestLoaderErrors:
    def test_unknown_model_id_gives_an_actionable_error(self):
        with pytest.raises(ModelLoadError, match="Could not load"):
            load_qa_model("this-model-definitely-does-not-exist-xyz")

    def test_unknown_tokenizer_id_gives_an_actionable_error(self):
        with pytest.raises(ModelLoadError, match="Could not load a tokenizer"):
            load_tokenizer("this-tokenizer-definitely-does-not-exist-xyz")


class TestDescribeModel:
    def test_reports_measured_facts(self, tiny_bert_qa):
        info = describe_model(tiny_bert_qa, "tiny-bert")
        assert info["model_name"] == "tiny-bert"
        assert info["architecture"] == "BertForQuestionAnswering"
        assert info["num_parameters"] == count_parameters(tiny_bert_qa)
        assert info["hidden_size"] == TINY_BERT["hidden_size"]
        assert info["num_hidden_layers"] == TINY_BERT["num_hidden_layers"]
        assert info["vocab_size"] == TINY_BERT["vocab_size"]

    def test_parameter_count_is_measured_not_declared(self, tiny_bert_qa):
        manual = sum(p.numel() for p in tiny_bert_qa.parameters())
        assert count_parameters(tiny_bert_qa) == manual


@pytest.mark.network
@pytest.mark.slow
class TestRealBertCheckpoint:
    """Confirms the finding on the actual model Experiment B used."""

    def test_hub_checkpoint_uses_legacy_layernorm_names(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        state = load_file(hf_hub_download("bert-base-uncased", "model.safetensors"))
        assert any(k.endswith((".gamma", ".beta")) for k in state), (
            "bert-base-uncased was expected to publish legacy LayerNorm names"
        )
        assert not any("LayerNorm.weight" in k for k in state)

    def test_saving_a_hub_loaded_bert_preserves_the_legacy_names(self, tmp_path):
        """Why our own saved checkpoints also contain .gamma/.beta.

        The convention follows the checkpoint the model came from, so a fine-tuned
        bert-base-uncased is written back out with legacy names. Harmless through
        from_pretrained, and exactly what makes a raw-load path dangerous.
        """
        from safetensors.torch import load_file

        model = load_qa_model("bert-base-uncased")
        model.save_pretrained(tmp_path)
        state = load_file(str(tmp_path / "model.safetensors"))
        assert any(k.endswith((".gamma", ".beta")) for k in state), (
            "a Hub-loaded BERT was expected to save legacy LayerNorm names; if this "
            "changed, verify_checkpoint_integrity's rationale should be revisited"
        )

    def test_pretrained_layernorm_values_are_actually_loaded(self):
        """The decisive check: pretrained LayerNorm must not arrive at defaults."""
        model = load_qa_model("bert-base-uncased")
        weight = dict(model.named_parameters())["bert.embeddings.LayerNorm.weight"].detach()
        assert not torch.allclose(weight, torch.ones_like(weight)), (
            "embeddings LayerNorm weight is all ones, i.e. the pretrained values were "
            "dropped rather than mapped from the legacy .gamma name"
        )

    def test_only_the_qa_head_is_missing_on_a_base_checkpoint(self):
        """Loading a base checkpoint for QA should report the head and nothing else."""
        from transformers import AutoModelForQuestionAnswering

        result = AutoModelForQuestionAnswering.from_pretrained(
            "bert-base-uncased", output_loading_info=True
        )
        _, info = result if isinstance(result, tuple) else (result, {})
        missing = list(info.get("missing_keys", []))
        assert sorted(missing) == ["qa_outputs.bias", "qa_outputs.weight"], (
            f"unexpected missing keys on a base checkpoint load: {missing}"
        )
        assert not [k for k in missing if "LayerNorm" in k]

    def test_real_bert_survives_a_save_reload_round_trip(self, tmp_path):
        model = load_qa_model("bert-base-uncased")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        model.save_pretrained(tmp_path)

        report = verify_checkpoint_integrity(model, tmp_path)
        assert report["ok"] is True
        assert report["num_drifted"] == 0
        assert report["layernorm_at_default_values"] == 0


# ---------------------------------------------------------------------------
# Regression tests for the Experiment D calibration failure.
#
# microsoft/deberta-v3-base publishes fp16 tensors. from_pretrained preserves the
# checkpoint dtype, so those fp16 tensors became the master parameters, and AdamW at
# eps=1e-8 produced Inf/NaN on optimizer step 1: (1 - beta2) * grad**2 is about 1e-9
# for a realistic 1e-3 gradient, which is below fp16's smallest subnormal (2**-24 =
# 5.96e-8), and eps=1e-8 rounds to zero in fp16 as well, so denom becomes exactly
# zero. Global gradient clipping then spread the NaN to all 200 parameter tensors.
#
# The checkpoint guard caught it but described it badly: torch.equal(NaN, NaN) is
# False so every tensor reported as drifted, while max(0.0, nan) returns 0.0 so the
# delta was reported as 0.000e+00.
# ---------------------------------------------------------------------------

# Small DeBERTa-v2 config mirroring deberta-v3-base's distinguishing settings:
# relative attention only (no absolute position embeddings), no token type
# embeddings, and the same tiny layer_norm_eps.
TINY_DEBERTA = {
    "vocab_size": 256,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "relative_attention": True,
    "position_buckets": 16,
    "pos_att_type": "p2c|c2p",
    "norm_rel_ebd": "layer_norm",
    "share_att_key": True,
    "position_biased_input": False,
    "type_vocab_size": 0,
    "layer_norm_eps": 1e-7,
}

# The project's real optimizer settings, so the underflow test exercises the
# configuration that actually failed rather than a synthetic one.
ADAMW_KWARGS = {"lr": 3e-5, "weight_decay": 0.01, "eps": 1e-8, "betas": (0.9, 0.999)}


@pytest.fixture
def tiny_deberta_qa():
    """A small DebertaV2ForQuestionAnswering with perturbed parameters."""
    from transformers import DebertaV2Config, DebertaV2ForQuestionAnswering

    torch.manual_seed(0)
    model = DebertaV2ForQuestionAnswering(DebertaV2Config(**TINY_DEBERTA))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return model


def _save_as_fp16(model, path) -> None:
    """Write a checkpoint whose tensors are fp16, as deberta-v3-base publishes.

    Note that ``Module.half()`` converts in place, so ``model`` is fp16 afterwards.
    Tests that compare against it should account for that.
    """
    model.half().save_pretrained(path)


class TestFp16CheckpointsLoadAsFp32:
    """Test A: an fp16 checkpoint must arrive as fp32 master weights."""

    def test_fp16_checkpoint_loads_as_fp32(self, tiny_deberta_qa, tmp_path):
        _save_as_fp16(tiny_deberta_qa, tmp_path)

        from safetensors.torch import load_file

        on_disk = load_file(str(tmp_path / "model.safetensors"))
        assert {t.dtype for t in on_disk.values()} == {torch.float16}, (
            "the fixture failed to write an fp16 checkpoint, so this test would not "
            "exercise the regression"
        )

        loaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        dtypes = {p.dtype for p in loaded.parameters()}
        assert dtypes == {torch.float32}, f"expected all fp32 master weights, got {dtypes}"

    def test_the_upcast_does_not_alter_any_value(self, tiny_deberta_qa, tmp_path):
        """fp16 -> fp32 is exact, so pinning the dtype cannot change a published value."""
        _save_as_fp16(tiny_deberta_qa, tmp_path)
        expected = {n: t.detach().clone() for n, t in tiny_deberta_qa.named_parameters()}

        loaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        for name, tensor in loaded.named_parameters():
            assert tensor.dtype is torch.float32
            assert torch.equal(tensor.detach(), expected[name].float()), (
                f"{name} changed value across the fp16 -> fp32 upcast"
            )

    def test_an_explicit_dtype_is_still_honoured(self, tiny_deberta_qa, tmp_path):
        """The default must be overridable, e.g. fp16 inference with no optimizer."""
        _save_as_fp16(tiny_deberta_qa, tmp_path)
        loaded = load_qa_model(str(tmp_path), dtype=torch.float16)
        assert {p.dtype for p in loaded.parameters()} == {torch.float16}


class TestAdamWUnderflow:
    """Test B: the exact optimizer scenario must stay finite with fp32 parameters."""

    @staticmethod
    def _run_adamw(parameter: torch.Tensor, steps: int = 20) -> tuple[int | None, int]:
        """Step AdamW with realistic gradients; report first non-finite step."""
        tracked = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([tracked], **ADAMW_KWARGS)
        generator = torch.Generator().manual_seed(0)
        first_non_finite = None
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            gradient = torch.randn(tracked.shape, generator=generator) * 1e-3
            tracked.grad = gradient.to(tracked.dtype)
            optimizer.step()
            if first_non_finite is None and not torch.isfinite(tracked).all():
                first_non_finite = step
        underflowed = int((optimizer.state[tracked]["exp_avg_sq"] == 0).sum())
        return first_non_finite, underflowed

    def test_fp32_parameters_stay_finite(self, tiny_deberta_qa, tmp_path):
        _save_as_fp16(tiny_deberta_qa, tmp_path)
        loaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        for name, parameter in loaded.named_parameters():
            first_non_finite, underflowed = self._run_adamw(parameter)
            assert first_non_finite is None, (
                f"{name} went non-finite at AdamW step {first_non_finite} despite "
                "being loaded as fp32"
            )
            assert underflowed == 0, (
                f"{name}: {underflowed} exp_avg_sq entries underflowed to zero, which "
                "is the precursor to a divide-by-zero in the AdamW update"
            )

    def test_fp16_parameters_would_have_failed(self):
        """Pins the mechanism, so this test fails if the hazard ever stops being real.

        Without this, the test above could keep passing for the wrong reason if some
        future torch release changed AdamW's state dtype handling.
        """
        first_non_finite, underflowed = self._run_adamw(
            torch.randn(4096, dtype=torch.float16) * 0.1
        )
        assert first_non_finite == 1, (
            "fp16 master weights were expected to go non-finite on AdamW step 1; if "
            "this changed, the rationale in load_qa_model should be revisited"
        )
        assert underflowed == 4096

    def test_fp16_eps_and_grad_square_underflow(self):
        """The two numeric facts the failure rests on, asserted directly."""
        assert float(torch.tensor(1e-8, dtype=torch.float16)) == 0.0
        assert float(torch.tensor(1e-4, dtype=torch.float16) ** 2) == 0.0
        # An lr-sized step is below fp16 resolution for a typical weight magnitude.
        weight = torch.tensor(0.1, dtype=torch.float16)
        infinity = torch.tensor(float("inf"), dtype=torch.float16)
        ulp = float(torch.nextafter(weight, infinity)) - float(weight)
        assert ulp > 3e-5


class TestNonFiniteCheckpointDiagnostics:
    """Test C: a NaN checkpoint is rejected and the diagnosis is accurate."""

    def test_nan_poisoned_checkpoint_is_rejected(self, tiny_deberta_qa, tmp_path):
        with torch.no_grad():
            for parameter in tiny_deberta_qa.parameters():
                parameter.fill_(float("nan"))
        tiny_deberta_qa.save_pretrained(tmp_path)

        with pytest.raises(CheckpointIntegrityError) as excinfo:
            verify_checkpoint_integrity(tiny_deberta_qa, tmp_path)
        message = str(excinfo.value)
        assert "does NOT reload" in message
        assert "non-finite parameter tensor" in message
        assert "TRAINING diverged" in message

    def test_report_counts_non_finite_tensors(self, tiny_deberta_qa, tmp_path):
        total = len(list(tiny_deberta_qa.parameters()))
        with torch.no_grad():
            for parameter in tiny_deberta_qa.parameters():
                parameter.fill_(float("nan"))
        tiny_deberta_qa.save_pretrained(tmp_path)

        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path, strict=False)

        assert report["ok"] is False
        assert report["num_non_finite_reference"] == total
        assert report["num_non_finite_reloaded"] == total
        assert report["num_drifted"] == total
        assert report["missing_after_reload"] == []
        assert report["unexpected_after_reload"] == []

    def test_nan_delta_is_never_reported_as_zero(self, tiny_deberta_qa, tmp_path):
        """The specific defect: max(0.0, nan) returned 0.0 and hid a destroyed model."""
        with torch.no_grad():
            for parameter in tiny_deberta_qa.parameters():
                parameter.fill_(float("nan"))
        tiny_deberta_qa.save_pretrained(tmp_path)

        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path, strict=False)

        assert report["max_abs_delta"] != 0.0
        assert math.isnan(report["max_abs_delta"]), (
            "a NaN delta must surface as NaN, not be silently discarded by max()"
        )

    def test_inf_poisoned_checkpoint_is_also_rejected(self, tiny_deberta_qa, tmp_path):
        """Inf == Inf compares equal, so equality alone would let this pass.

        The failed run produced mostly Inf (4094 of 4096 elements in the reproduction),
        so this is the common case rather than an exotic one.
        """
        with torch.no_grad():
            for parameter in tiny_deberta_qa.parameters():
                parameter.fill_(float("inf"))
        tiny_deberta_qa.save_pretrained(tmp_path)

        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path, strict=False)
        assert report["ok"] is False, "an Inf-poisoned checkpoint must not pass"
        assert report["num_non_finite_reference"] == len(list(tiny_deberta_qa.parameters()))
        assert report["num_drifted"] == 0, (
            "Inf compares equal to itself, so this is caught by the finiteness check "
            "rather than by drift; if drift is now non-zero the guard changed shape"
        )
        with pytest.raises(CheckpointIntegrityError):
            verify_checkpoint_integrity(tiny_deberta_qa, tmp_path)

    def test_a_single_nan_element_is_enough(self, tiny_deberta_qa, tmp_path):
        """Partial corruption must not be diluted by 200 healthy tensors."""
        with torch.no_grad():
            tiny_deberta_qa.qa_outputs.weight[0, 0] = float("nan")
        tiny_deberta_qa.save_pretrained(tmp_path)

        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path, strict=False)
        assert report["ok"] is False
        assert report["num_non_finite_reference"] == 1


class TestCleanDebertaIntegrity:
    """Test D: the guard must still pass a healthy DeBERTa round trip."""

    def test_clean_fp32_deberta_passes(self, tiny_deberta_qa, tmp_path):
        tiny_deberta_qa.save_pretrained(tmp_path)
        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path)

        assert report["ok"] is True
        assert report["num_drifted"] == 0
        assert report["max_abs_delta"] == 0.0
        assert report["num_non_finite_reference"] == 0
        assert report["num_non_finite_reloaded"] == 0
        assert report["layernorm_at_default_values"] == 0

    def test_clean_fp16_deberta_round_trip_still_passes(self, tiny_deberta_qa, tmp_path):
        """Loading fp16 explicitly is legitimate; the guard must not object to it.

        This is the case that made the real failure look like a dtype bug: a clean
        fp16 DeBERTa round-trips perfectly, so the guard was never the problem.
        """
        _save_as_fp16(tiny_deberta_qa, tmp_path)
        fp16_model = load_qa_model(str(tmp_path), dtype=torch.float16)

        report = verify_checkpoint_integrity(fp16_model, tmp_path)
        assert report["ok"] is True
        assert report["num_drifted"] == 0
        assert report["num_non_finite_reference"] == 0

    def test_real_drift_is_still_detected_on_deberta(self, tiny_deberta_qa, tmp_path):
        tiny_deberta_qa.save_pretrained(tmp_path)
        with torch.no_grad():
            tiny_deberta_qa.qa_outputs.weight.add_(1.0)

        report = verify_checkpoint_integrity(tiny_deberta_qa, tmp_path, strict=False)
        assert report["ok"] is False
        assert report["num_drifted"] == 1
        assert report["max_abs_delta"] == pytest.approx(1.0, abs=1e-6)
        assert report["num_non_finite_reference"] == 0


class TestOtherArchitecturesUnchanged:
    """Test E: BERT / RoBERTa / DistilBERT behaviour must not shift."""

    @pytest.mark.parametrize(
        ("config_cls", "model_cls", "overrides"),
        [
            ("BertConfig", "BertForQuestionAnswering", {}),
            ("RobertaConfig", "RobertaForQuestionAnswering", {}),
            (
                "DistilBertConfig",
                "DistilBertForQuestionAnswering",
                {"dim": 32, "hidden_dim": 64, "n_layers": 3, "n_heads": 2},
            ),
        ],
    )
    def test_fp32_checkpoints_still_load_as_fp32(
        self, config_cls, model_cls, overrides, tmp_path
    ):
        """These publish fp32, so pinning the dtype must be a no-op for them."""
        import transformers

        config_kwargs = dict(TINY_BERT)
        if overrides:
            for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                        "num_attention_heads"):
                config_kwargs.pop(key, None)
            config_kwargs.update(overrides)

        torch.manual_seed(0)
        config = getattr(transformers, config_cls)(**config_kwargs)
        model = getattr(transformers, model_cls)(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)

        expected = {n: t.detach().clone() for n, t in model.named_parameters()}
        model.save_pretrained(tmp_path)

        loaded = load_qa_model(str(tmp_path), expect_trained_head=True)

        assert {p.dtype for p in loaded.parameters()} == {torch.float32}
        for name, tensor in loaded.named_parameters():
            assert torch.equal(tensor.detach(), expected[name]), (
                f"{model_cls}: {name} changed across save/reload"
            )

        report = verify_checkpoint_integrity(loaded, tmp_path)
        assert report["ok"] is True
        assert report["num_non_finite_reference"] == 0

    def test_bert_integrity_report_keeps_its_existing_keys(self, tiny_bert_qa, tmp_path):
        """The new fields are additive; nothing downstream should lose a key."""
        tiny_bert_qa.save_pretrained(tmp_path)
        report = verify_checkpoint_integrity(tiny_bert_qa, tmp_path)

        for key in (
            "checkpoint_path",
            "num_parameters_checked",
            "num_drifted",
            "max_abs_delta",
            "missing_after_reload",
            "unexpected_after_reload",
            "layernorm_at_default_values",
            "ok",
        ):
            assert key in report, f"pre-existing report key {key!r} disappeared"
        assert report["num_non_finite_reference"] == 0
        assert report["num_non_finite_reloaded"] == 0
