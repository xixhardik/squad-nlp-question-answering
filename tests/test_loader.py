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
