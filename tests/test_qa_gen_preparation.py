r"""Tests for Phase 17C: dataset preparation, source loading and tokenizer-based sizing.

What these tests prove, and what they cannot
--------------------------------------------
**Proved here.** The preparation pipeline is deterministic and independent of the order records
arrive in. A leakage group never straddles two splits. The four configuration fields that had no
implementation before this phase -- ``max_examples``, ``max_examples_per_source``,
``drop_duplicates`` and ``shuffle_seed`` -- now take effect, in the way their docstrings describe.
Invalid examples are dropped and counted rather than silently kept. Fingerprints identify a
corpus by content and survive a round trip. Step arithmetic matches what transformers will do.
The source layer refuses to download unless told to, and reads local files without ``datasets``
installed at all. Token lengths are measured the way TRL measures them.

**Not proved here.** That ``rajpurkar/squad`` still has the columns the adapter expects. That the
real Qwen3 tokenizer produces the lengths a stub produces. That 87,599 SQuAD rows prepare in the
time the estimate claims. Those need the Studio, and the sizing command is what measures them.

The tokenizer stub
------------------
:class:`FakeTokenizer` renders the two Qwen3 template branches that matter -- the assistant turn
always carries an empty ``<think>\n\n</think>\n\n`` block, the generation prompt carries it only
when ``enable_thinking`` is false -- and tokenizes on a prefix-stable atom split. That is the same
mimic ``tests/test_qa_gen_smoke.py`` uses, and for the same reason: a stub that disagrees with the
library is worse than no test. Token *counts* from it are arbitrary; the prompt/completion
*boundary* is not.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any

import pytest

from qa_gen import (
    PREPARATION_STAGES,
    DatasetStatistics,
    PreparationError,
    QuestionGenerationDatasetConfig,
    QuestionGenerationExample,
    QuestionGenerationTarget,
    SourceIngestion,
    adapt_records,
    compute_dataset_fingerprint,
    deduplicate_by_content,
    deduplicate_by_id,
    example_from_dict,
    experiment_config_from_dict,
    order_examples,
    prepare_dataset,
    select_examples,
    statistics_from_dict,
)
from qa_gen_runtime import prepare as prepare_module
from qa_gen_runtime import sources as sources_module
from qa_gen_runtime.dataset import build_training_records
from qa_gen_runtime.prepare import (
    audit_dataset,
    build_parser,
    format_outcome,
    main,
    run_preparation,
)
from qa_gen_runtime.sizing import (
    DEFAULT_STEP_PLANS,
    SizingError,
    TokenLengthSummary,
    build_step_estimates,
    estimate_step_count,
    estimate_tokenization_seconds,
    measure_record_lengths,
    summarize_token_lengths,
)
from qa_gen_runtime.sources import (
    SOURCE_CATALOGUE,
    SourceLoadError,
    SourceRequest,
    adapt_source,
    catalogue_entry,
    describe_catalogue,
    describe_requirements,
    load_source_records,
    request_is_readable,
    resolve_requests,
)
from qa_paper import Difficulty, QuestionType

PASSAGE = (
    "Photosynthesis is the process by which green plants convert light energy into chemical "
    "energy. Chlorophyll in the leaves absorbs sunlight, and the plant combines carbon dioxide "
    "from the air with water drawn up through the roots to produce glucose and oxygen."
)

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
_ATOM = re.compile(r"<\|[^|>]*\|>|</?think>|\s+|\w+|.")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def squad_record(index: int = 0, *, title: str | None = None) -> dict[str, Any]:
    """Build a SQuAD-shaped record whose annotated offset verifies."""
    context = f"{PASSAGE} Paragraph {index}."
    answer = "Chlorophyll"
    return {
        "id": f"sq-{index}",
        "title": title if title is not None else f"Article {index % 4}",
        "context": context,
        "question": f"Which pigment absorbs sunlight, variant {index}?",
        "answers": {"text": [answer], "answer_start": [context.index(answer)]},
    }


def mcq_record(index: int = 0, **overrides: Any) -> dict[str, Any]:
    """Build an edu-mcq-shaped record."""
    defaults = {
        "id": f"mcq-{index}",
        "context": f"{PASSAGE} MCQ variant {index}.",
        "question": f"Which pigment absorbs sunlight? (item {index})",
        "options": ["Chlorophyll", "Haemoglobin", "Melanin", "Carotene"],
        "correct_index": 0,
        "topic": "Photosynthesis",
    }
    return {**defaults, **overrides}


#: RACE options, in corpus order. Index 2 is correct, so a transform that silently defaulted to
#: the first option would be caught rather than coincidentally right.
RACE_OPTIONS = ("Haemoglobin", "Melanin", "Chlorophyll", "Carotene")
RACE_ANSWER_LABEL = "C"
RACE_CORRECT_INDEX = 2


def race_record(index: int = 0, *, article: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Build an ``ehovy/race``-shaped record.

    Mirrors the verified upstream schema: the passage is ``article``, the correct option is a
    label rather than text, ``options`` is a four-string list, and ``example_id`` is the source
    filename, which repeats across the questions drawn from one article.
    """
    defaults = {
        "example_id": f"high{index // 3}.txt",
        "article": article if article is not None else f"{PASSAGE} Article {index // 3}.",
        "question": f"Which pigment absorbs sunlight? (race item {index})",
        "options": list(RACE_OPTIONS),
        "answer": RACE_ANSWER_LABEL,
    }
    return {**defaults, **overrides}


def mixed_corpus(
    *, squad: int = 60, race: int = 60
) -> dict[str, list[QuestionGenerationExample]]:
    """Build a SQuAD + RACE corpus through the real registered adapters.

    Distinct passages per source, so content deduplication has nothing to collapse and the
    per-source counts mean what they say.
    """
    return {
        "squad-qg": list(adapt_records("squad-qg", [squad_record(i) for i in range(squad)])),
        "race-mcq": list(adapt_records("race-mcq", [race_record(i) for i in range(race)])),
    }


def target(**overrides: Any) -> QuestionGenerationTarget:
    """Build a valid short-answer target."""
    defaults = {
        "question_type": QuestionType.SHORT_ANSWER,
        "question": "Which pigment absorbs sunlight?",
        "answer": "Chlorophyll",
        "difficulty": Difficulty.EASY,
        "marks": 1,
    }
    return QuestionGenerationTarget(**{**defaults, **overrides})


def example(index: int = 0, **overrides: Any) -> QuestionGenerationExample:
    """Build a valid single-target example."""
    defaults = {
        "id": f"src-{index:04d}",
        "context": f"{PASSAGE} Passage {index}.",
        "targets": (target(question=f"Question {index}?"),),
        "source": "squad-qg",
        "topic": "Photosynthesis",
    }
    return QuestionGenerationExample(**{**defaults, **overrides})


def adapted_corpus(
    *, squad: int = 40, mcq: int = 20
) -> dict[str, list[QuestionGenerationExample]]:
    """Build a two-source corpus through the real adapters."""
    return {
        "squad-qg": list(adapt_records("squad-qg", [squad_record(i) for i in range(squad)])),
        "edu-mcq": list(adapt_records("edu-mcq", [mcq_record(i) for i in range(mcq)])),
    }


def dataset_config(**overrides: Any) -> QuestionGenerationDatasetConfig:
    """Build a validated dataset configuration."""
    defaults = {
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "min_context_chars": 64,
        "max_context_chars": 4000,
    }
    return QuestionGenerationDatasetConfig(**{**defaults, **overrides})


def experiment_config(**overrides: Any) -> Any:
    """Build a validated experiment configuration."""
    payload: dict[str, Any] = {"name": "prep-test"}
    payload.update(overrides)
    return experiment_config_from_dict(payload)


CONFIG_YAML = """
name: qgen-prep-test
phase: "17"
dataset:
  sources: [edu-mcq]
  seed: 42
  train_ratio: 0.8
  validation_ratio: 0.1
  test_ratio: 0.1
  group_by: context
  min_context_chars: 64
  max_context_chars: 4000
model:
  model_id: Qwen/Qwen3-4B
  max_seq_length: 1024
  reasoning_mode: disabled
training:
  max_steps: 20
  warmup_ratio: 0.03
  evaluation_strategy: "no"
  save_strategy: "no"
  load_best_model_at_end: false
"""


@pytest.fixture
def local_corpus(tmp_path):
    """Write a config and a local MCQ corpus, and return both paths."""
    config_path = tmp_path / "prep.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    records_path = tmp_path / "mcq.jsonl"
    records_path.write_text(
        "".join(json.dumps(mcq_record(index)) + "\n" for index in range(40)),
        encoding="utf-8",
    )
    return SimpleNamespace(config=config_path, records=records_path, root=tmp_path)


# ---------------------------------------------------------------------------
# The tokenizer stub
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Renders the two Qwen3 template branches and tokenizes on a prefix-stable atom split."""

    def __init__(self) -> None:
        """Start with an empty vocabulary; atoms are interned as they are seen."""
        self.chat_template = (
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
        )
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"
        self._atoms: list[str] = []

    def _identifier(self, atom: str) -> int:
        """Return a stable id for one atom."""
        if atom not in self._atoms:
            self._atoms.append(atom)
        return self._atoms.index(atom)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render, then tokenize when asked.

        ``return_dict`` defaults to ``True``, which is what transformers 5.x does and what an
        earlier version of this stub got wrong. Returning a bare list regardless made the stub
        more forgiving than the library: a caller that took ``len()`` of the result got a token
        count here and the number of ``BatchEncoding`` keys -- 2 -- on the Studio. The stub now
        reproduces the library's shape so the tests can catch that.
        """
        parts: list[str] = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "assistant":
                parts.append(
                    f"<|im_start|>assistant\n{EMPTY_THINK_BLOCK}{content}<|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if kwargs.get("enable_thinking") is False:
                parts.append(EMPTY_THINK_BLOCK)
        text = "".join(parts)
        if not tokenize:
            return text
        ids = [self._identifier(atom) for atom in _ATOM.findall(text)]
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Join the atoms the ids stand for."""
        return "".join(self._atoms[int(index)] for index in ids)


# ---------------------------------------------------------------------------
# Deterministic selection primitives
# ---------------------------------------------------------------------------


class TestSelectionPrimitives:
    """The building blocks the caps and the ordering are made of."""

    def test_no_limit_keeps_everything_in_id_order(self):
        examples = [example(index) for index in range(10)]
        kept, dropped = select_examples(examples, None, seed=42)
        assert dropped == ()
        assert [item.id for item in kept] == sorted(item.id for item in examples)

    def test_a_limit_at_or_above_the_size_drops_nothing(self):
        examples = [example(index) for index in range(5)]
        kept, dropped = select_examples(examples, 5, seed=42)
        assert len(kept) == 5
        assert dropped == ()

    def test_selection_is_deterministic(self):
        examples = [example(index) for index in range(50)]
        first, _ = select_examples(examples, 20, seed=42)
        second, _ = select_examples(examples, 20, seed=42)
        assert [item.id for item in first] == [item.id for item in second]

    def test_selection_ignores_input_order(self):
        examples = [example(index) for index in range(50)]
        forward, _ = select_examples(examples, 20, seed=42)
        backward, _ = select_examples(list(reversed(examples)), 20, seed=42)
        assert [item.id for item in forward] == [item.id for item in backward]

    def test_a_different_seed_selects_a_different_subset(self):
        examples = [example(index) for index in range(50)]
        one, _ = select_examples(examples, 20, seed=42)
        two, _ = select_examples(examples, 20, seed=43)
        assert {item.id for item in one} != {item.id for item in two}

    def test_kept_and_dropped_partition_the_input(self):
        examples = [example(index) for index in range(30)]
        kept, dropped = select_examples(examples, 11, seed=7)
        assert len(kept) == 11
        assert len(dropped) == 19
        assert {item.id for item in kept} | set(dropped) == {item.id for item in examples}

    def test_selection_is_not_biased_by_the_source_prefix(self):
        """The trap this function exists to avoid: ids sort by source, digests do not."""
        mixed = [example(index, id=f"aaa-{index:04d}", source="aaa") for index in range(50)]
        mixed += [example(index, id=f"zzz-{index:04d}", source="zzz") for index in range(50)]
        kept, _ = select_examples(mixed, 50, seed=42)
        sources = {item.source for item in kept}
        assert sources == {"aaa", "zzz"}, "a cap must sample across sources, not empty one first"
        counts = {name: sum(1 for i in kept if i.source == name) for name in sources}
        assert min(counts.values()) >= 15, counts

    def test_a_zero_limit_is_refused(self):
        with pytest.raises(PreparationError, match="positive integer"):
            select_examples([example()], 0, seed=42)

    def test_a_negative_limit_is_refused(self):
        with pytest.raises(PreparationError, match="positive integer"):
            select_examples([example()], -1, seed=42)

    def test_ordering_is_deterministic_and_order_independent(self):
        examples = [example(index) for index in range(30)]
        forward = order_examples(examples, seed=1234)
        backward = order_examples(list(reversed(examples)), seed=1234)
        assert [item.id for item in forward] == [item.id for item in backward]

    def test_ordering_is_a_permutation(self):
        examples = [example(index) for index in range(30)]
        ordered = order_examples(examples, seed=1234)
        assert sorted(item.id for item in ordered) == sorted(item.id for item in examples)

    def test_ordering_actually_reorders(self):
        examples = [example(index) for index in range(30)]
        ordered = order_examples(examples, seed=1234)
        assert [item.id for item in ordered] != sorted(item.id for item in examples)

    def test_a_different_shuffle_seed_gives_a_different_order(self):
        examples = [example(index) for index in range(30)]
        assert [item.id for item in order_examples(examples, seed=1)] != [
            item.id for item in order_examples(examples, seed=2)
        ]


class TestDeduplication:
    """Two different decisions that are easy to conflate."""

    def test_duplicate_ids_are_removed_unconditionally(self):
        one = example(0)
        kept, dropped = deduplicate_by_id([one, one, example(1)])
        assert len(kept) == 2
        assert dropped == (one.id,)

    def test_deduplicating_ids_leaves_distinct_examples_alone(self):
        examples = [example(index) for index in range(5)]
        kept, dropped = deduplicate_by_id(examples)
        assert len(kept) == 5
        assert dropped == ()

    def test_duplicate_content_keeps_the_lowest_id(self):
        """Documented behaviour, and it must not depend on which corpus was read first."""
        shared = target(question="Same question?", answer="Same answer")
        low = example(0, id="aaa-1", targets=(shared,), source="alpha")
        high = example(0, id="zzz-9", targets=(shared,), source="omega")
        kept, dropped = deduplicate_by_content([high, low])
        assert [item.id for item in kept] == ["aaa-1"]
        assert dropped == ("zzz-9",)

    def test_deduplicating_content_is_order_independent(self):
        shared = target(question="Same question?", answer="Same answer")
        examples = [
            example(0, id="aaa-1", targets=(shared,)),
            example(0, id="zzz-9", targets=(shared,)),
            example(2),
        ]
        forward, _ = deduplicate_by_content(examples)
        backward, _ = deduplicate_by_content(list(reversed(examples)))
        assert [item.id for item in forward] == [item.id for item in backward]

    def test_different_content_survives(self):
        examples = [example(index) for index in range(4)]
        kept, dropped = deduplicate_by_content(examples)
        assert len(kept) == 4
        assert dropped == ()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class TestPreparationPipeline:
    """End to end over a two-source corpus built through the real adapters."""

    def test_it_prepares_and_reports_every_stage(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        counts = prepared.report.as_dict()["stage_counts"]
        assert list(counts) == list(PREPARATION_STAGES)
        assert counts["adapted"] == 60
        assert counts["capped_total"] == prepared.report.kept
        assert prepared.report.kept == len(prepared.examples)

    def test_the_partition_is_deterministic(self):
        corpus = adapted_corpus()
        first = prepare_dataset(corpus, dataset_config())
        second = prepare_dataset(corpus, dataset_config())
        assert [item.id for item in first.splits.train] == [
            item.id for item in second.splits.train
        ]
        assert first.fingerprint == second.fingerprint

    def test_the_partition_ignores_input_order(self):
        corpus = adapted_corpus()
        reversed_corpus = {key: list(reversed(value)) for key, value in corpus.items()}
        first = prepare_dataset(corpus, dataset_config())
        second = prepare_dataset(reversed_corpus, dataset_config())
        assert [item.id for item in first.splits.train] == [
            item.id for item in second.splits.train
        ]
        assert first.fingerprint == second.fingerprint

    def test_a_different_seed_moves_examples_between_splits(self):
        corpus = adapted_corpus()
        one = prepare_dataset(corpus, dataset_config(seed=42))
        two = prepare_dataset(corpus, dataset_config(seed=99))
        assert {i.id for i in one.splits.train} != {i.id for i in two.splits.train}
        assert one.fingerprint == two.fingerprint, "the corpus did not change, only the split"

    def test_every_example_lands_in_exactly_one_split(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        ids = [item.id for item in prepared.examples]
        assert len(ids) == len(set(ids))
        assert len(ids) == len(prepared.splits)

    def test_the_ingestion_records_travel_into_the_report(self):
        ingestion = [
            SourceIngestion(
                source_id="squad-qg",
                dataset_id="rajpurkar/squad",
                records_seen=100,
                examples_adapted=40,
                rejected=60,
            )
        ]
        prepared = prepare_dataset(adapted_corpus(), dataset_config(), ingestion=ingestion)
        assert prepared.report.records_seen == 100
        assert prepared.report.adapter_rejected == 60
        assert prepared.report.sources[0].rejection_rate == 0.6

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(PreparationError, match="nothing to prepare"):
            prepare_dataset({}, dataset_config())

    def test_a_corpus_filtered_to_nothing_is_refused(self):
        """A context bound that excludes everything must fail loudly, not train on zero."""
        with pytest.raises(PreparationError, match="filtered out"):
            prepare_dataset(
                adapted_corpus(squad=4, mcq=0),
                dataset_config(min_context_chars=10_000, max_context_chars=20_000),
            )

    def test_the_config_hash_reaches_the_metadata(self):
        prepared = prepare_dataset(
            adapted_corpus(), dataset_config(), config_hash="cafebabe1234"
        )
        assert prepared.metadata.config_hash == "cafebabe1234"

    def test_the_metadata_names_only_the_sources_that_survived(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert set(prepared.metadata.sources) == {"squad-qg", "edu-mcq"}


class TestLeakagePrevention:
    """The invariant the whole grouping mechanism exists for."""

    def test_no_leakage_group_appears_in_two_splits(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert prepared.splits.leaked_group_keys() == frozenset()

    def test_a_shared_passage_cannot_straddle_splits(self):
        """Twelve questions over three passages: each passage must stay whole."""
        examples = [
            example(
                index,
                id=f"shared-{index:03d}",
                context=f"{PASSAGE} Shared passage {index % 3}.",
            )
            for index in range(12)
        ]
        prepared = prepare_dataset({"squad-qg": examples}, dataset_config())
        by_split: dict[str, set[str]] = {}
        for name, members in (
            ("train", prepared.splits.train),
            ("validation", prepared.splits.validation),
            ("test", prepared.splits.test),
        ):
            by_split[name] = {item.effective_group_key() for item in members}
        pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
        for left, right in pairs:
            assert not by_split[left] & by_split[right], (left, right)

    def test_grouping_by_topic_keeps_an_article_whole(self):
        records = [squad_record(index, title=f"Article {index % 3}") for index in range(30)]
        corpus = {"squad-qg": list(adapt_records("squad-qg", records))}
        prepared = prepare_dataset(corpus, dataset_config(group_by="topic"))
        assert prepared.splits.leaked_group_keys() == frozenset()
        assert prepared.splits.group_by == "topic"

    def test_grouping_by_example_leaks_on_purpose_and_says_so(self):
        """That option exists to measure the effect of grouping, so it must leak visibly."""
        prepared = prepare_dataset(adapted_corpus(), dataset_config(group_by="example"))
        assert prepared.splits.group_by == "example"
        assert prepared.splits.leaked_group_keys(), "disabling grouping must not look clean"
        findings = {finding.code: finding for finding in audit_dataset(prepared, None)}
        assert "group_leakage" in findings
        assert findings["group_leakage"].blocking is False
        assert "deliberately" in findings["group_leakage"].message

    def test_grouping_by_context_blocks_when_leakage_is_not_deliberate(self):
        """The same finding blocks when the caller did ask for leakage control."""
        prepared = prepare_dataset(adapted_corpus(), dataset_config(group_by="context"))
        leaky = prepared.splits.__class__(
            train=prepared.splits.train,
            validation=prepared.splits.train[:1],
            test=prepared.splits.test,
            seed=prepared.splits.seed,
            group_by="context",
            ratios=prepared.splits.ratios,
            dataset_fingerprint=prepared.splits.dataset_fingerprint,
            assignments=prepared.splits.assignments,
            splitter=prepared.splits.splitter,
        )
        broken = prepared.__class__(
            splits=leaky,
            metadata=prepared.metadata,
            report=prepared.report,
            validation=prepared.validation,
            statistics=prepared.statistics,
        )
        findings = {finding.code: finding for finding in audit_dataset(broken, None)}
        assert findings["group_leakage"].blocking is True


class TestCapsAndDeduplicationInThePipeline:
    """The four configuration fields that had no implementation before this phase."""

    def test_the_total_cap_bounds_the_dataset(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config(max_examples=25))
        assert prepared.report.kept == 25
        assert prepared.report.total_capped == 35
        assert len(prepared.examples) == 25

    def test_the_total_cap_samples_across_sources(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config(max_examples=30))
        assert set(prepared.statistics.examples_by_source) == {"squad-qg", "edu-mcq"}

    def test_the_per_source_cap_applies_independently(self):
        prepared = prepare_dataset(
            adapted_corpus(squad=40, mcq=20), dataset_config(max_examples_per_source=10)
        )
        assert prepared.statistics.examples_by_source == {"edu-mcq": 10, "squad-qg": 10}
        assert prepared.report.per_source_capped == 40

    def test_a_source_below_the_per_source_cap_is_untouched(self):
        prepared = prepare_dataset(
            adapted_corpus(squad=40, mcq=5), dataset_config(max_examples_per_source=10)
        )
        assert prepared.statistics.examples_by_source == {"edu-mcq": 5, "squad-qg": 10}

    def test_both_caps_compose(self):
        prepared = prepare_dataset(
            adapted_corpus(squad=40, mcq=20),
            dataset_config(max_examples_per_source=15, max_examples=20),
        )
        assert prepared.report.kept == 20
        assert prepared.report.per_source_capped == 30
        assert prepared.report.total_capped == 10

    def duplicated(self, *, copies: int = 4, unique: int = 6):
        """Build ``copies`` examples sharing one fingerprint, plus ``unique`` distinct ones.

        The fingerprint combines the normalized context with every target, so a duplicate
        needs both to match -- differing only in id is what a duplicate *is* here.
        """
        shared_context = f"{PASSAGE} One shared passage."
        shared_target = target(question="Identical?", answer="Identical")
        examples = [
            example(
                index,
                id=f"dup-{index:03d}",
                context=shared_context,
                targets=(shared_target,),
            )
            for index in range(copies)
        ]
        examples.extend(example(index, id=f"uniq-{index:03d}") for index in range(unique))
        return examples

    def test_duplicate_content_is_dropped_when_asked(self):
        prepared = prepare_dataset(
            {"squad-qg": self.duplicated()}, dataset_config(drop_duplicates=True)
        )
        assert prepared.report.duplicate_content_dropped == 3
        assert prepared.report.kept == 7
        assert prepared.duplicate_groups == {}

    def test_duplicate_content_is_kept_and_reported_when_not_asked(self):
        prepared = prepare_dataset(
            {"squad-qg": self.duplicated()}, dataset_config(drop_duplicates=False)
        )
        assert prepared.report.duplicate_content_dropped == 0
        assert prepared.report.kept == 10
        assert len(prepared.duplicate_groups) == 1
        assert any("drop_duplicates is off" in note for note in prepared.report.notes)

    def test_the_surviving_duplicate_is_the_lowest_id(self):
        prepared = prepare_dataset(
            {"squad-qg": self.duplicated()}, dataset_config(drop_duplicates=True)
        )
        survivors = {item.id for item in prepared.examples if item.id.startswith("dup-")}
        assert survivors == {"dup-000"}

    def test_duplicate_ids_are_removed_before_splitting(self):
        """The splitter raises on a repeated id, so this cannot be left to drop_duplicates."""
        examples = [example(index) for index in range(6)]
        examples.append(examples[0])
        prepared = prepare_dataset(
            {"squad-qg": examples}, dataset_config(drop_duplicates=False)
        )
        assert prepared.report.duplicate_ids_dropped == 1
        assert prepared.report.kept == 6

    def test_the_shuffle_seed_changes_order_without_changing_membership(self):
        corpus = adapted_corpus()
        one = prepare_dataset(corpus, dataset_config(shuffle_seed=1))
        two = prepare_dataset(corpus, dataset_config(shuffle_seed=2))
        assert [i.id for i in one.splits.train] != [i.id for i in two.splits.train]
        assert {i.id for i in one.splits.train} == {i.id for i in two.splits.train}
        assert one.splits.sizes == two.splits.sizes
        assert one.fingerprint == two.fingerprint

    def test_the_shuffle_interleaves_sources_rather_than_grouping_them(self):
        """Id order groups by source prefix; training through that is an unchosen curriculum."""
        prepared = prepare_dataset(adapted_corpus(squad=40, mcq=40), dataset_config())
        sequence = [item.source for item in prepared.splits.train]
        transitions = sum(
            1 for a, b in zip(sequence[:-1], sequence[1:], strict=True) if a != b
        )
        assert transitions > 5, f"only {transitions} source changes in {len(sequence)} examples"


class TestValidationFiltering:
    """Invalid examples are dropped, counted and explained -- never silently kept."""

    def test_invalid_examples_are_dropped_and_counted(self):
        good = [example(index) for index in range(8)]
        blank = example(99, id="bad-blank", targets=(target(question="   "),))
        prepared = prepare_dataset({"squad-qg": [*good, blank]}, dataset_config())
        assert prepared.report.invalid_dropped == 1
        assert prepared.report.kept == 8
        assert "bad-blank" not in {item.id for item in prepared.examples}

    def test_the_validation_report_survives_into_the_output(self):
        blank = example(99, id="bad-blank", targets=(target(answer=""),))
        prepared = prepare_dataset(
            {"squad-qg": [*[example(i) for i in range(8)], blank]}, dataset_config()
        )
        assert not prepared.validation.ok
        assert prepared.validation.invalid_example_ids == frozenset({"bad-blank"})
        assert prepared.metadata.validation_summary["error_count"] >= 1

    def test_the_invalid_rate_is_reported(self):
        good = [example(index) for index in range(9)]
        blank = example(99, id="bad", targets=(target(question=""),))
        prepared = prepare_dataset({"squad-qg": [*good, blank]}, dataset_config())
        assert prepared.report.invalid_rate == 0.1

    def test_a_context_below_the_minimum_is_dropped(self):
        short = example(99, id="too-short", context="Tiny.")
        prepared = prepare_dataset(
            {"squad-qg": [*[example(i) for i in range(8)], short]},
            dataset_config(min_context_chars=64),
        )
        assert "too-short" not in {item.id for item in prepared.examples}
        assert prepared.validation.has(
            type(prepared.validation.issues[0].code).CONTEXT_TOO_SHORT
        )

    def test_a_context_above_the_maximum_is_dropped(self):
        long_example = example(99, id="too-long", context="x " * 3000)
        prepared = prepare_dataset(
            {"squad-qg": [*[example(i) for i in range(8)], long_example]},
            dataset_config(max_context_chars=1000),
        )
        assert "too-long" not in {item.id for item in prepared.examples}

    def test_warnings_do_not_drop_an_example(self):
        """Duplicate content is a warning; dropping on it would discard both copies."""
        shared_context = f"{PASSAGE} One shared passage."
        shared_target = target(question="Identical?", answer="Identical")
        examples = [
            example(
                index,
                id=f"dup-{index:03d}",
                context=shared_context,
                targets=(shared_target,),
            )
            for index in range(2)
        ]
        examples.extend(example(index, id=f"uniq-{index:03d}") for index in range(6))
        prepared = prepare_dataset({"squad-qg": examples}, dataset_config(drop_duplicates=False))
        assert prepared.report.invalid_dropped == 0
        assert prepared.report.kept == 8
        assert prepared.validation.warnings

    def test_dropped_examples_are_sampled_in_the_report(self):
        bad = [
            example(index, id=f"bad-{index:03d}", targets=(target(question=""),))
            for index in range(3)
        ]
        prepared = prepare_dataset(
            {"squad-qg": [*[example(i) for i in range(8)], *bad]}, dataset_config()
        )
        samples = prepared.report.as_dict()["dropped_samples"]
        assert set(samples) == {"bad-000", "bad-001", "bad-002"}
        assert set(samples.values()) == {"validation_error"}


class TestStatistics:
    """The planning numbers, and that they describe the kept corpus."""

    def test_statistics_describe_the_kept_examples_only(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config(max_examples=25))
        assert prepared.statistics.total_examples == 25
        assert sum(prepared.statistics.examples_by_source.values()) == 25

    def test_counts_by_question_type_and_difficulty(self):
        prepared = prepare_dataset(adapted_corpus(squad=30, mcq=10), dataset_config())
        assert prepared.statistics.targets_by_question_type == {
            "mcq": 10,
            "short_answer": 30,
        }
        assert prepared.statistics.targets_by_difficulty == {"easy": 30, "medium": 10}

    def test_counts_by_marks_are_derivable_from_the_kept_targets(self):
        prepared = prepare_dataset(adapted_corpus(squad=30, mcq=10), dataset_config())
        marks = {}
        for item in prepared.examples:
            for entry in item.targets:
                marks[entry.marks] = marks.get(entry.marks, 0) + 1
        assert marks == {1: 40}
        assert prepared.statistics.marks_total == 40

    def test_unique_groups_are_counted(self):
        prepared = prepare_dataset(adapted_corpus(squad=40, mcq=20), dataset_config())
        assert prepared.statistics.distinct_group_keys > 0
        assert prepared.statistics.distinct_group_keys <= prepared.statistics.total_examples

    def test_per_split_statistics_are_computed(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert set(prepared.split_statistics) == {"train", "validation", "test"}
        total = sum(stats.total_examples for stats in prepared.split_statistics.values())
        assert total == prepared.statistics.total_examples

    def test_grounded_examples_are_counted(self):
        """SQuAD offsets verify; MCQ has none. The split between them must be visible."""
        prepared = prepare_dataset(adapted_corpus(squad=20, mcq=20), dataset_config())
        assert prepared.statistics.grounded_examples == 20
        assert prepared.statistics.grounded_rate == 0.5

    def test_statistics_round_trip_through_serialization(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        restored = statistics_from_dict(prepared.statistics.as_dict())
        assert isinstance(restored, DatasetStatistics)
        assert restored.total_examples == prepared.statistics.total_examples
        assert restored.examples_by_source == prepared.statistics.examples_by_source


class TestFingerprintsAndReproducibility:
    """A recorded dataset has to be identifiable later."""

    def test_the_fingerprint_identifies_the_kept_corpus(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert prepared.fingerprint == compute_dataset_fingerprint(prepared.examples)

    def test_the_fingerprint_is_order_independent(self):
        examples = [example(index) for index in range(10)]
        assert compute_dataset_fingerprint(examples) == compute_dataset_fingerprint(
            list(reversed(examples))
        )

    def test_a_changed_corpus_changes_the_fingerprint(self):
        small = prepare_dataset(adapted_corpus(squad=20, mcq=10), dataset_config())
        large = prepare_dataset(adapted_corpus(squad=21, mcq=10), dataset_config())
        assert small.fingerprint != large.fingerprint

    def test_the_split_summary_carries_the_same_fingerprint(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert prepared.splits.as_dict()["dataset_fingerprint"] == prepared.fingerprint
        assert prepared.metadata.fingerprint == prepared.fingerprint

    def test_the_splitter_is_recorded_by_version(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        assert prepared.splits.splitter == "deterministic-group-splitter-v1"

    def test_the_group_assignments_reconstruct_the_partition(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        recorded: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
        for assignment in prepared.splits.assignments:
            recorded[assignment.split.value].update(assignment.example_ids)
        for name in recorded:
            assert recorded[name] == {item.id for item in prepared.splits[name]}


class TestSerialization:
    """Everything in a report has to survive being written down and read back."""

    def test_the_prepared_dataset_serializes_to_json(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        payload = json.loads(json.dumps(prepared.as_dict(), default=str))
        assert payload["fingerprint"] == prepared.fingerprint
        assert payload["splits"]["sizes"] == prepared.splits.sizes
        assert payload["preparation"]["kept"] == prepared.report.kept
        assert set(payload["split_statistics"]) == {"train", "validation", "test"}

    def test_the_examples_round_trip(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        for item in prepared.examples[:5]:
            assert example_from_dict(item.as_dict()) == item

    def test_the_report_serializes_with_every_stage_present(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        payload = prepared.report.as_dict()
        assert list(payload["stage_counts"]) == list(PREPARATION_STAGES)
        assert payload["kept"] + payload["dropped"] == payload["stage_counts"]["adapted"]

    def test_the_ingestion_record_serializes(self):
        record = SourceIngestion(
            source_id="squad-qg", records_seen=10, examples_adapted=8, rejected=2
        )
        payload = json.loads(json.dumps(record.as_dict()))
        assert payload["rejection_rate"] == 0.2

    def test_the_metadata_serializes(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        payload = json.loads(json.dumps(prepared.metadata.as_dict(), default=str))
        assert payload["fingerprint"] == prepared.fingerprint
        assert payload["statistics"]["total_examples"] == prepared.statistics.total_examples


# ---------------------------------------------------------------------------
# Step estimation
# ---------------------------------------------------------------------------


class TestStepEstimation:
    """The arithmetic transformers will do, computed where it can be checked."""

    def test_accumulation_multiplies_the_effective_batch(self):
        estimate = estimate_step_count(
            1000, batch_size=2, gradient_accumulation_steps=8, epochs=1
        )
        assert estimate.effective_batch_size == 16
        assert estimate.steps_per_epoch == 63
        assert estimate.total_steps == 63

    def test_a_partial_final_batch_still_costs_a_step(self):
        assert estimate_step_count(
            10, batch_size=1, gradient_accumulation_steps=4, epochs=1
        ).steps_per_epoch == 3

    def test_epochs_multiply_the_total(self):
        one = estimate_step_count(800, batch_size=1, gradient_accumulation_steps=8, epochs=1)
        three = estimate_step_count(800, batch_size=1, gradient_accumulation_steps=8, epochs=3)
        assert three.total_steps == one.total_steps * 3

    def test_forgetting_accumulation_would_overstate_the_run(self):
        """The wrong version of this arithmetic is the reason it lives in one tested place."""
        estimate = estimate_step_count(
            800, batch_size=1, gradient_accumulation_steps=8, epochs=1
        )
        assert estimate.total_steps == 100
        assert estimate.total_steps != 800

    def test_the_warmup_ratio_converts_to_steps(self):
        estimate = estimate_step_count(
            800, batch_size=1, gradient_accumulation_steps=8, epochs=1, warmup_ratio=0.03
        )
        assert estimate.total_steps == 100
        assert estimate.warmup_steps == 3

    def test_a_tiny_run_still_gets_one_warmup_step(self):
        estimate = estimate_step_count(
            2, batch_size=1, gradient_accumulation_steps=1, epochs=1, warmup_ratio=0.03
        )
        assert estimate.total_steps == 2
        assert estimate.warmup_steps == 1

    def test_no_warmup_when_the_ratio_is_zero(self):
        assert estimate_step_count(800, warmup_ratio=0.0).warmup_steps == 0

    def test_wall_clock_is_absent_without_a_measured_rate(self):
        estimate = estimate_step_count(800)
        assert estimate.estimated_seconds is None
        assert estimate.as_dict()["estimated_hours"] is None

    def test_wall_clock_uses_the_supplied_rate(self):
        estimate = estimate_step_count(
            800, batch_size=1, gradient_accumulation_steps=8, epochs=1, seconds_per_step=2.5
        )
        assert estimate.total_steps == 100
        assert estimate.estimated_seconds == 250.0
        assert estimate.as_dict()["estimated_hours"] == 0.07

    def test_an_empty_training_split_yields_no_steps(self):
        estimate = estimate_step_count(0)
        assert estimate.steps_per_epoch == 0
        assert estimate.total_steps == 0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("batch_size", 0),
            ("gradient_accumulation_steps", 0),
            ("epochs", 0),
            ("batch_size", -1),
        ],
    )
    def test_non_positive_values_are_refused(self, field, value):
        with pytest.raises(SizingError, match="positive integer"):
            estimate_step_count(100, **{field: value})

    def test_the_label_identifies_the_configuration(self):
        estimate = estimate_step_count(
            100, batch_size=2, gradient_accumulation_steps=8, epochs=3
        )
        assert estimate.label == "b2xa8xe3"

    def test_the_default_plans_include_the_measured_configuration(self):
        assert (1, 8, 1) in DEFAULT_STEP_PLANS
        estimates = build_step_estimates(1000)
        assert len(estimates) == len(DEFAULT_STEP_PLANS)

    def test_duplicate_plans_are_collapsed(self):
        estimates = build_step_estimates(100, plans=[(1, 8, 1), (1, 8, 1), (1, 8, 2)])
        assert [estimate.label for estimate in estimates] == ["b1xa8xe1", "b1xa8xe2"]

    def test_estimates_serialize(self):
        payload = json.loads(json.dumps(build_step_estimates(500)[0].as_dict()))
        assert payload["label"]
        assert payload["total_steps"] > 0


class TestTokenLengthSummary:
    """Percentiles, because a maximum alone does not settle a sequence length."""

    def test_an_empty_summary_is_all_zero(self):
        summary = TokenLengthSummary.from_values([])
        assert summary.count == 0
        assert summary.as_dict()["p95"] == 0

    def test_the_percentiles_are_real_values_from_the_distribution(self):
        summary = TokenLengthSummary.from_values(list(range(1, 101)))
        assert summary.minimum == 1
        assert summary.maximum == 100
        assert summary.p50 == 50
        assert summary.p95 == 95
        assert summary.mean == 50.5
        assert summary.total == 5050

    def test_a_long_tail_shows_up_as_a_gap_between_p95_and_max(self):
        values = [100] * 99 + [4000]
        summary = TokenLengthSummary.from_values(values)
        assert summary.p95 == 100
        assert summary.maximum == 4000

    def test_the_summary_serializes_with_stable_keys(self):
        payload = TokenLengthSummary.from_values([1, 2, 3]).as_dict()
        assert set(payload) == {"count", "min", "mean", "p50", "p95", "max", "total"}


class TestTokenizationTimeEstimate:
    """A planning figure that says it is one."""

    def test_the_estimate_states_its_assumption(self):
        estimate = estimate_tokenization_seconds(90_000)
        assert estimate["measured"] is False
        assert estimate["assumed_records_per_second"] == 900.0
        assert estimate["estimated_seconds"] == 100.0
        assert "not a measurement" in estimate["note"]

    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(SizingError, match="positive"):
            estimate_tokenization_seconds(10, records_per_second=0)


# ---------------------------------------------------------------------------
# Token measurement
# ---------------------------------------------------------------------------


class TestTokenMeasurement:
    """Measured the way TRL measures, or the numbers describe something else."""

    def records(self, count: int = 6, tokenizer: Any = None):
        """Build conversational records for the smoke-sized corpus."""
        config = experiment_config()
        examples = [example(index) for index in range(count)]
        return build_training_records(examples, config, tokenizer=tokenizer), config

    def test_lengths_are_measured_for_every_record(self):
        tokenizer = FakeTokenizer()
        records, config = self.records(tokenizer=tokenizer)
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.examples == len(records)
        assert sizing.prompt_tokens.count == len(records)
        assert sizing.completion_tokens.count == len(records)
        assert sizing.total_tokens.count == len(records)

    def test_the_halves_sum_to_the_total(self):
        tokenizer = FakeTokenizer()
        records, config = self.records(tokenizer=tokenizer)
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert (
            sizing.prompt_tokens.total + sizing.completion_tokens.total
            == sizing.total_tokens.total
        )

    def test_the_completion_is_shorter_than_the_prompt(self):
        """The prompt carries the passage and the output contract; the target is one object."""
        tokenizer = FakeTokenizer()
        records, config = self.records(tokenizer=tokenizer)
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.completion_tokens.maximum < sizing.prompt_tokens.minimum

    def test_the_reasoning_flag_moves_tokens_from_completion_to_prompt(self):
        """The Phase 17B.2 fix, visible in the sizing numbers."""
        tokenizer = FakeTokenizer()
        config = experiment_config()
        examples = [example(index) for index in range(4)]
        without = measure_record_lengths(
            build_training_records(examples, config), tokenizer, config
        )
        with_flag = measure_record_lengths(
            build_training_records(examples, config, tokenizer=tokenizer),
            tokenizer,
            config,
        )
        assert with_flag.total_tokens.total == without.total_tokens.total
        assert with_flag.prompt_tokens.total > without.prompt_tokens.total
        assert with_flag.completion_tokens.total < without.completion_tokens.total

    def test_records_without_the_reasoning_column_are_counted(self):
        """Sizing must measure what TRL will, and TRL reads only the per-record column."""
        tokenizer = FakeTokenizer()
        config = experiment_config()
        examples = [example(index) for index in range(4)]

        without = measure_record_lengths(
            build_training_records(examples, config), tokenizer, config
        )
        assert without.records_missing_template_kwargs == 4

        with_flag = measure_record_lengths(
            build_training_records(examples, config, tokenizer=tokenizer),
            tokenizer,
            config,
        )
        assert with_flag.records_missing_template_kwargs == 0

    def test_no_config_derived_default_is_substituted_for_the_column(self):
        """A default here would measure a boundary the trainer will not use."""
        tokenizer = FakeTokenizer()
        config = experiment_config()
        records = build_training_records([example()], config)
        assert "chat_template_kwargs" not in records[0]
        sizing = measure_record_lengths(records, tokenizer, config)
        # Without the column the empty think block sits in the completion, so the completion is
        # longer than it would be with the flag. Equality would mean a default leaked in.
        with_flag = measure_record_lengths(
            build_training_records([example()], config, tokenizer=tokenizer),
            tokenizer,
            config,
        )
        assert sizing.completion_tokens.total > with_flag.completion_tokens.total

    def test_truncation_is_counted_against_max_seq_length(self):
        tokenizer = FakeTokenizer()
        config = experiment_config(model={"max_seq_length": 32})
        records = build_training_records(
            [example(index) for index in range(4)], config, tokenizer=tokenizer
        )
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.truncated == 4
        assert sizing.truncation_rate == 1.0
        assert sizing.max_seq_length == 32

    def test_nothing_truncates_at_a_generous_limit(self):
        tokenizer = FakeTokenizer()
        config = experiment_config(model={"max_seq_length": 4096})
        records = build_training_records(
            [example(index) for index in range(4)], config, tokenizer=tokenizer
        )
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.truncated == 0
        assert sizing.truncation_rate == 0.0

    def test_a_tokenizer_without_a_chat_template_is_refused(self):
        config = experiment_config()
        records = build_training_records([example()], config)
        with pytest.raises(SizingError, match="no chat template"):
            measure_record_lengths(records, SimpleNamespace(chat_template=None), config)

    def test_a_non_conversational_record_is_refused(self):
        tokenizer = FakeTokenizer()
        config = experiment_config()
        with pytest.raises(SizingError, match="conversational shape"):
            measure_record_lengths([{"text": "not conversational"}], tokenizer, config)

    def test_the_split_sizing_serializes(self):
        tokenizer = FakeTokenizer()
        records, config = self.records(tokenizer=tokenizer)
        payload = json.loads(
            json.dumps(measure_record_lengths(records, tokenizer, config).as_dict())
        )
        assert payload["split"] == "train"
        assert set(payload["prompt_tokens"]) == {
            "count",
            "min",
            "mean",
            "p50",
            "p95",
            "max",
            "total",
        }

    def test_combining_splits_keeps_the_counts_exact(self):
        tokenizer = FakeTokenizer()
        config = experiment_config()
        per_split = {}
        for name, count in (("train", 6), ("validation", 2), ("test", 2)):
            records = build_training_records(
                [example(index) for index in range(count)], config, tokenizer=tokenizer
            )
            per_split[name] = measure_record_lengths(
                records, tokenizer, config, split=name
            )
        overall = summarize_token_lengths(per_split)
        assert overall.split == "overall"
        assert overall.examples == 10
        assert overall.total_tokens.count == 10
        assert overall.total_tokens.total == sum(
            sizing.total_tokens.total for sizing in per_split.values()
        )

    def test_combining_nothing_yields_an_empty_summary(self):
        assert summarize_token_lengths({}).examples == 0


class TestRealSquadRecordLengths:
    """Regression guard for the 2-token prompt / 0-token completion collapse.

    A first Phase 17C sizing run over 1,996 real ``rajpurkar/squad`` records reported
    ``prompt min=2 mean=2.0 max=2``, ``completion min=0 max=0`` and a truncation rate of zero.
    The cause was not the adapter, the target schema or the record shape: transformers 5.x
    defaults ``apply_chat_template(..., tokenize=True)`` to ``return_dict=True``, so it returns
    a ``BatchEncoding``, and ``len()`` of that is the key count -- ``input_ids`` plus
    ``attention_mask``, so 2.

    Every assertion here is on a realistic magnitude rather than an internal consistency
    property, because 2 and 0 are internally consistent: 2 - 2 == 0, and 2 < 1024.
    """

    def squad_records(self, count: int = 4):
        """Adapt realistic SQuAD rows and render them into conversational records.

        The passages are deliberately of differing lengths, as real SQuAD paragraphs are. A
        fixture of uniform length cannot distinguish a real measurement from a constant.
        """
        tokenizer = FakeTokenizer()
        config = experiment_config()
        rows = []
        for index in range(count):
            row = squad_record(index)
            # Real SQuAD paragraphs run from roughly one sentence to several hundred words.
            row["context"] = f"{row['context']} " + ("Additional detail. " * (index * 4))
            rows.append(row)
        examples = list(adapt_records("squad-qg", rows))
        records = build_training_records(examples, config, tokenizer=tokenizer)
        return records, tokenizer, config, examples

    def test_the_adapter_produces_the_expected_target_schema(self):
        """Rule out source adaptation before blaming the measurement."""
        _, _, _, examples = self.squad_records(1)
        assert len(examples) == 1
        item = examples[0]
        assert item.source == "squad-qg"
        assert len(item.targets) == 1
        target_object = item.targets[0]
        assert target_object.question_type is QuestionType.SHORT_ANSWER
        assert target_object.difficulty is Difficulty.EASY
        assert target_object.marks == 1
        assert target_object.question.strip()
        assert target_object.answer == "Chlorophyll"
        assert item.grounding is not None, "a verified SQuAD offset must ground the example"

    def test_the_records_carry_real_prompt_and_completion_content(self):
        """Rule out record construction too."""
        records, _, _, examples = self.squad_records(1)
        record = records[0]
        assert [message["role"] for message in record["prompt"]] == ["system", "user"]
        assert [message["role"] for message in record["completion"]] == ["assistant"]
        user = record["prompt"][1]["content"]
        assert examples[0].context in user
        assert len(user) > 300, "the user turn carries the passage and the output contract"
        completion = record["completion"][0]["content"]
        assert completion.startswith('{"question_type":')
        assert "Chlorophyll" in completion
        assert record["chat_template_kwargs"] == {"enable_thinking": False}

    def test_prompt_and_completion_lengths_are_realistic(self):
        """The assertion the shipped bug fails: 2 and 0 are not plausible token counts."""
        records, tokenizer, config, _ = self.squad_records()
        sizing = measure_record_lengths(records, tokenizer, config, split="train")

        assert sizing.prompt_tokens.minimum > 100, (
            "a prompt holding a SQuAD passage and the output contract cannot be this short; "
            f"got min={sizing.prompt_tokens.minimum}"
        )
        assert sizing.completion_tokens.minimum > 10, (
            "the supervised target is a JSON object with a question and an answer; "
            f"got min={sizing.completion_tokens.minimum}"
        )
        assert sizing.total_tokens.minimum > 110

    def test_no_record_has_an_empty_completion(self):
        """Zero supervised tokens means no learning signal, whatever the corpus."""
        records, tokenizer, config, _ = self.squad_records()
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.completion_tokens.minimum > 0
        assert sizing.prompt_tokens.minimum > 0

    def test_the_lengths_are_not_the_key_count_of_a_batch_encoding(self):
        """Named for the exact failure, so a future regression is unambiguous."""
        records, tokenizer, config, _ = self.squad_records()
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.prompt_tokens.maximum != 2
        assert sizing.total_tokens.maximum != 2
        assert sizing.completion_tokens.maximum != 0

    def test_the_lengths_vary_across_records(self):
        """Every record reporting an identical length is the signature of a constant."""
        records, tokenizer, config, _ = self.squad_records(6)
        sizing = measure_record_lengths(records, tokenizer, config, split="train")
        assert sizing.total_tokens.minimum != sizing.total_tokens.maximum

    def test_a_dict_returning_tokenizer_is_handled(self):
        """Transformers 5.x returns a BatchEncoding; the ids must be taken from it."""
        records, tokenizer, config, _ = self.squad_records(2)

        class DictOnlyTokenizer(FakeTokenizer):
            """Ignores return_dict and always returns a mapping, as a strict 5.x would."""

            def apply_chat_template(self, messages, **kwargs):
                kwargs["return_dict"] = False
                ids = super().apply_chat_template(messages, **kwargs)
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        strict = DictOnlyTokenizer()
        strict._atoms = tokenizer._atoms
        sizing = measure_record_lengths(records, strict, config, split="train")
        assert sizing.prompt_tokens.minimum > 100
        assert sizing.completion_tokens.minimum > 10

    def test_a_list_returning_tokenizer_is_handled(self):
        """A tokenizer honouring return_dict=False must work identically."""
        records, tokenizer, config, _ = self.squad_records(2)

        class ListOnlyTokenizer(FakeTokenizer):
            """Always returns a bare list, as transformers 4.x did."""

            def apply_chat_template(self, messages, **kwargs):
                kwargs["return_dict"] = False
                return super().apply_chat_template(messages, **kwargs)

        listing = ListOnlyTokenizer()
        listing._atoms = tokenizer._atoms
        assert measure_record_lengths(
            records, listing, config, split="train"
        ).prompt_tokens.minimum > 100

    def test_a_batched_list_of_lists_is_unwrapped(self):
        """Some processors return a batch of one even for a single example."""
        records, tokenizer, config, _ = self.squad_records(2)

        class BatchedTokenizer(FakeTokenizer):
            """Wraps the ids in an outer list, as a VLM processor does."""

            def apply_chat_template(self, messages, **kwargs):
                kwargs["return_dict"] = False
                ids = super().apply_chat_template(messages, **kwargs)
                return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

        batched = BatchedTokenizer()
        batched._atoms = tokenizer._atoms
        assert measure_record_lengths(
            records, batched, config, split="train"
        ).completion_tokens.minimum > 10

    def test_a_mapping_without_input_ids_is_refused(self):
        records, tokenizer, config, _ = self.squad_records(1)

        class WrongKeys(FakeTokenizer):
            """Returns a mapping that carries no token ids."""

            def apply_chat_template(self, messages, **kwargs):
                return {"attention_mask": [1, 1, 1]}

        with pytest.raises(SizingError, match="no 'input_ids'"):
            measure_record_lengths(records, WrongKeys(), config)

    def test_a_non_sequence_return_is_refused(self):
        records, tokenizer, config, _ = self.squad_records(1)

        class ReturnsAnInt(FakeTokenizer):
            """Returns something that is not a sequence at all."""

            def apply_chat_template(self, messages, **kwargs):
                return 457

        with pytest.raises(SizingError, match="not a sequence of"):
            measure_record_lengths(records, ReturnsAnInt(), config)

    def collapsed_sizing(self):
        """Build the exact degenerate measurement the Studio reported: 2 prompt, 0 completion."""
        from qa_gen_runtime.sizing import DatasetSizing, SplitSizing

        collapsed = SplitSizing(
            split="overall",
            examples=1996,
            prompt_tokens=TokenLengthSummary.from_values([2] * 1996),
            completion_tokens=TokenLengthSummary.from_values([0] * 1996),
            total_tokens=TokenLengthSummary.from_values([2] * 1996),
            truncated=0,
            max_seq_length=1024,
        )
        return DatasetSizing(
            tokenizer_id="Qwen/Qwen3-4B",
            max_seq_length=1024,
            measured=True,
            splits={"train": collapsed},
            overall=collapsed,
        )

    def test_the_audit_blocks_the_collapsed_measurement(self):
        """Had this finding existed, the first run would not have looked like a success."""
        prepared = prepare_dataset(adapted_corpus(squad=0, mcq=20), dataset_config())
        findings = {
            finding.code: finding for finding in audit_dataset(prepared, self.collapsed_sizing())
        }
        assert "empty_completion_tokens" in findings
        assert findings["empty_completion_tokens"].blocking is True
        assert "BatchEncoding" in findings["empty_completion_tokens"].message

    def test_the_audit_passes_a_healthy_measurement(self):
        records, tokenizer, config, _ = self.squad_records(8)
        prepared = prepare_dataset(adapted_corpus(squad=0, mcq=20), dataset_config())
        per_split = {"train": measure_record_lengths(records, tokenizer, config)}
        from qa_gen_runtime.sizing import DatasetSizing

        sizing = DatasetSizing(
            measured=True, splits=per_split, overall=summarize_token_lengths(per_split)
        )
        codes = {finding.code for finding in audit_dataset(prepared, sizing)}
        assert "empty_completion_tokens" not in codes
        assert "empty_prompt_tokens" not in codes


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestSourceCatalogue:
    """What is knowable before anything is read."""

    def test_every_registered_adapter_has_a_catalogue_entry(self):
        from qa_gen import registered_sources

        assert set(SOURCE_CATALOGUE) == set(registered_sources())

    def test_the_documented_sources_are_present(self):
        assert set(SOURCE_CATALOGUE) == {
            "squad-qg",
            "lmqg-squad-qag",
            "learningq-qg",
            "edu-mcq",
            "race-mcq",
        }

    def test_the_hub_backed_sources_name_a_repository(self):
        for source_id in ("squad-qg", "lmqg-squad-qag", "race-mcq"):
            entry = catalogue_entry(source_id)
            assert entry.hub_available
            assert "/" in entry.dataset_id
            assert not entry.requires_local_path

    def test_the_sources_without_a_mirror_require_a_local_path(self):
        for source_id in ("learningq-qg", "edu-mcq"):
            entry = catalogue_entry(source_id)
            assert entry.requires_local_path
            assert not entry.hub_available

    def test_no_source_is_gated(self):
        """A gated corpus would need a credential, which this pipeline does not introduce."""
        assert not any(entry.gated for entry in SOURCE_CATALOGUE.values())

    def test_race_names_the_configuration_it_reads(self):
        """``ehovy/race`` publishes three and defaults to none, so one has to be named."""
        entry = catalogue_entry("race-mcq")
        assert entry.default_config_name == "all"
        assert entry.request().config_name == "all"

    def test_resolving_race_carries_the_configuration_through(self):
        """The loader reads it off the request, so it has to survive resolution."""
        (request,) = resolve_requests(["race-mcq"])
        assert request.dataset_id == "ehovy/race"
        assert request.config_name == "all"
        assert request.split == "train"

    def test_the_other_sources_name_no_configuration(self):
        """Single-configuration repositories must keep passing nothing, not 'all'."""
        for source_id in ("squad-qg", "lmqg-squad-qag", "learningq-qg", "edu-mcq"):
            assert catalogue_entry(source_id).default_config_name is None

    def test_the_race_notes_record_the_non_commercial_terms(self):
        """Nothing in code enforces licensing, so the catalogue is where it is written."""
        notes = " ".join(catalogue_entry("race-mcq").notes).lower()
        assert "non-commercial" in notes
        assert "redistribut" in notes

    def test_every_entry_carries_notes(self):
        for entry in SOURCE_CATALOGUE.values():
            assert entry.notes, entry.source_id

    def test_an_unknown_source_is_self_diagnosing(self):
        with pytest.raises(SourceLoadError, match="Known sources"):
            catalogue_entry("squad")

    def test_the_catalogue_description_includes_the_adapter_spec(self):
        described = describe_catalogue()
        assert set(described) == set(SOURCE_CATALOGUE)
        assert described["squad-qg"]["adapter_spec"]["dataset_id"] == "rajpurkar/squad"
        json.dumps(described)


class TestRequestResolution:
    """Which corpora would be read, and what each still needs."""

    def test_naming_a_source_resolves_it(self):
        requests = resolve_requests(["squad-qg"])
        assert [request.source_id for request in requests] == ["squad-qg"]
        assert requests[0].dataset_id == "rajpurkar/squad"
        assert requests[0].split == "train"

    def test_defaulting_skips_sources_that_need_input(self):
        """The Hub-backed corpora resolve; the two needing a local file are left out."""
        requests = resolve_requests([])
        assert [request.source_id for request in requests] == [
            "lmqg-squad-qag",
            "race-mcq",
            "squad-qg",
        ]

    def test_defaulting_includes_a_local_source_when_a_path_is_given(self):
        requests = resolve_requests([], local_paths={"edu-mcq": "corpus.jsonl"})
        assert "edu-mcq" in {request.source_id for request in requests}

    def test_a_local_path_overrides_the_hub(self):
        requests = resolve_requests(
            ["squad-qg"], local_paths={"squad-qg": "local.jsonl"}
        )
        assert requests[0].local_path == "local.jsonl"

    def test_a_split_override_is_honoured(self):
        requests = resolve_requests(["squad-qg"], splits={"squad-qg": "validation"})
        assert requests[0].split == "validation"

    def test_a_read_limit_reaches_every_request(self):
        requests = resolve_requests(["squad-qg", "lmqg-squad-qag"], limit=500)
        assert all(request.limit == 500 for request in requests)

    def test_a_missing_local_path_is_refused_for_a_real_run(self):
        with pytest.raises(SourceLoadError, match="--source-path"):
            resolve_requests(["edu-mcq"])

    def test_a_missing_local_path_is_reported_rather_than_refused_for_a_plan(self):
        """--plan exists to tell you this; raising would mean already knowing the answer."""
        requests = resolve_requests(["edu-mcq"], require_readable=False)
        assert len(requests) == 1
        assert not request_is_readable(requests[0])

    def test_the_requirements_are_described_per_source(self):
        requests = resolve_requests(
            ["squad-qg", "edu-mcq"],
            local_paths={},
            require_readable=False,
        )
        lines = describe_requirements(requests)
        assert any("--allow-download" in line for line in lines)
        assert any("needs --source-path" in line for line in lines)

    def test_a_readable_request_is_reported_as_such(self):
        requests = resolve_requests(["edu-mcq"], local_paths={"edu-mcq": "x.jsonl"})
        assert request_is_readable(requests[0])
        assert "reads the local file" in describe_requirements(requests)[0]

    def test_the_request_serializes(self):
        payload = json.loads(json.dumps(resolve_requests(["squad-qg"])[0].as_dict()))
        assert payload["source_id"] == "squad-qg"


class TestLocalFileReading:
    """The route that needs no dataset library at all."""

    def test_jsonl_is_read_line_by_line(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            "".join(json.dumps(mcq_record(index)) + "\n" for index in range(5)),
            encoding="utf-8",
        )
        records = load_source_records(
            SourceRequest(source_id="edu-mcq", local_path=str(path))
        )
        assert len(records) == 5
        assert records[0]["id"] == "mcq-0"

    def test_a_json_list_is_read(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps([mcq_record(0), mcq_record(1)]), encoding="utf-8")
        assert len(load_source_records(SourceRequest("edu-mcq", local_path=str(path)))) == 2

    @pytest.mark.parametrize("key", ["data", "records", "examples"])
    def test_a_wrapped_json_list_is_unwrapped(self, tmp_path, key):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps({key: [mcq_record(0)]}), encoding="utf-8")
        assert len(load_source_records(SourceRequest("edu-mcq", local_path=str(path)))) == 1

    def test_the_read_limit_bounds_the_records(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            "".join(json.dumps(mcq_record(index)) + "\n" for index in range(20)),
            encoding="utf-8",
        )
        records = load_source_records(
            SourceRequest("edu-mcq", local_path=str(path), limit=5)
        )
        assert len(records) == 5

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            json.dumps(mcq_record(0)) + "\n\n   \n" + json.dumps(mcq_record(1)) + "\n",
            encoding="utf-8",
        )
        assert len(load_source_records(SourceRequest("edu-mcq", local_path=str(path)))) == 2

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(SourceLoadError, match="not found"):
            load_source_records(
                SourceRequest("edu-mcq", local_path=str(tmp_path / "absent.jsonl"))
            )

    def test_an_unsupported_suffix_is_reported(self, tmp_path):
        path = tmp_path / "corpus.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(SourceLoadError, match="unsupported source file suffix"):
            load_source_records(SourceRequest("edu-mcq", local_path=str(path)))

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(SourceLoadError, match="line 2"):
            load_source_records(SourceRequest("edu-mcq", local_path=str(path)))

    def test_a_non_object_record_is_refused(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps([mcq_record(0), 42]), encoding="utf-8")
        with pytest.raises(SourceLoadError, match="not an object"):
            load_source_records(SourceRequest("edu-mcq", local_path=str(path)))

    def test_a_request_with_no_origin_is_refused(self):
        with pytest.raises(SourceLoadError, match="nothing to read"):
            load_source_records(SourceRequest("edu-mcq"))


class TestNoSilentDownloads:
    """The default has to be the safe one, and it has to be mechanical."""

    def test_the_offline_switches_are_set_and_then_restored(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
        with sources_module._offline(True):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert "HF_HUB_OFFLINE" not in os.environ
        assert os.environ["HF_DATASETS_OFFLINE"] == "0"

    def test_allowing_downloads_leaves_the_environment_alone(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with sources_module._offline(False):
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_a_hub_read_forces_offline_by_default(self, monkeypatch):
        seen: dict[str, Any] = {}

        def fake_require_datasets():
            def load_dataset(**kwargs):
                seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
                seen["kwargs"] = kwargs
                return [mcq_record(0), mcq_record(1)]

            return SimpleNamespace(load_dataset=load_dataset)

        monkeypatch.setattr(sources_module, "require_datasets", fake_require_datasets)
        records = load_source_records(
            SourceRequest("squad-qg", dataset_id="rajpurkar/squad"), allow_download=False
        )
        assert seen["offline"] == "1"
        assert len(records) == 2

    def test_allowing_downloads_does_not_force_offline(self, monkeypatch):
        seen: dict[str, Any] = {}

        def fake_require_datasets():
            def load_dataset(**kwargs):
                seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
                return []

            return SimpleNamespace(load_dataset=load_dataset)

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(sources_module, "require_datasets", fake_require_datasets)
        load_source_records(
            SourceRequest("squad-qg", dataset_id="rajpurkar/squad"), allow_download=True
        )
        assert seen["offline"] is None

    def test_an_uncached_corpus_names_the_flag_that_would_fetch_it(self, monkeypatch):
        def fake_require_datasets():
            def load_dataset(**kwargs):
                raise OSError("offline mode is enabled")

            return SimpleNamespace(load_dataset=load_dataset)

        monkeypatch.setattr(sources_module, "require_datasets", fake_require_datasets)
        with pytest.raises(SourceLoadError, match="--allow-download"):
            load_source_records(
                SourceRequest("squad-qg", dataset_id="rajpurkar/squad"),
                allow_download=False,
            )

    def test_a_local_read_never_touches_the_dataset_library(self, monkeypatch, tmp_path):
        def explode():
            raise AssertionError("datasets must not be required for a local read")

        monkeypatch.setattr(sources_module, "require_datasets", explode)
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps(mcq_record(0)) + "\n", encoding="utf-8")
        assert len(load_source_records(SourceRequest("edu-mcq", local_path=str(path)))) == 1

    def test_no_module_reads_or_passes_a_token(self):
        """No credential is introduced anywhere in the Phase 17C runtime."""
        import ast
        import importlib
        import inspect

        forbidden = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "use_auth_token", "token"}
        for name in ("sources", "sizing", "prepare"):
            module = importlib.import_module(f"qa_gen_runtime.{name}")
            tree = ast.parse(inspect.getsource(module))
            referenced = {
                node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            keywords = {
                keyword.arg
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                for keyword in node.keywords
                if keyword.arg
            }
            leaked = sorted((referenced | keywords) & forbidden)
            assert not leaked, f"qa_gen_runtime.{name} references {leaked}"


class TestSourceAdaptation:
    """Per-source rejection counts, which cannot be recovered afterwards."""

    def test_records_are_adapted_and_counted(self):
        request = SourceRequest("edu-mcq", local_path="corpus.jsonl")
        loaded = adapt_source(request, [mcq_record(index) for index in range(5)])
        assert len(loaded.examples) == 5
        assert loaded.ingestion.records_seen == 5
        assert loaded.ingestion.rejected == 0

    def test_refused_records_are_counted_and_sampled(self):
        request = SourceRequest("edu-mcq", local_path="corpus.jsonl")
        records = [mcq_record(0), {"context": "x"}, {"nothing": True}]
        loaded = adapt_source(request, records)
        assert len(loaded.examples) == 1
        assert loaded.ingestion.rejected == 2
        assert len(loaded.rejection_messages) == 2
        assert loaded.ingestion.rejection_rate == round(2 / 3, 4)

    def test_strict_mode_makes_the_first_refusal_fatal(self):
        from qa_gen import AdapterError

        request = SourceRequest("edu-mcq", local_path="corpus.jsonl")
        with pytest.raises(AdapterError):
            adapt_source(request, [{"context": "x"}], skip_invalid=False)

    def test_the_rejection_sample_is_bounded(self):
        request = SourceRequest("edu-mcq", local_path="corpus.jsonl")
        loaded = adapt_source(request, [{"nope": index} for index in range(50)])
        assert loaded.ingestion.rejected == 50
        assert len(loaded.rejection_messages) == 5

    def test_the_provenance_records_where_the_records_came_from(self):
        loaded = adapt_source(
            SourceRequest("edu-mcq", local_path="corpus.jsonl"), [mcq_record(0)]
        )
        assert loaded.ingestion.dataset_id == "corpus.jsonl"
        assert "local file" in loaded.ingestion.notes[0]

    def test_a_hub_source_records_its_split_and_revision(self):
        loaded = adapt_source(
            SourceRequest("squad-qg", dataset_id="rajpurkar/squad", revision="abc123"),
            [squad_record(0)],
        )
        assert "split=train" in loaded.ingestion.notes[0]
        assert "revision=abc123" in loaded.ingestion.notes[0]

    def test_the_loaded_source_serializes_without_inlining_examples(self):
        loaded = adapt_source(
            SourceRequest("edu-mcq", local_path="corpus.jsonl"),
            [mcq_record(index) for index in range(3)],
        )
        payload = json.loads(json.dumps(loaded.as_dict()))
        assert payload["example_count"] == 3
        assert "examples" not in payload


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


class TestAudit:
    """Dataset-level problems that only exist once a partition does."""

    def test_a_healthy_dataset_raises_nothing_blocking(self):
        prepared = prepare_dataset(adapted_corpus(squad=0, mcq=40), dataset_config())
        findings = audit_dataset(prepared, None)
        assert not [finding for finding in findings if finding.blocking]

    def test_a_source_missing_from_a_split_is_reported_but_not_blocking(self):
        """SQuAD groups by title into few huge groups, so a small split can miss it."""
        corpus = {
            "squad-qg": list(
                adapt_records("squad-qg", [squad_record(i, title="Only") for i in range(30)])
            ),
            "edu-mcq": list(adapt_records("edu-mcq", [mcq_record(i) for i in range(30)])),
        }
        prepared = prepare_dataset(corpus, dataset_config())
        findings = audit_dataset(prepared, None)
        codes = {finding.code for finding in findings}
        assert any(code.startswith("source_missing_from_") for code in codes)
        assert all(not finding.blocking for finding in findings if "source_missing" in finding.code)

    def test_truncation_above_the_threshold_blocks(self):
        tokenizer = FakeTokenizer()
        config = experiment_config(model={"max_seq_length": 32})
        prepared = prepare_dataset(adapted_corpus(squad=0, mcq=20), dataset_config())
        records = build_training_records(prepared.splits.train, config, tokenizer=tokenizer)
        per_split = {"train": measure_record_lengths(records, tokenizer, config)}
        from qa_gen_runtime.sizing import DatasetSizing

        sizing = DatasetSizing(
            measured=True, splits=per_split, overall=summarize_token_lengths(per_split)
        )
        findings = audit_dataset(prepared, sizing)
        blocking = [finding for finding in findings if finding.blocking]
        assert any(finding.code == "truncation_above_threshold" for finding in blocking)

    def test_no_truncation_finding_at_a_generous_limit(self):
        tokenizer = FakeTokenizer()
        config = experiment_config(model={"max_seq_length": 4096})
        prepared = prepare_dataset(adapted_corpus(squad=0, mcq=20), dataset_config())
        records = build_training_records(prepared.splits.train, config, tokenizer=tokenizer)
        per_split = {"train": measure_record_lengths(records, tokenizer, config)}
        from qa_gen_runtime.sizing import DatasetSizing

        sizing = DatasetSizing(
            measured=True, splits=per_split, overall=summarize_token_lengths(per_split)
        )
        assert not [
            finding
            for finding in audit_dataset(prepared, sizing)
            if finding.code == "truncation_above_threshold"
        ]

    def test_findings_serialize(self):
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        payload = [finding.as_dict() for finding in audit_dataset(prepared, None)]
        json.dumps(payload)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestCliArguments:
    """The surface, and that nothing expensive is the default."""

    def test_a_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_downloads_are_off_by_default(self):
        args = build_parser().parse_args(["--config", "c.yaml"])
        assert args.allow_download is False
        assert args.plan is False
        assert args.write_dataset is False
        assert args.no_token_stats is False

    def test_the_repeatable_flags_default_to_empty(self):
        args = build_parser().parse_args(["--config", "c.yaml"])
        assert args.source_path == []
        assert args.source_split == []
        assert args.step_plan == []

    def test_source_paths_are_parsed(self):
        parsed = prepare_module._parse_mapping(
            ["edu-mcq=data/a.jsonl", "learningq-qg=data/b.json"], flag="--source-path"
        )
        assert parsed == {"edu-mcq": "data/a.jsonl", "learningq-qg": "data/b.json"}

    @pytest.mark.parametrize("entry", ["edu-mcq", "=path", "edu-mcq=", "  =  "])
    def test_a_malformed_mapping_exits(self, entry):
        with pytest.raises(SystemExit, match="SOURCE=VALUE"):
            prepare_module._parse_mapping([entry], flag="--source-path")

    def test_step_plans_are_parsed(self):
        assert prepare_module._parse_plans(["1,8,3", "2,4,1"]) == ((1, 8, 3), (2, 4, 1))

    def test_no_step_plan_falls_back_to_the_defaults(self):
        assert prepare_module._parse_plans([]) == DEFAULT_STEP_PLANS

    @pytest.mark.parametrize("entry", ["1,8", "1,8,3,4", "a,b,c"])
    def test_a_malformed_step_plan_exits(self, entry):
        with pytest.raises(SystemExit, match="--step-plan"):
            prepare_module._parse_plans([entry])


class TestCliBehaviour:
    """End to end against a local corpus. No network, no model weights."""

    def test_plan_reads_nothing_and_writes_nothing(self, local_corpus, capsys):
        code = main(
            ["--config", str(local_corpus.config), "--plan", "--log-level", "WARNING"]
        )
        output = capsys.readouterr().out
        assert code == 0
        assert "plan only" in output
        assert "NEEDS --source-path" in output
        assert not (local_corpus.root / "out").exists()

    def test_plan_names_what_each_source_needs(self, local_corpus, capsys):
        main(["--config", str(local_corpus.config), "--plan", "--log-level", "WARNING"])
        assert "needs --source-path edu-mcq" in capsys.readouterr().out

    def test_a_real_run_prepares_and_writes_a_report(self, local_corpus, capsys):
        code = main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        output = capsys.readouterr().out
        assert "prepared" in output
        assert "[ SPLITS ]" in output
        report = next((local_corpus.root / "out").rglob("sizing.json"))
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["dataset"]["splits"]["sizes"]["train"] > 0
        assert payload["sizing"]["measured"] is False

    def test_the_report_records_the_whole_catalogue(self, local_corpus):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        payload = json.loads(
            next((local_corpus.root / "out").rglob("sizing.json")).read_text(encoding="utf-8")
        )
        assert set(payload["catalogue"]) == set(SOURCE_CATALOGUE)

    def test_writing_the_dataset_emits_one_jsonl_per_split(self, local_corpus):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--write-dataset",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        directory = next((local_corpus.root / "out").iterdir())
        names = {path.name for path in directory.iterdir()}
        assert {"train.jsonl", "validation.jsonl", "test.jsonl"} <= names
        assert {"sizing.json", "dataset.json"} <= names

    def test_the_written_examples_read_back(self, local_corpus):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--write-dataset",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        train = next((local_corpus.root / "out").rglob("train.jsonl"))
        lines = [line for line in train.read_text(encoding="utf-8").splitlines() if line]
        restored = [example_from_dict(json.loads(line)) for line in lines]
        assert restored
        assert all(item.source == "edu-mcq" for item in restored)

    def test_the_report_directory_is_named_by_fingerprint(self, local_corpus):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        directory = next((local_corpus.root / "out").iterdir())
        payload = json.loads((directory / "sizing.json").read_text(encoding="utf-8"))
        assert directory.name.endswith(payload["dataset"]["fingerprint"])

    def test_json_output_is_parseable(self, local_corpus, capsys):
        code = main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--json",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["experiment"] == "qgen-prep-test"

    def test_the_read_limit_bounds_the_corpus(self, local_corpus, capsys):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--read-limit",
                "10",
                "--no-token-stats",
                "--json",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["dataset"]["preparation"]["stage_counts"]["adapted"] == 10

    def test_custom_step_plans_reach_the_report(self, local_corpus, capsys):
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--step-plan",
                "1,16,4",
                "--json",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        labels = [entry["label"] for entry in payload["sizing"]["step_estimates"]]
        assert labels == ["b1xa16xe4"]

    def test_a_missing_config_exits_one(self, tmp_path, capsys):
        code = main(
            ["--config", str(tmp_path / "absent.yaml"), "--plan", "--log-level", "WARNING"]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_an_invalid_config_exits_one(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text("name: bad\ndataset:\n  train_ratio: 2.0\n", encoding="utf-8")
        code = main(["--config", str(path), "--plan", "--log-level", "WARNING"])
        assert code == 1
        assert "train_ratio" in capsys.readouterr().err

    def test_a_missing_source_file_exits_one(self, local_corpus, capsys):
        code = main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.root / 'absent.jsonl'}",
                "--no-token-stats",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_a_blocking_finding_exits_three(self, local_corpus, monkeypatch, capsys):
        from qa_gen_runtime.prepare import AuditFinding

        monkeypatch.setattr(
            prepare_module,
            "audit_dataset",
            lambda prepared, sizing: (
                AuditFinding(code="synthetic", message="forced", blocking=True),
            ),
        )
        code = main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--no-token-stats",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 3
        assert "synthetic" in capsys.readouterr().err

    def test_token_statistics_use_the_tokenizer_when_available(
        self, local_corpus, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            prepare_module, "load_sizing_tokenizer", lambda config: FakeTokenizer()
        )
        code = main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--json",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        sizing = payload["sizing"]
        assert sizing["measured"] is True
        assert sizing["overall"]["total_tokens"]["p95"] > 0
        assert payload["tokenization"]["measured"] is True
        assert payload["tokenization"]["records_per_second"] is not None

    def test_sampling_bounds_the_measured_records(self, local_corpus, monkeypatch, capsys):
        monkeypatch.setattr(
            prepare_module, "load_sizing_tokenizer", lambda config: FakeTokenizer()
        )
        main(
            [
                "--config",
                str(local_corpus.config),
                "--source-path",
                f"edu-mcq={local_corpus.records}",
                "--sample",
                "3",
                "--json",
                "--output-dir",
                str(local_corpus.root / "out"),
                "--log-level",
                "WARNING",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["sizing"]["sampled"] == 3
        assert payload["sizing"]["splits"]["train"]["examples"] == 3
        assert any("sampled" in note for note in payload["sizing"]["notes"])

    def test_skipping_token_statistics_states_it_rather_than_estimating(self, local_corpus):
        outcome = run_preparation(
            str(local_corpus.config),
            local_paths={"edu-mcq": str(local_corpus.records)},
            token_stats=False,
            output_dir=str(local_corpus.root / "out"),
        )
        assert outcome.sizing is not None
        assert outcome.sizing.measured is False
        assert outcome.sizing.splits == {}
        assert any("No character-based estimate" in note for note in outcome.sizing.notes)

    def test_the_text_report_covers_every_section(self, local_corpus, monkeypatch):
        monkeypatch.setattr(
            prepare_module, "load_sizing_tokenizer", lambda config: FakeTokenizer()
        )
        outcome = run_preparation(
            str(local_corpus.config),
            local_paths={"edu-mcq": str(local_corpus.records)},
            output_dir=str(local_corpus.root / "out"),
        )
        text = format_outcome(outcome)
        for section in (
            "[ SOURCES ]",
            "[ PREPARATION ]",
            "[ SPLITS ]",
            "[ CONTENT ]",
            "[ TOKEN LENGTHS ]",
            "[ ESTIMATED TRAINING STEPS ]",
            "[ TOKENIZATION TIME ]",
            "[ ARTIFACTS ]",
        ):
            assert section in text, section

    def test_the_outcome_states_no_weights_were_loaded(self, local_corpus):
        outcome = run_preparation(
            str(local_corpus.config),
            local_paths={"edu-mcq": str(local_corpus.records)},
            token_stats=False,
            output_dir=str(local_corpus.root / "out"),
        )
        assert any("no model weights" in note for note in outcome.notes)
        assert any("refused" in note for note in outcome.notes)


class TestNoModelWeightsAreLoaded:
    """The sizing command's central constraint, asserted rather than intended."""

    def test_no_phase_17c_module_references_a_model_loader(self):
        import ast
        import importlib
        import inspect

        forbidden = {
            "AutoModelForCausalLM",
            "AutoModel",
            "load_base_model",
            "load_trainable_model",
            "attach_adapters",
            "get_peft_model",
            "SFTTrainer",
            "build_trainer",
        }
        for name in ("sources", "sizing", "prepare"):
            module = importlib.import_module(f"qa_gen_runtime.{name}")
            tree = ast.parse(inspect.getsource(module))
            referenced = {
                node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            leaked = sorted(referenced & forbidden)
            assert not leaked, f"qa_gen_runtime.{name} references {leaked}"

    def test_no_phase_17c_module_trains(self):
        import ast
        import importlib
        import inspect

        for name in ("sources", "sizing", "prepare"):
            module = importlib.import_module(f"qa_gen_runtime.{name}")
            tree = ast.parse(inspect.getsource(module))
            called = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert "train" not in called, f"qa_gen_runtime.{name} calls .train()"
            assert "backward" not in called, f"qa_gen_runtime.{name} calls .backward()"

    def test_the_sizing_tokenizer_loader_only_loads_a_tokenizer(self, monkeypatch):
        import qa_gen_runtime.loader as loader_module
        from qa_gen_runtime.sizing import load_sizing_tokenizer

        calls: list[str] = []
        monkeypatch.setattr(
            loader_module,
            "load_tokenizer",
            lambda model_config: calls.append(model_config.model_id) or FakeTokenizer(),
        )
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("no model loader may be called for sizing")

        for name in ("load_base_model", "load_trainable_model"):
            monkeypatch.setattr(loader_module, name, refuse)
        tokenizer = load_sizing_tokenizer(experiment_config())
        assert isinstance(tokenizer, FakeTokenizer)
        assert calls == ["Qwen/Qwen3-4B"]

    def test_a_tokenizer_failure_says_it_is_not_a_gpu_problem(self, monkeypatch):
        import qa_gen_runtime.loader as loader_module
        from qa_gen_runtime.sizing import load_sizing_tokenizer

        monkeypatch.setattr(
            loader_module,
            "load_tokenizer",
            lambda model_config: (_ for _ in ()).throw(OSError("not cached")),
        )
        with pytest.raises(SizingError, match="no model weights are loaded"):
            load_sizing_tokenizer(experiment_config())


class TestMixedCorpusPreparation:
    """Preparing SQuAD and RACE together, which is what ``qgen-mixed.yaml`` asks for.

    A mixed corpus can fail in ways a single-corpus one cannot: one source can silently
    dominate, a question type can vanish, provenance can be lost, and an MCQ answer index can
    survive adaptation but not the pipeline. Each of those is asserted here rather than assumed
    from the single-source tests.

    Scale is deliberately small. The mechanisms are scale-free -- the per-source cap buckets on
    ``example.source`` and samples each bucket independently -- so 60 examples prove the same
    invariant as 10,000 and the suite stays fast. The real 10,000/10,000 balance is a property of
    the shipped configuration, pinned separately in :class:`TestShippedQgenConfigs`.
    """

    def config(self, **overrides: Any) -> QuestionGenerationDatasetConfig:
        """A dataset configuration shaped like the shipped mixed one."""
        defaults = {
            "sources": ("squad-qg", "race-mcq"),
            "seed": 42,
            "shuffle_seed": 1234,
            "train_ratio": 0.9,
            "validation_ratio": 0.05,
            "test_ratio": 0.05,
            "group_by": "context",
            "min_context_chars": 64,
            "max_context_chars": 3000,
            "max_examples_per_source": 20,
            "max_examples": 40,
            "drop_duplicates": True,
        }
        return QuestionGenerationDatasetConfig(**{**defaults, **overrides})

    def prepare(self, **overrides: Any):
        """Prepare a mixed corpus with the mixed-shaped configuration."""
        return prepare_dataset(mixed_corpus(), self.config(**overrides))

    # -- balance ------------------------------------------------------------

    def test_each_source_contributes_exactly_the_per_source_cap(self):
        """The headline invariant: an equal mix, not a proportional one."""
        prepared = self.prepare()
        assert prepared.statistics.examples_by_source == {"race-mcq": 20, "squad-qg": 20}

    def test_the_total_equals_the_sum_of_the_per_source_caps(self):
        prepared = self.prepare()
        assert prepared.statistics.total_examples == 40
        assert prepared.report.kept == 40

    def test_the_total_cap_is_a_no_op_when_it_equals_twice_the_per_source_cap(self):
        """Why the shipped config sets max_examples to exactly 2x the per-source cap.

        A lower total would re-sample the pooled corpus proportionally and break the balance,
        so the report must show nothing was dropped at that stage.
        """
        prepared = self.prepare()
        assert prepared.report.total_capped == 0
        assert prepared.report.stage_counts["capped_per_source"] == 40
        assert prepared.report.stage_counts["capped_total"] == 40

    def test_a_smaller_total_cap_does_break_the_balance(self):
        """The failure mode the config's arithmetic avoids, demonstrated rather than asserted.

        Not a recommendation. It documents why max_examples is not set to something tidy like
        30,000 when the per-source caps sum to 20,000.
        """
        prepared = prepare_dataset(mixed_corpus(), self.config(max_examples=30))
        assert prepared.report.total_capped == 10
        assert sum(prepared.statistics.examples_by_source.values()) == 30
        assert prepared.statistics.examples_by_source != {"race-mcq": 15, "squad-qg": 15}

    def test_a_source_with_less_than_the_cap_is_not_topped_up(self):
        """No upsampling, and no reallocating the shortfall to the other corpus."""
        corpus = mixed_corpus(squad=60, race=9)
        prepared = prepare_dataset(corpus, self.config())
        assert prepared.statistics.examples_by_source == {"race-mcq": 9, "squad-qg": 20}

    # -- provenance ---------------------------------------------------------

    def test_every_example_traces_to_one_of_the_two_corpora(self):
        prepared = self.prepare()
        assert {item.source for item in prepared.examples} == {"squad-qg", "race-mcq"}

    def test_provenance_survives_into_every_split(self):
        """A split missing a source explains a metric of zero for that question type."""
        breakdown = self.prepare().splits.source_breakdown()
        assert set(breakdown["train"]) == {"squad-qg", "race-mcq"}
        assert sum(
            count for split in breakdown.values() for count in split.values()
        ) == 40

    def test_the_metadata_names_both_corpora(self):
        assert self.prepare().metadata.sources == ("race-mcq", "squad-qg")

    # -- question types -----------------------------------------------------

    def test_the_question_type_distribution_is_an_even_split(self):
        """SQuAD contributes short_answer and RACE contributes mcq, 20 targets each."""
        prepared = self.prepare()
        assert prepared.statistics.targets_by_question_type == {"mcq": 20, "short_answer": 20}

    def test_squad_stays_short_answer_and_race_stays_mcq(self):
        """Neither adapter's question type is coerced by being mixed with the other."""
        prepared = self.prepare()
        by_source: dict[str, set[str]] = {}
        for item in prepared.examples:
            for entry in item.targets:
                by_source.setdefault(item.source, set()).add(entry.question_type.value)
        assert by_source == {"squad-qg": {"short_answer"}, "race-mcq": {"mcq"}}

    def test_only_race_examples_carry_options(self):
        prepared = self.prepare()
        for item in prepared.examples:
            has_options = any(entry.options for entry in item.targets)
            assert has_options == (item.source == "race-mcq"), item.id

    # -- RACE answer index --------------------------------------------------

    def test_the_race_correct_option_index_is_preserved(self):
        """The label 'C' must still mean option 2 after the whole pipeline, not option 0."""
        race = [item for item in self.prepare().examples if item.source == "race-mcq"]
        assert race
        for item in race:
            entry = item.primary_target
            assert entry.options == RACE_OPTIONS
            assert entry.correct_option_index == RACE_CORRECT_INDEX
            assert entry.answer == RACE_OPTIONS[RACE_CORRECT_INDEX]

    def test_the_race_answer_index_survives_the_jsonl_round_trip(self):
        """The JSONL is what the trainer reads, so the index has to survive serialization."""
        race = [item for item in self.prepare().examples if item.source == "race-mcq"]
        restored = [example_from_dict(json.loads(json.dumps(item.as_dict()))) for item in race]
        assert [item.primary_target.correct_option_index for item in restored] == [
            RACE_CORRECT_INDEX
        ] * len(race)
        assert all(
            item.primary_target.answer == RACE_OPTIONS[RACE_CORRECT_INDEX] for item in restored
        )

    def test_the_option_count_distribution_covers_only_the_race_half(self):
        prepared = self.prepare()
        assert prepared.statistics.mcq_option_counts.count == 20
        assert prepared.statistics.mcq_option_counts.minimum == 4
        assert prepared.statistics.mcq_option_counts.maximum == 4

    # -- leakage ------------------------------------------------------------

    def test_no_leakage_group_spans_two_splits(self):
        assert self.prepare().splits.leaked_group_keys() == frozenset()

    def test_a_race_article_is_never_split_across_splits(self):
        """RACE attaches several questions to one article; they must move together."""
        prepared = self.prepare()
        placement: dict[str, set[str]] = {}
        for name in ("train", "validation", "test"):
            for item in prepared.splits[name]:
                if item.source == "race-mcq":
                    placement.setdefault(item.context, set()).add(name)
        assert placement
        assert all(len(names) == 1 for names in placement.values()), placement

    def test_the_two_halves_group_at_different_granularities(self):
        """``group_by: context`` does not mean both halves group by passage.

        ``effective_group_key`` prefers an adapter-set ``group_key``, and the SQuAD adapter
        sets ``title:<article>`` because paragraphs from one Wikipedia article overlap heavily.
        RACE sets none, so it falls back to the context fingerprint. The mix therefore has a
        few large SQuAD groups and many small RACE ones, which is stricter than passage
        grouping on the SQuAD side and worth stating rather than discovering.
        """
        prepared = self.prepare()
        by_source: dict[str, set[str]] = {}
        for item in prepared.examples:
            prefix = item.effective_group_key().split(":", 1)[0]
            by_source.setdefault(item.source, set()).add(prefix)
        assert by_source == {"squad-qg": {"title"}, "race-mcq": {"ctx"}}

    def test_group_keys_never_collide_across_the_two_corpora(self):
        """Distinct prefixes, so a SQuAD article and a RACE passage cannot share a group."""
        prepared = self.prepare()
        squad_keys = {
            item.effective_group_key()
            for item in prepared.examples
            if item.source == "squad-qg"
        }
        race_keys = {
            item.effective_group_key()
            for item in prepared.examples
            if item.source == "race-mcq"
        }
        assert squad_keys and race_keys
        assert squad_keys.isdisjoint(race_keys)

    def test_every_example_id_is_unique(self):
        """RACE's repeated example_id must not collapse questions sharing an article."""
        ids = [item.id for item in self.prepare().examples]
        assert len(set(ids)) == len(ids) == 40

    # -- determinism --------------------------------------------------------

    def test_preparation_is_deterministic(self):
        first, second = self.prepare(), self.prepare()
        assert first.fingerprint == second.fingerprint
        for name in ("train", "validation", "test"):
            assert [item.id for item in first.splits[name]] == [
                item.id for item in second.splits[name]
            ]

    def test_the_fingerprint_is_independent_of_source_read_order(self):
        """Same content, corpora supplied in the other order, same dataset."""
        corpus = mixed_corpus()
        forward = prepare_dataset(corpus, self.config())
        reversed_corpus = {key: corpus[key] for key in reversed(list(corpus))}
        backward = prepare_dataset(reversed_corpus, self.config())
        assert forward.fingerprint == backward.fingerprint
        assert forward.statistics.examples_by_source == backward.statistics.examples_by_source

    def test_the_fingerprint_changes_when_the_mix_changes(self):
        """Otherwise the fingerprint could not distinguish this dataset from a SQuAD-only one."""
        mixed = self.prepare().fingerprint
        squad_only = prepare_dataset(
            {"squad-qg": mixed_corpus()["squad-qg"]},
            self.config(sources=("squad-qg",)),
        ).fingerprint
        assert mixed != squad_only

    def test_the_shuffle_seed_reorders_without_moving_examples(self):
        """Training order may change; the partition and its fingerprint may not."""
        first = self.prepare()
        second = self.prepare(shuffle_seed=99)
        assert first.fingerprint == second.fingerprint
        assert {item.id for item in first.splits.train} == {
            item.id for item in second.splits.train
        }
        assert [item.id for item in first.splits.train] != [
            item.id for item in second.splits.train
        ]

    def test_the_mix_is_interleaved_rather_than_one_corpus_then_the_other(self):
        """Ids share a source prefix, so an unshuffled split would teach one corpus at a time."""
        sequence = [item.source for item in self.prepare().splits.train]
        transitions = sum(
            1 for a, b in zip(sequence[:-1], sequence[1:], strict=True) if a != b
        )
        assert transitions > 3, f"only {transitions} source changes in {len(sequence)} examples"

    # -- validation is not weakened ----------------------------------------

    def test_nothing_is_rejected_by_the_adapters_or_by_validation(self):
        """A clean corpus must pass cleanly; a silent drop here would hide a real one."""
        prepared = self.prepare()
        assert prepared.report.invalid_dropped == 0
        assert prepared.report.duplicate_ids_dropped == 0
        assert prepared.report.duplicate_content_dropped == 0
        assert prepared.validation.invalid_example_ids == frozenset()

    def test_an_over_long_context_is_still_rejected_in_a_mixed_corpus(self):
        """The context bound applies to both corpora, not just the one it was tuned for."""
        corpus = mixed_corpus(squad=60, race=59)
        corpus["race-mcq"].extend(
            adapt_records("race-mcq", [race_record(999, article="x" * 5000)])
        )
        prepared = prepare_dataset(corpus, self.config(max_examples_per_source=60))
        assert prepared.report.invalid_dropped == 1
        assert "x" * 5000 not in {item.context for item in prepared.examples}

    def test_a_short_context_is_still_rejected_in_a_mixed_corpus(self):
        corpus = mixed_corpus(squad=60, race=59)
        corpus["race-mcq"].extend(
            adapt_records("race-mcq", [race_record(998, article="Too short")])
        )
        prepared = prepare_dataset(corpus, self.config(max_examples_per_source=60))
        assert prepared.report.invalid_dropped == 1

    def test_content_deduplication_is_cross_source(self):
        """One shared passage must collapse before either corpus is capped."""
        shared = f"{PASSAGE} Shared passage."
        corpus = mixed_corpus(squad=60, race=60)
        corpus["race-mcq"].extend(
            adapt_records("race-mcq", [race_record(0, article=shared)])
        )
        corpus["squad-qg"].extend(
            adapt_records("race-mcq", [race_record(0, article=shared)])
        )
        prepared = prepare_dataset(corpus, self.config(max_examples_per_source=61))
        assert prepared.report.duplicate_ids_dropped == 1

    # -- reporting surface --------------------------------------------------

    def test_the_report_states_every_stage(self):
        report = self.prepare().report.as_dict()
        assert list(report["stage_counts"]) == list(PREPARATION_STAGES)
        assert report["stage_counts"]["adapted"] == 120

    def test_the_report_serializes(self):
        payload = json.loads(json.dumps(self.prepare().as_dict(), default=str))
        assert payload["statistics"]["examples_by_source"] == {"race-mcq": 20, "squad-qg": 20}
        assert payload["statistics"]["targets_by_question_type"] == {
            "mcq": 20,
            "short_answer": 20,
        }
        assert payload["splits"]["leaked_group_keys"] == []

    def test_only_the_squad_half_is_grounded(self):
        """SQuAD carries offsets and RACE carries none, so the rate should be about a half."""
        prepared = self.prepare()
        assert prepared.statistics.grounded_examples == 20
        assert prepared.statistics.grounded_rate == 0.5
        assert all(
            item.is_grounded == (item.source == "squad-qg") for item in prepared.examples
        )


class TestShippedQgenConfigs:
    """The two production corpus configurations, and how they differ.

    ``qgen-squad.yaml`` and ``qgen-learningq.yaml`` exist to be compared: the same base model,
    the same QLoRA settings and the same schedule over two corpora with very different question
    registers. That comparison is only meaningful if the training stack really is identical, so
    it is asserted here rather than maintained by hand.
    """

    def config_path(self, name: str) -> str:
        """Path to a shipped generative configuration."""
        from qa_ml.paths import find_repo_root

        return str(find_repo_root() / "ml" / "configs" / "qgen" / name)

    def load(self, name: str):
        """Load and validate a shipped generative configuration."""
        from qa_gen_runtime.config_io import load_experiment_config

        config = load_experiment_config(self.config_path(name))
        config.validate()
        return config

    def test_the_learningq_config_loads_and_validates(self):
        config = self.load("qgen-learningq.yaml")
        assert config.name == "qgen-learningq"
        assert config.phase == "17"

    def test_learningq_is_the_only_source(self):
        assert self.load("qgen-learningq.yaml").dataset.sources == ("learningq-qg",)

    def test_the_source_is_a_registered_adapter(self):
        from qa_gen import registered_sources

        for source in self.load("qgen-learningq.yaml").dataset.sources:
            assert source in registered_sources()

    def test_the_training_stack_is_identical_to_the_squad_config(self):
        """The point of a second corpus config: only the corpus may differ."""
        squad = self.load("qgen-squad.yaml")
        learningq = self.load("qgen-learningq.yaml")
        assert learningq.model == squad.model
        assert learningq.lora == squad.lora
        assert learningq.training == squad.training

    def test_only_the_dataset_section_differs(self):
        squad = self.load("qgen-squad.yaml")
        learningq = self.load("qgen-learningq.yaml")
        assert learningq.dataset != squad.dataset
        assert learningq.config_hash() != squad.config_hash()

    def test_the_qlora_settings_are_the_measured_ones(self):
        config = self.load("qgen-learningq.yaml")
        assert config.model.model_id == "Qwen/Qwen3-4B"
        assert config.model.quantization == "4bit"
        assert config.model.quantization_type == "nf4"
        assert config.model.double_quantization is True
        assert config.model.compute_dtype == "bf16"
        assert config.model.max_seq_length == 1024
        assert config.model.reasoning_mode == "disabled"
        assert config.training.optimizer == "paged_adamw_8bit"
        assert config.training.gradient_checkpointing is True
        assert config.training.completion_only_loss is True
        assert config.training.packing is False

    def test_the_learningq_config_matches_the_verified_baseline(self):
        from qa_gen import VERIFIED_QWEN3_4B_L4

        config = self.load("qgen-learningq.yaml")
        assert config.baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()

    def test_the_context_bound_leaves_room_for_a_long_answer(self):
        """LearningQ targets are marks=5 descriptive answers; the target must not truncate."""
        squad = self.load("qgen-squad.yaml")
        learningq = self.load("qgen-learningq.yaml")
        assert learningq.dataset.max_context_chars < squad.dataset.max_context_chars
        assert learningq.dataset.max_context_chars == 2000
        assert learningq.dataset.min_context_chars == 64

    def test_the_grouping_keeps_a_source_document_whole(self):
        """The adapter sets group_key="doc:<id>", which effective_group_key prefers."""
        config = self.load("qgen-learningq.yaml")
        assert config.dataset.group_by == "context"
        record = {
            "context": PASSAGE,
            "question": "Explain how chlorophyll captures light energy.",
            "answer": "Chlorophyll absorbs photons and transfers the energy to the "
            "photosystem reaction centre.",
            "doc_id": "khan-1234",
        }
        item = next(iter(adapt_records("learningq-qg", [record])))
        assert item.group_key == "doc:khan-1234"
        assert item.effective_group_key() == "doc:khan-1234"

    def test_the_adapter_assigns_the_long_answer_shape(self):
        record = {
            "context": PASSAGE,
            "question": "Explain how chlorophyll captures light energy.",
            "answer": "Chlorophyll absorbs photons and transfers the energy onward.",
            "doc_id": "khan-1",
        }
        item = next(iter(adapt_records("learningq-qg", [record])))
        assert item.source == "learningq-qg"
        assert item.targets[0].question_type is QuestionType.LONG_ANSWER
        assert item.targets[0].difficulty is Difficulty.MEDIUM
        assert item.targets[0].marks == 5

    def test_a_real_run_refuses_without_a_local_path(self):
        """LearningQ has no Hub mirror, so the requirement must fail loudly and say how."""
        config = self.load("qgen-learningq.yaml")
        with pytest.raises(SourceLoadError, match="--source-path learningq-qg"):
            resolve_requests(config.dataset.sources)

    def test_a_plan_reports_the_requirement_rather_than_failing(self):
        config = self.load("qgen-learningq.yaml")
        requests = resolve_requests(config.dataset.sources, require_readable=False)
        assert len(requests) == 1
        assert not request_is_readable(requests[0])
        assert "needs --source-path learningq-qg" in describe_requirements(requests)[0]

    def test_a_local_path_makes_it_readable(self):
        config = self.load("qgen-learningq.yaml")
        requests = resolve_requests(
            config.dataset.sources,
            local_paths={"learningq-qg": "data/learningq.jsonl"},
        )
        assert requests[0].local_path == "data/learningq.jsonl"
        assert requests[0].dataset_id == ""
        assert request_is_readable(requests[0])
        assert "reads the local file" in describe_requirements(requests)[0]

    def test_answer_free_records_are_dropped_not_fatal(self):
        """Most LearningQ items have no written answer; strict mode must stay off."""
        request = SourceRequest("learningq-qg", local_path="learningq.jsonl")
        records = [
            {
                "context": PASSAGE,
                "question": "Explain photosynthesis.",
                "answer": "Chlorophyll absorbs light and drives the reaction.",
                "doc_id": "d1",
            },
            {"context": PASSAGE, "question": "Why is the sky blue?", "doc_id": "d2"},
            {"context": PASSAGE, "question": "Discuss respiration.", "answer": "", "doc_id": "d3"},
        ]
        loaded = adapt_source(request, records, skip_invalid=True)
        assert len(loaded.examples) == 1
        assert loaded.ingestion.rejected == 2
        assert loaded.ingestion.rejection_rate == round(2 / 3, 4)
        assert any("written answer" in message for message in loaded.rejection_messages)

    def test_strict_adapters_would_make_the_first_refusal_fatal(self):
        """Why the config documents not passing --strict-adapters for this corpus."""
        from qa_gen import AdapterError

        request = SourceRequest("learningq-qg", local_path="learningq.jsonl")
        with pytest.raises(AdapterError, match="written answer"):
            adapt_source(
                request,
                [{"context": PASSAGE, "question": "Why?", "doc_id": "d"}],
                skip_invalid=False,
            )

    def test_strict_adapters_is_off_by_default_on_the_cli(self):
        args = build_parser().parse_args(["--config", "c.yaml"])
        assert args.strict_adapters is False

    def test_the_race_config_loads_and_validates(self):
        config = self.load("qgen-race.yaml")
        assert config.name == "qgen-race"
        assert config.phase == "17"

    def test_race_is_the_only_source(self):
        assert self.load("qgen-race.yaml").dataset.sources == ("race-mcq",)

    def test_the_race_source_is_a_registered_adapter(self):
        from qa_gen import registered_sources

        for source in self.load("qgen-race.yaml").dataset.sources:
            assert source in registered_sources()

    def test_the_race_training_stack_is_identical_to_the_squad_config(self):
        race = self.load("qgen-race.yaml")
        squad = self.load("qgen-squad.yaml")
        assert race.model == squad.model
        assert race.lora == squad.lora
        assert race.training == squad.training

    def test_only_the_race_dataset_section_differs(self):
        race = self.load("qgen-race.yaml")
        squad = self.load("qgen-squad.yaml")
        assert race.dataset != squad.dataset
        assert race.config_hash() != squad.config_hash()

    def test_the_race_config_reports_no_baseline_deviations(self):
        from qa_gen import VERIFIED_QWEN3_4B_L4

        assert self.load("qgen-race.yaml").baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()

    def test_the_race_context_cap_removes_the_measured_truncation(self):
        """3,000 chars, not SQuAD's 4,000: measured, 0.90% of rows overflow 1,024 at 4,000."""
        race = self.load("qgen-race.yaml")
        squad = self.load("qgen-squad.yaml")
        assert race.dataset.max_context_chars == 3000
        assert race.dataset.max_context_chars < squad.dataset.max_context_chars

    def test_the_race_config_caps_the_corpus_at_twenty_thousand(self):
        assert self.load("qgen-race.yaml").dataset.max_examples == 20000

    def test_the_race_cap_is_applied_before_the_split(self):
        """So the cap yields 18,000 train examples, i.e. 2,250 steps -- not 20,000/2,500.

        Pinned because the config comment states that arithmetic and a reader will plan a
        GPU budget from it.
        """
        config = self.load("qgen-race.yaml")
        assert config.dataset.max_examples is not None
        train = round(config.dataset.max_examples * config.dataset.train_ratio)
        accumulation = config.training.gradient_accumulation_steps
        batch = config.training.per_device_train_batch_size
        assert train == 18000
        assert train // (batch * accumulation) == 2250

    def test_race_resolves_against_the_hub_without_a_local_path(self):
        """Unlike LearningQ: ehovy/race is a real mirror, so a run needs no extra input."""
        config = self.load("qgen-race.yaml")
        (request,) = resolve_requests(config.dataset.sources)
        assert request.dataset_id == "ehovy/race"
        assert request.config_name == "all"
        assert request_is_readable(request)
        assert "reads ehovy/race" in describe_requirements([request])[0]

    def test_the_requirement_line_names_the_configuration(self):
        """A plan that omitted it would understate what is about to be fetched."""
        (request,) = resolve_requests(["race-mcq"])
        assert "configuration 'all'" in describe_requirements([request])[0]

    def test_a_single_configuration_source_prints_no_configuration(self):
        """Most repositories have one default; printing 'None' for them would be noise."""
        (request,) = resolve_requests(["squad-qg"])
        assert "configuration" not in describe_requirements([request])[0]

    def test_a_race_row_maps_through_the_configured_source(self):
        """End to end at the source layer: a real record shape, no renaming, letter answer."""
        request = SourceRequest("race-mcq", dataset_id="ehovy/race", config_name="all")
        records = [
            {
                "example_id": "high1729.txt",
                "article": PASSAGE,
                "question": "What does chlorophyll absorb?",
                "options": ["Water", "Light", "Sugar", "Oxygen"],
                "answer": "B",
            }
        ]
        loaded = adapt_source(request, records, skip_invalid=False)
        assert loaded.ingestion.rejected == 0
        (item,) = loaded.examples
        assert item.source == "race-mcq"
        assert item.targets[0].question_type is QuestionType.MCQ
        assert item.targets[0].answer == "Light"
        assert item.targets[0].correct_option_index == 1

    def test_race_questions_sharing_an_article_survive_deduplication(self):
        """The 69.6% loss this guards against is silent, so it needs an explicit test."""
        from qa_gen.preparation import deduplicate_by_id

        request = SourceRequest("race-mcq", dataset_id="ehovy/race", config_name="all")
        records = [
            {
                "example_id": "high1729.txt",
                "article": PASSAGE,
                "question": f"Question number {index}?",
                "options": ["Water", "Light", "Sugar", "Oxygen"],
                "answer": "B",
            }
            for index in range(4)
        ]
        loaded = adapt_source(request, records, skip_invalid=False)
        kept, dropped = deduplicate_by_id(loaded.examples)
        assert len(kept) == 4
        assert dropped == ()

    def test_the_mixed_config_loads_and_validates(self):
        config = self.load("qgen-mixed.yaml")
        assert config.name == "qgen-mixed"
        assert config.phase == "17"

    def test_the_mixed_config_names_both_corpora_explicitly(self):
        """Not left to the "empty means everything" default, which would add lmqg-squad-qag."""
        assert set(self.load("qgen-mixed.yaml").dataset.sources) == {"squad-qg", "race-mcq"}

    def test_the_mixed_sources_are_registered_adapters(self):
        from qa_gen import registered_sources

        for source in self.load("qgen-mixed.yaml").dataset.sources:
            assert source in registered_sources()

    def test_the_mixed_config_asks_for_ten_thousand_from_each_corpus(self):
        dataset = self.load("qgen-mixed.yaml").dataset
        assert dataset.max_examples_per_source == 10000
        assert dataset.max_examples == 20000

    def test_the_total_cap_is_exactly_twice_the_per_source_cap(self):
        """The arithmetic that keeps the 10,000/10,000 balance exact.

        A total cap below the sum re-samples the pooled corpus proportionally, so this
        relationship is the whole reason the mix is balanced rather than approximately balanced.
        """
        dataset = self.load("qgen-mixed.yaml").dataset
        assert dataset.max_examples_per_source is not None
        assert dataset.max_examples == 2 * dataset.max_examples_per_source
        assert len(dataset.sources) == 2

    def test_the_mixed_config_uses_a_deterministic_seed(self):
        dataset = self.load("qgen-mixed.yaml").dataset
        assert dataset.seed == 42
        assert dataset.shuffle_seed == 1234

    def test_the_mixed_config_controls_leakage_by_context(self):
        assert self.load("qgen-mixed.yaml").dataset.group_by == "context"

    def test_the_mixed_config_deduplicates_before_capping(self):
        """Order is fixed in preparation; this pins that the flag enabling it is on."""
        assert self.load("qgen-mixed.yaml").dataset.drop_duplicates is True

    def test_the_mixed_split_ratios_target_eighteen_thousand_training_examples(self):
        config = self.load("qgen-mixed.yaml")
        dataset = config.dataset
        assert dataset.ratios == (0.9, 0.05, 0.05)
        assert dataset.max_examples is not None
        target_train = round(dataset.max_examples * dataset.train_ratio)
        assert target_train == 18000
        assert round(dataset.max_examples * dataset.validation_ratio) == 1000
        assert round(dataset.max_examples * dataset.test_ratio) == 1000

    def test_the_mixed_config_expects_the_documented_step_counts(self):
        """2,250 steps for one epoch and 4,500 for two, at batch 1 x accumulation 8."""
        config = self.load("qgen-mixed.yaml")
        train = round(config.dataset.max_examples * config.dataset.train_ratio)
        effective = (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        )
        assert effective == 8
        assert estimate_step_count(
            train, batch_size=1, gradient_accumulation_steps=8, epochs=1
        ).total_steps == 2250
        assert estimate_step_count(
            train, batch_size=1, gradient_accumulation_steps=8, epochs=2
        ).total_steps == 4500

    def test_the_default_step_plans_cover_both_required_configurations(self):
        """Both requested plans are reported without passing --step-plan."""
        assert (1, 8, 1) in DEFAULT_STEP_PLANS
        assert (1, 8, 2) in DEFAULT_STEP_PLANS

    def test_the_mixed_context_bound_is_the_race_bound(self):
        """The mix is bounded by RACE's longer passages, not SQuAD's."""
        mixed = self.load("qgen-mixed.yaml")
        race = self.load("qgen-race.yaml")
        squad = self.load("qgen-squad.yaml")
        assert mixed.dataset.max_context_chars == 3000
        assert mixed.dataset.max_context_chars == race.dataset.max_context_chars
        assert mixed.dataset.max_context_chars < squad.dataset.max_context_chars
        assert mixed.dataset.min_context_chars == 64

    def test_the_mixed_training_stack_is_identical_to_the_squad_config(self):
        """Only the corpus and the checkpoint interval may differ.

        The mixed config checkpoints periodically so a four-hour run is resumable, which
        qgen-squad.yaml does not. That is a durability setting, not an optimisation one: it
        changes what an interruption costs and nothing the model learns. So the comparison
        excludes exactly those three fields and asserts everything else matches, rather than
        being dropped for being inconvenient.
        """
        import dataclasses

        mixed = self.load("qgen-mixed.yaml")
        squad = self.load("qgen-squad.yaml")
        assert mixed.model == squad.model
        assert mixed.lora == squad.lora

        checkpointing = {"save_strategy", "save_steps", "save_total_limit"}
        normalized = dataclasses.replace(
            mixed.training,
            **{name: getattr(squad.training, name) for name in checkpointing},
        )
        assert normalized == squad.training

    def test_only_the_checkpoint_fields_differ_from_the_squad_config(self):
        """Names the deviation, so a future edit cannot hide behind the exclusion above."""
        import dataclasses

        mixed = self.load("qgen-mixed.yaml").training
        squad = self.load("qgen-squad.yaml").training
        differing = {
            item.name
            for item in dataclasses.fields(mixed)
            if getattr(mixed, item.name) != getattr(squad, item.name)
        }
        assert differing == {"save_strategy", "save_steps"}

    def test_the_mixed_optimisation_settings_are_untouched(self):
        """The settings the L4 measurement covered, pinned against the resumability change."""
        training = self.load("qgen-mixed.yaml").training
        assert training.learning_rate == 0.0002
        assert training.per_device_train_batch_size == 1
        assert training.gradient_accumulation_steps == 8
        assert training.effective_batch_size == 8
        assert training.num_train_epochs == 1
        assert training.warmup_ratio == 0.03
        assert training.lr_scheduler_type == "cosine"
        assert training.optimizer == "paged_adamw_8bit"
        assert training.gradient_checkpointing is True
        assert training.completion_only_loss is True
        assert training.packing is False
        assert training.seed == 42
        assert training.max_steps is None

    def test_the_mixed_config_is_resumable(self):
        """The point of the change: a run that dies at step 2,000 need not start over."""
        assert self.load("qgen-mixed.yaml").training.is_resumable is True

    def test_the_mixed_config_checkpoints_every_five_hundred_steps(self):
        training = self.load("qgen-mixed.yaml").training
        assert training.save_strategy == "steps"
        assert training.save_steps == 500
        assert training.save_total_limit == 2

    def test_the_checkpoint_interval_divides_the_run_into_useful_pieces(self):
        """500 steps of 2,248 is about a quarter of the run, so an interruption costs a quarter.

        Asserted as a relationship rather than a number, so changing the epoch count or the
        corpus size makes this fail rather than silently leaving a 500-step interval on a
        300-step run, where it would never fire.
        """
        config = self.load("qgen-mixed.yaml")
        train = round(config.dataset.max_examples * config.dataset.train_ratio)
        total = estimate_step_count(
            train, batch_size=1, gradient_accumulation_steps=8, epochs=1
        ).total_steps
        assert config.training.save_steps is not None
        assert config.training.save_steps < total
        assert total // config.training.save_steps >= 2

    def test_the_other_configs_are_not_resumable_and_say_so(self):
        """Unchanged, and worth pinning: only the production mixed run got this treatment."""
        for name in ("qgen-squad.yaml", "qgen-race.yaml", "qgen-learningq.yaml"):
            training = self.load(name).training
            assert training.save_steps is None, name
        assert self.load("qgen-squad.yaml").training.save_strategy == "no"

    def test_the_mixed_config_keeps_the_measured_qlora_settings(self):
        config = self.load("qgen-mixed.yaml")
        assert config.model.model_id == "Qwen/Qwen3-4B"
        assert config.model.max_seq_length == 1024
        assert config.model.quantization == "4bit"
        assert config.model.quantization_type == "nf4"
        assert config.model.reasoning_mode == "disabled"
        assert config.model.chat_template == "tokenizer"
        assert config.training.completion_only_loss is True
        assert config.training.gradient_checkpointing is True
        assert config.training.optimizer == "paged_adamw_8bit"

    def test_the_mixed_config_reports_no_baseline_deviations(self):
        from qa_gen import VERIFIED_QWEN3_4B_L4

        assert self.load("qgen-mixed.yaml").baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()

    def test_the_mixed_config_is_distinct_from_the_single_corpus_ones(self):
        mixed = self.load("qgen-mixed.yaml")
        for name in ("qgen-squad.yaml", "qgen-race.yaml"):
            other = self.load(name)
            assert mixed.dataset != other.dataset
            assert mixed.config_hash() != other.config_hash()

    def test_the_mixed_config_resolves_both_corpora_against_the_hub(self):
        """Neither half needs --source-path, so a real run needs no extra input."""
        config = self.load("qgen-mixed.yaml")
        requests = resolve_requests(config.dataset.sources)
        by_source = {request.source_id: request for request in requests}
        assert set(by_source) == {"squad-qg", "race-mcq"}
        assert by_source["squad-qg"].dataset_id == "rajpurkar/squad"
        assert by_source["race-mcq"].dataset_id == "ehovy/race"
        assert by_source["race-mcq"].config_name == "all"
        assert all(request_is_readable(request) for request in requests)

    def test_the_mixed_config_does_not_require_grounding(self):
        """Requiring it would discard the entire RACE half, which carries no offsets."""
        assert self.load("qgen-mixed.yaml").dataset.require_grounding is False

    def test_the_mixed_config_records_the_stricter_licence(self):
        """RACE's terms govern the mix, and the config is where a reader will look."""
        from qa_ml.paths import find_repo_root

        text = (
            find_repo_root() / "ml/configs/qgen/qgen-mixed.yaml"
        ).read_text(encoding="utf-8").lower()
        assert "non-commercial" in text
        assert "may not be published" in text

    def test_the_squad_config_is_unchanged(self):
        """Adding a corpus must not perturb the production SQuAD configuration."""
        squad = self.load("qgen-squad.yaml")
        assert squad.name == "qgen-squad"
        assert squad.dataset.sources == ("squad-qg",)
        assert squad.dataset.max_context_chars == 4000
        assert squad.dataset.train_ratio == 0.9
        assert squad.dataset.max_examples is None
        assert squad.training.max_steps is None

    def test_the_learningq_config_is_unchanged(self):
        learningq = self.load("qgen-learningq.yaml")
        assert learningq.dataset.sources == ("learningq-qg",)
        assert learningq.dataset.max_context_chars == 2000

    def test_all_configs_are_discovered_by_the_directory_guard(self):
        from qa_ml.paths import find_repo_root

        names = {
            path.name
            for path in (find_repo_root() / "ml" / "configs" / "qgen").glob("*.yaml")
        }
        assert {
            "qgen-smoke.yaml",
            "qgen-squad.yaml",
            "qgen-learningq.yaml",
            "qgen-race.yaml",
            "qgen-mixed.yaml",
        } <= names


class TestPhaseBoundary:
    """What Phase 17C deliberately does not do."""

    def test_the_shipped_smoke_configuration_is_untouched(self):
        from qa_ml.paths import find_repo_root

        text = (find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml").read_text(
            encoding="utf-8"
        )
        assert "max_steps: 20" in text
        assert "gradient_accumulation_steps: 8" in text

    def test_the_smoke_harness_still_refuses_to_train_without_run(self):
        from qa_gen_runtime.smoke import SmokeHarnessError, _call_trainer_train

        with pytest.raises(SmokeHarnessError, match="--run"):
            _call_trainer_train(SimpleNamespace(), execute=False)

    def test_preparation_is_importable_without_the_ml_stack(self):
        """qa_gen.preparation is stdlib-only, like the rest of the package."""
        import ast
        import importlib
        import inspect

        tree = ast.parse(inspect.getsource(importlib.import_module("qa_gen.preparation")))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {
            "__future__",
            "hashlib",
            "collections",
            "dataclasses",
            "typing",
            "qa_gen",
        }

    def test_the_prepared_dataset_carries_no_tokenized_features(self):
        """Tokenization is a sizing measurement here, not a stored artifact."""
        prepared = prepare_dataset(adapted_corpus(), dataset_config())
        payload = prepared.as_dict()
        assert "input_ids" not in json.dumps(payload)
        assert "features" not in payload
