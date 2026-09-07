"""Tests for dataset adapters, splitting, validation and statistics.

The adapter fixtures are literal dicts shaped like the real corpus records, checked against
the schema each adapter *declares* in its :class:`~qa_gen.adapters.AdapterSpec`. That is what
makes offline testing of the mapping meaningful: if the fixture and the declaration agree, and
the declaration is right, the mapping is right when real records arrive.

Nothing here downloads anything. :class:`TestNoDownloads` asserts that rather than assuming it.
"""

from __future__ import annotations

import json

import pytest

from qa_gen import (
    ADAPTER_REGISTRY,
    AdapterError,
    AdapterSpec,
    DatasetAdapter,
    DatasetIssueCode,
    DatasetMetadata,
    DatasetStatistics,
    DeterministicGroupSplitter,
    EducationalMcqAdapter,
    LearningQAdapter,
    LmqgSquadQagAdapter,
    QuestionGenerationDatasetConfig,
    QuestionGenerationExample,
    QuestionGenerationTarget,
    SplitError,
    SplitName,
    SplitRatios,
    SquadQuestionGenerationAdapter,
    UnknownAdapterError,
    adapt_records,
    adapter_for,
    compute_dataset_fingerprint,
    compute_statistics,
    find_duplicate_examples,
    registered_sources,
    statistics_from_dict,
    validate_dataset,
    validate_example,
)
from qa_paper import Difficulty, QuestionType

MITOCHONDRION = (
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. "
    "It generates most of the cell's supply of adenosine triphosphate, used as a source of "
    "chemical energy for the cell."
)

CHLOROPLAST = (
    "Chloroplasts are organelles that conduct photosynthesis in plant and algal cells. They "
    "capture light energy to make sugars, storing it in the bonds of glucose molecules."
)


def squad_record(index: int = 0, *, title: str = "Mitochondrion", context: str | None = None):
    """Build a SQuAD-shaped record with a verified answer offset."""
    passage = context if context is not None else f"{MITOCHONDRION} Paragraph {index}."
    answer = "adenosine triphosphate"
    return {
        "id": f"sq-{index}",
        "title": title,
        "context": passage,
        "question": f"What does the mitochondrion generate, variant {index}?",
        "answers": {"text": [answer], "answer_start": [passage.index(answer)]},
    }


def mcq_record(index: int = 0, **overrides):
    """Build an educational MCQ record."""
    defaults = {
        "id": f"mcq-{index}",
        "context": f"{MITOCHONDRION} Note {index}.",
        "question": f"Which organelle produces ATP? Variant {index}.",
        "options": ["Nucleus", "Mitochondrion", "Ribosome", "Vacuole"],
        "correct_index": 1,
        "topic": "Cell Biology",
    }
    return {**defaults, **overrides}


def race_record(index: int = 0, **overrides):
    """Build a record shaped like a real ``ehovy/race`` row.

    The field names and value conventions are the ones verified against the published
    dataset card: the passage is ``article``, the correct option is a label under ``answer``,
    ``options`` is a four-string list, and ``example_id`` is the source *filename*, which
    repeats across every question drawn from the same article.
    """
    defaults = {
        "example_id": "high1729.txt",
        "article": f"{MITOCHONDRION} Passage note {index}.",
        "question": f"Which organelle produces ATP? Variant {index}.",
        "options": ["Nucleus", "Mitochondrion", "Ribosome", "Vacuole"],
        "answer": "B",
    }
    return {**defaults, **overrides}


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
    """Build a valid example with a distinct context."""
    defaults = {
        "id": f"ex-{index}",
        "context": f"{MITOCHONDRION} Passage {index}.",
        "targets": (target(question=f"Question number {index}?"),),
        "source": "squad-qg",
        "topic": "Cell Biology",
    }
    return QuestionGenerationExample(**{**defaults, **overrides})


class TestAdapterRegistry:
    """The declared corpora."""

    def test_every_declared_source_is_registered(self):
        assert registered_sources() == (
            "edu-mcq",
            "learningq-qg",
            "lmqg-squad-qag",
            "race-mcq",
            "squad-qg",
        )

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_each_adapter_satisfies_the_protocol(self, source_id):
        assert isinstance(ADAPTER_REGISTRY[source_id], DatasetAdapter)

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_each_spec_declares_itself_consistently(self, source_id):
        spec = ADAPTER_REGISTRY[source_id].spec
        assert isinstance(spec, AdapterSpec)
        assert spec.source_id == source_id
        assert spec.dataset_id
        assert spec.description
        assert spec.required_fields
        assert spec.record_shape

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_every_required_field_is_documented_in_the_record_shape(self, source_id):
        """The declaration is what the fixtures are built against, so it must be complete."""
        spec = ADAPTER_REGISTRY[source_id].spec
        for name in spec.required_fields:
            assert name in spec.record_shape, f"{source_id} requires undocumented {name!r}"

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_no_adapter_claims_to_have_ingested_data(self, source_id):
        """Nothing has been downloaded in this phase, and the specs say so."""
        assert ADAPTER_REGISTRY[source_id].spec.status == "mapping_only"

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_every_adapter_records_its_licensing_position(self, source_id):
        assert ADAPTER_REGISTRY[source_id].spec.license_note

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_every_adapter_states_the_judgements_it_makes(self, source_id):
        """Difficulty and marks are assigned, not read, and that has to be written down."""
        assert ADAPTER_REGISTRY[source_id].spec.assignment_notes

    @pytest.mark.parametrize("source_id", registered_sources())
    def test_each_spec_is_json_serializable(self, source_id):
        payload = json.loads(json.dumps(ADAPTER_REGISTRY[source_id].spec.as_dict()))
        assert payload["source_id"] == source_id

    def test_only_squad_declares_offsets(self):
        """It is the one corpus of the four that can produce grounded examples."""
        with_offsets = {
            source
            for source in registered_sources()
            if ADAPTER_REGISTRY[source].spec.provides_offsets
        }
        assert with_offsets == {"squad-qg"}

    def test_only_lmqg_declares_multiple_targets(self):
        multi = {
            source
            for source in registered_sources()
            if ADAPTER_REGISTRY[source].spec.multi_target
        }
        assert multi == {"lmqg-squad-qag"}

    def test_an_unknown_source_is_refused_with_the_available_list(self):
        with pytest.raises(UnknownAdapterError, match="squad-qg"):
            adapter_for("squad-v2-qg")

    def test_a_custom_registry_is_honoured(self):
        registry = {"only": SquadQuestionGenerationAdapter()}
        assert adapter_for("only", registry=registry).spec.source_id == "squad-qg"
        with pytest.raises(UnknownAdapterError):
            adapter_for("squad-qg", registry=registry)


class TestSquadAdapter:
    """SQuAD read backwards, and the only source of real grounding."""

    def test_a_record_maps_to_a_single_target_example(self):
        result = SquadQuestionGenerationAdapter().adapt(squad_record(1))
        assert result.source == "squad-qg"
        assert len(result.targets) == 1
        assert result.question_type is QuestionType.SHORT_ANSWER
        assert result.answer == "adenosine triphosphate"

    def test_difficulty_and_marks_reflect_that_the_answer_is_a_span(self):
        result = SquadQuestionGenerationAdapter().adapt(squad_record())
        assert result.difficulty is Difficulty.EASY
        assert result.marks == 1

    def test_the_title_becomes_the_topic(self):
        result = SquadQuestionGenerationAdapter().adapt(squad_record(title="Chloroplast"))
        assert result.topic == "Chloroplast"

    def test_the_title_becomes_the_leakage_group_by_default(self):
        """SQuAD paragraphs from one article overlap, so the article is the safe unit."""
        result = SquadQuestionGenerationAdapter().adapt(squad_record(title="Mitochondrion"))
        assert result.group_key == "title:Mitochondrion"

    def test_title_grouping_can_be_disabled(self):
        adapter = SquadQuestionGenerationAdapter(group_by_title=False)
        assert adapter.adapt(squad_record()).group_key is None

    def test_a_verified_offset_produces_traceable_grounding(self):
        result = SquadQuestionGenerationAdapter().adapt(squad_record())
        assert result.is_grounded is True
        assert result.grounding is not None
        assert result.grounding.reference() is not None

    def test_the_grounding_span_reproduces_the_answer(self):
        result = SquadQuestionGenerationAdapter().adapt(squad_record())
        span = result.grounding.span
        assert span is not None
        assert result.context[span.char_start : span.char_end] == result.answer

    def test_a_wrong_offset_yields_no_grounding_rather_than_a_wrong_span(self):
        """A grounding that misleads is worse than none, so it is dropped."""
        record = squad_record()
        record["answers"]["answer_start"] = [3]
        result = SquadQuestionGenerationAdapter().adapt(record)
        assert result.is_grounded is False
        assert result.grounding is None

    def test_a_missing_offset_yields_no_grounding(self):
        record = squad_record()
        record["answers"]["answer_start"] = []
        assert SquadQuestionGenerationAdapter().adapt(record).grounding is None

    def test_ids_are_deterministic(self):
        first = SquadQuestionGenerationAdapter().adapt(squad_record(7))
        second = SquadQuestionGenerationAdapter().adapt(squad_record(7))
        assert first.id == second.id

    def test_ids_are_namespaced_by_source(self):
        assert SquadQuestionGenerationAdapter().adapt(squad_record()).id.startswith("squad-qg-")

    def test_different_records_get_different_ids(self):
        first = SquadQuestionGenerationAdapter().adapt(squad_record(1))
        second = SquadQuestionGenerationAdapter().adapt(squad_record(2))
        assert first.id != second.id

    @pytest.mark.parametrize("missing", ["context", "question", "answers"])
    def test_a_missing_required_field_is_refused(self, missing):
        record = squad_record()
        del record[missing]
        with pytest.raises(AdapterError, match=missing):
            SquadQuestionGenerationAdapter().adapt(record)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_context_is_refused(self, blank):
        record = squad_record()
        record["context"] = blank
        with pytest.raises(AdapterError, match="context"):
            SquadQuestionGenerationAdapter().adapt(record)

    def test_a_malformed_answers_field_is_refused(self):
        record = squad_record()
        record["answers"] = ["adenosine triphosphate"]
        with pytest.raises(AdapterError, match="answers"):
            SquadQuestionGenerationAdapter().adapt(record)

    def test_an_empty_answer_list_is_refused_as_squad_v2(self):
        """Unanswerable questions need a null threshold this project does not model."""
        record = squad_record()
        record["answers"] = {"text": [], "answer_start": []}
        with pytest.raises(AdapterError, match="SQuAD 2.0"):
            SquadQuestionGenerationAdapter().adapt(record)


class TestLmqgAdapter:
    """The multi-target shape."""

    def record(self, **overrides):
        """Build an LMQG QAG record."""
        defaults = {
            "paragraph_id": "p-1",
            "paragraph": MITOCHONDRION,
            "questions": ["What generates ATP?", "What is ATP used for?"],
            "answers": ["the mitochondrion", "chemical energy"],
        }
        return {**defaults, **overrides}

    def test_one_record_becomes_one_multi_target_example(self):
        result = LmqgSquadQagAdapter().adapt(self.record())
        assert len(result.targets) == 2
        assert result.context == MITOCHONDRION

    def test_pairs_keep_their_order(self):
        result = LmqgSquadQagAdapter().adapt(self.record())
        assert result.targets[0].question == "What generates ATP?"
        assert result.targets[1].answer == "chemical energy"

    def test_the_paragraph_is_stored_once(self):
        """The reason targets are a tuple rather than one example per pair."""
        result = LmqgSquadQagAdapter().adapt(self.record())
        assert result.context.count("mitochondrion is a double") == 1

    def test_no_grounding_is_claimed_without_offsets(self):
        """Locating each answer by string search would often find the wrong occurrence."""
        assert LmqgSquadQagAdapter().adapt(self.record()).is_grounded is False

    def test_blank_pairs_are_dropped_and_counted(self):
        result = LmqgSquadQagAdapter().adapt(
            self.record(questions=["Real question?", "  "], answers=["real answer", "x"])
        )
        assert len(result.targets) == 1
        assert result.metadata["pairs_dropped"] == 1

    def test_mismatched_list_lengths_are_refused_not_truncated(self):
        """If the lists are not parallel then every pair is suspect."""
        with pytest.raises(AdapterError, match="parallel"):
            LmqgSquadQagAdapter().adapt(self.record(answers=["only one"]))

    def test_a_record_with_no_usable_pair_is_refused(self):
        with pytest.raises(AdapterError, match="no usable"):
            LmqgSquadQagAdapter().adapt(self.record(questions=["  "], answers=["  "]))

    def test_a_string_instead_of_a_list_is_refused(self):
        with pytest.raises(AdapterError, match="questions"):
            LmqgSquadQagAdapter().adapt(self.record(questions="What generates ATP?"))

    def test_the_pair_count_is_recorded(self):
        assert LmqgSquadQagAdapter().adapt(self.record()).metadata["pair_count"] == 2

    def test_ids_are_deterministic(self):
        assert (
            LmqgSquadQagAdapter().adapt(self.record()).id
            == LmqgSquadQagAdapter().adapt(self.record()).id
        )


class TestLearningQAdapter:
    """Higher-order educational questions, and the strict answer requirement."""

    def record(self, **overrides):
        """Build a LearningQ record."""
        defaults = {
            "doc_id": "khan-1",
            "context": CHLOROPLAST,
            "question": "Explain how chloroplasts store light energy.",
            "answer": "They capture light and store it in the chemical bonds of glucose.",
            "topic": "Photosynthesis",
        }
        return {**defaults, **overrides}

    def test_a_record_maps_to_a_long_answer_target(self):
        result = LearningQAdapter().adapt(self.record())
        assert result.question_type is QuestionType.LONG_ANSWER
        assert result.marks == 5
        assert result.difficulty is Difficulty.MEDIUM

    def test_the_topic_is_carried(self):
        assert LearningQAdapter().adapt(self.record()).topic == "Photosynthesis"

    def test_the_document_becomes_the_leakage_group(self):
        assert LearningQAdapter().adapt(self.record()).group_key == "doc:khan-1"

    @pytest.mark.parametrize("answer", [None, "", "   ", 42])
    def test_a_record_without_a_written_answer_is_refused(self, answer):
        """Synthesising one would train the model on invented content."""
        with pytest.raises(AdapterError, match="no written answer"):
            LearningQAdapter().adapt(self.record(answer=answer))

    def test_the_refusal_explains_the_upstream_remedy(self):
        with pytest.raises(AdapterError, match="Filter these records out"):
            LearningQAdapter().adapt(self.record(answer=None))

    def test_the_spec_warns_that_many_records_will_be_dropped(self):
        notes = " ".join(LearningQAdapter().spec.assignment_notes)
        assert "large share" in notes

    def test_the_question_type_is_configurable(self):
        adapter = LearningQAdapter(question_type=QuestionType.SHORT_ANSWER, marks=2)
        result = adapter.adapt(self.record())
        assert result.question_type is QuestionType.SHORT_ANSWER
        assert result.marks == 2


class TestEducationalMcqAdapter:
    """Generic MCQ ingestion."""

    def test_a_record_maps_to_an_mcq_target(self):
        result = EducationalMcqAdapter().adapt(mcq_record())
        assert result.question_type is QuestionType.MCQ
        assert result.options == ("Nucleus", "Mitochondrion", "Ribosome", "Vacuole")
        assert result.primary_target.correct_option_index == 1

    def test_the_answer_is_the_correct_option_text(self):
        """Stored once and derived, so the two cannot disagree."""
        result = EducationalMcqAdapter().adapt(mcq_record())
        assert result.answer == "Mitochondrion"
        assert result.primary_target.correct_option == result.answer

    def test_the_correct_option_can_be_given_as_text(self):
        record = mcq_record()
        del record["correct_index"]
        record["answer"] = "Mitochondrion"
        assert EducationalMcqAdapter().adapt(record).primary_target.correct_option_index == 1

    def test_answer_text_matching_is_case_insensitive(self):
        record = mcq_record()
        del record["correct_index"]
        record["answer"] = "  mitochondrion  "
        assert EducationalMcqAdapter().adapt(record).answer == "Mitochondrion"

    def test_an_index_wins_over_disagreeing_answer_text(self):
        record = mcq_record(correct_index=2, answer="Mitochondrion")
        assert EducationalMcqAdapter().adapt(record).answer == "Ribosome"

    def test_too_few_options_are_refused(self):
        """A two-option item is a true/false question wearing a disguise."""
        with pytest.raises(AdapterError, match="at least 3"):
            EducationalMcqAdapter().adapt(mcq_record(options=["Yes", "No"], correct_index=0))

    def test_the_minimum_matches_the_paper_validator(self):
        from qa_paper import MINIMUM_MCQ_OPTIONS

        assert EducationalMcqAdapter().minimum_options == MINIMUM_MCQ_OPTIONS

    def test_a_blank_option_is_refused(self):
        with pytest.raises(AdapterError, match="blank option"):
            EducationalMcqAdapter().adapt(
                mcq_record(options=["A", "  ", "C"], correct_index=0)
            )

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(AdapterError, match="does not address"):
            EducationalMcqAdapter().adapt(mcq_record(correct_index=9))

    def test_a_non_integer_index_is_refused(self):
        with pytest.raises(AdapterError, match="must be an integer"):
            EducationalMcqAdapter().adapt(mcq_record(correct_index="second"))

    def test_no_index_and_no_answer_is_refused(self):
        """Defaulting to the first option would mislabel every such record."""
        record = mcq_record()
        del record["correct_index"]
        with pytest.raises(AdapterError, match="neither correct_index nor answer"):
            EducationalMcqAdapter().adapt(record)

    def test_answer_text_matching_no_option_is_refused(self):
        record = mcq_record()
        del record["correct_index"]
        record["answer"] = "Golgi apparatus"
        with pytest.raises(AdapterError, match="does not match any option"):
            EducationalMcqAdapter().adapt(record)

    def test_ambiguous_answer_text_is_refused(self):
        record = mcq_record(options=["ATP", "ATP", "GTP"])
        del record["correct_index"]
        record["answer"] = "ATP"
        with pytest.raises(AdapterError, match="matches 2 options"):
            EducationalMcqAdapter().adapt(record)

    def test_a_field_map_absorbs_different_column_names(self):
        adapter = EducationalMcqAdapter(
            field_map={
                "context": "passage",
                "question": "stem",
                "options": "choices",
                "correct_index": "label",
                "topic": "subject",
                "id": "uid",
            }
        )
        record = {
            "uid": "r-1",
            "passage": MITOCHONDRION,
            "stem": "Which organelle produces ATP?",
            "choices": ["Nucleus", "Mitochondrion", "Ribosome"],
            "label": 1,
            "subject": "Biology",
        }
        result = adapter.adapt(record)
        assert result.answer == "Mitochondrion"
        assert result.topic == "Biology"

    def test_the_default_style_reads_the_answer_as_text(self):
        """The pre-existing behaviour, pinned so adding a style did not change it."""
        assert EducationalMcqAdapter().answer_style == "text"

    def test_an_unknown_style_is_refused_at_construction(self):
        """A typo should surface on registration, not part-way through a corpus."""
        with pytest.raises(AdapterError, match="answer_style must be one of"):
            EducationalMcqAdapter(answer_style="letters")


class TestMcqLetterAnswers:
    """Reading the answer field as an option label rather than as option text."""

    def adapter(self, **overrides):
        """An MCQ adapter that reads the answer field as an option label."""
        return EducationalMcqAdapter(answer_style="letter", **overrides)

    def record(self, **overrides):
        """An MCQ record with ``correct_index`` removed, so the answer field decides."""
        base = mcq_record(**overrides)
        base.pop("correct_index", None)
        return base

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("A", 0), ("B", 1), ("C", 2), ("D", 3)],
    )
    def test_each_label_maps_to_its_ordinal(self, answer, expected):
        result = self.adapter().adapt(self.record(answer=answer))
        assert result.primary_target.correct_option_index == expected

    @pytest.mark.parametrize("answer", ["b", "B", " b ", "\tB\n"])
    def test_case_and_surrounding_space_are_normalized(self, answer):
        """Only case and outer whitespace. Anything else is refused, not repaired."""
        assert self.adapter().adapt(self.record(answer=answer)).answer == "Mitochondrion"

    def test_the_answer_text_is_derived_from_the_label(self):
        result = self.adapter().adapt(self.record(answer="C"))
        assert result.answer == "Ribosome"
        assert result.primary_target.correct_option == "Ribosome"

    def test_a_label_beyond_the_options_is_refused(self):
        """'E' against four options means the record or the schema is wrong."""
        with pytest.raises(AdapterError, match="names option 5, but the record has only 4"):
            self.adapter().adapt(self.record(answer="E"))

    @pytest.mark.parametrize("answer", ["(A)", "A.", "A)", "AB", "1", "first", "Mitochondrion"])
    def test_anything_that_is_not_a_bare_label_is_refused(self, answer):
        """Guessing here would silently relabel every record if the schema changed."""
        with pytest.raises(AdapterError, match="must be a single option label"):
            self.adapter().adapt(self.record(answer=answer))

    def test_the_refusal_names_the_other_style_as_the_remedy(self):
        with pytest.raises(AdapterError, match="answer_style='text'"):
            self.adapter().adapt(self.record(answer="Mitochondrion"))

    def test_an_explicit_index_still_wins_over_a_label(self):
        """Documented precedence, unchanged by the new style."""
        result = self.adapter().adapt(mcq_record(correct_index=2, answer="A"))
        assert result.answer == "Ribosome"

    def test_a_missing_answer_is_still_refused(self):
        """``self.record()`` drops ``correct_index``, so this record carries neither."""
        record = self.record()
        assert "answer" not in record and "correct_index" not in record
        with pytest.raises(AdapterError, match="neither correct_index nor answer text"):
            self.adapter().adapt(record)

    def test_a_label_is_not_matched_against_option_text(self):
        """The two styles must not blend: under 'letter', 'A' is an ordinal, never a match.

        These options are literally the letters A-D, so a text match would also succeed and
        would happen to agree. It agrees here and would disagree on a shuffled corpus, so the
        style has to decide rather than whichever check ran first.
        """
        result = self.adapter().adapt(
            self.record(options=["D", "C", "B", "A"], answer="A")
        )
        assert result.primary_target.correct_option_index == 0
        assert result.answer == "D"

    def test_the_spec_states_which_style_is_in_use(self):
        notes = " ".join(self.adapter().spec.assignment_notes)
        assert "'letter'" in notes
        assert "'A' is the first option" in notes


class TestRaceMcqSource:
    """The registered ``race-mcq`` adapter, against real ``ehovy/race`` record shapes."""

    def adapt(self, **overrides):
        """Map a RACE-shaped record through the registered ``race-mcq`` adapter."""
        return adapter_for("race-mcq").adapt(race_record(**overrides))

    def test_a_race_row_maps_without_any_renaming(self):
        """The whole point of the registration: real rows go in unmodified."""
        result = self.adapt()
        assert result.question_type is QuestionType.MCQ
        assert result.context.startswith("The mitochondrion")
        assert result.options == ("Nucleus", "Mitochondrion", "Ribosome", "Vacuole")
        assert result.answer == "Mitochondrion"
        assert result.primary_target.correct_option_index == 1

    def test_the_example_reports_its_own_source(self):
        """Not 'edu-mcq': the per-source cap and the sizing report group by this."""
        assert self.adapt().source == "race-mcq"
        assert adapter_for("race-mcq").spec.source_id == "race-mcq"
        assert adapter_for("race-mcq").spec.dataset_id == "ehovy/race"

    def test_questions_sharing_an_article_keep_distinct_ids(self):
        """``example_id`` is the article filename and repeats; ids must not.

        Measured on 1,000 real rows: about 3.3 questions share each ``example_id``, so an id
        derived from it would collide and deduplication would discard two thirds of the
        corpus. This is the regression test for that.
        """
        ids = {
            self.adapt(question=f"Question number {index}?").id
            for index in range(5)
        }
        assert len(ids) == 5

    def test_questions_sharing_an_article_share_a_leakage_group(self):
        """Distinct ids, one group: both are required for a context-grouped split."""
        groups = {
            self.adapt(question=f"Question number {index}?").effective_group_key()
            for index in range(5)
        }
        assert len(groups) == 1
        assert next(iter(groups)).startswith("ctx:")

    def test_a_different_article_lands_in_a_different_group(self):
        assert self.adapt().effective_group_key() != self.adapt(
            article=CHLOROPLAST
        ).effective_group_key()

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("A", "Nucleus"), ("B", "Mitochondrion"), ("C", "Ribosome"), ("D", "Vacuole")],
    )
    def test_every_label_the_corpus_uses_resolves(self, answer, expected):
        """A-D are the only values observed across 1,000 sampled rows."""
        assert self.adapt(answer=answer).answer == expected

    def test_the_four_options_clear_the_paper_validator_minimum(self):
        from qa_paper import MINIMUM_MCQ_OPTIONS

        assert len(self.adapt().options) >= MINIMUM_MCQ_OPTIONS

    def test_the_target_round_trips_into_the_paper_payload(self):
        """RACE needs no change to the canonical target or the paper representation."""
        payload = self.adapt().primary_target.to_payload()
        assert payload.options == ("Nucleus", "Mitochondrion", "Ribosome", "Vacuole")
        assert payload.correct_index == 1

    def test_the_spec_records_the_non_commercial_terms(self):
        note = adapter_for("race-mcq").spec.license_note
        assert "non-commercial" in note.lower()

    def test_the_default_edu_mcq_registration_is_untouched(self):
        """Adding RACE must not have changed the corpus that shares its class."""
        edu = adapter_for("edu-mcq")
        assert edu.spec.source_id == "edu-mcq"
        assert edu.answer_style == "text"
        assert edu.adapt(mcq_record()).source == "edu-mcq"


class TestBatchAdaptation:
    """Converting many records."""

    def test_records_are_converted_in_order(self):
        results = list(adapt_records("squad-qg", [squad_record(0), squad_record(1)]))
        assert [r.metadata["record_id"] for r in results] == ["sq-0", "sq-1"]

    def test_a_bad_record_fails_loudly_by_default(self):
        """A renamed column should fail on the first record, not yield an empty corpus."""
        bad = squad_record()
        del bad["question"]
        with pytest.raises(AdapterError):
            list(adapt_records("squad-qg", [squad_record(0), bad]))

    def test_bad_records_can_be_skipped_deliberately(self):
        bad = squad_record()
        del bad["question"]
        results = list(
            adapt_records("squad-qg", [squad_record(0), bad], skip_invalid=True)
        )
        assert len(results) == 1

    def test_an_empty_record_list_yields_nothing(self):
        assert list(adapt_records("squad-qg", [])) == []


class TestNoDownloads:
    """Nothing in this package fetches data or a model."""

    def test_the_adapters_module_does_not_reference_a_dataset_library(self):
        """Checked against parsed identifiers, so a docstring cannot trigger it."""
        from test_qa_gen_isolation import code_identifiers

        referenced = code_identifiers("adapters")
        forbidden = {"datasets", "load_dataset", "hf_hub_download", "requests", "urllib"}
        assert not (referenced & forbidden), sorted(referenced & forbidden)

    def test_no_qa_gen_module_opens_a_url(self):
        import pkgutil

        from test_qa_gen_isolation import code_identifiers

        import qa_gen

        forbidden = {"urlopen", "socket", "httpx", "requests", "urllib"}
        for info in pkgutil.iter_modules(qa_gen.__path__):
            leaked = code_identifiers(info.name) & forbidden
            assert not leaked, f"qa_gen.{info.name} references {sorted(leaked)}"

    def test_adapting_a_record_touches_no_filesystem_path(self, tmp_path, monkeypatch):
        """A cheap guard: adapting must not create anything on disk."""
        monkeypatch.chdir(tmp_path)
        SquadQuestionGenerationAdapter().adapt(squad_record())
        assert list(tmp_path.iterdir()) == []


class TestDeterministicSplitting:
    """Partitioning is a pure function of seed and content."""

    def corpus(self, count: int = 30, *, titles: int = 6):
        """Build a corpus with several examples per article title."""
        return [
            example(
                index,
                group_key=f"title:Article-{index % titles}",
                context=f"{MITOCHONDRION} Passage {index}.",
            )
            for index in range(count)
        ]

    def test_every_example_lands_in_exactly_one_split(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        assert len(splits) == 30
        ids = [e.id for e in splits.train] + [e.id for e in splits.validation] + [
            e.id for e in splits.test
        ]
        assert len(set(ids)) == 30

    def test_the_same_seed_reproduces_the_partition(self):
        first = DeterministicGroupSplitter().split(self.corpus(), seed=7)
        second = DeterministicGroupSplitter().split(self.corpus(), seed=7)
        assert [e.id for e in first.train] == [e.id for e in second.train]
        assert [e.id for e in first.test] == [e.id for e in second.test]

    def test_a_different_seed_changes_the_partition(self):
        ratios = SplitRatios(0.6, 0.2, 0.2)
        first = DeterministicGroupSplitter().split(self.corpus(), seed=1, ratios=ratios)
        second = DeterministicGroupSplitter().split(self.corpus(), seed=2, ratios=ratios)
        assert [e.id for e in first.test] != [e.id for e in second.test]

    def test_small_splits_are_not_starved_by_a_large_train_share(self):
        """Absolute shortfall would fill train first and leave validation and test empty."""
        splits = DeterministicGroupSplitter().split(
            self.corpus(30, titles=6), seed=42, ratios=SplitRatios(0.9, 0.05, 0.05)
        )
        assert splits.sizes["validation"] > 0
        assert splits.sizes["test"] > 0

    def test_a_zero_ratio_split_stays_empty(self):
        splits = DeterministicGroupSplitter().split(
            self.corpus(20, titles=10), seed=42, ratios=SplitRatios(0.8, 0.2, 0.0)
        )
        assert splits.sizes["test"] == 0
        assert splits.sizes["validation"] > 0

    def test_few_large_groups_cannot_honour_fine_ratios(self):
        """A real limitation of group splitting, which distinct_group_keys diagnoses."""
        corpus = [
            example(index, group_key=f"title:Article-{index % 2}") for index in range(100)
        ]
        splits = DeterministicGroupSplitter().split(
            corpus, seed=42, ratios=SplitRatios(0.9, 0.05, 0.05), group_by="topic"
        )
        assert len(splits.assignments) == 2
        assert sorted(splits.sizes.values()) == [0, 50, 50]
        assert compute_statistics(corpus).distinct_group_keys == 2

    def test_the_partition_does_not_depend_on_input_order(self):
        """A seeded shuffle would fail this; hashing the group key does not."""
        corpus = self.corpus()
        forward = DeterministicGroupSplitter().split(corpus, seed=42)
        backward = DeterministicGroupSplitter().split(list(reversed(corpus)), seed=42)
        assert [e.id for e in forward.train] == [e.id for e in backward.train]
        assert [e.id for e in forward.validation] == [e.id for e in backward.validation]
        assert [e.id for e in forward.test] == [e.id for e in backward.test]

    def test_splitting_uses_no_global_random_state(self):
        """Seeding the interpreter's RNG would change unrelated code that runs after."""
        import random

        random.seed(1234)
        expected = random.random()
        random.seed(1234)
        DeterministicGroupSplitter().split(self.corpus(), seed=99)
        assert random.random() == expected

    def test_no_random_module_is_imported_by_the_splitter(self):
        """Checked against the parsed imports, not the text.

        A substring search would trip over the module docstring, which mentions
        ``random.seed()`` precisely to explain why it is not used. The AST sees only real
        import statements.
        """
        import ast
        import inspect

        from qa_gen import splitting

        tree = ast.parse(inspect.getsource(splitting))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert "random" not in imported
        assert "numpy" not in imported
        assert imported <= {"__future__", "hashlib", "collections", "dataclasses", "enum",
                            "typing", "qa_gen"}, f"unexpected imports: {sorted(imported)}"

    def test_sizes_approximate_the_requested_ratios(self):
        splits = DeterministicGroupSplitter().split(
            self.corpus(60, titles=20), seed=42, ratios=SplitRatios(0.6, 0.2, 0.2)
        )
        assert splits.sizes["train"] >= splits.sizes["validation"]
        assert sum(splits.sizes.values()) == 60

    def test_target_counts_sum_exactly_to_the_total(self):
        """Rounding three floats independently loses or invents an example."""
        for total in (1, 7, 13, 99, 100, 1001):
            counts = SplitRatios(0.9, 0.05, 0.05).target_counts(total)
            assert sum(counts.values()) == total

    def test_an_empty_corpus_yields_three_empty_splits(self):
        splits = DeterministicGroupSplitter().split([])
        assert len(splits) == 0
        assert splits.sizes == {"train": 0, "validation": 0, "test": 0}

    def test_a_single_example_goes_to_train(self):
        splits = DeterministicGroupSplitter().split([example(0)])
        assert len(splits.train) == 1

    def test_invalid_ratios_are_refused(self):
        with pytest.raises(SplitError, match="sum to 1.0"):
            DeterministicGroupSplitter().split(
                self.corpus(), ratios=SplitRatios(0.5, 0.2, 0.1)
            )

    def test_an_unknown_grouping_strategy_is_refused(self):
        with pytest.raises(SplitError, match="group_by"):
            DeterministicGroupSplitter().split(self.corpus(), group_by="paragraph")

    def test_duplicate_ids_are_refused(self):
        """An ambiguous corpus cannot be partitioned traceably."""
        corpus = [example(0), example(0)]
        with pytest.raises(SplitError, match="duplicate example id"):
            DeterministicGroupSplitter().split(corpus)

    def test_a_split_can_be_indexed_by_name_or_enum(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        assert splits["train"] == splits[SplitName.TRAIN]

    def test_an_unknown_split_name_is_refused(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        with pytest.raises(SplitError, match="unknown split"):
            splits["dev"]

    def test_the_splitter_records_its_version(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        assert splits.splitter == "deterministic-group-splitter-v1"

    def test_the_summary_is_json_serializable_and_omits_examples(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        payload = json.loads(json.dumps(splits.as_dict()))
        assert payload["total"] == 30
        assert "train" not in payload

    def test_examples_are_ordered_stably_within_a_split(self):
        splits = DeterministicGroupSplitter().split(self.corpus())
        assert [e.id for e in splits.train] == sorted(e.id for e in splits.train)


class TestLeakagePrevention:
    """The property that makes a test score mean anything."""

    def test_a_shared_context_never_spans_two_splits(self):
        """The same paragraph in train and test measures memorisation, and looks like skill."""
        corpus = [
            example(index, context=f"{MITOCHONDRION} Shared passage {index % 5}.")
            for index in range(40)
        ]
        splits = DeterministicGroupSplitter().split(corpus, seed=42, group_by="context")
        assert splits.leaked_group_keys() == frozenset()

    def test_a_shared_title_never_spans_two_splits_under_topic_grouping(self):
        corpus = [
            example(index, group_key=f"title:Article-{index % 4}") for index in range(40)
        ]
        splits = DeterministicGroupSplitter().split(corpus, seed=42, group_by="topic")
        assert splits.leaked_group_keys() == frozenset()

    def test_grouping_by_example_permits_what_grouping_prevents(self):
        """Offered only so the size of the effect can be measured."""
        corpus = [
            example(index, context=f"{MITOCHONDRION} One passage.") for index in range(20)
        ]
        grouped = DeterministicGroupSplitter().split(corpus, seed=42, group_by="context")
        ungrouped = DeterministicGroupSplitter().split(corpus, seed=42, group_by="example")
        assert grouped.sizes["train"] == 20
        assert ungrouped.sizes["train"] < 20

    def test_whitespace_variants_of_a_context_group_together(self):
        corpus = [
            example(0, context="The cell divides by mitosis."),
            example(1, context="the  cell divides by mitosis"),
        ]
        splits = DeterministicGroupSplitter().split(corpus, seed=1, group_by="context")
        assert len(splits.assignments) == 1

    def test_topic_grouping_falls_back_to_context_for_unlabelled_examples(self):
        """Otherwise every unlabelled example lands in one enormous group."""
        corpus = [example(index, group_key=None) for index in range(10)]
        splits = DeterministicGroupSplitter().split(corpus, seed=1, group_by="topic")
        assert len(splits.assignments) == 10


class TestSplitAuditTrail:
    """A partition has to be explainable after the fact."""

    def test_every_group_is_recorded(self):
        corpus = [example(index, group_key=f"title:A-{index % 5}") for index in range(20)]
        splits = DeterministicGroupSplitter().split(corpus, seed=3, group_by="topic")
        assert len(splits.assignments) == 5
        assert sum(a.example_count for a in splits.assignments) == 20

    def test_an_assignment_names_its_examples_and_digest(self):
        splits = DeterministicGroupSplitter().split([example(0)], seed=3)
        assignment = splits.assignments[0]
        assert assignment.example_ids == ("ex-0",)
        assert assignment.digest
        assert assignment.sources == ("squad-qg",)

    def test_assignments_are_json_serializable(self):
        splits = DeterministicGroupSplitter().split([example(0)])
        assert json.loads(json.dumps(splits.assignments[0].as_dict()))["split"] == "train"

    def test_the_source_breakdown_covers_every_split(self):
        corpus = [
            *[example(i, source="squad-qg", group_key=f"g{i}") for i in range(10)],
            *[example(i + 100, source="edu-mcq", group_key=f"h{i}") for i in range(10)],
        ]
        splits = DeterministicGroupSplitter().split(
            corpus, seed=5, ratios=SplitRatios(0.5, 0.25, 0.25)
        )
        breakdown = splits.source_breakdown()
        assert set(breakdown) == {"train", "validation", "test"}
        assert sum(sum(counts.values()) for counts in breakdown.values()) == 20


class TestDatasetFingerprint:
    """Corpus identity by content."""

    def test_the_same_corpus_hashes_the_same(self):
        corpus = [example(i) for i in range(5)]
        assert compute_dataset_fingerprint(corpus) == compute_dataset_fingerprint(corpus)

    def test_the_fingerprint_ignores_order(self):
        corpus = [example(i) for i in range(5)]
        assert compute_dataset_fingerprint(corpus) == compute_dataset_fingerprint(
            list(reversed(corpus))
        )

    def test_a_changed_corpus_changes_the_fingerprint(self):
        corpus = [example(i) for i in range(5)]
        assert compute_dataset_fingerprint(corpus) != compute_dataset_fingerprint(
            [*corpus, example(99)]
        )

    def test_an_empty_corpus_hashes_consistently(self):
        assert compute_dataset_fingerprint([]) == compute_dataset_fingerprint([])

    def test_the_partition_records_the_fingerprint(self):
        corpus = [example(i) for i in range(5)]
        splits = DeterministicGroupSplitter().split(corpus)
        assert splits.dataset_fingerprint == compute_dataset_fingerprint(corpus)


class TestDuplicateDetection:
    """Duplicates are reported, not silently removed."""

    def test_identical_examples_are_detected(self):
        duplicates = find_duplicate_examples([example(0), example(0, id="ex-copy")])
        assert len(duplicates) == 1
        assert list(duplicates.values())[0] == ("ex-0", "ex-copy")

    def test_distinct_examples_are_not_flagged(self):
        assert find_duplicate_examples([example(0), example(1)]) == {}

    def test_examples_differing_only_in_whitespace_are_detected(self):
        first = example(0, context="The cell divides.", targets=(target(),))
        second = example(1, context="the  cell  divides", targets=(target(),))
        assert find_duplicate_examples([first, second])

    def test_numeric_questions_are_not_falsely_collapsed(self):
        """Reuses qa_paper.fingerprint, so 2 + 2 and 2 - 2 remain distinct examples."""
        plus = example(0, targets=(target(question="What is 2 + 2?", answer="4"),))
        minus = example(1, targets=(target(question="What is 2 - 2?", answer="0"),))
        assert find_duplicate_examples([plus, minus]) == {}

    def test_the_report_is_deterministically_ordered(self):
        corpus = [example(0), example(0, id="b"), example(1), example(1, id="d")]
        first = find_duplicate_examples(corpus)
        second = find_duplicate_examples(list(reversed(corpus)))
        assert list(first) == list(second)

    def test_duplicates_are_reported_by_dataset_validation(self):
        report = validate_dataset([example(0), example(0, id="ex-copy")])
        assert report.has(DatasetIssueCode.DUPLICATE_EXAMPLE_CONTENT)

    def test_duplicate_content_is_a_warning_not_an_error(self):
        """Two corpora legitimately overlapping is different from one repeating itself."""
        report = validate_dataset([example(0), example(0, id="ex-copy")])
        assert report.ok is True

    def test_duplicate_ids_are_an_error(self):
        report = validate_dataset([example(0), example(0)])
        assert report.has(DatasetIssueCode.DUPLICATE_EXAMPLE_ID)
        assert not report.ok


class TestValidation:
    """Every required check, and the report that carries them."""

    def test_a_valid_corpus_passes_cleanly(self):
        report = validate_dataset([example(0), example(1)])
        assert report.ok, report.as_dict()
        assert report.examples_checked == 2

    def test_the_report_is_truthy_when_ok(self):
        assert bool(validate_dataset([example(0)])) is True

    @pytest.mark.parametrize("context", ["", "   "])
    def test_missing_context_is_reported(self, context):
        report = validate_example(example(0, context=context))
        assert report.has(DatasetIssueCode.MISSING_CONTEXT)

    @pytest.mark.parametrize("question", ["", "   "])
    def test_missing_question_is_reported(self, question):
        report = validate_example(example(0, targets=(target(question=question),)))
        assert report.has(DatasetIssueCode.MISSING_QUESTION)

    @pytest.mark.parametrize("answer", ["", "   "])
    def test_missing_answer_is_reported(self, answer):
        report = validate_example(example(0, targets=(target(answer=answer),)))
        assert report.has(DatasetIssueCode.MISSING_ANSWER)

    def test_an_invalid_question_type_is_reported(self):
        report = validate_example(example(0, targets=(target(question_type="essay"),)))
        assert report.has(DatasetIssueCode.INVALID_QUESTION_TYPE)

    def test_an_invalid_difficulty_is_reported(self):
        report = validate_example(example(0, targets=(target(difficulty="mixed"),)))
        assert report.has(DatasetIssueCode.INVALID_DIFFICULTY)

    @pytest.mark.parametrize("marks", [0, -3])
    def test_non_positive_marks_are_reported(self, marks):
        report = validate_example(example(0, targets=(target(marks=marks),)))
        assert report.has(DatasetIssueCode.INVALID_MARKS)

    @pytest.mark.parametrize("marks", ["2", 1.5, None])
    def test_non_integer_marks_are_reported(self, marks):
        report = validate_example(example(0, targets=(target(marks=marks),)))
        assert report.has(DatasetIssueCode.INVALID_MARKS)

    def test_boolean_marks_are_reported(self):
        """``bool`` is an int subclass, so True would otherwise pass as one mark."""
        report = validate_example(example(0, targets=(target(marks=True),)))
        assert report.has(DatasetIssueCode.INVALID_MARKS)

    def test_too_few_mcq_options_are_reported(self):
        item = target(
            question_type=QuestionType.MCQ,
            options=("A", "B"),
            correct_option_index=0,
            answer="A",
        )
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.INVALID_MCQ_OPTION_COUNT)

    def test_a_blank_mcq_option_is_reported(self):
        item = target(
            question_type=QuestionType.MCQ,
            options=("A", "  ", "C"),
            correct_option_index=0,
            answer="A",
        )
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.INVALID_MCQ_OPTION_COUNT)

    def test_a_repeated_mcq_option_is_reported(self):
        item = target(
            question_type=QuestionType.MCQ,
            options=("A", "a", "C"),
            correct_option_index=0,
            answer="A",
        )
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.INVALID_MCQ_OPTION_COUNT)

    def test_an_out_of_range_mcq_index_is_reported(self):
        item = target(
            question_type=QuestionType.MCQ,
            options=("A", "B", "C"),
            correct_option_index=9,
            answer="A",
        )
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.MCQ_INDEX_OUT_OF_RANGE)

    def test_a_missing_mcq_index_is_reported(self):
        item = target(question_type=QuestionType.MCQ, options=("A", "B", "C"), answer="A")
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.MCQ_INDEX_OUT_OF_RANGE)

    def test_an_mcq_answer_disagreeing_with_its_index_is_reported(self):
        item = target(
            question_type=QuestionType.MCQ,
            options=("A", "B", "C"),
            correct_option_index=0,
            answer="C",
        )
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.MCQ_ANSWER_MISMATCH)

    def test_options_on_a_non_mcq_are_reported(self):
        item = target(question_type=QuestionType.SHORT_ANSWER, options=("A", "B", "C"))
        report = validate_example(example(0, targets=(item,)))
        assert report.has(DatasetIssueCode.OPTIONS_ON_NON_MCQ)

    def test_empty_targets_are_reported(self):
        report = validate_example(example(0, targets=()))
        assert report.has(DatasetIssueCode.EMPTY_TARGETS)

    def test_an_empty_corpus_is_reported(self):
        """"The filters removed everything" is otherwise a quiet way to train on nothing."""
        report = validate_dataset([])
        assert report.has(DatasetIssueCode.EMPTY_DATASET)

    def test_an_empty_corpus_can_be_permitted_explicitly(self):
        assert validate_dataset([], allow_empty=True).ok

    def test_the_issue_names_the_offending_target_index(self):
        item = example(0, targets=(target(), target(answer="")))
        report = validate_example(item)
        issue = next(i for i in report.issues if i.code is DatasetIssueCode.MISSING_ANSWER)
        assert issue.target_index == 1

    def test_the_issue_records_the_source(self):
        report = validate_example(example(0, source="edu-mcq", targets=(target(answer=""),)))
        assert report.errors[0].source == "edu-mcq"

    def test_invalid_example_ids_are_collectable(self):
        """The practical output: filter these out and train on the rest."""
        report = validate_dataset([example(0), example(1, targets=(target(answer=""),))])
        assert report.invalid_example_ids == frozenset({"ex-1"})

    def test_code_counts_are_reported(self):
        report = validate_dataset(
            [example(0, targets=(target(answer=""),)), example(1, targets=(target(answer=""),))]
        )
        assert report.code_counts()["missing_answer"] == 2

    def test_the_report_is_json_serializable(self):
        report = validate_dataset([example(0, context="")])
        assert json.loads(json.dumps(report.as_dict()))["error_count"] >= 1

    def test_raise_if_invalid_raises_on_errors(self):
        from qa_gen import DatasetValidationError

        with pytest.raises(DatasetValidationError, match="validation error"):
            validate_dataset([example(0, context="")]).raise_if_invalid()

    def test_raise_if_invalid_is_silent_when_clean(self):
        validate_dataset([example(0)]).raise_if_invalid()

    def test_raise_if_invalid_truncates_a_huge_error_list(self):
        from qa_gen import DatasetValidationError

        corpus = [example(index, context="") for index in range(30)]
        with pytest.raises(DatasetValidationError, match="and 10 more"):
            validate_dataset(corpus).raise_if_invalid()

    def test_reports_can_be_merged(self):
        merged = validate_dataset([example(0)]).merged_with(validate_dataset([example(1)]))
        assert merged.examples_checked == 2

    def test_every_declared_code_is_a_distinct_value(self):
        values = [member.value for member in DatasetIssueCode]
        assert len(set(values)) == len(values)


class TestConfigDrivenValidation:
    """Checks that only apply when a dataset configuration is supplied."""

    def test_a_short_context_is_reported(self):
        config = QuestionGenerationDatasetConfig(min_context_chars=500)
        report = validate_example(example(0), config=config)
        assert report.has(DatasetIssueCode.CONTEXT_TOO_SHORT)

    def test_a_long_context_is_reported(self):
        config = QuestionGenerationDatasetConfig(min_context_chars=1, max_context_chars=10)
        report = validate_example(example(0), config=config)
        assert report.has(DatasetIssueCode.CONTEXT_TOO_LONG)

    def test_a_disallowed_question_type_is_a_warning(self):
        config = QuestionGenerationDatasetConfig(allowed_question_types=("mcq",))
        report = validate_example(example(0), config=config)
        assert report.has(DatasetIssueCode.QUESTION_TYPE_NOT_ALLOWED)
        assert report.ok is True

    def test_a_disallowed_difficulty_is_a_warning(self):
        config = QuestionGenerationDatasetConfig(allowed_difficulties=("hard",))
        report = validate_example(example(0), config=config)
        assert report.has(DatasetIssueCode.DIFFICULTY_NOT_ALLOWED)

    def test_an_unconfigured_source_is_a_warning(self):
        config = QuestionGenerationDatasetConfig(sources=("edu-mcq",))
        report = validate_example(example(0, source="squad-qg"), config=config)
        assert report.has(DatasetIssueCode.UNKNOWN_SOURCE)

    def test_required_grounding_is_enforced_when_configured(self):
        config = QuestionGenerationDatasetConfig(require_grounding=True)
        report = validate_example(example(0), config=config)
        assert report.has(DatasetIssueCode.UNGROUNDED_EXAMPLE)
        assert not report.ok

    def test_grounding_is_not_required_by_default(self):
        """Most corpora carry no offsets, so requiring it would discard nearly everything."""
        assert QuestionGenerationDatasetConfig().require_grounding is False
        assert validate_example(example(0)).ok

    def test_a_grounded_example_satisfies_the_requirement(self):
        config = QuestionGenerationDatasetConfig(require_grounding=True)
        grounded = SquadQuestionGenerationAdapter().adapt(squad_record())
        assert validate_example(grounded, config=config).ok

    def test_an_answer_absent_from_the_context_is_an_optional_warning(self):
        item = example(0, targets=(target(answer="quantum chromodynamics"),))
        assert validate_example(item).ok
        report = validate_example(item, check_answer_in_context=True)
        assert report.has(DatasetIssueCode.ANSWER_NOT_IN_CONTEXT)
        assert report.ok is True

    def test_the_answer_check_ignores_long_answer_targets(self):
        """A descriptive answer is meant to be a synthesis, not a quotation."""
        item = example(
            0,
            targets=(
                target(
                    question_type=QuestionType.LONG_ANSWER,
                    answer="A synthesis that quotes nothing verbatim.",
                ),
            ),
        )
        report = validate_example(item, check_answer_in_context=True)
        assert not report.has(DatasetIssueCode.ANSWER_NOT_IN_CONTEXT)

    def test_the_answer_check_is_case_insensitive(self):
        item = example(0, targets=(target(answer="Adenosine Triphosphate"),))
        report = validate_example(item, check_answer_in_context=True)
        assert not report.has(DatasetIssueCode.ANSWER_NOT_IN_CONTEXT)


class TestStatistics:
    """Corpus composition, deterministically."""

    def corpus(self):
        """A mixed corpus: eight short-answer examples and two MCQs."""
        squad = [
            SquadQuestionGenerationAdapter().adapt(
                squad_record(index, title="Mitochondrion" if index < 4 else "Chloroplast")
            )
            for index in range(8)
        ]
        mcq = [EducationalMcqAdapter().adapt(mcq_record(index)) for index in range(2)]
        return [*squad, *mcq]

    def test_totals_count_examples_and_targets_separately(self):
        stats = compute_statistics(self.corpus())
        assert stats.total_examples == 10
        assert stats.total_targets == 10

    def test_a_multi_target_example_counts_once_but_supplies_many_targets(self):
        item = example(0, targets=(target(), target(question="Second?"), target(question="Third?")))
        stats = compute_statistics([item])
        assert stats.total_examples == 1
        assert stats.total_targets == 3

    def test_examples_are_counted_by_source(self):
        assert compute_statistics(self.corpus()).examples_by_source == {
            "edu-mcq": 2,
            "squad-qg": 8,
        }

    def test_targets_are_counted_by_question_type(self):
        assert compute_statistics(self.corpus()).targets_by_question_type == {
            "mcq": 2,
            "short_answer": 8,
        }

    def test_targets_are_counted_by_difficulty_in_reading_order(self):
        stats = compute_statistics(self.corpus())
        assert list(stats.targets_by_difficulty) == ["easy", "medium"]

    def test_an_invalid_difficulty_is_still_counted(self):
        """Statistics must not disagree with validation about how many targets exist."""
        stats = compute_statistics([example(0, targets=(target(difficulty="unknown"),))])
        assert stats.targets_by_difficulty == {"unknown": 1}
        assert stats.total_targets == 1

    def test_examples_are_counted_by_topic(self):
        stats = compute_statistics(self.corpus())
        assert stats.examples_by_topic["Mitochondrion"] == 4
        assert stats.distinct_topics == 3

    def test_the_topic_table_is_ranked_highest_first(self):
        stats = compute_statistics(self.corpus())
        counts = list(stats.examples_by_topic.values())
        assert counts == sorted(counts, reverse=True)

    def test_the_topic_table_can_be_capped_while_the_count_stays_complete(self):
        corpus = [example(i, topic=f"Topic-{i}") for i in range(20)]
        stats = compute_statistics(corpus, topic_limit=5)
        assert len(stats.examples_by_topic) == 5
        assert stats.distinct_topics == 20

    def test_unlabelled_examples_are_counted(self):
        stats = compute_statistics([example(0, topic=None), example(1, topic="  ")])
        assert stats.examples_without_topic == 2
        assert stats.distinct_topics == 0

    def test_average_lengths_are_reported(self):
        stats = compute_statistics(self.corpus())
        assert stats.context_chars.mean > 0
        assert stats.question_chars.mean > 0
        assert stats.answer_chars.mean > 0
        assert stats.context_chars.count == 10

    def test_length_summaries_carry_min_mean_and_max(self):
        stats = compute_statistics(
            [example(0, context="a" * 100), example(1, context="b" * 300)]
        )
        assert stats.context_chars.minimum == 100
        assert stats.context_chars.maximum == 300
        assert stats.context_chars.mean == 200.0

    def test_mcq_option_counts_cover_only_mcq_targets(self):
        stats = compute_statistics(self.corpus())
        assert stats.mcq_option_counts.count == 2
        assert stats.mcq_option_counts.minimum == 4

    def test_grounding_rate_is_reported(self):
        stats = compute_statistics(self.corpus())
        assert stats.grounded_examples == 8
        assert stats.grounded_rate == 0.8

    def test_distinct_group_keys_are_counted(self):
        """The ceiling on how finely a corpus can be split."""
        stats = compute_statistics(self.corpus())
        assert stats.distinct_group_keys == 4

    def test_marks_are_summed_and_averaged(self):
        stats = compute_statistics(self.corpus())
        assert stats.marks_total == 10
        assert stats.mean_marks == 1.0

    def test_non_integer_marks_are_excluded_from_the_total(self):
        stats = compute_statistics([example(0, targets=(target(marks="two"),))])
        assert stats.marks_total == 0

    def test_an_empty_corpus_yields_zeros_rather_than_an_error(self):
        stats = compute_statistics([])
        assert stats.is_empty is True
        assert stats.total_examples == 0
        assert stats.context_chars.as_dict() == {"min": 0, "mean": 0.0, "max": 0, "count": 0}
        assert stats.grounded_rate == 0.0
        assert stats.mean_marks == 0.0

    def test_statistics_are_deterministic(self):
        corpus = self.corpus()
        assert compute_statistics(corpus).as_dict() == compute_statistics(corpus).as_dict()

    def test_statistics_do_not_depend_on_input_order(self):
        corpus = self.corpus()
        forward = compute_statistics(corpus).as_dict()
        backward = compute_statistics(list(reversed(corpus))).as_dict()
        assert forward == backward

    def test_statistics_are_json_serializable(self):
        payload = json.loads(json.dumps(compute_statistics(self.corpus()).as_dict()))
        assert payload["total_examples"] == 10

    def test_statistics_round_trip(self):
        original = compute_statistics(self.corpus())
        assert statistics_from_dict(original.as_dict()) == original

    def test_a_generator_input_is_accepted(self):
        stats = compute_statistics(item for item in self.corpus())
        assert stats.total_examples == 10


class TestDatasetMetadata:
    """The prepared-corpus record."""

    def test_the_record_carries_provenance_and_composition(self):
        corpus = [example(i) for i in range(6)]
        splits = DeterministicGroupSplitter().split(corpus, seed=11)
        report = validate_dataset(corpus)
        metadata = DatasetMetadata(
            fingerprint=splits.dataset_fingerprint,
            config_hash="abc123",
            sources=("squad-qg",),
            adapter_specs=(SquadQuestionGenerationAdapter().spec,),
            split_summary=splits.as_dict(),
            statistics=compute_statistics(corpus),
            split_statistics={
                name.value: compute_statistics(splits[name]) for name in SplitName
            },
            validation_summary=report.as_dict(),
            notes=("LearningQ unavailable in this phase",),
        )
        payload = json.loads(metadata.to_json())
        assert payload["fingerprint"] == splits.dataset_fingerprint
        assert payload["statistics"]["total_examples"] == 6
        assert set(payload["split_statistics"]) == {"train", "validation", "test"}
        assert payload["adapter_specs"][0]["source_id"] == "squad-qg"
        assert payload["notes"]

    def test_the_record_serializes_without_statistics(self):
        metadata = DatasetMetadata(fingerprint="deadbeef")
        assert json.loads(metadata.to_json())["statistics"] is None

    def test_split_statistics_are_ordered_stably(self):
        metadata = DatasetMetadata(
            fingerprint="x",
            split_statistics={"test": DatasetStatistics(), "train": DatasetStatistics()},
        )
        assert list(metadata.as_dict()["split_statistics"]) == ["test", "train"]


class TestEndToEndPreparation:
    """The whole offline path: records in, split and audited corpus out."""

    def test_a_mixed_corpus_prepares_cleanly(self):
        examples = [
            *adapt_records("squad-qg", [squad_record(i) for i in range(12)]),
            *adapt_records("edu-mcq", [mcq_record(i) for i in range(4)]),
        ]
        report = validate_dataset(examples)
        assert report.ok, report.as_dict()

        splits = DeterministicGroupSplitter().split(
            examples, seed=42, ratios=SplitRatios(0.5, 0.25, 0.25), group_by="context"
        )
        assert len(splits) == 16
        assert splits.leaked_group_keys() == frozenset()

        stats = compute_statistics(examples)
        assert stats.examples_by_source == {"edu-mcq": 4, "squad-qg": 12}
        assert set(stats.targets_by_question_type) == {"mcq", "short_answer"}

    def test_the_prepared_corpus_renders_prompts_for_every_example(self):
        from qa_gen import DEFAULT_TEMPLATE, target_from_json

        examples = list(adapt_records("squad-qg", [squad_record(i) for i in range(3)]))
        for item in examples:
            rendered = DEFAULT_TEMPLATE.render(item)
            assert item.context in rendered.user
            assert target_from_json(rendered.completion) == item.primary_target
