import pytest

from text_change_detector.tiling.extraction.shared import (
    NUMBERED_HEADING,
    has_finite_verb,
    is_content,
    is_label,
    load_nlp,
    split_sentences,
)


class TestNumberedHeading:
    def test_simple_number(self):
        assert NUMBERED_HEADING.match("1. Introduction").group(1) == "1"

    def test_multi_level_number(self):
        assert NUMBERED_HEADING.match("1.2.3 Deep Dive").group(1) == "1.2.3"

    def test_paren_separator(self):
        assert NUMBERED_HEADING.match("2) Second item").group(1) == "2"

    def test_number_without_separator(self):
        assert NUMBERED_HEADING.match("12 Overview").group(1) == "12"

    def test_requires_whitespace_before_text(self):
        assert NUMBERED_HEADING.match("1.Introduction") is None

    def test_plain_text_does_not_match(self):
        assert NUMBERED_HEADING.match("No number here") is None

    def test_dot_count_gives_depth(self):
        assert NUMBERED_HEADING.match("1.2.3 X").group(1).count(".") + 1 == 3


class TestLoadNlp:
    def test_returns_usable_pipeline(self):
        nlp = load_nlp("en_core_web_sm")
        doc = nlp("The tenant pays rent.")

        assert [t.text for t in doc][:2] == ["The", "tenant"]

    def test_missing_model_raises_with_hint(self):
        with pytest.raises(OSError, match="is not installed"):
            load_nlp("nonexistent_model_xyz123")


class TestSpacyPredicates:
    def test_has_finite_verb(self, nlp):
        assert has_finite_verb(nlp("She writes clean code.")[:])
        assert not has_finite_verb(nlp("Payment Terms")[:])

    def test_is_content_on_sentence(self, nlp):
        assert is_content("The database replicates writes to two standby nodes.", nlp)

    def test_is_content_false_on_noun_phrase(self, nlp):
        assert not is_content("Caching Strategy", nlp)

    def test_is_label_on_short_phrase(self, nlp):
        assert is_label("Payment Terms", nlp)

    def test_is_label_false_on_content(self, nlp):
        assert not is_label("The service authenticates each request.", nlp)

    def test_is_label_false_when_too_long(self, nlp):
        assert not is_label("General Terms And Conditions Of The Service Agreement", nlp)

    def test_is_label_false_on_empty(self, nlp):
        assert not is_label("", nlp)
        assert not is_label("   ", nlp)


class TestSplitSentences:
    def test_splits_and_strips(self, nlp):
        assert split_sentences("First one. Second two! Third three?", nlp) == [
            "First one.",
            "Second two!",
            "Third three?",
        ]

    def test_drops_blank_input(self, nlp):
        assert split_sentences("    ", nlp) == []
