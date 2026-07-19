from text_change_detector.shared.models import Segment
from text_change_detector.tiling.extraction.text import extract_text


class TestExtractText:
    def test_returns_one_segment_per_sentence(self, nlp):
        segments = extract_text("The cat sat down. The dog ran away.", nlp)

        assert [s.text for s in segments] == ["The cat sat down.", "The dog ran away."]

    def test_segments_are_flat_with_no_section_or_payload(self, nlp):
        for segment in extract_text("The cat sat down. The dog ran away.", nlp):
            assert isinstance(segment, Segment)
            assert segment.section == ""
            assert segment.payload == []

    def test_blank_text_yields_no_segments(self, nlp):
        assert extract_text("   \n\t  ", nlp) == []
