from text_change_detector.shared.models import Segment
from text_change_detector.tiling.extraction.shared import split_sentences


def extract_text(text: str, nlp) -> list[Segment]:
    """Turn a raw text string into one `Segment` per sentence.

    Plain text carries no typographic structure to track, so every sentence
    becomes a standalone `Segment` with an empty section and no payload;
    grouping the sentences into units is left to tiling.
    """
    return [Segment(text=s) for s in split_sentences(text, nlp)]
