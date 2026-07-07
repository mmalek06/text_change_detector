from collections.abc import Callable
from pathlib import Path

import fitz
from docx.document import Document as DocxDocument

from text_change_detector.shared.models import Segment
from text_change_detector.tiling.extraction.docx import DocxSource, extract_docx
from text_change_detector.tiling.extraction.pdf import PdfSource, extract_pdf
from text_change_detector.tiling.extraction.shared import load_nlp

Extractor = Callable[[Path], list[Segment]]
Source = str | Path | DocxDocument | fitz.Document | list[Segment]


def builtin_extract(source: DocxSource | PdfSource, spacy_model: str | None) -> list[Segment]:
    extract = _resolve(source)

    if spacy_model is None:
        raise ValueError("Provide spacy_model for the built-in extractor, or pass a custom extractor.")

    nlp = load_nlp(spacy_model)

    return extract(source, nlp)


def _resolve(source: DocxSource | PdfSource) -> Callable[..., list[Segment]]:
    if isinstance(source, DocxDocument):
        return extract_docx

    if isinstance(source, fitz.Document):
        return extract_pdf

    if isinstance(source, Path):
        suffix = source.suffix.lower()

        if suffix == ".docx":
            return extract_docx

        if suffix == ".pdf":
            return extract_pdf

    raise ValueError(f"Cannot infer a built-in extractor for {source!r}; pass a custom extractor or list[Segment].")


__all__ = [
    "Extractor",
    "Source",
    "DocxSource",
    "PdfSource",
    "extract_docx",
    "extract_pdf",
    "builtin_extract",
    "load_nlp",
]
