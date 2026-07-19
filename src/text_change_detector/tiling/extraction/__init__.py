from collections.abc import Callable
from pathlib import Path

from docx.document import Document as DocxDocument

from text_change_detector.shared.models import Segment
from text_change_detector.tiling.extraction.docx import DocxSource, extract_docx
from text_change_detector.tiling.extraction.pdf import (
    Block,
    PdfReader,
    blocks_to_segments,
    extract_pdf,
    join_wrapped,
)
from text_change_detector.tiling.extraction.shared import load_nlp
from text_change_detector.tiling.extraction.text import extract_text

Extractor = Callable[[Path], list[Segment]]
Source = str | Path | DocxDocument | list[Segment]


def builtin_extract_text(text: str, spacy_model: str | None) -> list[Segment]:
    if spacy_model is None:
        raise ValueError("Provide spacy_model to tile a raw text string, or pass a list[Segment].")

    return extract_text(text, load_nlp(spacy_model))


def builtin_extract(source: Source, spacy_model: str | None, pdf_reader: PdfReader | None = None) -> list[Segment]:
    kind = _kind(source)

    if kind == "pdf" and pdf_reader is None:
        raise ValueError(
            "PDF extraction needs a reading strategy. Pass pdf_reader=... a PdfReader, "
            "e.g. read_blocks from text_change_detector_pymupdf_rewrite (permissive) or "
            "text_change_detector_pymupdf_adapter (PyMuPDF), or pass a custom extractor."
        )

    if spacy_model is None:
        raise ValueError("Provide spacy_model for the built-in extractor, or pass a custom extractor.")

    nlp = load_nlp(spacy_model)

    if kind == "docx":
        return extract_docx(source, nlp)

    return extract_pdf(source, nlp, pdf_reader)


def _kind(source: Source) -> str:
    if isinstance(source, DocxDocument):
        return "docx"

    if isinstance(source, Path):
        suffix = source.suffix.lower()

        if suffix == ".docx":
            return "docx"

        if suffix == ".pdf":
            return "pdf"

    raise ValueError(f"Cannot infer a built-in extractor for {source!r}; pass a custom extractor or list[Segment].")


__all__ = [
    "Extractor",
    "Source",
    "DocxSource",
    "Block",
    "PdfReader",
    "extract_docx",
    "extract_pdf",
    "extract_text",
    "blocks_to_segments",
    "join_wrapped",
    "builtin_extract",
    "builtin_extract_text",
    "load_nlp",
]
