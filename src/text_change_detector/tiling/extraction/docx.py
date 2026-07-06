import re
from pathlib import Path

import numpy as np
from docx import Document
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from text_change_detector.models import Segment
from text_change_detector.tiling.extraction.shared import (
    NUMBERED_HEADING,
    is_content,
    is_label,
    split_sentences,
)

DocxSource = str | Path | DocxDocument

TOC_LINE = re.compile(r"^\d+(\.\d+)*\t.+\t\d+$")
OUTLINE_BODY_LEVEL = 9


def outline_heading_level(p: Paragraph) -> int | None:
    path = "./w:pPr/w:outlineLvl/@w:val"
    sources = [p._p.xpath(path)]
    style = p.style
    seen = set()

    while style is not None and style.style_id not in seen:
        seen.add(style.style_id)
        sources.append(style.element.xpath(path))

        style = style.base_style

    for values in sources:
        if values:
            level = int(values[0])

            return level + 1 if level < OUTLINE_BODY_LEVEL else None

    return None


def heuristic_heading_level(p: Paragraph, body_font_size: float | None = None) -> int | None:
    text = p.text.strip()

    if not text or "\n" in text or len(text.split()) > 12 or text[-1] in ".!?;:,":
        return None

    number = NUMBERED_HEADING.match(text)
    bold_runs = [r.font.bold for r in p.runs if r.text.strip()]
    is_bold = (bool(bold_runs) and all(bold_runs)) or bool(p.style.font.bold)
    keeps_next = bool(p.paragraph_format.keep_with_next or p.style.paragraph_format.keep_with_next)
    sizes = [r.font.size.pt for r in p.runs if r.text.strip() and r.font.size]

    if not sizes and p.style.font.size:
        sizes = [p.style.font.size.pt]

    is_larger = body_font_size is not None and bool(sizes) and max(sizes) > body_font_size
    is_caps = text == text.upper() and any(ch.isalpha() for ch in text)
    score = sum([bool(number), is_bold, keeps_next, is_larger, is_caps])

    if score < 2:
        return None

    return number.group(1).count(".") + 1 if number else 1


def is_toc_paragraph(p: Paragraph) -> bool:
    if p._p.xpath(".//w:hyperlink[starts-with(@w:anchor, '_Toc')]"):
        return True

    return bool(TOC_LINE.match(p.text.strip()))


def has_numbering(p: Paragraph) -> bool:
    if p._p.xpath("./w:pPr/w:numPr"):
        return True

    style = p.style
    seen = set()

    while style is not None and style.style_id not in seen:
        seen.add(style.style_id)

        if style.element.xpath("./w:pPr/w:numPr"):
            return True

        style = style.base_style

    return False


def median_font_size(doc: Document) -> float | None:
    sizes = [r.font.size.pt for p in doc.paragraphs for r in p.runs if r.font.size]

    return float(np.median(sizes)) if sizes else None


def column_headers(table: Table, nlp) -> list[str] | None:
    if len(table.rows) < 2:
        return None

    first = [cell.text.strip() for cell in table.rows[0].cells]

    return first if all(is_label(t, nlp) for t in first) else None


def extract_docx(source: DocxSource, nlp) -> list[Segment]:
    doc = source if isinstance(source, DocxDocument) else Document(source)
    body_size = median_font_size(doc)
    segments: list[Segment] = []
    pending: list[str] = []
    sections: dict[int, str] = {}
    section = ""
    attach_forward = True
    lead_in = ""

    def emit(text: str) -> None:
        nonlocal attach_forward

        if is_content(text, nlp):
            segments.append(Segment(text=text, section=section, payload=pending.copy()))
            pending.clear()

            attach_forward = False
        elif attach_forward or not segments:
            pending.append(text)
        else:
            segments[-1].payload.append(text)

    def handle_paragraph(p: Paragraph) -> None:
        nonlocal section, lead_in, attach_forward

        text = p.text.strip()

        if not text or is_toc_paragraph(p):
            return

        level = outline_heading_level(p) or heuristic_heading_level(p, body_size)

        if level is not None:
            sections[level] = text

            for deeper in [lvl for lvl in sections if lvl > level]:
                del sections[deeper]

            section = " > ".join(sections[lvl] for lvl in sorted(sections))
            lead_in = ""
            attach_forward = True

            return

        parts = split_sentences(text, nlp)

        if has_numbering(p):
            for s in parts:
                emit(f"{lead_in} {s}" if lead_in else s)
        else:
            for s in parts:
                emit(s)

            lead_in = parts[-1] if parts and text.endswith(":") else ""

    def handle_table(table: Table) -> None:
        headers = column_headers(table, nlp)
        two_col_form = headers is None and len(table.columns) == 2
        seen = set()

        for r, row in enumerate(table.rows):
            row_label = row.cells[0].text.strip() if two_col_form and row.cells else ""

            if not is_label(row_label, nlp):
                row_label = ""

            for c, cell in enumerate(row.cells):
                if cell._tc in seen:
                    continue

                seen.add(cell._tc)

                if headers is not None and r == 0:
                    continue

                text = cell.text.strip()

                if not text or (row_label and c == 0):
                    continue

                context = row_label or (headers[c] if headers and c < len(headers) else "")

                for s in split_sentences(text, nlp):
                    emit(s if not context or is_content(s, nlp) else f"{context}: {s}")

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            handle_paragraph(Paragraph(child, doc))
        elif child.tag.endswith("}tbl"):
            handle_table(Table(child, doc))

    if pending and segments:
        segments[-1].payload.extend(pending)

    return segments
