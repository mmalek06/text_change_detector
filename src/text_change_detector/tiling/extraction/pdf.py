import re
from pathlib import Path
from typing import NamedTuple

import fitz

from text_change_detector.models import Segment
from text_change_detector.tiling.extraction.shared import NUMBERED_HEADING, is_content, split_sentences

PdfSource = str | Path | fitz.Document


class Block(NamedTuple):
    text: str
    size: float
    bold: bool
    single_line: bool
    page: int


def join_wrapped(parts: list[str]) -> str:
    out = ""

    for p in parts:
        if not out:
            out = p
        elif out.endswith("-"):
            out = out[:-1] + p
        else:
            out = f"{out} {p}"

    return out


def read_blocks(doc) -> list[Block]:
    blocks = []

    for page_no, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b:
                continue

            spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            text = join_wrapped(["".join(s["text"] for s in line["spans"]).strip()
                                 for line in b["lines"] if "".join(s["text"] for s in line["spans"]).strip()])

            if not text:
                continue

            blocks.append(Block(
                text=text,
                size=max((s["size"] for s in spans), default=0.0),
                bold=any(s["flags"] & 16 for s in spans),
                single_line=len(b["lines"]) == 1,
                page=page_no,
            ))

    return blocks


def body_font_size(blocks: list[Block]) -> float | None:
    weight: dict[float, int] = {}

    for b in blocks:
        weight[b.size] = weight.get(b.size, 0) + len(b.text)

    return max(weight, key=weight.get) if weight else None


def running_furniture(blocks: list[Block], pages: int) -> set[str]:
    """Detect running page furniture (repeated headers, footers, page numbers).

    "Furniture" is the typographic term for non-content page fixtures and
    "running" means they repeat across pages. Returns the digit-masked block
    texts that appear on at least half the pages, for the extractor to skip.
    """
    seen_on: dict[str, set[int]] = {}

    for b in blocks:
        seen_on.setdefault(re.sub(r"\d+", "#", b.text), set()).add(b.page)

    return {text for text, ps in seen_on.items() if len(ps) >= pages * 0.5}


def heading_level(block: Block, body_size: float | None, nlp) -> int | None:
    text = block.text

    if not block.single_line or len(text.split()) > 12 or text[-1] in ".!?;:," or is_content(text, nlp):
        return None

    number = NUMBERED_HEADING.match(text)

    if number:
        return number.group(1).count(".") + 1

    is_caps = text == text.upper() and any(ch.isalpha() for ch in text)
    is_larger = body_size is not None and block.size > body_size

    if is_caps or is_larger:
        return 1

    return 2 if block.bold else None


def extract_pdf(source: PdfSource, nlp) -> list[Segment]:
    doc = source if isinstance(source, fitz.Document) else fitz.open(source)
    blocks = read_blocks(doc)
    body_size = body_font_size(blocks)
    furniture = running_furniture(blocks, doc.page_count)

    segments: list[Segment] = []
    sections: dict[int, str] = {}
    section = ""
    pending: list[str] = []
    attach_forward = True

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

    for block in blocks:
        if re.sub(r"\d+", "#", block.text) in furniture:
            continue

        level = heading_level(block, body_size, nlp)

        if level is not None:
            sections[level] = block.text

            for deeper in [lvl for lvl in sections if lvl > level]:
                del sections[deeper]

            section = " > ".join(sections[lvl] for lvl in sorted(sections))
            attach_forward = True

            continue

        for s in split_sentences(block.text, nlp):
            emit(s)

    if pending and segments:
        segments[-1].payload.extend(pending)

    return segments
