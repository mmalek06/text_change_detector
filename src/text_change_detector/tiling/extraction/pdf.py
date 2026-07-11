import re
from pathlib import Path
from typing import NamedTuple, Protocol

from text_change_detector.shared.models import Segment
from text_change_detector.tiling.extraction.shared import NUMBERED_HEADING, is_content, split_sentences


class Block(NamedTuple):
    text: str
    size: float
    bold: bool
    single_line: bool
    page: int


class PdfReader(Protocol):
    """Turns a PDF into a list of typographic blocks.

    A reading strategy is injected into the library rather than bundled, so
    the core carries no PDF engine (and no engine's licence). Implementations
    live in companion packages, for example the PyMuPDF adapter or the
    pypdfium2 rewrite. A reader takes a path to a PDF and returns the page
    blocks in reading order; the shared `blocks_to_segments` turns those into
    `Segment`s the same way regardless of which engine produced them.
    """

    def __call__(self, source: str | Path) -> list[Block]: ...


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


def blocks_to_segments(blocks: list[Block], nlp) -> list[Segment]:
    """Turn typographic blocks into `Segment`s.

    Engine-agnostic: any `PdfReader` produces `Block`s and this drives the
    same section tracking, running-furniture removal and sentence splitting.
    """
    body_size = body_font_size(blocks)
    pages = max((b.page for b in blocks), default=-1) + 1
    furniture = running_furniture(blocks, pages)

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


def extract_pdf(source: str | Path, nlp, reader: PdfReader) -> list[Segment]:
    """Extract `Segment`s from a PDF using an injected reading strategy."""
    return blocks_to_segments(reader(source), nlp)
