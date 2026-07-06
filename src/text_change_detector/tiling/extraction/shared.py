import re

import spacy

NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*)[.)]?\s+\S")


def load_nlp(name: str):
    try:
        return spacy.load(name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{name}' is not installed. Run: python -m spacy download {name}"
        ) from exc


def has_finite_verb(span) -> bool:
    return any("Fin" in tok.morph.get("VerbForm") for tok in span)


def is_content(text: str, nlp) -> bool:
    return any(has_finite_verb(s) for s in nlp(text).sents)


def is_label(text: str, nlp) -> bool:
    return 0 < len(text.split()) <= 6 and not is_content(text, nlp)


def split_sentences(text: str, nlp) -> list[str]:
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
