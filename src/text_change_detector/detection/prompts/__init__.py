"""Prompt templates for the three change-detection LLM passes.

`ENGLISH_PROMPTS` and `POLISH_PROMPTS` are ready to use; pass a `prompts=`
argument to `detect_changes` with your own `Prompts` to customise them. Keep the
document and the changes in the same language as the prompt set so the LLM reads
everything consistently.
"""

from dataclasses import dataclass

from text_change_detector.detection.prompts import en, pl


@dataclass(frozen=True)
class Prompts:
    """The three prompt templates driving a detection run.

    relation: rates how a change relates to a unit. Must accept ``{change}`` and
        ``{unit}``.
    verify: a skeptical second pass on a strong relation. Must accept
        ``{change}``, ``{unit}`` and ``{justification}``.
    merge: weaves a verified change into a unit's text. Must accept ``{change}``
        and ``{unit}``.
    """

    relation: str
    verify: str
    merge: str


ENGLISH_PROMPTS = Prompts(relation=en.RELATION, verify=en.VERIFY, merge=en.MERGE)
POLISH_PROMPTS = Prompts(relation=pl.RELATION, verify=pl.VERIFY, merge=pl.MERGE)

__all__ = ["Prompts", "ENGLISH_PROMPTS", "POLISH_PROMPTS"]
