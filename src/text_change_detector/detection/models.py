from typing import Literal

from pydantic import BaseModel, Field

Relatedness = Literal["none", "medium", "strong"]


class UnitRelation(BaseModel):
    unit_topic: str = Field(description="the unit's main subject in a few words")
    relation: Relatedness = Field(description="one of: strong, medium, none (see the instructions)")
    justification: str = Field(description="one or two sentences with the concrete deciding detail")


class Verdict(BaseModel):
    objection: str = Field(description="the strongest reason the change might not belong in this unit")
    reason: str = Field(description="one sentence deciding, addressing the objection")
    agrees: bool = Field(description="whether the change genuinely belongs in this unit")


class Merge(BaseModel):
    added: str = Field(description="the new wording the change inserts; empty when it only updates or deletes")
    merged_text: str = Field(description="the unit text after the change, with every unaffected sentence preserved")


class Change(BaseModel):
    """A proposed new or changed requirement to test against the document."""

    name: str
    text: str


class Relation(BaseModel):
    """How one impacted unit relates to a change, with the verification verdict.

    `verified` is None when the relation was not `strong` (the skeptical pass only
    runs on strong hits); True/False once it has run.
    """

    requirement: str
    unit_id: int
    section: str
    relation: Relatedness
    unit_topic: str
    justification: str
    verified: bool | None = None
    verify_reason: str = ""


class Suggestion(BaseModel):
    """A concrete edit for a strong hit that passed verification."""

    requirement: str
    unit_id: int
    section: str
    justification: str
    verify_reason: str
    current_text: str
    added: str
    merged_text: str


class ChangeImpact(BaseModel):
    """Everything the pipeline found for a single change.

    `primary` are the unit ids the change most resembles (direct hits); `ripple`
    are the units one graph hop away from those. `relations` covers every reviewed
    candidate; `suggestions` carries only the verified-strong edits.
    """

    name: str
    text: str
    primary: list[int]
    ripple: list[int]
    relations: list[Relation]
    suggestions: list[Suggestion]


class DetectionResult(BaseModel):
    """The return type of `detect_changes`: one `ChangeImpact` per change."""

    changes: list[ChangeImpact]

    @property
    def relations(self) -> list[Relation]:
        """Every reviewed relation across all changes, flattened."""
        return [relation for change in self.changes for relation in change.relations]

    @property
    def suggestions(self) -> list[Suggestion]:
        """Every verified-strong suggestion across all changes, flattened."""
        return [suggestion for change in self.changes for suggestion in change.suggestions]
