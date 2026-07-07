from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A unit of extracted text fed into tiling.

    Currently a Segment is a single sentence, but as the tiling algorithms
    evolve it may become a larger unit than a sentence.
    """

    text: str
    section: str = ""
    payload: list[str] = Field(default_factory=list)


class SemanticUnit(BaseModel):
    id: int
    section: str
    sentences: list[str]
    payload: list[str]


class Community(BaseModel):
    id: int
    units: list[SemanticUnit]


class TilingResult(BaseModel):
    communities: list[Community]
