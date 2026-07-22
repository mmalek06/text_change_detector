"""Split text into semantic units and detect which units a change impacts."""

from text_change_detector.detection import (
    ENGLISH_PROMPTS,
    POLISH_PROMPTS,
    Change,
    ChangeImpact,
    DetectionResult,
    Prompts,
    Relation,
    Suggestion,
    detect_changes,
)
from text_change_detector.shared.embedder import Embedder, SentenceTransformerEmbedder
from text_change_detector.shared.models import Community, Segment, SemanticUnit, TilingResult
from text_change_detector.tiling import tile
from text_change_detector.tiling.extraction import (
    Block,
    Extractor,
    PdfReader,
    blocks_to_segments,
    extract_text,
)

__version__ = "0.6.0"

__all__ = [
    "tile",
    "detect_changes",
    "DetectionResult",
    "Change",
    "ChangeImpact",
    "Relation",
    "Suggestion",
    "Prompts",
    "ENGLISH_PROMPTS",
    "POLISH_PROMPTS",
    "Extractor",
    "PdfReader",
    "Block",
    "blocks_to_segments",
    "extract_text",
    "Embedder",
    "SentenceTransformerEmbedder",
    "TilingResult",
    "Community",
    "SemanticUnit",
    "Segment",
]
