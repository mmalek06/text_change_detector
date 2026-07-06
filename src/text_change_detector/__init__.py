"""Split text into semantic units and detect which units a change impacts."""

from text_change_detector.embedder import Embedder, SentenceTransformerEmbedder
from text_change_detector.models import Community, Segment, SemanticUnit, TilingResult
from text_change_detector.tiling import tile
from text_change_detector.tiling.extraction import Extractor

__version__ = "0.1.0"

__all__ = [
    "tile",
    "Extractor",
    "Embedder",
    "SentenceTransformerEmbedder",
    "TilingResult",
    "Community",
    "SemanticUnit",
    "Segment",
]
