"""Detect which semantic units a proposed change impacts.

The single entry point is `detect_changes`; `DetectionResult` is its return type.
"""

from text_change_detector.detection.models import (
    Change,
    ChangeImpact,
    DetectionResult,
    Relation,
    Suggestion,
)
from text_change_detector.detection.pipeline import detect_changes
from text_change_detector.detection.prompts import ENGLISH_PROMPTS, POLISH_PROMPTS, Prompts

__all__ = [
    "detect_changes",
    "DetectionResult",
    "Change",
    "ChangeImpact",
    "Relation",
    "Suggestion",
    "Prompts",
    "ENGLISH_PROMPTS",
    "POLISH_PROMPTS",
]
