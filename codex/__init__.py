"""
Agent's Codex for Terra Constellata

A comprehensive archival and legacy preservation system that captures agent
contributions, workflow histories, and successful strategies for future learning
and narrative generation in the Galactic Storybook.
"""

from .models import *
from .archival_system import ArchivalSystem
from .knowledge_base import KnowledgeBase
from .chapter_generator import ChapterGenerator
from .attribution_tracker import AttributionTracker
from .codex_manager import CodexManager

__version__ = "1.0.0"
__all__ = [
    "ArchivalSystem",
    "KnowledgeBase",
    "ChapterGenerator",
    "AttributionTracker",
    "CodexManager",
]
