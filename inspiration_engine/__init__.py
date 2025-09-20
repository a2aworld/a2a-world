"""
Inspiration Engine for Terra Constellata

A system for detecting novelty and quantifying "interestingness" in geospatial and mythological data
using advanced statistical methods including RPAD, Peculiarity/J-Measure, and Belief-Change Measure.

This engine integrates with the Cultural Knowledge Graph (CKG) and PostGIS databases,
and communicates with other agents via the A2A protocol.
"""

from .core import InspirationEngine
from .algorithms import NoveltyDetector
from .data_ingestion import DataIngestor
from .prompt_ranking import PromptRanker
from .a2a_integration import A2AInspirationClient

__version__ = "1.0.0"
__all__ = [
    "InspirationEngine",
    "NoveltyDetector",
    "DataIngestor",
    "PromptRanker",
    "A2AInspirationClient",
]
