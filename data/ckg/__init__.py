"""
Cultural Knowledge Graph (CKG) module for ArangoDB integration.

This module provides functionality to interact with the Cultural Knowledge Graph,
including schema setup, data insertion, and querying capabilities.
"""

from .connection import get_db_connection
from .schema import create_collections
from .operations import (
    insert_mythological_entity,
    insert_geographic_feature,
    insert_cultural_concept,
    insert_text_source,
    insert_geospatial_point,
    insert_edge,
    get_all_entities,
    get_entity_by_id,
    get_related_entities,
    search_entities_by_name,
    get_edges_between,
)

from .ckg import CulturalKnowledgeGraph

__all__ = [
    "get_db_connection",
    "create_collections",
    "insert_mythological_entity",
    "insert_geographic_feature",
    "insert_cultural_concept",
    "insert_text_source",
    "insert_geospatial_point",
    "insert_edge",
    "get_all_entities",
    "get_entity_by_id",
    "get_related_entities",
    "search_entities_by_name",
    "get_edges_between",
    "CulturalKnowledgeGraph",
]
