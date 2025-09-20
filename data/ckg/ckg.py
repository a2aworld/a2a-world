"""
Main CKG class for Cultural Knowledge Graph operations.
Provides a high-level interface for interacting with the graph database.
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


class CulturalKnowledgeGraph:
    """
    Main class for Cultural Knowledge Graph operations.
    """

    def __init__(
        self,
        host="http://localhost:8529",
        username="root",
        password="",
        database="ckg_db",
    ):
        """
        Initialize the CKG instance.

        Args:
            host (str): ArangoDB host URL
            username (str): Username
            password (str): Password
            database (str): Database name
        """
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.db = None

    def connect(self):
        """
        Establish connection to the database and create schema if needed.
        """
        from .connection import get_db_connection

        self.db = get_db_connection(
            self.host, self.username, self.password, self.database
        )
        create_collections()
        return self.db

    def insert_entity(self, entity_type, **kwargs):
        """
        Insert an entity of the specified type.

        Args:
            entity_type (str): Type of entity ('mythological', 'geographic', 'cultural', 'text', 'geospatial')
            **kwargs: Entity attributes

        Returns:
            dict: Insertion result
        """
        if not self.db:
            self.connect()

        entity_type = entity_type.lower()
        if entity_type == "mythological":
            return insert_mythological_entity(**kwargs)
        elif entity_type == "geographic":
            return insert_geographic_feature(**kwargs)
        elif entity_type == "cultural":
            return insert_cultural_concept(**kwargs)
        elif entity_type == "text":
            return insert_text_source(**kwargs)
        elif entity_type == "geospatial":
            return insert_geospatial_point(**kwargs)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def insert_relationship(self, relationship_type, from_id, to_id, **kwargs):
        """
        Insert a relationship between entities.

        Args:
            relationship_type (str): Type of relationship
            from_id (str): Source entity ID
            to_id (str): Target entity ID
            **kwargs: Additional attributes

        Returns:
            dict: Insertion result
        """
        if not self.db:
            self.connect()

        return insert_edge(
            relationship_type, from_id, to_id, kwargs if kwargs else None
        )

    def query_entities(self, collection_name, filters=None):
        """
        Query entities from a collection.

        Args:
            collection_name (str): Collection name
            filters (dict): Optional filters

        Returns:
            list: List of entities
        """
        if not self.db:
            self.connect()

        if filters and "name" in filters:
            return search_entities_by_name(collection_name, filters["name"])
        else:
            return get_all_entities(collection_name)

    def get_entity(self, collection_name, entity_id):
        """
        Get a specific entity by ID.

        Args:
            collection_name (str): Collection name
            entity_id (str): Entity ID

        Returns:
            dict: Entity data
        """
        if not self.db:
            self.connect()

        return get_entity_by_id(collection_name, entity_id)

    def get_related(self, entity_id, relationship_type, direction="OUTBOUND"):
        """
        Get entities related to the given entity.

        Args:
            entity_id (str): Entity ID
            relationship_type (str): Relationship type
            direction (str): 'OUTBOUND' or 'INBOUND'

        Returns:
            list: List of related entities
        """
        if not self.db:
            self.connect()

        return get_related_entities(entity_id, relationship_type, direction)

    def get_relationships(self, from_id, to_id, relationship_type=None):
        """
        Get relationships between two entities.

        Args:
            from_id (str): Source entity ID
            to_id (str): Target entity ID
            relationship_type (str): Optional relationship type

        Returns:
            list: List of relationships
        """
        if not self.db:
            self.connect()

        return get_edges_between(from_id, to_id, relationship_type)
