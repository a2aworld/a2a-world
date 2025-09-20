from .connection import get_db_connection
import logging

logger = logging.getLogger(__name__)


class CKGOperations:
    """Operations for the Cognitive Knowledge Graph."""

    def __init__(self):
        self.db = get_db_connection()

    def find_unmapped_mythological_entities(self) -> list:
        """
        Find mythological entities that are not connected to geographic features.

        Returns:
            list: List of unmapped mythological entities
        """
        try:
            # AQL query to find MythologicalEntity nodes not connected via LOCATED_AT
            aql = """
            FOR entity IN MythologicalEntity
                LET connected = (
                    FOR v IN 1..1 OUTBOUND entity._id LOCATED_AT
                    RETURN v
                )
                FILTER LENGTH(connected) == 0
                RETURN {
                    _id: entity._id,
                    name: entity.name,
                    description: entity.description,
                    attributes: entity
                }
            """
            return list(self.db.aql.execute(aql))
        except Exception as e:
            logger.error(f"Error finding unmapped entities: {e}")
            return []

    def find_contradictory_information(self) -> list:
        """
        Find entities with contradictory or conflicting information.

        Returns:
            list: List of entities with contradictory information
        """
        try:
            # AQL query to find entities mentioned in multiple sources with different descriptions
            aql = """
            FOR entity IN MythologicalEntity
                LET sources = (
                    FOR source IN 1..1 INBOUND entity._id MENTIONED_IN
                    RETURN source
                )
                FILTER LENGTH(sources) > 1
                LET descriptions = (
                    FOR source IN sources
                    RETURN source.description || source.content
                )
                FILTER LENGTH(descriptions) > 1
                RETURN {
                    entity: entity.name,
                    conflict_description: "Multiple sources with potentially conflicting information",
                    source_count: LENGTH(sources),
                    sources: sources[*].title || sources[*].name
                }
            """
            return list(self.db.aql.execute(aql))
        except Exception as e:
            logger.error(f"Error finding contradictory information: {e}")
            return []

    def find_knowledge_gaps(self) -> list:
        """
        Find gaps in knowledge connections between related concepts.

        Returns:
            list: List of knowledge gaps
        """
        try:
            # AQL query to find concepts that should be connected but aren't
            aql = """
            FOR concept1 IN CulturalConcept
                FOR concept2 IN CulturalConcept
                    FILTER concept1 != concept2
                    LET connection_exists = (
                        FOR edge IN RELATED_TO
                            FILTER (edge._from == concept1._id AND edge._to == concept2._id)
                                OR (edge._from == concept2._id AND edge._to == concept1._id)
                            RETURN edge
                    )
                    FILTER LENGTH(connection_exists) == 0
                    AND LOWER(concept1.name) != LOWER(concept2.name)
                    LIMIT 10
                    RETURN {
                        from_entity: concept1.name,
                        to_entity: concept2.name,
                        gap_type: "missing_relationship"
                    }
            """
            return list(self.db.aql.execute(aql))
        except Exception as e:
            logger.error(f"Error finding knowledge gaps: {e}")
            return []


# Insertion functions for vertex collections (nodes)


def insert_mythological_entity(name, description, attributes=None):
    """
    Inserts a new MythologicalEntity node.

    Args:
        name (str): Name of the entity
        description (str): Description of the entity
        attributes (dict): Additional attributes

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection("MythologicalEntity")
    doc = {"name": name, "description": description}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


def insert_geographic_feature(name, type, coordinates, attributes=None):
    """
    Inserts a new GeographicFeature node.

    Args:
        name (str): Name of the feature
        type (str): Type of geographic feature
        coordinates (list): Geo coordinates [lat, lon]
        attributes (dict): Additional attributes

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection("GeographicFeature")
    doc = {"name": name, "type": type, "coordinates": coordinates}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


def insert_cultural_concept(name, description, attributes=None):
    """
    Inserts a new CulturalConcept node.

    Args:
        name (str): Name of the concept
        description (str): Description of the concept
        attributes (dict): Additional attributes

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection("CulturalConcept")
    doc = {"name": name, "description": description}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


def insert_text_source(title, content, source_type, attributes=None):
    """
    Inserts a new TextSource node.

    Args:
        title (str): Title of the text source
        content (str): Content of the text
        source_type (str): Type of source (e.g., book, article)
        attributes (dict): Additional attributes

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection("TextSource")
    doc = {"title": title, "content": content, "source_type": source_type}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


def insert_geospatial_point(name, coordinates, attributes=None):
    """
    Inserts a new GeospatialPoint node.

    Args:
        name (str): Name of the point
        coordinates (list): Geo coordinates [lat, lon]
        attributes (dict): Additional attributes

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection("GeospatialPoint")
    doc = {"name": name, "coordinates": coordinates}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


# Insertion function for edges


def insert_edge(edge_collection, from_id, to_id, attributes=None):
    """
    Inserts a new edge between two nodes.

    Args:
        edge_collection (str): Name of the edge collection
        from_id (str): ID of the source node
        to_id (str): ID of the target node
        attributes (dict): Additional attributes for the edge

    Returns:
        dict: Insertion result
    """
    db = get_db_connection()
    collection = db.collection(edge_collection)
    doc = {"_from": from_id, "_to": to_id}
    if attributes:
        doc.update(attributes)
    return collection.insert(doc)


# Query functions


def get_all_entities(collection_name):
    """
    Retrieves all entities from a collection.

    Args:
        collection_name (str): Name of the collection

    Returns:
        list: List of documents
    """
    db = get_db_connection()
    collection = db.collection(collection_name)
    return list(collection.all())


def get_entity_by_id(collection_name, doc_id):
    """
    Retrieves an entity by its ID.

    Args:
        collection_name (str): Name of the collection
        doc_id (str): Document ID

    Returns:
        dict: Document data
    """
    db = get_db_connection()
    collection = db.collection(collection_name)
    return collection.get(doc_id)


def get_related_entities(entity_id, edge_collection, direction="OUTBOUND"):
    """
    Retrieves entities related via a specific edge collection.

    Args:
        entity_id (str): ID of the source entity
        edge_collection (str): Name of the edge collection
        direction (str): 'OUTBOUND' or 'INBOUND'

    Returns:
        list: List of related documents
    """
    db = get_db_connection()
    if direction.upper() == "OUTBOUND":
        aql = f"FOR v IN 1..1 OUTBOUND '{entity_id}' {edge_collection} RETURN v"
    elif direction.upper() == "INBOUND":
        aql = f"FOR v IN 1..1 INBOUND '{entity_id}' {edge_collection} RETURN v"
    else:
        raise ValueError("Direction must be 'OUTBOUND' or 'INBOUND'")
    return list(db.aql.execute(aql))


def search_entities_by_name(collection_name, name):
    """
    Searches for entities by name (case-insensitive partial match).

    Args:
        collection_name (str): Name of the collection
        name (str): Name to search for

    Returns:
        list: List of matching documents
    """
    db = get_db_connection()
    aql = f"FOR doc IN {collection_name} FILTER LOWER(doc.name) LIKE LOWER('%{name}%') RETURN doc"
    return list(db.aql.execute(aql))


def get_edges_between(from_id, to_id, edge_collection=None):
    """
    Retrieves edges between two entities.

    Args:
        from_id (str): Source entity ID
        to_id (str): Target entity ID
        edge_collection (str): Specific edge collection (optional)

    Returns:
        list: List of edge documents
    """
    db = get_db_connection()
    if edge_collection:
        aql = f"FOR edge IN {edge_collection} FILTER edge._from == '{from_id}' AND edge._to == '{to_id}' RETURN edge"
    else:
        # Search all edge collections
        edge_cols = ["DEPICTS", "LOCATED_AT", "MENTIONED_IN", "RELATED_TO", "PART_OF"]
        results = []
        for col in edge_cols:
            aql = f"FOR edge IN {col} FILTER edge._from == '{from_id}' AND edge._to == '{to_id}' RETURN edge"
            results.extend(list(db.aql.execute(aql)))
        return results
    return list(db.aql.execute(aql))
