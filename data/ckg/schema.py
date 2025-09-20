from .connection import get_db_connection


def create_collections():
    """
    Creates the necessary vertex and edge collections for the Cultural Knowledge Graph.
    Vertex collections: MythologicalEntity, GeographicFeature, CulturalConcept, TextSource, GeospatialPoint
    Edge collections: DEPICTS, LOCATED_AT, MENTIONED_IN, RELATED_TO, PART_OF
    """
    db = get_db_connection()

    # Vertex collections (nodes)
    vertex_collections = [
        "MythologicalEntity",
        "GeographicFeature",
        "CulturalConcept",
        "TextSource",
        "GeospatialPoint",
    ]

    # Edge collections (relationships)
    edge_collections = [
        "DEPICTS",
        "LOCATED_AT",
        "MENTIONED_IN",
        "RELATED_TO",
        "PART_OF",
    ]

    # Create vertex collections
    for col_name in vertex_collections:
        if not db.has_collection(col_name):
            db.create_collection(col_name)
            print(f"Created vertex collection: {col_name}")
        else:
            print(f"Vertex collection {col_name} already exists")

    # Create edge collections
    for col_name in edge_collections:
        if not db.has_collection(col_name):
            db.create_collection(col_name, edge=True)
            print(f"Created edge collection: {col_name}")
        else:
            print(f"Edge collection {col_name} already exists")


if __name__ == "__main__":
    create_collections()
