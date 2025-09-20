from arango import ArangoClient


def get_db_connection(
    host="http://localhost:8529", username="root", password="", database="ckg_db"
):
    """
    Establishes a connection to the ArangoDB database.
    Creates the database if it doesn't exist.

    Args:
        host (str): ArangoDB host URL
        username (str): Username for authentication
        password (str): Password for authentication
        database (str): Database name

    Returns:
        arango.database.StandardDatabase: Database connection object
    """
    client = ArangoClient(hosts=host)

    # Create database if it doesn't exist
    try:
        client.db(database, username=username, password=password)
    except:
        client.create_database(
            database,
            users=[{"username": username, "password": password, "active": True}],
        )

    db = client.db(database, username=username, password=password)
    return db
