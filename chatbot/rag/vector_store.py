"""
Vector Store Setup for Lore Weaver Chatbot RAG System

This module handles vector embeddings and storage for the Cultural Knowledge Graph
and PostGIS data to enable semantic search and retrieval-augmented generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb import Settings
import faiss
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from ...data.ckg.ckg import CulturalKnowledgeGraph
from ...data.postgis.connection import PostGISConnection

logger = logging.getLogger(__name__)


class LoreWeaverVectorStore:
    """
    Vector store manager for the Lore Weaver chatbot.
    Handles embeddings creation, storage, and retrieval for both CKG and PostGIS data.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        use_chroma: bool = True,
    ):
        """
        Initialize the vector store.

        Args:
            embedding_model: HuggingFace model name for embeddings
            persist_directory: Directory to persist ChromaDB
            use_chroma: Whether to use ChromaDB (True) or FAISS (False)
        """
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.use_chroma = use_chroma

        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize vector stores
        if use_chroma:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self.vector_store = None  # Will be initialized per collection
        else:
            self.vector_store = None  # Will be initialized when data is loaded

        # Initialize database connections
        self.ckg = CulturalKnowledgeGraph()
        self.postgis = PostGISConnection()

        logger.info(f"Initialized LoreWeaverVectorStore with {embedding_model}")

    def _get_or_create_chroma_collection(
        self, collection_name: str
    ) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except ValueError:
            collection = self.chroma_client.create_collection(name=collection_name)
        return collection

    def load_ckg_data(self, collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load data from Cultural Knowledge Graph into vector store.

        Args:
            collections: List of CKG collections to load (default: all)

        Returns:
            Dictionary with loading statistics
        """
        if collections is None:
            collections = [
                "MythologicalEntity",
                "GeographicFeature",
                "CulturalConcept",
                "TextSource",
                "GeospatialPoint",
            ]

        stats = {"total_documents": 0, "collections_loaded": 0}

        try:
            self.ckg.connect()

            for collection_name in collections:
                documents = []

                # Get all entities from collection
                entities = self.ckg.query_entities(collection_name)

                for entity in entities:
                    # Create document from entity data
                    content = self._entity_to_text(entity, collection_name)
                    metadata = {
                        "source": "ckg",
                        "collection": collection_name,
                        "entity_id": entity.get("_key", ""),
                        "entity_type": collection_name,
                    }

                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

                if documents:
                    # Store in vector database
                    if self.use_chroma:
                        collection = self._get_or_create_chroma_collection(
                            f"ckg_{collection_name.lower()}"
                        )
                        self._store_in_chroma(collection, documents)
                    else:
                        self._store_in_faiss(
                            documents, f"ckg_{collection_name.lower()}"
                        )

                    stats["collections_loaded"] += 1
                    stats["total_documents"] += len(documents)
                    logger.info(
                        f"Loaded {len(documents)} documents from {collection_name}"
                    )

        except Exception as e:
            logger.error(f"Error loading CKG data: {e}")
        finally:
            # Note: CKG connection cleanup if needed
            pass

        return stats

    def load_postgis_data(self, table_name: str = "puzzle_pieces") -> Dict[str, Any]:
        """
        Load geospatial data from PostGIS into vector store.

        Args:
            table_name: PostGIS table to load

        Returns:
            Dictionary with loading statistics
        """
        stats = {"total_documents": 0}

        try:
            if self.postgis.connect():
                # Query all records from the table
                query = f"SELECT * FROM {table_name}"
                results = self.postgis.execute_query(query)

                documents = []
                for row in results:
                    # Create document from PostGIS data
                    content = self._postgis_row_to_text(row)
                    metadata = {
                        "source": "postgis",
                        "table": table_name,
                        "id": row.get("id", ""),
                        "latitude": row.get("latitude"),
                        "longitude": row.get("longitude"),
                        "entity": row.get("entity", ""),
                        "sub_entity": row.get("sub_entity", ""),
                    }

                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

                if documents:
                    # Store in vector database
                    if self.use_chroma:
                        collection = self._get_or_create_chroma_collection(
                            f"postgis_{table_name}"
                        )
                        self._store_in_chroma(collection, documents)
                    else:
                        self._store_in_faiss(documents, f"postgis_{table_name}")

                    stats["total_documents"] = len(documents)
                    logger.info(
                        f"Loaded {len(documents)} documents from PostGIS {table_name}"
                    )

        except Exception as e:
            logger.error(f"Error loading PostGIS data: {e}")
        finally:
            self.postgis.disconnect()

        return stats

    def _entity_to_text(self, entity: Dict[str, Any], collection_name: str) -> str:
        """
        Convert CKG entity to text representation for embedding.

        Args:
            entity: Entity dictionary
            collection_name: Collection name

        Returns:
            Text representation
        """
        text_parts = []

        # Add entity name/title
        if "name" in entity:
            text_parts.append(f"Name: {entity['name']}")
        elif "title" in entity:
            text_parts.append(f"Title: {entity['title']}")

        # Add description
        if "description" in entity:
            text_parts.append(f"Description: {entity['description']}")

        # Add entity type
        text_parts.append(f"Type: {collection_name}")

        # Add other relevant fields
        for key, value in entity.items():
            if (
                key not in ["_key", "_id", "_rev", "name", "title", "description"]
                and value
            ):
                text_parts.append(f"{key}: {value}")

        return " | ".join(text_parts)

    def _postgis_row_to_text(self, row: Dict[str, Any]) -> str:
        """
        Convert PostGIS row to text representation for embedding.

        Args:
            row: Database row dictionary

        Returns:
            Text representation
        """
        text_parts = []

        # Add name and entity info
        if row.get("name"):
            text_parts.append(f"Name: {row['name']}")

        if row.get("entity"):
            text_parts.append(f"Entity: {row['entity']}")

        if row.get("sub_entity"):
            text_parts.append(f"Sub-entity: {row['sub_entity']}")

        # Add description
        if row.get("description"):
            text_parts.append(f"Description: {row['description']}")

        # Add location info
        if row.get("latitude") and row.get("longitude"):
            text_parts.append(f"Location: {row['latitude']}, {row['longitude']}")

        # Add source
        if row.get("source_url"):
            text_parts.append(f"Source: {row['source_url']}")

        return " | ".join(text_parts)

    def _store_in_chroma(
        self, collection: chromadb.Collection, documents: List[Document]
    ):
        """
        Store documents in ChromaDB collection.

        Args:
            collection: ChromaDB collection
            documents: List of documents to store
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [
            f"{meta.get('source', 'unknown')}_{meta.get('id', str(i))}"
            for i, meta in enumerate(metadatas)
        ]

        collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def _store_in_faiss(self, documents: List[Document], collection_name: str):
        """
        Store documents in FAISS vector store.

        Args:
            documents: List of documents to store
            collection_name: Name for the collection
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

    def similarity_search(
        self, query: str, k: int = 5, collection_name: Optional[str] = None
    ) -> List[Document]:
        """
        Perform similarity search across vector stores.

        Args:
            query: Search query
            k: Number of results to return
            collection_name: Specific collection to search (optional)

        Returns:
            List of relevant documents
        """
        if self.use_chroma:
            return self._chroma_similarity_search(query, k, collection_name)
        else:
            return self._faiss_similarity_search(query, k)

    def _chroma_similarity_search(
        self, query: str, k: int, collection_name: Optional[str]
    ) -> List[Document]:
        """
        Perform similarity search in ChromaDB.

        Args:
            query: Search query
            k: Number of results
            collection_name: Collection name

        Returns:
            List of documents
        """
        results = []

        if collection_name:
            collections = [collection_name]
        else:
            # Search all collections
            collections = [name for name in self.chroma_client.list_collections()]

        for col_name in collections:
            try:
                collection = self.chroma_client.get_collection(name=col_name)
                chroma_results = collection.query(query_texts=[query], n_results=k)

                for i, doc in enumerate(chroma_results["documents"][0]):
                    metadata = chroma_results["metadatas"][0][i]
                    results.append(Document(page_content=doc, metadata=metadata))
            except Exception as e:
                logger.warning(f"Error searching collection {col_name}: {e}")

        # Sort by relevance and return top k
        return results[:k]

    def _faiss_similarity_search(self, query: str, k: int) -> List[Document]:
        """
        Perform similarity search in FAISS.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of documents
        """
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collections.

        Returns:
            Dictionary with collection statistics
        """
        stats = {}

        if self.use_chroma:
            collections = self.chroma_client.list_collections()
            for collection_name in collections:
                col = self.chroma_client.get_collection(name=collection_name)
                stats[collection_name] = {"count": col.count()}
        else:
            stats["faiss"] = {"initialized": self.vector_store is not None}

        return stats
