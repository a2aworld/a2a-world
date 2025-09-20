"""
Vector Store Integration for Tool Discovery

Uses ChromaDB for storing and retrieving tool embeddings for semantic search.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path

from .models import Tool, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class ToolVectorStore:
    """
    Vector database for tool storage and semantic search.

    Uses ChromaDB as the backend and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "./tool_shed_db",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "tool_shed",
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            model_name: Sentence transformer model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize sentence transformer
        self.embedding_model = SentenceTransformer(model_name)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)

        logger.info(f"Initialized ToolVectorStore with collection: {collection_name}")

    def add_tool(self, tool: Tool) -> bool:
        """
        Add a tool to the vector store.

        Args:
            tool: Tool to add

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for the tool
            text_to_embed = f"{tool.metadata.name} {tool.metadata.description} {' '.join(tool.capabilities.functions)}"
            embedding = self.embedding_model.encode(text_to_embed).tolist()

            # Update tool with embedding
            tool.embedding = embedding

            # Prepare metadata for ChromaDB
            metadata = {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "author": tool.metadata.author,
                "version": tool.metadata.version,
                "category": tool.metadata.category,
                "tags": json.dumps(tool.metadata.tags),
                "security_level": tool.capabilities.security_level,
                "is_active": str(tool.is_active),
                "rating": str(tool.rating),
                "usage_count": str(tool.usage_count),
            }

            # Add to collection
            self.collection.add(
                ids=[tool.id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[tool.documentation],
            )

            logger.info(f"Added tool {tool.metadata.name} to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add tool {tool.metadata.name}: {e}")
            return False

    def update_tool(self, tool: Tool) -> bool:
        """
        Update an existing tool in the vector store.

        Args:
            tool: Updated tool

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove old version
            self.collection.delete(ids=[tool.id])

            # Add updated version
            return self.add_tool(tool)

        except Exception as e:
            logger.error(f"Failed to update tool {tool.metadata.name}: {e}")
            return False

    def remove_tool(self, tool_id: str) -> bool:
        """
        Remove a tool from the vector store.

        Args:
            tool_id: ID of tool to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[tool_id])
            logger.info(f"Removed tool {tool_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to remove tool {tool_id}: {e}")
            return False

    def search_tools(self, query: SearchQuery) -> SearchResult:
        """
        Search for tools using semantic similarity and metadata filters.

        Args:
            query: Search query parameters

        Returns:
            SearchResult with matching tools
        """
        import time

        start_time = time.time()

        try:
            # Build where clause for metadata filtering
            where_clause = {}

            if query.category:
                where_clause["category"] = query.category

            if query.author:
                where_clause["author"] = query.author

            if query.security_level:
                where_clause["security_level"] = query.security_level

            if query.min_rating is not None:
                where_clause["$and"] = [{"rating": {"$gte": str(query.min_rating)}}]

            # Add tag filtering if specified
            if query.tags:
                # ChromaDB doesn't support complex tag filtering easily
                # We'll filter after retrieval
                pass

            # Perform semantic search
            results = self.collection.query(
                query_texts=[query.query],
                n_results=query.limit,
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"],
            )

            # Process results
            tools = []
            semantic_matches = []

            if results["ids"]:
                for i, tool_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    document = results["documents"][0][i]
                    distance = results["distances"][0][i]

                    # Apply tag filtering if needed
                    if query.tags:
                        tool_tags = json.loads(metadata.get("tags", "[]"))
                        if not any(tag in tool_tags for tag in query.tags):
                            continue

                    # Create tool object (simplified for search results)
                    tool = Tool(
                        id=tool_id,
                        metadata={
                            "name": metadata["name"],
                            "description": metadata["description"],
                            "author": metadata["author"],
                            "version": metadata["version"],
                            "category": metadata["category"],
                            "tags": json.loads(metadata.get("tags", "[]")),
                        },
                        capabilities={},  # Would need full retrieval for this
                        code="",  # Not stored in vector DB
                        documentation=document,
                    )

                    tools.append(tool)
                    semantic_matches.append(
                        {
                            "tool_id": tool_id,
                            "similarity_score": 1.0
                            - distance,  # Convert distance to similarity
                            "metadata": metadata,
                        }
                    )

            query_time = time.time() - start_time

            return SearchResult(
                tools=tools,
                total_count=len(tools),
                query_time=query_time,
                semantic_matches=semantic_matches,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(
                tools=[],
                total_count=0,
                query_time=time.time() - start_time,
                semantic_matches=[],
            )

    def get_tool_embedding(self, tool_id: str) -> Optional[List[float]]:
        """
        Get the embedding for a specific tool.

        Args:
            tool_id: ID of the tool

        Returns:
            Embedding vector or None if not found
        """
        try:
            result = self.collection.get(ids=[tool_id], include=["embeddings"])
            if result["embeddings"]:
                return result["embeddings"][0]
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding for {tool_id}: {e}")
            return None

    def get_similar_tools(self, tool_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find tools similar to the given tool.

        Args:
            tool_id: ID of the reference tool
            limit: Maximum number of similar tools to return

        Returns:
            List of similar tools with similarity scores
        """
        embedding = self.get_tool_embedding(tool_id)
        if not embedding:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit + 1,  # +1 to exclude the tool itself
                include=["metadatas", "distances"],
            )

            similar_tools = []
            if results["ids"]:
                for i, similar_id in enumerate(results["ids"][0]):
                    if similar_id != tool_id:  # Exclude the tool itself
                        distance = results["distances"][0][i]
                        metadata = results["metadatas"][0][i]

                        similar_tools.append(
                            {
                                "tool_id": similar_id,
                                "similarity_score": 1.0 - distance,
                                "metadata": metadata,
                            }
                        )

                        if len(similar_tools) >= limit:
                            break

            return similar_tools

        except Exception as e:
            logger.error(f"Failed to find similar tools for {tool_id}: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            count = self.collection.count()
            return {
                "total_tools": count,
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
