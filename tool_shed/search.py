"""
Semantic Search System for Tool Discovery

Advanced search capabilities for finding relevant tools using semantic similarity,
metadata filtering, and hybrid search approaches.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .models import SearchQuery, SearchResult, Tool
from .vector_store import ToolVectorStore
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Advanced semantic search engine for tool discovery.

    Combines vector similarity search with metadata filtering and
    ranking algorithms for optimal tool discovery.
    """

    def __init__(self, registry: ToolRegistry, vector_store: ToolVectorStore):
        """
        Initialize the search engine.

        Args:
            registry: Tool registry instance
            vector_store: Vector store instance
        """
        self.registry = registry
        self.vector_store = vector_store

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Perform semantic search for tools.

        Args:
            query: Search query parameters

        Returns:
            Search results with ranking
        """
        import time

        start_time = time.time()

        try:
            # Get initial results from vector store
            vector_results = self.vector_store.search_tools(query)

            # Apply additional filtering and ranking
            filtered_tools = self._apply_filters(vector_results.tools, query)

            # Re-rank results based on multiple factors
            ranked_tools = self._rank_results(filtered_tools, query.query)

            # Add semantic matches from vector search
            semantic_matches = vector_results.semantic_matches

            # Create final result
            result = SearchResult(
                tools=ranked_tools,
                total_count=len(ranked_tools),
                query_time=time.time() - start_time,
                semantic_matches=semantic_matches,
            )

            logger.info(
                f"Search completed in {result.query_time:.3f}s, found {result.total_count} tools"
            )
            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(
                tools=[],
                total_count=0,
                query_time=time.time() - start_time,
                semantic_matches=[],
            )

    def _apply_filters(self, tools: List[Tool], query: SearchQuery) -> List[Tool]:
        """
        Apply additional filters to search results.

        Args:
            tools: Initial tool results
            query: Search query with filters

        Returns:
            Filtered list of tools
        """
        filtered = tools

        # Filter by rating
        if query.min_rating is not None:
            filtered = [t for t in filtered if t.rating >= query.min_rating]

        # Filter by security level
        if query.security_level:
            filtered = [
                t
                for t in filtered
                if t.capabilities.security_level == query.security_level
            ]

        # Filter by tags (if not already done by vector store)
        if query.tags:
            filtered = [
                t for t in filtered if any(tag in t.metadata.tags for tag in query.tags)
            ]

        # Filter by recency (last updated within X days)
        # This could be extended with more date filters

        return filtered

    def _rank_results(self, tools: List[Tool], query_text: str) -> List[Tool]:
        """
        Rank search results based on multiple relevance factors.

        Args:
            tools: Tools to rank
            query_text: Original search query

        Returns:
            Ranked list of tools
        """
        if not tools:
            return tools

        # Calculate relevance scores
        tool_scores = []

        for tool in tools:
            score = self._calculate_relevance_score(tool, query_text)
            tool_scores.append((tool, score))

        # Sort by score (descending)
        tool_scores.sort(key=lambda x: x[1], reverse=True)

        return [tool for tool, _ in tool_scores]

    def _calculate_relevance_score(self, tool: Tool, query: str) -> float:
        """
        Calculate relevance score for a tool based on multiple factors.

        Args:
            tool: Tool to score
            query: Search query

        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        total_weight = 0.0

        # Semantic similarity (from vector search) - weight: 0.4
        # This would be available from the vector store results
        semantic_weight = 0.4
        # For now, use a placeholder - in real implementation,
        # this would come from the vector similarity score
        semantic_score = 0.5  # Placeholder
        score += semantic_score * semantic_weight
        total_weight += semantic_weight

        # Metadata relevance - weight: 0.3
        metadata_weight = 0.3
        metadata_score = self._calculate_metadata_relevance(tool, query)
        score += metadata_score * metadata_weight
        total_weight += metadata_weight

        # Usage/popularity - weight: 0.15
        usage_weight = 0.15
        usage_score = min(tool.usage_count / 100.0, 1.0)  # Normalize
        score += usage_score * usage_weight
        total_weight += usage_weight

        # Rating - weight: 0.15
        rating_weight = 0.15
        rating_score = tool.rating / 5.0  # Normalize to 0-1
        score += rating_score * rating_weight
        total_weight += rating_weight

        # Normalize final score
        if total_weight > 0:
            score = score / total_weight

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_metadata_relevance(self, tool: Tool, query: str) -> float:
        """
        Calculate relevance based on metadata matching.

        Args:
            tool: Tool to evaluate
            query: Search query

        Returns:
            Metadata relevance score (0-1)
        """
        query_lower = query.lower()
        score = 0.0

        # Name matching
        if query_lower in tool.metadata.name.lower():
            score += 0.3

        # Description matching
        if query_lower in tool.metadata.description.lower():
            score += 0.2

        # Tag matching
        matching_tags = [
            tag for tag in tool.metadata.tags if query_lower in tag.lower()
        ]
        if matching_tags:
            score += 0.2 * min(len(matching_tags) / len(tool.metadata.tags), 1.0)

        # Category matching
        if query_lower in tool.metadata.category.lower():
            score += 0.2

        # Function matching
        matching_functions = [
            func for func in tool.capabilities.functions if query_lower in func.lower()
        ]
        if matching_functions:
            score += 0.1 * min(
                len(matching_functions) / len(tool.capabilities.functions), 1.0
            )

        return min(score, 1.0)

    def find_similar_tools(self, tool_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find tools similar to a given tool.

        Args:
            tool_id: ID of the reference tool
            limit: Maximum number of similar tools

        Returns:
            List of similar tools with similarity info
        """
        return self.vector_store.get_similar_tools(tool_id, limit)

    def suggest_tools(
        self, context: str, required_capabilities: List[str] = None
    ) -> List[Tool]:
        """
        Suggest tools based on context and required capabilities.

        Args:
            context: Usage context description
            required_capabilities: List of required capabilities

        Returns:
            Suggested tools
        """
        # Create a search query based on context
        query = SearchQuery(query=context, limit=10)

        results = self.search(query)
        tools = results.tools

        if required_capabilities:
            # Filter tools that have the required capabilities
            filtered_tools = []
            for tool in tools:
                if any(
                    cap in tool.capabilities.functions for cap in required_capabilities
                ):
                    filtered_tools.append(tool)
            tools = filtered_tools

        return tools[:5]  # Return top 5 suggestions

    def get_trending_tools(self, days: int = 7, limit: int = 10) -> List[Tool]:
        """
        Get trending tools based on recent usage.

        Args:
            days: Number of days to look back
            limit: Maximum number of tools to return

        Returns:
            List of trending tools
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get all tools and sort by recent usage
        all_tools = self.registry.list_tools(active_only=True)

        # Filter by recent activity
        recent_tools = []
        for tool in all_tools:
            # Check if tool was accessed recently
            # This would require tracking access timestamps
            # For now, use usage count as proxy
            if tool.usage_count > 0:
                recent_tools.append(tool)

        # Sort by usage count (descending)
        recent_tools.sort(key=lambda t: t.usage_count, reverse=True)

        return recent_tools[:limit]

    def get_tools_by_category(
        self, category: str, sort_by: str = "rating"
    ) -> List[Tool]:
        """
        Get tools by category with sorting.

        Args:
            category: Tool category
            sort_by: Sort criteria ("rating", "usage", "name")

        Returns:
            Sorted list of tools in category
        """
        tools = self.registry.list_tools(category=category, active_only=True)

        if sort_by == "rating":
            tools.sort(key=lambda t: t.rating, reverse=True)
        elif sort_by == "usage":
            tools.sort(key=lambda t: t.usage_count, reverse=True)
        elif sort_by == "name":
            tools.sort(key=lambda t: t.metadata.name.lower())

        return tools

    def get_tools_by_author(self, author: str, sort_by: str = "recent") -> List[Tool]:
        """
        Get tools by author with sorting.

        Args:
            author: Tool author
            sort_by: Sort criteria ("recent", "rating", "usage")

        Returns:
            Sorted list of tools by author
        """
        tools = self.registry.list_tools(author=author, active_only=True)

        if sort_by == "recent":
            tools.sort(key=lambda t: t.metadata.updated_at, reverse=True)
        elif sort_by == "rating":
            tools.sort(key=lambda t: t.rating, reverse=True)
        elif sort_by == "usage":
            tools.sort(key=lambda t: t.usage_count, reverse=True)

        return tools

    def advanced_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        sort_by: str = "relevance",
        limit: int = 20,
    ) -> SearchResult:
        """
        Perform advanced search with complex filters and sorting.

        Args:
            query: Search query
            filters: Additional filters
            sort_by: Sort criteria
            limit: Result limit

        Returns:
            Advanced search results
        """
        # Build search query
        search_query = SearchQuery(query=query, limit=limit)

        # Apply filters
        if filters:
            if "category" in filters:
                search_query.category = filters["category"]
            if "author" in filters:
                search_query.author = filters["author"]
            if "tags" in filters:
                search_query.tags = filters["tags"]
            if "min_rating" in filters:
                search_query.min_rating = filters["min_rating"]
            if "security_level" in filters:
                search_query.security_level = filters["security_level"]

        # Perform search
        results = self.search(search_query)

        # Apply additional sorting if needed
        if sort_by != "relevance":
            if sort_by == "rating":
                results.tools.sort(key=lambda t: t.rating, reverse=True)
            elif sort_by == "usage":
                results.tools.sort(key=lambda t: t.usage_count, reverse=True)
            elif sort_by == "recent":
                results.tools.sort(key=lambda t: t.metadata.updated_at, reverse=True)

        return results
