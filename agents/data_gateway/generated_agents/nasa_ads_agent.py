"""
NASA_ADS_AGENT - Data Gateway Agent for Terra Constellata

Specialized agent for accessing Astrophysics data from NASA.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_data_gateway_agent import DataGatewayAgent

logger = logging.getLogger(__name__)


class NasaAds(DataGatewayAgent):
    """
    NASA_ADS_AGENT data gateway agent.

    Provides access to Astrophysics data from NASA.

    Capabilities: search_publications, get_object_data
    """

    def __init__(self, llm, tools: List = None, **kwargs):
        """
        Initialize the NASA_ADS_AGENT agent.

        Args:
            llm: Language model for the agent
            tools: List of tools (auto-generated if None)
            **kwargs: Additional configuration
        """
        # Default configuration
        default_config = {
            "base_url": "https://ui.adsabs.harvard.edu/",
            "api_key": "{{SECRETS.NASA_ADS_AGENT_API_KEY}}",
            "authentication_methods": ["api_key"],
            "provenance_level": "CANONICAL",
            "version": "1.0.0",
            "capabilities": ['search_publications', 'get_object_data'],
        }

        # Merge with provided kwargs
        config_merged = {**default_config, **kwargs}

        super().__init__(
            agent_name="NASA_ADS_AGENT",
            data_domain="Astrophysics",
            data_set_owner={'name': 'NASA', 'ownerType': 'Government Agency', 'officialContactUri': 'https://ui.adsabs.harvard.edu/'},
            llm=llm,
            tools=tools or self._get_default_tools(),
            **config_merged
        )

        logger.info(f"Initialized NASA_ADS_AGENT for Astrophysics data access")

    def _get_default_tools(self) -> List:
        """Get default tools for this agent."""
        # Tools will be implemented in specialized agent classes
        return []

    async def _execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a specific capability against the NASA ADS API.

        Args:
            capability: Capability name
            **kwargs: Capability parameters

        Returns:
            API response data
        """
        if capability == "search_publications":
            return await self._search_publications(**kwargs)
        elif capability == "get_object_data":
            return await self._get_object_data(**kwargs)
        else:
            raise NotImplementedError(f"Capability '{capability}' not implemented for NASA_ADS_AGENT")

    async def _search_publications(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search NASA ADS publications.

        Args:
            query: Search query
            **kwargs: Additional search parameters (year, author, etc.)

        Returns:
            Publication search results
        """
        endpoint = "search/query"
        params = {
            "q": query,
            "fl": "title,author,year,bibcode,abstract",  # Fields to return
            "rows": kwargs.get("limit", 10),
            "start": kwargs.get("start", 0),
        }

        # Add optional filters
        if "year" in kwargs:
            params["year"] = kwargs["year"]
        if "author" in kwargs:
            params["author"] = kwargs["author"]

        try:
            result = await self._make_api_request("GET", endpoint, params=params)
            return {
                "query": query,
                "total_results": result.get("response", {}).get("numFound", 0),
                "publications": result.get("response", {}).get("docs", []),
                "facets": result.get("facet_counts", {}),
                "source": "NASA_ADS"
            }
        except Exception as e:
            logger.warning(f"NASA ADS API unavailable, returning mock data: {e}")
            return {
                "query": query,
                "total_results": 1,
                "publications": [{
                    "title": "Sample Astrophysics Publication",
                    "author": ["Sample Author"],
                    "year": "2023",
                    "bibcode": "2023Sample",
                    "abstract": "This is mock data for demonstration purposes."
                }],
                "source": "NASA_ADS_MOCK",
                "note": "Mock data - API integration required"
            }

    async def _get_object_data(self, bibcode: str, **kwargs) -> Dict[str, Any]:
        """
        Get detailed data for a specific publication.

        Args:
            bibcode: ADS bibcode identifier

        Returns:
            Publication details
        """
        endpoint = f"search/query"
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "title,author,year,abstract,keyword,doi,citation_count",
        }

        try:
            result = await self._make_api_request("GET", endpoint, params=params)
            docs = result.get("response", {}).get("docs", [])
            if docs:
                pub = docs[0]
                return {
                    "bibcode": bibcode,
                    "title": pub.get("title", [""])[0],
                    "authors": pub.get("author", []),
                    "year": pub.get("year", ""),
                    "abstract": pub.get("abstract", ""),
                    "keywords": pub.get("keyword", []),
                    "doi": pub.get("doi", ""),
                    "citation_count": pub.get("citation_count", 0),
                    "source": "NASA_ADS"
                }
            else:
                raise ValueError(f"Publication not found: {bibcode}")
        except Exception as e:
            logger.warning(f"NASA ADS API unavailable, returning mock data: {e}")
            return {
                "bibcode": bibcode,
                "title": "Sample Publication Title",
                "authors": ["Sample Author"],
                "year": "2023",
                "abstract": "Mock abstract for demonstration.",
                "keywords": ["astrophysics", "sample"],
                "doi": "",
                "citation_count": 0,
                "source": "NASA_ADS_MOCK",
                "note": "Mock data - API integration required"
            }


# Register agent in global registry
from .base_data_gateway_agent import data_gateway_registry

# Note: Actual instantiation requires LLM and tools
# This is handled by the agent factory
