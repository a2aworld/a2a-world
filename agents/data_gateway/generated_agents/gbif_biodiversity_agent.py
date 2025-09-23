"""
GBIF_BIODIVERSITY_AGENT - Data Gateway Agent for Terra Constellata

Specialized agent for accessing Biology data from Global Biodiversity Information Facility.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_data_gateway_agent import DataGatewayAgent

logger = logging.getLogger(__name__)


class GbifBiodiversity(DataGatewayAgent):
    """
    GBIF_BIODIVERSITY_AGENT data gateway agent.

    Provides access to Biology data from Global Biodiversity Information Facility.

    Capabilities: get_species_occurrences_by_region
    """

    def __init__(self, llm, tools: List = None, **kwargs):
        """
        Initialize the GBIF_BIODIVERSITY_AGENT agent.

        Args:
            llm: Language model for the agent
            tools: List of tools (auto-generated if None)
            **kwargs: Additional configuration
        """
        # Default configuration
        default_config = {
            "base_url": "https://www.gbif.org/",
            "api_key": "{{SECRETS.GBIF_BIODIVERSITY_AGENT_API_KEY}}",
            "authentication_methods": ["api_key"],
            "provenance_level": "CANONICAL",
            "version": "1.0.0",
            "capabilities": ['get_species_occurrences_by_region'],
        }

        # Merge with provided kwargs
        config_merged = {**default_config, **kwargs}

        super().__init__(
            agent_name="GBIF_BIODIVERSITY_AGENT",
            data_domain="Biology",
            data_set_owner={'name': 'Global Biodiversity Information Facility', 'ownerType': 'Non-Profit', 'officialContactUri': 'https://www.gbif.org/'},
            llm=llm,
            tools=tools or self._get_default_tools(),
            **config_merged
        )

        logger.info(f"Initialized GBIF_BIODIVERSITY_AGENT for Biology data access")

    def _get_default_tools(self) -> List:
        """Get default tools for this agent."""
        # Tools will be implemented in specialized agent classes
        return []

    async def _execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a specific capability against the Global Biodiversity Information Facility API.

        This is a base implementation - specialized agents should override this method.

        Args:
            capability: Capability name
            **kwargs: Capability parameters

        Returns:
            API response data
        """
        # Default implementation - raise NotImplementedError
        # Specialized agents will implement specific capability logic
        raise NotImplementedError(f"Capability '{capability}' not implemented for {agent_name}")


# Register agent in global registry
from .base_data_gateway_agent import data_gateway_registry

# Note: Actual instantiation requires LLM and tools
# This is handled by the agent factory
