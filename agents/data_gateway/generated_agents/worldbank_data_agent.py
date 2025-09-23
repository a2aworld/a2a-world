"""
WORLDBANK_DATA_AGENT - Data Gateway Agent for Terra Constellata

Specialized agent for accessing Economics data from The World Bank.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_data_gateway_agent import DataGatewayAgent

logger = logging.getLogger(__name__)


class WorldbankData(DataGatewayAgent):
    """
    WORLDBANK_DATA_AGENT data gateway agent.

    Provides access to Economics data from The World Bank.

    Capabilities: get_indicator_data_by_country
    """

    def __init__(self, llm, tools: List = None, **kwargs):
        """
        Initialize the WORLDBANK_DATA_AGENT agent.

        Args:
            llm: Language model for the agent
            tools: List of tools (auto-generated if None)
            **kwargs: Additional configuration
        """
        # Default configuration
        default_config = {
            "base_url": "https://data.worldbank.org/",
            "api_key": "{{SECRETS.WORLDBANK_DATA_AGENT_API_KEY}}",
            "authentication_methods": ["api_key"],
            "provenance_level": "CANONICAL",
            "version": "1.0.0",
            "capabilities": ['get_indicator_data_by_country'],
        }

        # Merge with provided kwargs
        config_merged = {**default_config, **kwargs}

        super().__init__(
            agent_name="WORLDBANK_DATA_AGENT",
            data_domain="Economics",
            data_set_owner={'name': 'The World Bank', 'ownerType': 'Non-Profit', 'officialContactUri': 'https://data.worldbank.org/'},
            llm=llm,
            tools=tools or self._get_default_tools(),
            **config_merged
        )

        logger.info(f"Initialized WORLDBANK_DATA_AGENT for Economics data access")

    def _get_default_tools(self) -> List:
        """Get default tools for this agent."""
        # Tools will be implemented in specialized agent classes
        return []

    async def _execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a specific capability against the The World Bank API.

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
