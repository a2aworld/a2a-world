"""
Data Gateway Agents for Terra Constellata

This module contains the base classes and implementations for data gateway agents
that provide authenticated access to external data sources and APIs.
"""

from .base_data_gateway_agent import DataGatewayAgent, DataGatewayAgentRegistry

__all__ = ["DataGatewayAgent", "DataGatewayAgentRegistry"]