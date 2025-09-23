"""
Registry Initialization for Data Gateway Agents

This script initializes and registers all data gateway agents.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from .base_data_gateway_agent import data_gateway_registry
from . import *  # Import all agent classes

logger = logging.getLogger(__name__)


def initialize_data_gateway_agents(llm, common_tools: List = None) -> Dict[str, Any]:
    """
    Initialize and register all data gateway agents.

    Args:
        llm: Language model instance
        common_tools: Common tools to add to all agents

    Returns:
        Initialization report
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "agents_initialized": [],
        "errors": []
    }

    # List of all agent classes (generated dynamically)
    agent_classes = [
        GebcoBathymetry,
        NasaLandsat,
        EsaSentinel,
        UsgsSeismic,
        NoaaClimate,
        EcmwfEra5,
        Usgs3dep,
        AsterGdem,
        NasaWmm,
        GhslSettlement,
        Onegeology,
        Openstreetmap,
        Naturalearth,
        NoaaPaleoclimate,
        WikidataKnowledge,
        DplaHeritage,
        EuropeanaHeritage,
        Internetarchive,
        ProjectGutenberg,
        LocChronamerica,
        OpencontextArchaeology,
        PleiadesPlaces,
        Sacredtexts,
        SefariaTalmud,
        PaliCanon,
        DavidRumseyMaps,
        MetmuseumArt,
        BritishmuseumCollections,
        GlottologLanguages,
        WordnetSemantics,
        IconclassSymbols,
        GettyAat,
        GettyTgn,
        GettyUlan,
        GettyIconography,
        WiktionaryEtymology,
        ClicsColexification,
        AtuMotifIndex,
        NasaAds,
        GbifBiodiversity,
        CdcHealthdata,
        WorldbankData,
        Pubchem,
        A2aworldRegistry,
        A2aworldOrchestrator,
        A2aworldValidator,
        A2aworldReputation,
        A2aworldButler,
        A2aworldOntology,
        A2aworldNews
    ]

    for agent_class in agent_classes:
        try:
            # Instantiate agent
            agent = agent_class(llm=llm, tools=common_tools)

            # Register in global registry
            data_gateway_registry.register_agent(agent)

            report["agents_initialized"].append({
                "name": agent.agent_name,
                "class": agent_class.__name__,
                "domain": agent.data_domain,
                "capabilities": agent.capabilities
            })

            logger.info(f"Initialized and registered agent: {agent.agent_name}")

        except Exception as e:
            error_info = {
                "class": agent_class.__name__,
                "error": str(e)
            }
            report["errors"].append(error_info)
            logger.error(f"Failed to initialize agent {agent_class.__name__}: {e}")

    report["total_agents"] = len(report["agents_initialized"])
    report["total_errors"] = len(report["errors"])

    logger.info(f"Data gateway agents initialization complete: {report['total_agents']} agents, {report['total_errors']} errors")

    return report


def get_agent_manifest() -> Dict[str, Any]:
    """Get the complete agent manifest."""
    return {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "agents": [
            {
                "name": agent.agent_name,
                "domain": agent.data_domain,
                "capabilities": agent.capabilities,
                "owner": agent.data_set_owner,
                "endpoint": agent.a2a_endpoint
            }
            for agent in data_gateway_registry.agents.values()
        ]
    }
