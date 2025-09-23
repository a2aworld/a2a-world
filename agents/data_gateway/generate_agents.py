#!/usr/bin/env python3
"""
Agent Code Generation Script for Terra Constellata Data Gateway Agents

This script generates the 50 foundational data gateway agent classes
from the manifest configuration.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentCodeGenerator:
    """Generates agent classes from manifest configuration."""

    def __init__(self, manifest_path: str, output_dir: str):
        """
        Initialize the code generator.

        Args:
            manifest_path: Path to the agent manifest JSON file
            output_dir: Directory to output generated agent classes
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded manifest with {len(self.manifest)} agents")

    def generate_all_agents(self):
        """Generate all agent classes from the manifest."""
        generated_files = []

        for agent_config in self.manifest:
            agent_file = self.generate_agent_class(agent_config)
            generated_files.append(agent_file)

        # Generate __init__.py for agents package
        self.generate_agents_init(generated_files)

        # Generate registry initialization
        self.generate_registry_init()

        logger.info(f"Generated {len(generated_files)} agent classes")

        return generated_files

    def generate_agent_class(self, config: Dict[str, Any]) -> str:
        """Generate a single agent class."""
        agent_name = config["agentName"]
        class_name = self._to_class_name(agent_name)
        file_name = f"{agent_name.lower()}.py"

        # Prepare data set owner dict
        data_set_owner = {
            "name": config["ownerName"],
            "ownerType": config["ownerType"],
            "officialContactUri": config["contactUri"]
        }

        # Generate class code
        class_code = f'''"""
{agent_name} - Data Gateway Agent for Terra Constellata

Specialized agent for accessing {config["dataDomain"]} data from {config["ownerName"]}.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_data_gateway_agent import DataGatewayAgent

logger = logging.getLogger(__name__)


class {class_name}(DataGatewayAgent):
    """
    {agent_name} data gateway agent.

    Provides access to {config["dataDomain"]} data from {config["ownerName"]}.

    Capabilities: {", ".join(config["capabilities"])}
    """

    def __init__(self, llm, tools: List = None, **kwargs):
        """
        Initialize the {agent_name} agent.

        Args:
            llm: Language model for the agent
            tools: List of tools (auto-generated if None)
            **kwargs: Additional configuration
        """
        # Default configuration
        default_config = {{
            "base_url": "{config["contactUri"]}",
            "api_key": "{{{{SECRETS.{agent_name}_API_KEY}}}}",
            "authentication_methods": ["api_key"],
            "provenance_level": "{config.get('provenanceLevel', 'CANONICAL')}",
            "version": "1.0.0",
            "capabilities": {config["capabilities"]},
        }}

        # Merge with provided kwargs
        config_merged = {{**default_config, **kwargs}}

        super().__init__(
            agent_name="{agent_name}",
            data_domain="{config["dataDomain"]}",
            data_set_owner={data_set_owner},
            llm=llm,
            tools=tools or self._get_default_tools(),
            **config_merged
        )

        logger.info(f"Initialized {agent_name} for {config["dataDomain"]} data access")

    def _get_default_tools(self) -> List:
        """Get default tools for this agent."""
        # Tools will be implemented in specialized agent classes
        return []

    async def _execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a specific capability against the {config["ownerName"]} API.

        This is a base implementation - specialized agents should override this method.

        Args:
            capability: Capability name
            **kwargs: Capability parameters

        Returns:
            API response data
        """
        # Default implementation - raise NotImplementedError
        # Specialized agents will implement specific capability logic
        raise NotImplementedError(f"Capability '{{capability}}' not implemented for {{agent_name}}")


# Register agent in global registry
from .base_data_gateway_agent import data_gateway_registry

# Note: Actual instantiation requires LLM and tools
# This is handled by the agent factory
'''

        # Write to file
        file_path = self.output_dir / file_name
        with open(file_path, 'w') as f:
            f.write(class_code)

        logger.info(f"Generated agent class: {class_name} -> {file_path}")
        return file_name

    def _to_class_name(self, agent_name: str) -> str:
        """Convert agent name to class name."""
        # Remove _AGENT suffix if present and convert to CamelCase
        name = agent_name.replace("_AGENT", "")
        parts = name.split("_")
        return "".join(word.capitalize() for word in parts)

    def _get_base_url(self, config: Dict[str, Any]) -> str:
        """Extract base URL from config."""
        contact_uri = config["contactUri"]
        # Simple extraction - can be overridden
        return contact_uri

    def generate_agents_init(self, generated_files: List[str]):
        """Generate __init__.py for the agents package."""
        imports = []
        all_exports = []

        for file_name in generated_files:
            module_name = file_name.replace(".py", "")
            class_name = self._to_class_name(module_name.upper().replace("_", "_"))

            imports.append(f"from .{module_name} import {class_name}")
            all_exports.append(class_name)

        init_code = f'''"""
Terra Constellata Data Gateway Agents

Generated agent classes for accessing external data sources.
"""

{chr(10).join(imports)}

__all__ = {all_exports}
'''

        init_path = self.output_dir / "__init__.py"
        with open(init_path, 'w') as f:
            f.write(init_code)

        logger.info(f"Generated agents __init__.py")

    def generate_registry_init(self):
        """Generate registry initialization script."""
        registry_code = f'''"""
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
    report = {{
        "timestamp": datetime.utcnow().isoformat(),
        "agents_initialized": [],
        "errors": []
    }}

    # List of all agent classes (generated dynamically)
    agent_classes = [
        {self._get_all_class_names()}
    ]

    for agent_class in agent_classes:
        try:
            # Instantiate agent
            agent = agent_class(llm=llm, tools=common_tools)

            # Register in global registry
            data_gateway_registry.register_agent(agent)

            report["agents_initialized"].append({{
                "name": agent.agent_name,
                "class": agent_class.__name__,
                "domain": agent.data_domain,
                "capabilities": agent.capabilities
            }})

            logger.info(f"Initialized and registered agent: {{agent.agent_name}}")

        except Exception as e:
            error_info = {{
                "class": agent_class.__name__,
                "error": str(e)
            }}
            report["errors"].append(error_info)
            logger.error(f"Failed to initialize agent {{agent_class.__name__}}: {{e}}")

    report["total_agents"] = len(report["agents_initialized"])
    report["total_errors"] = len(report["errors"])

    logger.info(f"Data gateway agents initialization complete: {{report['total_agents']}} agents, {{report['total_errors']}} errors")

    return report


def get_agent_manifest() -> Dict[str, Any]:
    """Get the complete agent manifest."""
    return {{
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(),
        "agents": [
            {{
                "name": agent.agent_name,
                "domain": agent.data_domain,
                "capabilities": agent.capabilities,
                "owner": agent.data_set_owner,
                "endpoint": agent.a2a_endpoint
            }}
            for agent in data_gateway_registry.agents.values()
        ]
    }}
'''

        registry_path = self.output_dir / "registry_init.py"
        with open(registry_path, 'w') as f:
            f.write(registry_code)

        logger.info(f"Generated registry initialization script")

    def _get_all_class_names(self) -> str:
        """Get all class names for registry init."""
        class_names = []
        for config in self.manifest:
            agent_name = config["agentName"]
            class_name = self._to_class_name(agent_name)
            class_names.append(class_name)
        return ",\n        ".join(class_names)


def main():
    """Main entry point for code generation."""
    # Configure paths
    script_dir = Path(__file__).parent
    manifest_path = script_dir / "foundational_agents_manifest.json"
    output_dir = script_dir / "generated_agents"

    # Generate agents
    generator = AgentCodeGenerator(manifest_path, output_dir)
    generated_files = generator.generate_all_agents()

    print(f"Successfully generated {len(generated_files)} agent classes in {output_dir}")


if __name__ == "__main__":
    main()