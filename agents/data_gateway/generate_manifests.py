#!/usr/bin/env python3
"""
Agent Registration Manifest Generator for Terra Constellata

This script generates registration manifests for all data gateway agents
conforming to the adapted A2A_World_VIA_Agent_Schema_v1.0.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManifestGenerator:
    """Generates agent registration manifests."""

    def __init__(self, manifest_path: str, output_dir: str):
        """
        Initialize the manifest generator.

        Args:
            manifest_path: Path to the agent manifest JSON file
            output_dir: Directory to output registration manifests
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded manifest with {len(self.manifest)} agents")

    def generate_all_manifests(self) -> List[str]:
        """Generate registration manifests for all agents."""
        manifests = []

        for agent_config in self.manifest:
            manifest_file = self.generate_agent_manifest(agent_config)
            manifests.append(manifest_file)

        # Generate master registry manifest
        self.generate_master_registry()

        logger.info(f"Generated {len(manifests)} agent manifests")
        return manifests

    def generate_agent_manifest(self, config: Dict[str, Any]) -> str:
        """Generate registration manifest for a single agent."""
        agent_name = config["agentName"]

        # Create manifest conforming to adapted schema
        manifest = {
            "agentId": str(uuid.uuid4()),  # Generate unique ID
            "viaStatus": "AUTHENTICATED",  # All agents start as authenticated
            "agentName": agent_name,
            "dataSetOwner": {
                "name": config["ownerName"],
                "ownerType": config["ownerType"],
                "officialContactUri": config["contactUri"]
            },
            "dataDomain": config["dataDomain"],
            "a2aEndpoint": f"http://localhost:8080/agents/{agent_name}",  # Default A2A endpoint
            "authenticationMethods": ["api_key"],  # Default auth methods
            "provenanceLevel": config.get("provenanceLevel", "CANONICAL"),
            "version": "1.0.0",
            "lastAuthenticatedTimestamp": datetime.utcnow().isoformat(),
            "capabilities": [
                {
                    "capabilityId": cap,
                    "description": f"Execute {cap} capability for {config['dataDomain']} data",
                    "queryFormats": self._get_query_formats(cap, config["dataDomain"])
                }
                for cap in config["capabilities"]
            ],
            # Terra Constellata specific fields
            "terra_constellata": {
                "agent_type": "data_gateway",
                "framework_version": "1.0.0",
                "deployment_status": "ready",
                "monitoring_enabled": True,
                "health_check_endpoint": f"/agents/{agent_name}/health",
                "registration_timestamp": datetime.utcnow().isoformat()
            }
        }

        # Write to file
        file_name = f"{agent_name.lower()}_registration.json"
        file_path = self.output_dir / file_name

        with open(file_path, 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated manifest: {file_path}")
        return file_name

    def _get_query_formats(self, capability: str, domain: str) -> List[str]:
        """Determine appropriate query formats for a capability."""
        # Default formats based on domain
        domain_formats = {
            "Geospatial": ["geojson", "json", "wkt"],
            "Climatology": ["json", "netcdf", "csv"],
            "Cultural Heritage": ["json", "xml", "html"],
            "Scientific": ["json", "bibtex", "xml"],
            "Linguistics": ["json", "xml"],
            "Infrastructure": ["json"]
        }

        base_formats = domain_formats.get(domain, ["json"])

        # Add capability-specific formats
        if "spatial" in capability.lower() or "geo" in capability.lower():
            base_formats.extend(["geojson", "wkt"])
        elif "search" in capability.lower():
            base_formats.extend(["xml"])

        # Remove duplicates
        return list(set(base_formats))

    def generate_master_registry(self):
        """Generate master registry manifest containing all agents."""
        master_manifest = {
            "registry_name": "Terra Constellata Data Gateway Agents Registry",
            "schema_version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "total_agents": len(self.manifest),
            "agents": []
        }

        # Add summary for each agent
        for config in self.manifest:
            agent_summary = {
                "agentId": str(uuid.uuid4()),  # Would be same as individual manifest
                "agentName": config["agentName"],
                "dataDomain": config["dataDomain"],
                "ownerName": config["ownerName"],
                "capabilities": config["capabilities"],
                "status": "registered",
                "registration_file": f"{config['agentName'].lower()}_registration.json"
            }
            master_manifest["agents"].append(agent_summary)

        # Group by domain
        domain_groups = {}
        for agent in master_manifest["agents"]:
            domain = agent["dataDomain"]
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(agent["agentName"])

        master_manifest["domain_groups"] = domain_groups
        master_manifest["domains"] = list(domain_groups.keys())

        # Write master registry
        master_path = self.output_dir / "master_registry.json"
        with open(master_path, 'w') as f:
            json.dump(master_manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated master registry: {master_path}")

    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate a generated manifest against basic schema requirements."""
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Check required fields
            required_fields = [
                "agentId", "viaStatus", "agentName", "dataSetOwner",
                "dataDomain", "a2aEndpoint", "capabilities"
            ]

            for field in required_fields:
                if field not in manifest:
                    logger.error(f"Missing required field '{field}' in {manifest_path}")
                    return False

            # Validate capabilities structure
            for cap in manifest["capabilities"]:
                if not all(k in cap for k in ["capabilityId", "description", "queryFormats"]):
                    logger.error(f"Invalid capability structure in {manifest_path}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating manifest {manifest_path}: {e}")
            return False


def main():
    """Main entry point for manifest generation."""
    # Configure paths
    script_dir = Path(__file__).parent
    manifest_path = script_dir / "foundational_agents_manifest.json"
    output_dir = script_dir / "registration_manifests"

    # Generate manifests
    generator = ManifestGenerator(manifest_path, output_dir)
    generated_files = generator.generate_all_manifests()

    # Validate generated manifests
    valid_count = 0
    for file_name in generated_files:
        file_path = output_dir / file_name
        if generator.validate_manifest(file_path):
            valid_count += 1
        else:
            logger.error(f"Validation failed for {file_name}")

    print(f"Successfully generated {len(generated_files)} manifests ({valid_count} valid)")


if __name__ == "__main__":
    main()