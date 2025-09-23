#!/usr/bin/env python3
"""
Generate OpenAPI 3.0 specifications for Terra Constellata Data Gateway Agents
"""

import json
import yaml
from typing import Dict, List, Any
from pathlib import Path


def generate_openapi_spec(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0 specification for a data gateway agent.

    Args:
        agent_config: Agent configuration dictionary

    Returns:
        OpenAPI 3.0 specification dictionary
    """
    agent_name = agent_config["agentName"]
    agent_name_lower = agent_name.lower()
    data_domain = agent_config["dataDomain"]
    capabilities = agent_config.get("capabilities", [])

    # Base OpenAPI structure
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": f"Terra Constellata {agent_name} Data Gateway Agent API",
            "description": f"Authenticated access to {data_domain} data through the {agent_name} agent. Part of the Terra Constellata 'Library of Alexandria for AI Wisdom' ecosystem.",
            "version": agent_config.get("version", "1.0.0"),
            "contact": {
                "name": "Terra Constellata Development Team",
                "url": "https://terra-constellata.ai"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": f"https://api.terra-constellata.ai/agents/{agent_name}",
                "description": "Production server"
            },
            {
                "url": "http://localhost:8080/agents/{agent_name}",
                "description": "Local development server"
            }
        ],
        "security": [
            {
                "ApiKeyAuth": []
            },
            {
                "BearerAuth": []
            }
        ],
        "paths": {},
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message"
                        },
                        "code": {
                            "type": "integer",
                            "description": "Error code"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "HealthCheck": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Unique agent identifier"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Agent name"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "unhealthy", "a2a_healthy_api_unhealthy"],
                            "description": "Overall health status"
                        },
                        "health_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Health score (0.0-1.0)"
                        },
                        "a2a_connected": {
                            "type": "boolean",
                            "description": "A2A protocol connection status"
                        },
                        "api_accessible": {
                            "type": "boolean",
                            "description": "External API accessibility"
                        },
                        "data_domain": {
                            "type": "string",
                            "description": "Data domain"
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Available capabilities"
                        },
                        "performance_metrics": {
                            "type": "object",
                            "properties": {
                                "total_requests": {
                                    "type": "integer",
                                    "description": "Total requests processed"
                                },
                                "error_count": {
                                    "type": "integer",
                                    "description": "Total errors"
                                },
                                "error_rate": {
                                    "type": "number",
                                    "description": "Error rate (0.0-1.0)"
                                },
                                "avg_response_time": {
                                    "type": "number",
                                    "description": "Average response time in seconds"
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    # Add capability-specific paths
    for capability in capabilities:
        spec["paths"].update(generate_capability_paths(agent_name, capability, data_domain))

    # Add standard paths
    spec["paths"].update({
        "/health": {
            "get": {
                "summary": f"Get {agent_name} health status",
                "description": f"Retrieve comprehensive health information for the {agent_name} agent, including connection status, performance metrics, and external API accessibility.",
                "operationId": "getHealth",
                "tags": ["Health"],
                "responses": {
                    "200": {
                        "description": "Health check successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthCheck"
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Service unavailable",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/metrics": {
            "get": {
                "summary": f"Get {agent_name} Prometheus metrics",
                "description": f"Retrieve Prometheus-formatted metrics for monitoring the {agent_name} agent performance and health.",
                "operationId": "getMetrics",
                "tags": ["Monitoring"],
                "responses": {
                    "200": {
                        "description": "Metrics retrieved successfully",
                        "content": {
                            "text/plain": {
                                "schema": {
                                    "type": "string",
                                    "description": "Prometheus metrics in text format"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/schema": {
            "get": {
                "summary": f"Get {agent_name} agent schema",
                "description": f"Retrieve the complete agent schema and configuration information for {agent_name}.",
                "operationId": "getSchema",
                "tags": ["Metadata"],
                "responses": {
                    "200": {
                        "description": "Schema retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "description": "Agent schema information"
                                }
                            }
                        }
                    }
                }
            }
        }
    })

    return spec


def generate_capability_paths(agent_name: str, capability: str, data_domain: str) -> Dict[str, Any]:
    """
    Generate OpenAPI paths for a specific capability.

    Args:
        agent_name: Name of the agent
        capability: Capability name
        data_domain: Data domain

    Returns:
        Dictionary of OpenAPI path definitions
    """
    capability_path = f"/execute/{capability}"
    capability_tag = data_domain.replace(" ", "")

    # Define parameters based on capability type
    parameters = []
    request_body = None

    if "by_point" in capability or "at_point" in capability:
        parameters = [
            {
                "name": "lat",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Latitude coordinate"
            },
            {
                "name": "lon",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Longitude coordinate"
            }
        ]
    elif "by_bbox" in capability:
        parameters = [
            {
                "name": "min_lat",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Minimum latitude"
            },
            {
                "name": "min_lon",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Minimum longitude"
            },
            {
                "name": "max_lat",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Maximum latitude"
            },
            {
                "name": "max_lon",
                "in": "query",
                "required": True,
                "schema": {"type": "number", "format": "float"},
                "description": "Maximum longitude"
            }
        ]
    elif "by_date" in capability:
        parameters = [
            {
                "name": "start_date",
                "in": "query",
                "required": False,
                "schema": {"type": "string", "format": "date"},
                "description": "Start date (YYYY-MM-DD)"
            },
            {
                "name": "end_date",
                "in": "query",
                "required": False,
                "schema": {"type": "string", "format": "date"},
                "description": "End date (YYYY-MM-DD)"
            }
        ]
    elif "by_keyword" in capability or "search" in capability:
        parameters = [
            {
                "name": "query",
                "in": "query",
                "required": True,
                "schema": {"type": "string"},
                "description": "Search query or keywords"
            },
            {
                "name": "limit",
                "in": "query",
                "required": False,
                "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "description": "Maximum number of results"
            }
        ]
    elif "by_id" in capability:
        parameters = [
            {
                "name": "id",
                "in": "query",
                "required": True,
                "schema": {"type": "string"},
                "description": "Resource identifier"
            }
        ]

    return {
        capability_path: {
            "post": {
                "summary": f"Execute {capability} capability",
                "description": f"Execute the {capability} capability to retrieve {data_domain.lower()} data from external sources.",
                "operationId": f"execute{capability.replace('_', '').title()}",
                "tags": [capability_tag],
                "parameters": parameters,
                "requestBody": request_body,
                "responses": {
                    "200": {
                        "description": f"{capability} executed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "result": {
                                            "type": "object",
                                            "description": f"{capability} result data"
                                        },
                                        "metadata": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {"type": "string", "format": "date-time"},
                                                "processing_time": {"type": "number"},
                                                "data_source": {"type": "string"},
                                                "provenance_level": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request parameters",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Authentication required",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "403": {
                        "description": "Access forbidden",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "External data source unavailable",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        }
    }


def main():
    """Generate OpenAPI specs for all agents."""
    # Load agent manifest
    manifest_path = Path(__file__).parent / "foundational_agents_manifest.json"
    with open(manifest_path, 'r') as f:
        agents = json.load(f)

    output_dir = Path(__file__).parent / "openapi_specs"
    output_dir.mkdir(exist_ok=True)

    print(f"Generating OpenAPI specs for {len(agents)} agents...")

    for agent_config in agents:
        agent_name = agent_config["agentName"]
        print(f"Generating spec for {agent_name}...")

        # Generate spec
        spec = generate_openapi_spec(agent_config)

        # Save as JSON
        json_path = output_dir / f"{agent_name.lower()}_openapi.json"
        with open(json_path, 'w') as f:
            json.dump(spec, f, indent=2)

        # Save as YAML (alternative format)
        yaml_path = output_dir / f"{agent_name.lower()}_openapi.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False)

    print(f"OpenAPI specs generated in {output_dir}/")


if __name__ == "__main__":
    main()