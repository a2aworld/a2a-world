"""
Base Data Gateway Agent for Terra Constellata

This module provides the base class for data gateway agents that provide
authenticated access to external data sources and APIs.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import aiohttp
import json
import time

from langchain.agents import AgentExecutor
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory

from ..base_agent import BaseSpecialistAgent
from ...a2a_protocol.schemas import A2AMessage
from .secrets_manager import resolve_secret_placeholder

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics disabled")

logger = logging.getLogger(__name__)


class DataGatewayAgent(BaseSpecialistAgent):
    """
    Base class for data gateway agents in Terra Constellata.

    These agents provide authenticated, secure access to external data sources
    and APIs, serving as the "Library of Alexandria for AI Wisdom."
    """

    def __init__(
        self,
        agent_name: str,
        data_domain: str,
        data_set_owner: Dict[str, Any],
        llm: BaseLLM,
        tools: List[BaseTool],
        a2a_server_url: str = "http://localhost:8080",
        memory_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize the data gateway agent.

        Args:
            agent_name: Standardized agent name (e.g., 'GEBCO_BATHYMETRY_AGENT')
            data_domain: Domain of data (e.g., 'Geospatial', 'Climatology')
            data_set_owner: Owner information dict with name, ownerType, officialContactUri
            llm: Language model for the agent
            tools: List of tools available to the agent
            a2a_server_url: URL of the A2A protocol server
            memory_size: Maximum memory buffer size
            **kwargs: Additional agent-specific parameters
        """
        super().__init__(
            name=agent_name,
            llm=llm,
            tools=tools,
            a2a_server_url=a2a_server_url,
            memory_size=memory_size,
            **kwargs
        )

        # Agent schema properties (adapted from A2A_World_VIA_Agent_Schema_v1.0)
        self.agent_id = str(uuid.uuid4())  # Globally unique agent ID
        self.via_status = "AUTHENTICATED"  # AUTHENTICATED or PROVISIONAL
        self.agent_name = agent_name
        self.data_set_owner = data_set_owner
        self.data_domain = data_domain
        self.a2a_endpoint = f"{a2a_server_url}/agents/{agent_name}"
        self.authentication_methods = kwargs.get("authentication_methods", ["api_key"])
        self.provenance_level = kwargs.get("provenance_level", "CANONICAL")  # CANONICAL or BEST_EFFORT
        self.version = kwargs.get("version", "1.0.0")
        self.last_authenticated_timestamp = datetime.utcnow().isoformat()
        self.capabilities = kwargs.get("capabilities", [])

        # Data access properties
        self.base_url = kwargs.get("base_url", "")
        self.api_key = kwargs.get("api_key", "{{SECRETS.API_KEY}}")  # Placeholder for secrets
        self.client_id = kwargs.get("client_id", "{{SECRETS.CLIENT_ID}}")
        self.client_secret = kwargs.get("client_secret", "{{SECRETS.CLIENT_SECRET}}")
        self.timeout = kwargs.get("timeout", 30)
        self.retry_attempts = kwargs.get("retry_attempts", 3)

        # HTTP client session
        self.session: Optional[aiohttp.ClientSession] = None

        # Health status
        self.health_status = "initializing"
        self.last_health_check = datetime.utcnow()

        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.last_request_time = None

        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self._setup_metrics()
        else:
            self.metrics = None

        logger.info(f"Initialized DataGatewayAgent: {agent_name} for domain {data_domain}")

    def _setup_metrics(self):
        """Set up Prometheus metrics for monitoring."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Create metrics registry for this agent
        self.metrics_registry = CollectorRegistry()

        # Request metrics
        self.requests_total = Counter(
            f'{self.agent_name.lower()}_requests_total',
            f'Total requests processed by {self.agent_name}',
            ['method', 'status'],
            registry=self.metrics_registry
        )

        self.requests_duration = Histogram(
            f'{self.agent_name.lower()}_request_duration_seconds',
            f'Request duration for {self.agent_name}',
            ['method'],
            registry=self.metrics_registry
        )

        # Health metrics
        self.health_status_gauge = Gauge(
            f'{self.agent_name.lower()}_health_status',
            f'Health status of {self.agent_name} (1=healthy, 0=unhealthy)',
            registry=self.metrics_registry
        )

        # API metrics
        self.api_requests_total = Counter(
            f'{self.agent_name.lower()}_api_requests_total',
            f'Total external API requests by {self.agent_name}',
            ['endpoint', 'status'],
            registry=self.metrics_registry
        )

        self.api_request_duration = Histogram(
            f'{self.agent_name.lower()}_api_request_duration_seconds',
            f'External API request duration for {self.agent_name}',
            ['endpoint'],
            registry=self.metrics_registry
        )

    async def connect_a2a(self):
        """Connect to the A2A protocol server."""
        await super().connect_a2a()
        # Initialize HTTP session for external API calls
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        self.health_status = "connected"

    async def disconnect_a2a(self):
        """Disconnect from the A2A protocol server."""
        await super().disconnect_a2a()
        if self.session:
            await self.session.close()
            self.session = None
        self.health_status = "disconnected"

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for data gateway operations."""
        from langchain.agents import initialize_agent, AgentType

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            max_iterations=10,
            max_execution_time=60,
        )

    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process a data access task.

        Args:
            task: Task description (e.g., "get_elevation_by_point")
            **kwargs: Task parameters

        Returns:
            Task result from external API
        """
        start_time = time.time()
        self.request_count += 1
        self.last_request_time = datetime.utcnow()

        try:
            # Parse task to determine capability
            capability = self._parse_task_capability(task)

            # Track request metrics
            if self.metrics:
                self.requests_total.labels(method=capability, status='started').inc()

            # Execute the capability
            result = await self._execute_capability(capability, **kwargs)

            # Track success metrics
            duration = time.time() - start_time
            self.response_times.append(duration)
            if len(self.response_times) > 100:  # Keep last 100 response times
                self.response_times.pop(0)

            if self.metrics:
                self.requests_total.labels(method=capability, status='success').inc()
                self.requests_duration.labels(method=capability).observe(duration)

            # Update memory and activity
            self.update_memory(task, str(result)[:500])
            self.last_activity = datetime.utcnow()

            return result

        except Exception as e:
            # Track error metrics
            duration = time.time() - start_time
            self.error_count += 1

            if self.metrics:
                self.requests_total.labels(method=self._parse_task_capability(task), status='error').inc()
                self.requests_duration.labels(method=self._parse_task_capability(task)).observe(duration)

            logger.error(f"Error processing task {task}: {e}")
            self.health_status = "error"
            raise

    @abstractmethod
    async def _execute_capability(self, capability: str, **kwargs) -> Any:
        """
        Execute a specific capability against the external data source.

        Args:
            capability: Capability name (e.g., "get_elevation_by_point")
            **kwargs: Capability parameters

        Returns:
            Capability result
        """
        pass

    def _parse_task_capability(self, task: str) -> str:
        """Parse task string to extract capability name."""
        # Simple parsing - can be enhanced with NLP
        task_lower = task.lower()
        for cap in self.capabilities:
            if cap in task_lower:
                return cap
        return task.split()[0] if task else "unknown"

    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict:
        """
        Make authenticated API request to external data source.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Additional headers

        Returns:
            API response data
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized")

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Prepare headers with authentication
        request_headers = headers or {}
        if "Authorization" not in request_headers:
            request_headers.update(self._get_auth_headers())

        # Add standard headers
        request_headers.update({
            "User-Agent": f"TerraConstellata-{self.agent_name}/1.0",
            "Accept": "application/json",
        })

        for attempt in range(self.retry_attempts):
            start_time = time.time()
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers,
                ) as response:
                    duration = time.time() - start_time

                    # Track API metrics
                    if self.metrics:
                        status_code = str(response.status)
                        self.api_requests_total.labels(endpoint=endpoint, status=status_code).inc()
                        self.api_request_duration.labels(endpoint=endpoint).observe(duration)

                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientError as e:
                duration = time.time() - start_time

                # Track failed API metrics
                if self.metrics:
                    self.api_requests_total.labels(endpoint=endpoint, status='error').inc()
                    self.api_request_duration.labels(endpoint=endpoint).observe(duration)

                if attempt == self.retry_attempts - 1:
                    logger.error(f"API request failed after {self.retry_attempts} attempts: {e}")
                    self.health_status = "api_error"
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}

        # Resolve secrets
        resolved_api_key = resolve_secret_placeholder(self.api_key) if self.api_key else None
        resolved_client_id = resolve_secret_placeholder(self.client_id) if self.client_id else None
        resolved_client_secret = resolve_secret_placeholder(self.client_secret) if self.client_secret else None

        # API Key authentication
        if resolved_api_key:
            if "bearer" in self.authentication_methods:
                headers["Authorization"] = f"Bearer {resolved_api_key}"
            elif "api_key" in self.authentication_methods:
                headers["X-API-Key"] = resolved_api_key
            else:
                # Default to Authorization header
                headers["Authorization"] = f"Bearer {resolved_api_key}"

        # OAuth2 if configured
        if resolved_client_id and resolved_client_secret:
            # Implement OAuth2 token retrieval if needed
            # For now, use basic auth as fallback
            import base64
            auth_string = base64.b64encode(f"{resolved_client_id}:{resolved_client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_string}"

        return headers

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the agent and its data source.

        Returns:
            Health status information including performance metrics
        """
        try:
            # Check A2A connection
            a2a_healthy = self.a2a_client is not None

            # Check external API health
            api_healthy = await self._check_external_api_health()

            # Calculate performance metrics
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            error_rate = self.error_count / max(self.request_count, 1)

            # Update status
            if a2a_healthy and api_healthy:
                self.health_status = "healthy"
                health_score = 1.0
            elif a2a_healthy:
                self.health_status = "a2a_healthy_api_unhealthy"
                health_score = 0.5
            else:
                self.health_status = "unhealthy"
                health_score = 0.0

            # Update Prometheus metrics
            if self.metrics:
                self.health_status_gauge.set(health_score)

            self.last_health_check = datetime.utcnow()

            health_data = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": self.health_status,
                "health_score": health_score,
                "a2a_connected": a2a_healthy,
                "api_accessible": api_healthy,
                "last_check": self.last_health_check.isoformat(),
                "data_domain": self.data_domain,
                "capabilities": self.capabilities,
                "performance_metrics": {
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": error_rate,
                    "avg_response_time": avg_response_time,
                    "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
                },
                "system_info": {
                    "version": self.version,
                    "provenance_level": self.provenance_level,
                    "authentication_methods": self.authentication_methods,
                }
            }

            return health_data

        except Exception as e:
            self.health_status = "health_check_failed"
            if self.metrics:
                self.health_status_gauge.set(0.0)
            logger.error(f"Health check failed: {e}")
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "health_check_failed",
                "health_score": 0.0,
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    async def _check_external_api_health(self) -> bool:
        """Check if external API is accessible."""
        try:
            # Attempt a simple health check request
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except:
            # If health endpoint doesn't exist, try a basic capability
            try:
                await self._execute_capability(self.capabilities[0], test=True)
                return True
            except:
                return False

    def get_agent_schema(self) -> Dict[str, Any]:
        """Get the complete agent schema information."""
        return {
            "agentId": self.agent_id,
            "viaStatus": self.via_status,
            "agentName": self.agent_name,
            "dataSetOwner": self.data_set_owner,
            "dataDomain": self.data_domain,
            "a2aEndpoint": self.a2a_endpoint,
            "authenticationMethods": self.authentication_methods,
            "provenanceLevel": self.provenance_level,
            "version": self.version,
            "lastAuthenticatedTimestamp": self.last_authenticated_timestamp,
            "capabilities": [
                {
                    "capabilityId": cap,
                    "description": f"Execute {cap} capability",
                    "queryFormats": ["json", "geojson"] if "geo" in cap.lower() else ["json"]
                }
                for cap in self.capabilities
            ],
        }

    def get_metrics(self) -> str:
        """
        Get Prometheus metrics in text format.

        Returns:
            Metrics data in Prometheus format
        """
        if not PROMETHEUS_AVAILABLE or not self.metrics_registry:
            return "# Prometheus metrics not available"

        from prometheus_client import generate_latest
        return generate_latest(self.metrics_registry).decode('utf-8')

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics.

        Returns:
            Performance metrics dictionary
        """
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        min_response_time = min(self.response_times) if self.response_times else 0
        max_response_time = max(self.response_times) if self.response_times else 0

        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "response_time_samples": len(self.response_times),
            "uptime_seconds": (datetime.utcnow() - self.last_authenticated_timestamp.replace('Z', '+00:00')).total_seconds() if isinstance(self.last_authenticated_timestamp, str) else 0,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
        }

    async def _autonomous_loop(self):
        """Autonomous operation loop for data gateway agents."""
        while self.is_active:
            try:
                # Perform periodic health checks
                if (datetime.utcnow() - self.last_health_check).seconds > 300:  # 5 minutes
                    await self.health_check()

                # Process any pending A2A messages
                await self._process_pending_messages()

                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(30)

    async def _process_pending_messages(self):
        """Process any pending A2A messages."""
        # Implementation depends on A2A server capabilities
        # For now, this is a placeholder
        pass

    async def request_data_from_agent(self, target_agent: str, query: str, **kwargs) -> Any:
        """
        Request data from another agent via A2A protocol.

        Args:
            target_agent: Name of the agent to query
            query: Data query (e.g., "get_elevation_by_point lat=40.0 lon=-74.0")
            **kwargs: Additional query parameters

        Returns:
            Query results from target agent
        """
        from ..a2a_protocol.schemas import A2AMessage

        message = A2AMessage(
            message_id=f"data_query_{self.agent_id}_{target_agent}",
            sender_agent=self.name,
            target_agent=target_agent,
            timestamp=datetime.utcnow().isoformat(),
            message_type="DATA_QUERY",
            payload={
                "query": query,
                "parameters": kwargs,
                "requester_domain": self.data_domain
            }
        )

        return await self.send_message(message, target_agent)

    async def share_data_with_agent(self, target_agent: str, data: Dict[str, Any], context: str = "") -> None:
        """
        Share data with another agent via A2A protocol.

        Args:
            target_agent: Name of the agent to share with
            data: Data to share
            context: Context for the data sharing
        """
        from ..a2a_protocol.schemas import A2AMessage

        message = A2AMessage(
            message_id=f"data_share_{self.agent_id}_{target_agent}",
            sender_agent=self.name,
            target_agent=target_agent,
            timestamp=datetime.utcnow().isoformat(),
            message_type="DATA_SHARE",
            payload={
                "data": data,
                "context": context,
                "provider_domain": self.data_domain,
                "provenance_level": self.provenance_level
            }
        )

        await self.send_notification(message, target_agent)

    async def broadcast_data_availability(self, data_summary: Dict[str, Any]) -> None:
        """
        Broadcast data availability to all agents in the registry.

        Args:
            data_summary: Summary of available data
        """
        from ..a2a_protocol.schemas import A2AMessage

        message = A2AMessage(
            message_id=f"data_available_{self.agent_id}",
            sender_agent=self.name,
            timestamp=datetime.utcnow().isoformat(),
            message_type="DATA_AVAILABLE",
            payload={
                "agent_name": self.agent_name,
                "data_domain": self.data_domain,
                "capabilities": self.capabilities,
                "data_summary": data_summary,
                "contact_uri": self.a2a_endpoint
            }
        )

        await data_gateway_registry.broadcast_message(message, exclude_agent=self.name)

    async def query_related_agents(self, query: str, domains: List[str] = None) -> Dict[str, Any]:
        """
        Query related agents for complementary data.

        Args:
            query: The query to execute
            domains: List of domains to query (if None, queries all)

        Returns:
            Aggregated results from related agents
        """
        results = {}
        target_domains = domains or self._get_related_domains()

        for domain in target_domains:
            agents = data_gateway_registry.get_agents_by_domain(domain)
            for agent in agents:
                if agent.name != self.name:
                    try:
                        result = await self.request_data_from_agent(agent.name, query)
                        results[agent.name] = result
                    except Exception as e:
                        logger.warning(f"Failed to query agent {agent.name}: {e}")
                        results[agent.name] = {"error": str(e)}

        return {
            "query": query,
            "results": results,
            "queried_domains": target_domains,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_related_domains(self) -> List[str]:
        """Get domains related to this agent's domain."""
        domain_relations = {
            "Geospatial": ["Climatology", "Geology", "Cartography"],
            "Climatology": ["Geospatial", "Geophysics"],
            "Cultural Heritage": ["Historical Archive", "Art History"],
            "Scientific": ["Astrophysics", "Biology", "Chemistry"],
            "Linguistics": ["Knowledge Graph"],
            "Infrastructure": ["All"]  # Infrastructure agents can relate to all
        }
        return domain_relations.get(self.data_domain, [])


class DataGatewayAgentRegistry:
    """Registry for managing data gateway agents."""

    def __init__(self):
        self.agents: Dict[str, DataGatewayAgent] = {}

    def register_agent(self, agent: DataGatewayAgent):
        """Register a data gateway agent."""
        self.agents[agent.agent_name] = agent
        logger.info(f"Registered data gateway agent: {agent.agent_name}")

    def get_agent(self, agent_name: str) -> Optional[DataGatewayAgent]:
        """Get an agent by name."""
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def get_agents_by_domain(self, domain: str) -> List[DataGatewayAgent]:
        """Get agents by data domain."""
        return [
            agent for agent in self.agents.values()
            if agent.data_domain == domain
        ]

    def get_agents_by_capability(self, capability: str) -> List[DataGatewayAgent]:
        """Get agents that support a specific capability."""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]

    async def broadcast_health_check(self):
        """Broadcast health check to all agents."""
        results = {}
        for agent in self.agents.values():
            try:
                results[agent.agent_name] = await agent.health_check()
            except Exception as e:
                results[agent.agent_name] = {"status": "error", "error": str(e)}
        return results


# Global data gateway agent registry
data_gateway_registry = DataGatewayAgentRegistry()