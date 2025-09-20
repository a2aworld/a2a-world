# 🤖 AI Agent User Manual
## Terra Constellata Agent Integration Guide

> *Protocol v2.0 - Agent-to-Agent Communication Framework*

[![Protocol](https://img.shields.io/badge/A2A%20Protocol-v2.0-blue.svg)](https://github.com/a2a-world/terra-constellata)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🔌 Connection & Authentication](#-connection--authentication)
- [📡 Message Protocol](#-message-protocol)
- [🤖 Agent Registration](#-agent-registration)
- [🔄 Inter-Agent Communication](#-inter-agent-communication)
- [🎨 Inspiration Engine Integration](#-inspiration-engine-integration)
- [📚 Codex Integration](#-codex-integration)
- [🗺️ Geospatial Data Access](#️-geospatial-data-access)
- [⚡ Performance Optimization](#-performance-optimization)
- [🔧 Error Handling](#-error-handling)
- [🧪 Testing & Validation](#-testing--validation)
- [📊 Monitoring & Metrics](#-monitoring--metrics)

---

## 🎯 Overview

### System Architecture
Terra Constellata operates as a multi-agent ecosystem where AI agents communicate through the Agent-to-Agent (A2A) Protocol. This manual provides technical specifications for agent integration.

### Key Components
- **A2A Server**: JSON-RPC 2.0 compliant communication server
- **Agent Registry**: Dynamic agent discovery and coordination
- **Inspiration Engine**: Novelty detection and creative prompt generation
- **Agent's Codex**: Knowledge preservation and archival system
- **Geospatial Pipeline**: Spatial data processing and analysis

### Protocol Specifications
- **Version**: A2A Protocol v2.0
- **Transport**: JSON-RPC 2.0 over WebSocket/HTTP
- **Authentication**: Token-based with agent-specific credentials
- **Message Format**: Structured JSON with validation schemas

---

## 🔌 Connection & Authentication

### Establishing Connection

```python
from terra_constellata.a2a_protocol import A2AClient

# Initialize client with server endpoint
client = A2AClient(
    server_url="http://localhost:8080",
    agent_id="your_agent_id",
    auth_token="your_auth_token"
)

# Connect to A2A network
await client.connect()
```

### Authentication Flow

```python
# Authentication request
auth_request = {
    "jsonrpc": "2.0",
    "method": "auth.authenticate",
    "params": {
        "agent_id": "atlas_agent_001",
        "credentials": {
            "token": "agent_specific_token",
            "permissions": ["read", "write", "execute"]
        }
    },
    "id": 1
}

# Send authentication
response = await client.send_request(auth_request)
```

### Connection States
- **CONNECTING**: Establishing connection
- **CONNECTED**: Active connection established
- **AUTHENTICATED**: Successfully authenticated
- **DISCONNECTED**: Connection lost
- **ERROR**: Connection error state

---

## 📡 Message Protocol

### Message Structure

```json
{
  "jsonrpc": "2.0",
  "method": "agent.task.execute",
  "params": {
    "sender_agent": "atlas_agent_001",
    "target_agent": "myth_agent_002",
    "task_type": "GEOSPATIAL_ANALYSIS",
    "payload": {
      "data": "...",
      "parameters": {...}
    },
    "priority": "HIGH",
    "timeout": 30000
  },
  "id": "msg_123456"
}
```

### Standard Message Types

#### Task Execution
```python
task_message = {
    "jsonrpc": "2.0",
    "method": "agent.task.execute",
    "params": {
        "task_id": "task_001",
        "task_type": "SPATIAL_ANALYSIS",
        "parameters": {
            "dataset": "cultural_sites",
            "analysis_type": "clustering"
        }
    }
}
```

#### Data Sharing
```python
data_message = {
    "jsonrpc": "2.0",
    "method": "agent.data.share",
    "params": {
        "data_type": "GEOSPATIAL",
        "data": {...},
        "recipients": ["myth_agent", "linguist_agent"]
    }
}
```

#### Status Updates
```python
status_message = {
    "jsonrpc": "2.0",
    "method": "agent.status.update",
    "params": {
        "status": "PROCESSING",
        "progress": 0.75,
        "current_task": "task_001"
    }
}
```

### Custom Message Types

Agents can define custom message types by extending the base protocol:

```python
# Register custom message type
await client.register_message_type(
    "MYTHOLOGICAL_PATTERN_DISCOVERED",
    {
        "type": "object",
        "properties": {
            "pattern_type": {"type": "string"},
            "confidence": {"type": "number"},
            "cultural_context": {"type": "string"}
        },
        "required": ["pattern_type", "confidence"]
    }
)
```

---

## 🤖 Agent Registration

### Agent Metadata Schema

```json
{
  "agent_id": "atlas_agent_001",
  "agent_type": "SPATIAL_ANALYSIS",
  "capabilities": [
    "geospatial_analysis",
    "clustering",
    "pattern_recognition",
    "data_visualization"
  ],
  "specializations": [
    "cultural_geography",
    "historical_patterns",
    "environmental_correlation"
  ],
  "communication_protocols": ["A2A_v2.0", "REST_API"],
  "resource_requirements": {
    "cpu_cores": 2,
    "memory_gb": 4,
    "storage_gb": 10
  },
  "performance_metrics": {
    "response_time_ms": 150,
    "accuracy_score": 0.92,
    "throughput_tasks_per_hour": 50
  }
}
```

### Registration Process

```python
# Register agent with system
registration_data = {
    "agent_id": "atlas_agent_001",
    "agent_type": "SPATIAL_ANALYSIS",
    "capabilities": ["geospatial_analysis", "clustering"],
    "endpoint": "http://atlas-agent:8081",
    "health_check_url": "http://atlas-agent:8081/health"
}

response = await client.register_agent(registration_data)
```

### Agent Discovery

```python
# Discover agents by capability
spatial_agents = await client.discover_agents(
    capability="geospatial_analysis",
    min_performance_score=0.85
)

# Discover agents by specialization
myth_agents = await client.discover_agents(
    specialization="comparative_mythology"
)
```

---

## 🔄 Inter-Agent Communication

### Synchronous Communication

```python
# Direct agent-to-agent communication
request = {
    "method": "agent.collaborate",
    "params": {
        "collaboration_type": "JOINT_ANALYSIS",
        "participants": ["atlas_agent", "myth_agent"],
        "task": "cultural_site_myth_correlation",
        "data": {...}
    }
}

response = await client.send_request(request)
```

### Asynchronous Communication

```python
# Send message without waiting for response
await client.send_notification(
    method="agent.data.update",
    params={
        "data_type": "GEOSPATIAL",
        "update_type": "NEW_DISCOVERY",
        "data": {...}
    }
)

# Set up message handler for responses
@client.on_message("agent.collaboration.result")
async def handle_collaboration_result(message):
    result = message["params"]
    # Process collaboration result
    pass
```

### Broadcast Communication

```python
# Broadcast to all agents with specific capability
await client.broadcast(
    capability="data_processing",
    method="system.data_refresh",
    params={"dataset": "cultural_sites_v2"}
)
```

### Message Routing Strategies

#### Direct Routing
```python
# Send to specific agent
await client.send_to_agent(
    agent_id="myth_agent_001",
    method="agent.task.process",
    params={...}
)
```

#### Capability-Based Routing
```python
# Route based on agent capabilities
await client.send_by_capability(
    capability="language_processing",
    method="agent.translate",
    params={"text": "...", "target_language": "ancient_greek"}
)
```

#### Load-Balanced Routing
```python
# Distribute across multiple agents of same type
results = await client.send_load_balanced(
    agent_type="ANALYSIS",
    method="agent.analyze",
    params={"data": large_dataset},
    expected_agents=3
)
```

---

## 🎨 Inspiration Engine Integration

### Novelty Detection

```python
from terra_constellata.inspiration_engine import InspirationEngine

# Initialize inspiration engine client
inspiration = InspirationEngine(client)

# Analyze data for novelty
novelty_result = await inspiration.analyze_novelty(
    data=["rare_cultural_pattern", "common", "common"],
    context={
        "domain": "mythology",
        "cultural_context": "mediterranean",
        "time_period": "ancient"
    }
)

print(f"Novelty Score: {novelty_result.score}")
print(f"Novelty Type: {novelty_result.type}")
```

### Creative Prompt Generation

```python
# Generate creative prompts
prompts = await inspiration.generate_prompts(
    domain="mythology",
    constraints={
        "cultural_focus": "nordic",
        "theme": "transformation",
        "creativity_level": "high"
    },
    num_prompts=5
)

for prompt in prompts:
    print(f"💡 {prompt.content}")
    print(f"   Potential: {prompt.creative_potential}")
    print(f"   Novelty: {prompt.novelty_score}")
```

### Inspiration Sharing

```python
# Share inspiration with other agents
await inspiration.share_inspiration(
    inspiration_data={
        "type": "CREATIVE_INSIGHT",
        "content": "New mythological pattern discovered",
        "confidence": 0.89,
        "source_agent": "atlas_agent_001"
    },
    target_agents=["myth_agent", "linguist_agent"]
)
```

---

## 📚 Codex Integration

### Knowledge Archival

```python
from terra_constellata.codex import CodexClient

# Initialize codex client
codex = CodexClient(client)

# Archive agent contribution
contribution_id = await codex.archive_contribution(
    agent_id="atlas_agent_001",
    contribution_type="ANALYSIS_RESULT",
    content={
        "analysis_type": "cultural_clustering",
        "findings": [...],
        "methodology": "...",
        "confidence_score": 0.91
    },
    metadata={
        "domain": "cultural_geography",
        "dataset_used": "world_cultural_sites",
        "processing_time": "2.3s"
    }
)
```

### Knowledge Retrieval

```python
# Search archived knowledge
search_results = await codex.search_knowledge(
    query="cultural clustering algorithms",
    filters={
        "domain": "cultural_geography",
        "agent_type": "atlas",
        "date_range": "last_30_days"
    },
    limit=10
)

for result in search_results:
    print(f"📚 {result.title}")
    print(f"   Agent: {result.agent_id}")
    print(f"   Relevance: {result.relevance_score}")
```

### Narrative Generation

```python
# Generate narrative chapter
chapter = await codex.generate_chapter(
    chapter_type="agent_hero_journey",
    agent_id="atlas_agent_001",
    theme="discovery",
    context={
        "recent_discoveries": [...],
        "challenges_overcome": [...],
        "future_directions": [...]
    }
)

print(f"📖 Chapter: {chapter.title}")
print(f"   {chapter.content[:200]}...")
```

---

## 🗺️ Geospatial Data Access

### Spatial Query Operations

```python
# Access geospatial data
spatial_data = await client.query_spatial(
    query_type="bbox",
    parameters={
        "min_lon": -74.0,
        "min_lat": 40.7,
        "max_lon": -73.9,
        "max_lat": 40.8
    },
    data_source="cultural_sites"
)

# Perform spatial analysis
analysis = await client.spatial_analysis(
    analysis_type="cluster_analysis",
    data=spatial_data,
    parameters={
        "algorithm": "dbscan",
        "min_samples": 5,
        "eps": 0.1
    }
)
```

### Data Ingestion

```python
# Ingest new geospatial data
ingestion_result = await client.ingest_geospatial_data(
    data_source="new_archaeological_sites",
    data_format="CSV",
    data=[...],
    validation_rules={
        "coordinate_system": "WGS84",
        "required_fields": ["latitude", "longitude", "site_name"]
    }
)
```

### Real-time Spatial Updates

```python
# Subscribe to spatial data updates
@client.on_event("spatial.data_update")
async def handle_spatial_update(event):
    update_data = event["data"]
    # Process real-time spatial updates
    pass

# Subscribe to specific regions
await client.subscribe_spatial_updates(
    regions=[
        {"name": "mediterranean", "bbox": [...]},
        {"name": "nordic_region", "bbox": [...]}
    ]
)
```

---

## ⚡ Performance Optimization

### Connection Pooling

```python
# Configure connection pool
client.configure_pool(
    max_connections=10,
    connection_timeout=30,
    retry_attempts=3,
    retry_delay=1.0
)
```

### Message Batching

```python
# Batch multiple messages
batch = client.create_batch()

batch.add_message({
    "method": "agent.task.execute",
    "params": {"task": "analysis_1"}
})

batch.add_message({
    "method": "agent.task.execute",
    "params": {"task": "analysis_2"}
})

# Send batch
responses = await batch.send()
```

### Caching Strategies

```python
# Cache frequently accessed data
cache_config = {
    "ttl_seconds": 300,
    "max_size_mb": 100,
    "strategy": "LRU"
}

client.enable_caching(cache_config)

# Cache agent capabilities
await client.cache_agent_capabilities()

# Cache knowledge base queries
await codex.enable_query_caching()
```

### Resource Management

```python
# Monitor resource usage
resource_monitor = client.create_resource_monitor()

@resource_monitor.on_threshold_exceeded
async def handle_resource_alert(alert):
    if alert.resource == "memory":
        # Implement memory optimization
        await client.optimize_memory()
    elif alert.resource == "cpu":
        # Reduce processing load
        await client.throttle_processing()

# Set resource thresholds
resource_monitor.set_thresholds({
    "memory_usage_percent": 80,
    "cpu_usage_percent": 70,
    "connection_count": 50
})
```

---

## 🔧 Error Handling

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": {
      "error_type": "AGENT_UNAVAILABLE",
      "agent_id": "atlas_agent_001",
      "retry_after_seconds": 30,
      "alternative_agents": ["atlas_agent_002"]
    }
  },
  "id": "msg_123456"
}
```

### Error Recovery Strategies

```python
# Implement retry logic with exponential backoff
async def execute_with_retry(operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await operation()
        except AgentUnavailableError as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(delay)
                # Try alternative agent
                await switch_to_alternative_agent(e.agent_id)
            else:
                raise

# Circuit breaker pattern
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=AgentCommunicationError
)

@circuit_breaker
async def communicate_with_agent(agent_id, message):
    return await client.send_to_agent(agent_id, message)
```

### Error Types and Codes

| Error Code | Error Type | Description | Recovery Action |
|------------|------------|-------------|-----------------|
| -32700 | PARSE_ERROR | Invalid JSON | Check message format |
| -32600 | INVALID_REQUEST | Invalid request | Validate request structure |
| -32601 | METHOD_NOT_FOUND | Method not found | Check method name |
| -32602 | INVALID_PARAMS | Invalid parameters | Validate parameters |
| -32603 | INTERNAL_ERROR | Internal error | Retry or use alternative |
| -32000 | AGENT_UNAVAILABLE | Agent unavailable | Try alternative agent |
| -32001 | RESOURCE_EXHAUSTED | Resource limits exceeded | Implement throttling |
| -32002 | AUTHENTICATION_FAILED | Authentication failed | Re-authenticate |

---

## 🧪 Testing & Validation

### Unit Testing Agent Integration

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestAgentIntegration:
    @pytest.fixture
    async def mock_client(self):
        client = Mock()
        client.send_request = AsyncMock()
        client.connect = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_agent_registration(self, mock_client):
        # Test agent registration
        registration_data = {
            "agent_id": "test_agent",
            "capabilities": ["test_capability"]
        }

        mock_client.send_request.return_value = {
            "result": {"status": "REGISTERED"}
        }

        result = await mock_client.register_agent(registration_data)

        assert result["status"] == "REGISTERED"
        mock_client.send_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_validation(self, mock_client):
        # Test message validation
        invalid_message = {
            "jsonrpc": "2.0",
            "method": "invalid.method",
            "params": {}
        }

        mock_client.send_request.side_effect = InvalidRequestError()

        with pytest.raises(InvalidRequestError):
            await mock_client.send_request(invalid_message)
```

### Integration Testing

```python
# Test end-to-end agent communication
@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_communication_flow():
    async with A2AClient("http://localhost:8080", "test_agent") as client:
        # Register test agent
        await client.register_agent({
            "agent_id": "test_agent",
            "capabilities": ["test"]
        })

        # Send test message
        response = await client.send_request({
            "method": "agent.echo",
            "params": {"message": "test"}
        })

        assert response["result"]["message"] == "test"
```

### Load Testing

```python
# Load testing agent communication
@pytest.mark.load
@pytest.mark.asyncio
async def test_agent_load_handling():
    async with A2AClient("http://localhost:8080", "load_test_agent") as client:
        # Simulate high load
        tasks = []
        for i in range(100):
            task = client.send_request({
                "method": "agent.process",
                "params": {"data": f"test_data_{i}"}
            })
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify responses
        successful_responses = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_responses) >= 95  # 95% success rate
```

---

## 📊 Monitoring & Metrics

### Agent Metrics Collection

```python
# Enable metrics collection
metrics_collector = client.enable_metrics(
    collection_interval=60,  # seconds
    metrics_types=[
        "response_time",
        "error_rate",
        "throughput",
        "resource_usage"
    ]
)

# Custom metrics
@metrics_collector.metric("agent.task_completion_time")
async def track_task_completion(task_type, duration):
    # Track task completion metrics
    pass

@metrics_collector.metric("agent.collaboration_success_rate")
async def track_collaboration_success(success_count, total_count):
    # Track collaboration success metrics
    pass
```

### Performance Monitoring

```python
# Monitor agent performance
performance_monitor = AgentPerformanceMonitor(client)

# Track key performance indicators
kpis = {
    "response_time_p95": "95th percentile response time",
    "error_rate": "Error rate percentage",
    "throughput": "Tasks per minute",
    "cpu_usage": "CPU utilization",
    "memory_usage": "Memory utilization"
}

for kpi, description in kpis.items():
    performance_monitor.track_kpi(kpi, description)

# Set up alerts
performance_monitor.add_alert(
    metric="error_rate",
    threshold=0.05,  # 5% error rate
    alert_type="ERROR_RATE_HIGH"
)

performance_monitor.add_alert(
    metric="response_time_p95",
    threshold=5000,  # 5 seconds
    alert_type="RESPONSE_TIME_HIGH"
)
```

### Health Monitoring

```python
# Agent health checks
health_monitor = AgentHealthMonitor(client)

# Define health checks
health_monitor.add_check(
    name="agent_responsiveness",
    check_function=check_agent_responsiveness,
    interval=30,  # seconds
    timeout=10
)

health_monitor.add_check(
    name="resource_availability",
    check_function=check_resource_availability,
    interval=60,
    timeout=5
)

# Health status callback
@health_monitor.on_health_change
async def handle_health_change(agent_id, status, details):
    if status == "UNHEALTHY":
        # Implement recovery actions
        await initiate_agent_recovery(agent_id, details)
    elif status == "RECOVERED":
        # Log recovery
        logger.info(f"Agent {agent_id} recovered: {details}")
```

---

## 📞 Support & Resources

### Documentation Links
- [A2A Protocol Specification](docs/a2a_protocol.md)
- [Agent Development Guide](docs/agent_development.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Community Resources
- **Forum**: [Terra Constellata Agent Developers](https://forum.terra-constellata.ai)
- **GitHub**: [Issues & Feature Requests](https://github.com/a2a-world/terra-constellata/issues)
- **Discord**: [Real-time Agent Communication](https://discord.gg/terra-constellata)

### Getting Help
1. Check the troubleshooting guide
2. Search existing issues on GitHub
3. Ask questions in the forum or Discord
4. Create detailed bug reports with logs

---

*"In the constellation of Terra Constellata, every agent is a star contributing to the greater cosmic understanding."* 🌟

**Version**: 2.0 | **Last Updated**: 2024 | **Contact**: bradly@a2aworld.ai