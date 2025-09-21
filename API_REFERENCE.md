# 🔌 Terra Constellata API Reference

[![API Version](https://img.shields.io/badge/API%20Version-v2.0-blue.svg)](https://github.com/a2a-world/terra-constellata)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-green.svg)](https://swagger.io/specification/)
[![REST](https://img.shields.io/badge/REST-JSON-orange.svg)](https://restfulapi.net/)
[![GraphQL](https://img.shields.io/badge/GraphQL-API-red.svg)](https://graphql.org/)

---

## 📋 Table of Contents

- [🌐 REST API](#-rest-api)
  - [Authentication](#authentication)
  - [Agents](#agents)
  - [Data Management](#data-management)
  - [Research](#research)
  - [Inspiration](#inspiration)
  - [Codex](#codex)
  - [Spatial Analysis](#spatial-analysis)
  - [Monitoring](#monitoring)
- [📊 GraphQL API](#-graphql-api)
- [🤖 A2A Protocol API](#-a2a-protocol-api)
- [🔧 SDKs and Libraries](#-sdks-and-libraries)
- [📚 Code Examples](#-code-examples)
- [🔒 Security](#-security)
- [📊 Rate Limits](#-rate-limits)
- [🚨 Error Handling](#-error-handling)
- [🧪 Testing](#-testing)

---

## 🌐 REST API

### Base URL
```
https://api.terra-constellata.ai/v2
```

### Authentication

#### POST /auth/login
Authenticate user and receive access token.

**Request:**
```http
POST /auth/login
Content-Type: application/json

{
  "username": "researcher@example.com",
  "password": "secure_password",
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "researcher@example.com",
    "role": "researcher",
    "permissions": ["read", "write", "execute"]
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid credentials
- `401 Unauthorized`: MFA required
- `429 Too Many Requests`: Rate limit exceeded

#### POST /auth/refresh
Refresh access token using refresh token.

**Request:**
```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

#### POST /auth/logout
Invalidate current session.

**Request:**
```http
POST /auth/logout
Authorization: Bearer <access_token>
```

### Agents

#### GET /agents
List all available agents.

**Parameters:**
- `capability` (optional): Filter by capability
- `status` (optional): Filter by status (`active`, `inactive`, `maintenance`)
- `limit` (optional): Maximum number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "agents": [
    {
      "id": "atlas_agent_001",
      "name": "Atlas Agent",
      "type": "SPATIAL_ANALYSIS",
      "capabilities": ["geospatial_analysis", "clustering", "mapping"],
      "status": "active",
      "performance_score": 0.92,
      "last_active": "2024-01-15T10:30:00Z",
      "description": "Specialized in spatial analysis and geographical pattern recognition"
    }
  ],
  "total": 12,
  "limit": 50,
  "offset": 0
}
```

#### GET /agents/{agent_id}
Get detailed information about a specific agent.

**Response:**
```json
{
  "id": "atlas_agent_001",
  "name": "Atlas Agent",
  "type": "SPATIAL_ANALYSIS",
  "capabilities": [
    "geospatial_analysis",
    "spatial_clustering",
    "geographical_mapping",
    "terrain_analysis",
    "coordinate_transformation"
  ],
  "specializations": [
    "cultural_geography",
    "historical_patterns",
    "environmental_correlation"
  ],
  "status": "active",
  "performance_metrics": {
    "response_time_ms": 245,
    "accuracy_score": 0.94,
    "tasks_completed": 15420,
    "uptime_percentage": 99.7
  },
  "resource_requirements": {
    "cpu_cores": 2,
    "memory_gb": 4,
    "storage_gb": 20
  },
  "supported_formats": ["GeoJSON", "Shapefile", "CSV", "KML"],
  "version": "2.1.3",
  "last_updated": "2024-01-10T14:20:00Z"
}
```

#### POST /agents/{agent_id}/task
Submit a task to a specific agent.

**Request:**
```http
POST /agents/atlas_agent_001/task
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "task_type": "SPATIAL_ANALYSIS",
  "parameters": {
    "analysis_type": "clustering",
    "dataset": "cultural_sites_europe",
    "algorithm": "dbscan",
    "parameters": {
      "eps": 0.5,
      "min_samples": 5
    }
  },
  "priority": "high",
  "timeout_seconds": 300,
  "callback_url": "https://my-app.com/webhook/task_complete"
}
```

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "queued",
  "estimated_completion_seconds": 120,
  "agent_id": "atlas_agent_001",
  "submitted_at": "2024-01-15T10:30:00Z",
  "priority": "high"
}
```

#### GET /agents/{agent_id}/tasks
Get task history for an agent.

**Parameters:**
- `status` (optional): Filter by status
- `limit` (optional): Maximum results (default: 20)
- `offset` (optional): Pagination offset

#### GET /tasks/{task_id}
Get task status and results.

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "progress": 1.0,
  "agent_id": "atlas_agent_001",
  "submitted_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "execution_time_seconds": 130,
  "results": {
    "clusters_found": 15,
    "cluster_centers": [
      {"lat": 51.5074, "lng": -0.1278, "size": 45},
      {"lat": 48.8566, "lng": 2.3522, "size": 38}
    ],
    "silhouette_score": 0.72,
    "visualization_url": "https://api.terra-constellata.ai/v2/files/viz_123.png"
  },
  "metadata": {
    "input_dataset": "cultural_sites_europe",
    "algorithm_used": "dbscan",
    "parameters": {"eps": 0.5, "min_samples": 5}
  }
}
```

### Data Management

#### POST /data/upload
Upload data file for processing.

**Request:**
```http
POST /data/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

# Form data:
# file: <CSV/GeoJSON/Shapefile>
# data_type: cultural_sites | environmental_data | textual_records
# name: "European Cultural Sites Dataset"
# description: "Comprehensive dataset of cultural sites across Europe"
# tags: ["archaeology", "europe", "cultural_heritage"]
```

**Response:**
```json
{
  "upload_id": "upload_123",
  "status": "processing",
  "file_name": "cultural_sites.csv",
  "file_size_bytes": 2457600,
  "data_type": "cultural_sites",
  "records_count": 1250,
  "columns": ["name", "latitude", "longitude", "entity", "description"],
  "validation_results": {
    "valid_records": 1245,
    "invalid_records": 5,
    "errors": [
      {"row": 125, "column": "latitude", "error": "Invalid coordinate range"}
    ]
  },
  "estimated_processing_time_seconds": 45
}
```

#### GET /data/uploads/{upload_id}
Check upload processing status.

#### GET /data/datasets
List available datasets.

**Parameters:**
- `type` (optional): Filter by data type
- `tags` (optional): Filter by tags
- `owner` (optional): Filter by owner
- `limit` (optional): Maximum results

**Response:**
```json
{
  "datasets": [
    {
      "id": "dataset_456",
      "name": "European Cultural Sites",
      "type": "cultural_sites",
      "description": "Comprehensive archaeological sites database",
      "owner": "user_123",
      "created_at": "2024-01-10T09:00:00Z",
      "updated_at": "2024-01-15T14:30:00Z",
      "record_count": 1245,
      "size_bytes": 1843200,
      "tags": ["archaeology", "europe", "cultural_heritage"],
      "access_level": "public",
      "quality_score": 0.95,
      "last_accessed": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 47,
  "limit": 20,
  "offset": 0
}
```

#### GET /data/datasets/{dataset_id}
Get detailed dataset information.

#### POST /data/datasets/{dataset_id}/query
Query dataset with filters and aggregations.

**Request:**
```json
{
  "filters": {
    "entity": "monument",
    "latitude": {"$gte": 35.0, "$lte": 55.0},
    "longitude": {"$gte": -10.0, "$lte": 20.0}
  },
  "aggregations": {
    "by_country": {"$group": {"_id": "$country", "count": {"$sum": 1}}},
    "avg_latitude": {"$avg": "$latitude"}
  },
  "sort": {"name": 1},
  "limit": 100,
  "fields": ["name", "latitude", "longitude", "entity", "description"]
}
```

#### DELETE /data/datasets/{dataset_id}
Delete a dataset (owner only).

### Research

#### POST /research/projects
Create a new research project.

**Request:**
```json
{
  "name": "Ancient Trade Route Analysis",
  "description": "Investigation of cultural exchange patterns in ancient Mediterranean",
  "domain": "archaeology",
  "objectives": [
    "Map trade routes between civilizations",
    "Identify cultural exchange patterns",
    "Analyze environmental influences"
  ],
  "agents": ["atlas_agent", "mythology_agent", "linguist_agent"],
  "datasets": ["mediterranean_sites", "trade_records"],
  "timeline": {
    "start_date": "2024-02-01",
    "end_date": "2024-06-30",
    "milestones": [
      {"name": "Data Integration", "date": "2024-02-15"},
      {"name": "Spatial Analysis", "date": "2024-03-15"},
      {"name": "Cultural Analysis", "date": "2024-04-15"}
    ]
  },
  "collaborators": ["user_456", "user_789"],
  "visibility": "private"
}
```

#### GET /research/projects
List research projects.

#### GET /research/projects/{project_id}
Get project details.

#### POST /research/projects/{project_id}/tasks
Add task to research project.

**Request:**
```json
{
  "name": "Spatial Clustering Analysis",
  "description": "Identify clusters of cultural sites",
  "agent": "atlas_agent",
  "task_type": "SPATIAL_ANALYSIS",
  "parameters": {
    "analysis_type": "clustering",
    "algorithm": "dbscan"
  },
  "dependencies": ["data_integration_task"],
  "priority": "high"
}
```

#### GET /research/projects/{project_id}/results
Get consolidated project results.

### Inspiration

#### POST /inspiration/analyze
Analyze data for novelty and inspiration.

**Request:**
```json
{
  "data": [
    "ancient_trade_route_pattern",
    "cultural_diffusion_mechanism",
    "environmental_adaptation_strategy"
  ],
  "context": {
    "domain": "archaeology",
    "cultural_focus": "mediterranean",
    "time_period": "bronze_age"
  },
  "analysis_type": "novelty_detection",
  "sensitivity": 0.8
}
```

**Response:**
```json
{
  "analysis_id": "analysis_789",
  "novelty_score": 0.76,
  "patterns_identified": 3,
  "insights": [
    {
      "type": "cultural_diffusion",
      "confidence": 0.89,
      "description": "Non-linear cultural transmission patterns detected"
    }
  ],
  "creative_prompts": [
    {
      "content": "What if ancient trade routes were not just economic pathways but also cultural evolution accelerators?",
      "potential": 0.82,
      "novelty": 0.91
    }
  ]
}
```

#### POST /inspiration/prompts
Generate creative prompts.

**Request:**
```json
{
  "domain": "mythology",
  "num_prompts": 5,
  "constraints": {
    "theme": "transformation",
    "cultural_elements": true,
    "modern_application": true
  },
  "creativity_level": "high"
}
```

#### POST /inspiration/share
Share inspiration with agents.

### Codex

#### POST /codex/archive
Archive research contribution.

**Request:**
```json
{
  "title": "Novel Spatial Analysis Method",
  "content": {
    "methodology": "Advanced clustering for cultural site analysis",
    "findings": "Identified 15 distinct cultural clusters",
    "significance": "New approach to understanding cultural diffusion"
  },
  "tags": ["archaeology", "spatial_analysis", "methodology"],
  "attribution": {
    "contributors": ["atlas_agent", "human_researcher"],
    "project": "trade_routes_analysis"
  },
  "access_level": "public"
}
```

#### GET /codex/search
Search archived knowledge.

**Parameters:**
- `query`: Search terms
- `domain`: Knowledge domain filter
- `tags`: Tag filters
- `date_range`: Temporal filters

#### POST /codex/narrative
Generate research narrative.

### Spatial Analysis

#### GET /spatial/sites
Query spatial sites with filters.

**Parameters:**
- `bbox`: Bounding box (min_lng,min_lat,max_lng,max_lat)
- `radius`: Radius search (lat,lng,radius_km)
- `entity`: Entity type filter
- `limit`: Result limit

#### POST /spatial/cluster
Perform spatial clustering analysis.

**Request:**
```json
{
  "dataset": "cultural_sites",
  "algorithm": "dbscan",
  "parameters": {
    "eps": 0.5,
    "min_samples": 5
  },
  "features": ["latitude", "longitude"]
}
```

#### POST /spatial/routes
Analyze routes between locations.

#### GET /spatial/patterns
Identify spatial patterns.

### Monitoring

#### GET /health
System health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.1",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "agents": "healthy",
    "cache": "healthy"
  },
  "metrics": {
    "uptime_seconds": 2592000,
    "total_requests": 15420,
    "active_users": 23,
    "response_time_avg_ms": 245
  }
}
```

#### GET /metrics
Detailed system metrics.

#### GET /logs
System logs (admin only).

**Parameters:**
- `service`: Service filter
- `level`: Log level filter
- `start_time`: Start time filter
- `end_time`: End time filter
- `limit`: Result limit

---

## 📊 GraphQL API

### Endpoint
```
POST https://api.terra-constellata.ai/v2/graphql
```

### Schema Overview

```graphql
type Query {
  # Agent queries
  agents(filter: AgentFilter, limit: Int, offset: Int): [Agent!]!
  agent(id: ID!): Agent

  # Data queries
  datasets(filter: DatasetFilter, limit: Int, offset: Int): [Dataset!]!
  dataset(id: ID!): Dataset

  # Research queries
  projects(filter: ProjectFilter, limit: Int, offset: Int): [Project!]!
  project(id: ID!): Project

  # Spatial queries
  spatialSites(filter: SpatialFilter, limit: Int): [SpatialSite!]!
  spatialClusters(dataset: ID!, algorithm: ClusteringAlgorithm): [Cluster!]!

  # Codex queries
  codexEntries(filter: CodexFilter, limit: Int, offset: Int): [CodexEntry!]!
  codexSearch(query: String!, filters: CodexFilter): [CodexEntry!]!
}

type Mutation {
  # Authentication
  login(credentials: LoginInput!): AuthPayload!
  refreshToken(token: String!): AuthPayload!

  # Agent operations
  submitTask(agentId: ID!, task: TaskInput!): Task!
  cancelTask(taskId: ID!): Task!

  # Data operations
  uploadData(file: Upload!, metadata: DataMetadataInput!): UploadResult!
  deleteDataset(datasetId: ID!): Boolean!

  # Research operations
  createProject(project: ProjectInput!): Project!
  updateProject(id: ID!, updates: ProjectUpdateInput!): Project!
  deleteProject(id: ID!): Boolean!

  # Inspiration operations
  analyzeNovelty(data: [String!]!, context: ContextInput): NoveltyAnalysis!
  generatePrompts(domain: String!, count: Int, constraints: PromptConstraints): [CreativePrompt!]!

  # Codex operations
  archiveContribution(contribution: ContributionInput!): CodexEntry!
  updateCodexEntry(id: ID!, updates: CodexUpdateInput!): CodexEntry!
}
```

### Example Queries

#### Get Agent Information
```graphql
query GetAgentDetails($agentId: ID!) {
  agent(id: $agentId) {
    id
    name
    type
    capabilities
    status
    performanceMetrics {
      responseTimeMs
      accuracyScore
      tasksCompleted
    }
    specializations
  }
}
```

#### Complex Research Query
```graphql
query GetResearchOverview($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    description
    status
    collaborators {
      id
      name
      email
    }
    datasets {
      id
      name
      type
      recordCount
    }
    tasks {
      id
      name
      status
      agent {
        name
        type
      }
      results {
        ... on SpatialAnalysisResult {
          clustersFound
          silhouetteScore
          visualizationUrl
        }
        ... on CulturalAnalysisResult {
          patternsIdentified
          confidenceScore
        }
      }
    }
    codexContributions {
      id
      title
      tags
      createdAt
    }
  }
}
```

#### Submit Research Task
```graphql
mutation SubmitSpatialAnalysis($agentId: ID!, $datasetId: ID!) {
  submitTask(
    agentId: $agentId,
    task: {
      type: SPATIAL_ANALYSIS,
      parameters: {
        analysisType: "clustering",
        dataset: $datasetId,
        algorithm: "dbscan",
        eps: 0.5,
        minSamples: 5
      },
      priority: HIGH,
      timeoutSeconds: 300
    }
  ) {
    id
    status
    estimatedCompletionSeconds
    submittedAt
  }
}
```

---

## 🤖 A2A Protocol API

### Connection Establishment

#### WebSocket Connection
```javascript
// Connect to A2A server
const ws = new WebSocket('wss://a2a.terra-constellata.ai/v2');

// Authentication
ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: '2.0',
    method: 'auth.authenticate',
    params: {
      agent_id: 'my_agent_001',
      token: 'agent_auth_token'
    },
    id: 1
  }));
};
```

#### HTTP Connection
```javascript
// Direct HTTP request
const response = await fetch('https://a2a.terra-constellata.ai/v2/rpc', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer agent_token'
  },
  body: JSON.stringify({
    jsonrpc: '2.0',
    method: 'agent.register',
    params: { /* agent metadata */ },
    id: 1
  })
});
```

### Agent Registration

```json
{
  "jsonrpc": "2.0",
  "method": "agent.register",
  "params": {
    "agent_id": "custom_agent_001",
    "agent_type": "ANALYSIS",
    "capabilities": ["data_analysis", "pattern_recognition"],
    "specializations": ["statistical_modeling", "machine_learning"],
    "communication_protocols": ["A2A_v2.0"],
    "resource_requirements": {
      "cpu_cores": 2,
      "memory_gb": 4
    },
    "endpoint": "https://my-agent.example.com",
    "health_check_url": "https://my-agent.example.com/health"
  },
  "id": 1
}
```

### Task Execution

#### Submit Task
```json
{
  "jsonrpc": "2.0",
  "method": "agent.task.execute",
  "params": {
    "task_id": "task_123",
    "task_type": "DATA_ANALYSIS",
    "parameters": {
      "dataset": "research_data_001",
      "analysis_type": "correlation_analysis",
      "variables": ["variable_a", "variable_b"]
    },
    "priority": "normal",
    "timeout_seconds": 300
  },
  "id": 2
}
```

#### Task Response
```json
{
  "jsonrpc": "2.0",
  "result": {
    "task_id": "task_123",
    "status": "completed",
    "execution_time_seconds": 45.2,
    "results": {
      "correlation_coefficient": 0.78,
      "p_value": 0.001,
      "confidence_interval": [0.65, 0.87],
      "visualization_url": "https://api.terra-constellata.ai/files/corr_plot_123.png"
    },
    "metadata": {
      "algorithm_used": "pearson_correlation",
      "sample_size": 1250,
      "data_quality_score": 0.94
    }
  },
  "id": 2
}
```

### Collaboration Messages

#### Request Collaboration
```json
{
  "jsonrpc": "2.0",
  "method": "agent.collaborate.request",
  "params": {
    "collaboration_id": "collab_456",
    "initiator": "atlas_agent_001",
    "participants": ["mythology_agent_002", "linguist_agent_003"],
    "collaboration_type": "MULTI_MODAL_ANALYSIS",
    "shared_context": {
      "research_question": "How do spatial and cultural factors interact?",
      "shared_data": "integrated_dataset_789",
      "collaboration_goal": "Comprehensive analysis of cultural-spatial relationships"
    },
    "coordination_strategy": "MASTER_SLAVE",
    "timeout_minutes": 60
  },
  "id": 3
}
```

#### Collaboration Update
```json
{
  "jsonrpc": "2.0",
  "method": "agent.collaborate.update",
  "params": {
    "collaboration_id": "collab_456",
    "update_type": "PROGRESS_UPDATE",
    "progress": 0.6,
    "current_phase": "INTEGRATION_ANALYSIS",
    "partial_results": {
      "spatial_clusters": 12,
      "cultural_patterns": 8,
      "correlation_strength": 0.72
    },
    "next_steps": ["final_synthesis", "result_validation"]
  },
  "id": 4
}
```

### Knowledge Sharing

#### Publish Knowledge
```json
{
  "jsonrpc": "2.0",
  "method": "knowledge.publish",
  "params": {
    "knowledge_id": "knowledge_789",
    "knowledge_type": "ANALYSIS_METHOD",
    "title": "Advanced Spatial-Temporal Correlation Analysis",
    "content": {
      "methodology": "Novel approach combining spatial clustering with temporal analysis",
      "algorithm_details": "...",
      "validation_results": "...",
      "applications": ["archaeology", "cultural_geography"]
    },
    "tags": ["spatial_analysis", "temporal_modeling", "correlation"],
    "access_level": "public",
    "attribution": {
      "contributors": ["atlas_agent_001", "researcher_123"],
      "creation_date": "2024-01-15T10:00:00Z",
      "license": "CC-BY-4.0"
    }
  },
  "id": 5
}
```

#### Query Knowledge
```json
{
  "jsonrpc": "2.0",
  "method": "knowledge.query",
  "params": {
    "query_type": "SEMANTIC_SEARCH",
    "query": "spatial temporal correlation methods",
    "filters": {
      "knowledge_type": "ANALYSIS_METHOD",
      "tags": ["spatial_analysis"],
      "date_range": {
        "start": "2023-01-01",
        "end": "2024-12-31"
      },
      "min_relevance_score": 0.7
    },
    "result_limit": 10,
    "sort_by": "relevance"
  },
  "id": 6
}
```

---

## 🔧 SDKs and Libraries

### Python SDK

#### Installation
```bash
pip install terra-constellata-sdk
```

#### Basic Usage
```python
from terra_constellata import TerraConstellata

# Initialize client
tc = TerraConstellata(
    api_key="your_api_key",
    base_url="https://api.terra-constellata.ai/v2"
)

# Authenticate
await tc.authenticate("username", "password")

# Submit task
task = await tc.agents.submit_task(
    agent_id="atlas_agent_001",
    task_type="SPATIAL_ANALYSIS",
    parameters={
        "dataset": "cultural_sites",
        "analysis_type": "clustering"
    }
)

# Get results
results = await tc.tasks.get_results(task.id)
print(f"Found {results.clusters_found} clusters")
```

#### Advanced Features
```python
# Research project management
project = await tc.research.create_project({
    "name": "Cultural Diffusion Study",
    "agents": ["atlas_agent", "mythology_agent"],
    "datasets": ["cultural_sites", "trade_routes"]
})

# Batch operations
batch = tc.create_batch()
batch.add_task(agent="atlas_agent", task=spatial_task)
batch.add_task(agent="mythology_agent", task=cultural_task)
results = await batch.execute()

# Real-time monitoring
async with tc.monitoring.subscribe("task_completion") as subscription:
    async for event in subscription:
        print(f"Task {event.task_id} completed: {event.status}")
```

### JavaScript SDK

#### Installation
```bash
npm install terra-constellata-sdk
```

#### Usage
```javascript
import { TerraConstellata } from 'terra-constellata-sdk';

const tc = new TerraConstellata({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.terra-constellata.ai/v2'
});

// Authenticate
await tc.authenticate('username', 'password');

// Upload data
const upload = await tc.data.upload(file, {
  type: 'cultural_sites',
  name: 'European Sites Dataset'
});

// Submit analysis
const task = await tc.agents.submitTask('atlas_agent_001', {
  type: 'SPATIAL_ANALYSIS',
  parameters: {
    dataset: upload.datasetId,
    analysisType: 'clustering'
  }
});

// Get results with polling
const results = await tc.tasks.pollResults(task.id);
console.log(`Found ${results.clustersFound} clusters`);
```

### Go SDK

#### Installation
```bash
go get github.com/a2a-world/terra-constellata-go-sdk
```

#### Usage
```go
package main

import (
    "context"
    "log"
    tc "github.com/a2a-world/terra-constellata-go-sdk"
)

func main() {
    client := tc.NewClient("your_api_key")
    ctx := context.Background()

    // Authenticate
    err := client.Authenticate(ctx, "username", "password")
    if err != nil {
        log.Fatal(err)
    }

    // Submit task
    task, err := client.Agents.SubmitTask(ctx, tc.TaskRequest{
        AgentID: "atlas_agent_001",
        Type:    "SPATIAL_ANALYSIS",
        Parameters: map[string]interface{}{
            "dataset": "cultural_sites",
            "analysis_type": "clustering",
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Get results
    results, err := client.Tasks.GetResults(ctx, task.ID)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Found %d clusters", results.ClustersFound)
}
```

---

## 📚 Code Examples

### Complete Research Workflow

```python
import asyncio
from terra_constellata import TerraConstellata

async def run_research_workflow():
    # Initialize client
    tc = TerraConstellata(api_key="your_key")

    # Create research project
    project = await tc.research.create_project({
        "name": "Ancient Mediterranean Trade Analysis",
        "description": "Comprehensive study of cultural exchange patterns",
        "agents": ["atlas_agent", "mythology_agent", "linguist_agent"],
        "datasets": ["mediterranean_sites", "trade_records"]
    })

    # Upload and process data
    upload = await tc.data.upload("sites.csv", {
        "type": "cultural_sites",
        "name": "Mediterranean Cultural Sites"
    })

    # Submit spatial analysis
    spatial_task = await tc.agents.submit_task(
        "atlas_agent_001",
        "SPATIAL_ANALYSIS",
        {"dataset": upload.dataset_id, "analysis_type": "clustering"}
    )

    # Submit cultural analysis
    cultural_task = await tc.agents.submit_task(
        "mythology_agent_002",
        "CULTURAL_ANALYSIS",
        {"dataset": upload.dataset_id, "focus": "exchange_patterns"}
    )

    # Wait for results
    spatial_results = await tc.tasks.wait_for_completion(spatial_task.id)
    cultural_results = await tc.tasks.wait_for_completion(cultural_task.id)

    # Generate integrated insights
    insights = await tc.inspiration.analyze_novelty([
        f"Spatial clusters: {spatial_results.clusters_found}",
        f"Cultural patterns: {cultural_results.patterns_identified}"
    ], {
        "domain": "archaeology",
        "context": "mediterranean_trade"
    })

    # Archive findings
    await tc.codex.archive({
        "title": "Mediterranean Cultural Exchange Patterns",
        "content": {
            "spatial_analysis": spatial_results,
            "cultural_analysis": cultural_results,
            "integrated_insights": insights
        },
        "tags": ["archaeology", "cultural_exchange", "mediterranean"]
    })

    print("Research workflow completed successfully!")

# Run the workflow
asyncio.run(run_research_workflow())
```

### Agent Development Template

```python
from terra_constellata.agents import BaseAgent
from terra_constellata.a2a import A2AClient

class CustomAnalysisAgent(BaseAgent):
    def __init__(self, agent_id, specialization):
        super().__init__(agent_id, [
            "custom_analysis",
            "data_processing",
            "insight_generation"
        ])

        self.specialization = specialization
        self.a2a_client = A2AClient()
        self.analysis_model = self.load_model(specialization)

    async def initialize(self):
        """Initialize agent and register with system"""
        await self.a2a_client.connect()
        await self.register_with_system()
        self.logger.info(f"Custom agent {self.agent_id} initialized")

    async def handle_task(self, task_message):
        """Process incoming tasks"""
        task_type = task_message["params"]["task_type"]

        if task_type == "CUSTOM_ANALYSIS":
            result = await self.perform_custom_analysis(
                task_message["params"]
            )
        else:
            result = {"error": f"Unsupported task type: {task_type}"}

        # Send response
        await self.a2a_client.send_response(task_message["id"], result)

    async def perform_custom_analysis(self, parameters):
        """Execute custom analysis logic"""
        try:
            # Load data
            data = await self.load_data(parameters["dataset"])

            # Apply analysis
            analysis_result = await self.analysis_model.analyze(data)

            # Generate insights
            insights = await self.generate_insights(analysis_result)

            return {
                "status": "success",
                "analysis_result": analysis_result,
                "insights": insights,
                "confidence_score": self.calculate_confidence(analysis_result)
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__
            }

    async def collaborate(self, collaboration_request):
        """Handle collaboration requests"""
        # Implementation for multi-agent collaboration
        pass

    async def share_knowledge(self, knowledge_request):
        """Share specialized knowledge"""
        # Implementation for knowledge sharing
        pass

# Usage
agent = CustomAnalysisAgent("custom_agent_001", "archaeological_analysis")
await agent.initialize()
```

---

## 🔒 Security

### Authentication Methods

#### API Key Authentication
```http
Authorization: Bearer <api_key>
```

#### JWT Token Authentication
```http
Authorization: Bearer <jwt_token>
```

#### OAuth 2.0
```http
Authorization: Bearer <oauth_token>
```

### Data Encryption

#### At Rest
- AES-256 encryption for stored data
- Key rotation every 90 days
- Secure key management service

#### In Transit
- TLS 1.3 encryption for all communications
- Perfect forward secrecy
- Certificate pinning for mobile clients

### Access Control

#### Role-Based Access Control (RBAC)
```json
{
  "roles": {
    "researcher": {
      "permissions": ["read_datasets", "execute_tasks", "create_projects"]
    },
    "agent": {
      "permissions": ["read_assigned_data", "execute_tasks", "report_results"]
    },
    "admin": {
      "permissions": ["*"]
    }
  }
}
```

#### API Permissions
- `read`: View data and results
- `write`: Create/modify resources
- `execute`: Run tasks and analyses
- `admin`: System administration
- `agent`: Agent-specific operations

### Security Best Practices

#### API Usage
- Use HTTPS for all requests
- Implement proper error handling
- Validate all input data
- Use pagination for large result sets
- Implement exponential backoff for retries

#### Data Protection
- Never log sensitive information
- Use parameterized queries
- Implement data sanitization
- Regular security audits
- Compliance with GDPR/CCPA

---

## 📊 Rate Limits

### API Rate Limits

| Endpoint Type | Limit | Window | Burst |
|---------------|-------|--------|-------|
| Authentication | 10 req | 1 min | 2 req |
| Data Upload | 5 req | 1 hour | 1 req |
| Task Submission | 50 req | 1 hour | 10 req |
| Data Queries | 1000 req | 1 hour | 100 req |
| Agent Operations | 100 req | 1 hour | 20 req |
| Monitoring | 500 req | 1 hour | 50 req |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Burst: 100
```

### Handling Rate Limits

#### Client-Side Implementation
```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RateLimitedSession(requests.Session):
    def __init__(self, rate_limit=100, window_seconds=3600):
        super().__init__()
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests_made = []
        self.adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504]
            )
        )
        self.mount('https://', self.adapter)

    def request(self, *args, **kwargs):
        # Clean old requests
        current_time = time.time()
        self.requests_made = [
            req_time for req_time in self.requests_made
            if current_time - req_time < self.window_seconds
        ]

        # Check rate limit
        if len(self.requests_made) >= self.rate_limit:
            sleep_time = self.window_seconds - (current_time - self.requests_made[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.requests_made = []

        # Make request
        response = super().request(*args, **kwargs)

        # Record request
        self.requests_made.append(time.time())

        return response
```

#### Server-Side Handling
```python
# Handle rate limit exceeded
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "retry_after": e.description,
        "limit": request.rate_limit.limit,
        "remaining": request.rate_limit.remaining,
        "reset_time": request.rate_limit.reset_time
    }), 429
```

---

## 🚨 Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 502 | Bad Gateway | Gateway error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "latitude",
      "issue": "Value must be between -90 and 90",
      "provided_value": 91.5
    },
    "request_id": "req_123456",
    "timestamp": "2024-01-15T10:30:00Z",
    "documentation_url": "https://docs.terra-constellata.ai/errors/validation"
  }
}
```

### Common Error Codes

#### Authentication Errors
- `INVALID_CREDENTIALS`: Username/password incorrect
- `TOKEN_EXPIRED`: Authentication token expired
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `ACCOUNT_DISABLED`: User account is disabled

#### Validation Errors
- `MISSING_REQUIRED_FIELD`: Required field not provided
- `INVALID_FIELD_FORMAT`: Field format incorrect
- `VALUE_OUT_OF_RANGE`: Value exceeds allowed range
- `INVALID_REFERENCE`: Referenced resource doesn't exist

#### System Errors
- `SERVICE_UNAVAILABLE`: Service temporarily down
- `DATABASE_ERROR`: Database operation failed
- `AGENT_UNAVAILABLE`: Requested agent not available
- `RESOURCE_EXHAUSTED`: System resources exhausted

#### Task-Specific Errors
- `TASK_TIMEOUT`: Task execution exceeded time limit
- `INVALID_TASK_PARAMETERS`: Task parameters invalid
- `DATASET_NOT_FOUND`: Referenced dataset doesn't exist
- `ANALYSIS_FAILED`: Analysis execution failed

### Error Recovery Strategies

#### Client-Side Error Handling
```python
class APIClient:
    async def make_request(self, method, url, **kwargs):
        try:
            response = await self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            await self.handle_http_error(e)
        except requests.exceptions.ConnectionError as e:
            await self.handle_connection_error(e)
        except requests.exceptions.Timeout as e:
            await self.handle_timeout_error(e)

    async def handle_http_error(self, error):
        status_code = error.response.status_code
        error_data = error.response.json()

        if status_code == 401:
            # Token expired, refresh and retry
            await self.refresh_token()
            return await self.retry_request(error.request)
        elif status_code == 429:
            # Rate limited, wait and retry
            retry_after = error.response.headers.get('Retry-After', 60)
            await asyncio.sleep(int(retry_after))
            return await self.retry_request(error.request)
        elif status_code >= 500:
            # Server error, exponential backoff
            await self.exponential_backoff(error.request)
        else:
            # Client error, raise exception
            raise APIError(f"API Error: {error_data['error']['message']}")

    async def exponential_backoff(self, request, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await self.session.send(request)
            except Exception:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                else:
                    raise
```

#### Error Logging and Monitoring
```python
import logging
import sentry_sdk

# Configure error logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure Sentry for error tracking
sentry_sdk.init(
    dsn="your_sentry_dsn",
    environment="production",
    release="terra-constellata@v2.0.1"
)

# Error handler decorator
def handle_api_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"API Error in {func.__name__}: {e}")
            sentry_sdk.capture_exception(e)

            # Return appropriate error response
            return {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "request_id": sentry_sdk.last_event_id()
                }
            }
    return wrapper
```

---

## 🧪 Testing

### API Testing Tools

#### Postman Collection
```json
{
  "info": {
    "name": "Terra Constellata API",
    "version": "2.0",
    "description": "Complete API test collection"
  },
  "variable": [
    {
      "key": "base_url",
      "value": "https://api.terra-constellata.ai/v2"
    },
    {
      "key": "api_key",
      "value": "{{api_key}}"
    }
  ]
}
```

#### Newman CLI Testing
```bash
# Install Newman
npm install -g newman

# Run API tests
newman run terra_constellata_api.postman_collection.json \
       --environment test_environment.json \
       --reporters cli,json \
       --reporter-json-export results.json
```

### Unit Testing Examples

#### Authentication Testing
```python
import pytest
from terra_constellata.api.auth import AuthService

class TestAuthService:
    @pytest.fixture
    async def auth_service(self):
        return AuthService()

    @pytest.mark.asyncio
    async def test_successful_login(self, auth_service):
        credentials = {"username": "test@example.com", "password": "password123"}

        result = await auth_service.login(credentials)

        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_invalid_credentials(self, auth_service):
        credentials = {"username": "test@example.com", "password": "wrong"}

        with pytest.raises(AuthenticationError):
            await auth_service.login(credentials)

    @pytest.mark.asyncio
    async def test_token_refresh(self, auth_service):
        refresh_token = "valid_refresh_token"

        result = await auth_service.refresh_token(refresh_token)

        assert "access_token" in result
        assert result["token_type"] == "Bearer"
```

#### Agent Task Testing
```python
import pytest
from terra_constellata.api.agents import AgentService
from unittest.mock import Mock, AsyncMock

class TestAgentService:
    @pytest.fixture
    def agent_service(self):
        return AgentService()

    @pytest.fixture
    def mock_agent_repo(self):
        repo = Mock()
        repo.get_agent = AsyncMock()
        repo.submit_task = AsyncMock()
        return repo

    def test_task_submission_validation(self, agent_service):
        # Test invalid task parameters
        invalid_task = {
            "agent_id": "atlas_agent_001",
            "task_type": "INVALID_TYPE"
        }

        with pytest.raises(ValidationError):
            agent_service.validate_task(invalid_task)

    @pytest.mark.asyncio
    async def test_successful_task_submission(self, agent_service, mock_agent_repo):
        agent_service.agent_repo = mock_agent_repo

        task_request = {
            "agent_id": "atlas_agent_001",
            "task_type": "SPATIAL_ANALYSIS",
            "parameters": {"dataset": "test_dataset"}
        }

        mock_agent_repo.submit_task.return_value = {
            "task_id": "task_123",
            "status": "queued"
        }

        result = await agent_service.submit_task(task_request)

        assert result["task_id"] == "task_123"
        assert result["status"] == "queued"
        mock_agent_repo.submit_task.assert_called_once_with(task_request)
```

### Integration Testing

#### End-to-End Workflow Test
```python
import pytest
from terra_constellata.test_utils import TerraConstellataTestClient

class TestResearchWorkflow:
    @pytest.fixture
    async def test_client(self):
        client = TerraConstellataTestClient()
        await client.setup_test_data()
        yield client
        await client.cleanup()

    @pytest.mark.asyncio
    async def test_complete_research_workflow(self, test_client):
        # 1. Create project
        project = await test_client.create_test_project({
            "name": "Test Research Project",
            "agents": ["atlas_agent", "mythology_agent"]
        })

        # 2. Upload test data
        dataset = await test_client.upload_test_dataset(
            "cultural_sites.csv",
            {"type": "cultural_sites"}
        )

        # 3. Submit analysis tasks
        spatial_task = await test_client.submit_task({
            "agent": "atlas_agent",
            "type": "SPATIAL_ANALYSIS",
            "dataset": dataset["id"]
        })

        cultural_task = await test_client.submit_task({
            "agent": "mythology_agent",
            "type": "CULTURAL_ANALYSIS",
            "dataset": dataset["id"]
        })

        # 4. Wait for completion
        spatial_result = await test_client.wait_for_task(spatial_task["id"])
        cultural_result = await test_client.wait_for_task(cultural_task["id"])

        # 5. Verify results
        assert spatial_result["status"] == "completed"
        assert "clusters_found" in spatial_result["results"]
        assert cultural_result["status"] == "completed"
        assert "patterns_identified" in cultural_result["results"]

        # 6. Test result integration
        integrated_results = await test_client.integrate_results([
            spatial_result, cultural_result
        ])

        assert "integrated_analysis" in integrated_results
        assert integrated_results["confidence_score"] > 0.5
```

### Load Testing

#### Using Locust
```python
from locust import HttpUser, task, between

class TerraConstellataUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def get_agents(self):
        self.client.get("/agents", headers=self.auth_headers)

    @task(2)
    def submit_task(self):
        task_data = {
            "agent_id": "atlas_agent_001",
            "task_type": "SPATIAL_ANALYSIS",
            "parameters": {"dataset": "test_dataset"}
        }
        self.client.post(
            "/agents/atlas_agent_001/task",
            json=task_data,
            headers=self.auth_headers
        )

    @task(1)
    def get_task_status(self):
        self.client.get("/tasks/task_123", headers=self.auth_headers)

    def on_start(self):
        # Login and get auth token
        response = self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_pass"
        })
        self.auth_headers = {
            "Authorization": f"Bearer {response.json()['access_token']}"
        }
```

#### Running Load Tests
```bash
# Install Locust
pip install locust

# Run load test
locust -f load_test.py --host https://api.terra-constellata.ai/v2

# Open web interface at http://localhost:8089
# Configure test parameters and run
```

### API Contract Testing

#### Using Pact
```python
import pytest
from pact import Consumer, Provider
from terra_constellata.api.client import APIClient

pact = Consumer('TerraConstellataClient').has_pact_with(
    Provider('TerraConstellataAPI'),
    pact_dir='./pacts'
)

class TestAPIContract:
    @pact.given('an agent exists')
    @pact.when('I request the agent details')
    @pact.then('I receive the agent details')
    def test_get_agent_details(self):
        expected_response = {
            "id": "atlas_agent_001",
            "name": "Atlas Agent",
            "capabilities": ["spatial_analysis"]
        }

        (pact
         .get('/agents/atlas_agent_001')
         .will_respond_with(200)
         .with_json(expected_response))

        # Test implementation
        client = APIClient()
        response = client.get_agent('atlas_agent_001')

        assert response == expected_response
```

---

*"This API Reference provides comprehensive documentation for integrating with Terra Constellata's powerful multi-agent research platform. Whether you're building custom agents, developing research applications, or integrating with existing systems, these APIs provide the foundation for collaborative AI-human research."*

**API Version:** 2.0 | **Last Updated:** January 2024 | **Contact:** api@terra-constellata.ai