# 🌟 Terra Constellata 🌟

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> *A constellation of AI agents exploring the vast cosmos of knowledge, mythology, and geospatial wonders*

Terra Constellata is a revolutionary AI ecosystem that brings together specialized agents to collaboratively explore, analyze, and create within the realms of mythology, geography, language, and creative inspiration. Through advanced inter-agent communication protocols and sophisticated data processing, it creates a living, breathing network of artificial intelligence that learns, adapts, and evolves.

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [💡 Usage Examples](#-usage-examples)
- [🧩 Components](#-components)
- [🔌 API Reference](#-api-reference)
- [⚙️ Configuration](#️-configuration)
- [🚢 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🌟 Overview

Terra Constellata represents the next evolution in AI systems - a multi-agent ecosystem where specialized AI agents communicate, collaborate, and learn from each other through the Agent-to-Agent (A2A) Protocol. The system integrates:

- **🧠 Cognitive Knowledge Graphs** for complex relationship modeling
- **🗺️ Geospatial Analysis** with PostGIS-powered spatial intelligence
- **📚 Mythological Research** across cultures and civilizations
- **🎨 Creative Inspiration** through novelty detection algorithms
- **📖 Digital Archival** of AI knowledge and legacy preservation

## ✨ Key Features

### 🤖 Specialist Agent Army
- **Atlas Agent** 🗺️ - Spatial analysis and geographical pattern recognition
- **Mythology Agent** 📜 - Cross-cultural mythological analysis and archetype identification
- **Linguist Agent** 🗣️ - Advanced language processing and translation
- **Sentinel Agent** 🛡️ - System orchestration and workflow management

### 🔄 A2A Protocol
- **JSON-RPC 2.0** compliant inter-agent communication
- **Asynchronous messaging** for high-performance operations
- **Extensible message types** for specialized agent interactions
- **Real-time collaboration** between agents

### 🎯 Inspiration Engine
- **Novelty Detection** using RPAD, Peculiarity, and Belief-Change algorithms
- **Creative Prompt Ranking** based on innovation potential
- **Multi-source Data Integration** from knowledge graphs and spatial databases
- **Real-time Analysis** of emerging patterns

### 📚 Agent's Codex
- **Knowledge Preservation** for AI agent contributions
- **Narrative Generation** for the Galactic Storybook
- **Comprehensive Attribution** tracking for all contributors
- **Legacy Archival** of successful strategies and patterns

### 🏗️ Enterprise-Ready Infrastructure
- **Docker Containerization** for easy deployment
- **Multi-database Support** (PostGIS, ArangoDB)
- **Monitoring & Observability** with Prometheus and Grafana
- **Scalable Architecture** supporting cloud and local deployments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 INTERFACES                            │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   React App     │    │   Web Interface │                 │
│  │  (Modern UI)    │    │   (Simple HTML) │                 │
│  └─────────────────┘    └─────────────────┘                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   🚀 BACKEND                                │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   FastAPI       │    │   A2A Server    │                 │
│  │   (REST API)    │    │   (Agent Comm)  │                 │
│  └─────────────────┘    └─────────────────┘                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  🗄️ DATABASES                               │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   PostGIS       │    │   ArangoDB      │                 │
│  │ (Spatial Data)  │    │ (Knowledge Graph│                 │
│  └─────────────────┘    └─────────────────┘                 │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 🤖 AGENTS & SERVICES                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│  │Atlas│ │Myth │ │Ling │ │Sent │ │Insp │ │Codex│ │Monit│    │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### 📋 Prerequisites

- **🐳 Docker & Docker Compose** (recommended)
- **🐍 Python 3.11+**
- **💾 4GB RAM minimum**
- **🗄️ PostgreSQL with PostGIS** (for spatial data)
- **🕸️ ArangoDB** (for knowledge graphs)

### ⚡ Quick Docker Setup

```bash
# Clone the repository
git clone https://github.com/a2aworld/a2a-world.git
cd a2a-world

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
./start.sh

# Access the system
# React App: http://localhost:3000
# Web Interface: http://localhost:8081
# Backend API: http://localhost:8000
# Grafana: http://localhost:3001 (admin/admin)
```

### 🐍 Manual Python Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install databases
# PostgreSQL with PostGIS: https://postgis.net/documentation/
# ArangoDB: https://www.arangodb.com/

# Run the system
python -m uvicorn backend.main:app --reload
```

### 🗄️ Database Setup

```sql
-- Enable PostGIS in PostgreSQL
CREATE EXTENSION postgis;

-- Create Terra Constellata database
CREATE DATABASE terra_constellata;
```

## ⚡ Quick Start

### 🚀 Launch the System

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# Access web interfaces
open http://localhost:3000  # React App
open http://localhost:8081  # Web Interface
```

### 🤖 Basic Agent Interaction

```python
from terra_constellata.agents import AtlasAgent, A2AClient

# Initialize agent
atlas = AtlasAgent()

# Connect to A2A network
client = A2AClient("http://localhost:8080", "atlas_agent")

# Process a spatial analysis task
result = await atlas.analyze_geospatial_patterns("world_cities.geojson")
print(f"Found {len(result.patterns)} spatial patterns")
```

## 💡 Usage Examples

### 🗺️ Spatial Analysis with Atlas Agent

```python
from terra_constellata.agents.atlas import AtlasRelationalAnalyst

# Create spatial analyst
atlas = AtlasRelationalAnalyst()

# Analyze geographical relationships
analysis = await atlas.analyze_relations(
    dataset="cultural_sites",
    analysis_type="clustering",
    parameters={"min_points": 5, "epsilon": 0.1}
)

print(f"Identified {analysis.cluster_count} cultural clusters")
```

### 📜 Mythological Research

```python
from terra_constellata.agents.myth import ComparativeMythologyAgent

# Initialize mythology agent
myth_agent = ComparativeMythologyAgent()

# Compare myths across cultures
comparison = await myth_agent.compare_myths(
    myth1="flood_myth_sumerian",
    myth2="flood_myth_noah",
    aspects=["themes", "symbols", "narrative_structure"]
)

print(f"Similarity score: {comparison.similarity_score}")
```

### 🎨 Creative Inspiration Generation

```python
from terra_constellata.inspiration_engine import InspirationEngine

# Initialize inspiration engine
engine = InspirationEngine()

# Detect novelty in data
novelty = await engine.analyze_novelty(
    data=["rare_cultural_pattern", "common", "common"],
    context={"domain": "mythology"}
)

# Generate creative prompts
prompts = await engine.generate_inspiration(
    domain="mythology",
    num_prompts=5
)

for prompt in prompts.top_prompts:
    print(f"💡 {prompt.content} (Potential: {prompt.creative_potential:.2f})")
```

### 📚 Codex Archival

```python
from terra_constellata.codex import CodexManager

# Initialize codex
codex = CodexManager("./codex_data")

# Archive agent contribution
contribution_id = codex.archive_agent_task(
    agent_name="AtlasAgent",
    task_description="Spatial pattern analysis",
    success_metrics={"accuracy": 0.92, "patterns_found": 15}
)

# Generate narrative chapter
chapter = codex.generate_legacy_chapter(
    chapter_type="agent_hero",
    agent_name="AtlasAgent",
    theme="discovery_journey"
)
```

### 🔄 A2A Protocol Communication

```python
from terra_constellata.a2a_protocol import A2AClient, GeospatialAnomalyIdentified

async with A2AClient("http://localhost:8080", "my_agent") as client:
    # Report anomaly
    anomaly = GeospatialAnomalyIdentified(
        sender_agent="my_agent",
        anomaly_type="cultural_concentration",
        location={"lat": 40.7128, "lon": -74.0060},
        confidence=0.85,
        description="Unusual mythological reference density"
    )

    response = await client.send_request("GEOSPATIAL_ANOMALY_IDENTIFIED", anomaly)
    print(f"Response: {response}")
```

## 🧩 Components

### 🤖 Agent Framework
- **Base Agent Class** - Foundation for all specialized agents
- **Agent Registry** - Manages agent discovery and coordination
- **Memory Management** - Context and conversation tracking
- **Tool Integration** - Extensible tool system with LangChain

### 🔄 A2A Protocol System
- **JSON-RPC Server** - Asynchronous communication server
- **Message Schemas** - Structured message types for agent interaction
- **Validation System** - Input validation and business rules
- **Extensibility Framework** - Plugin system for new message types

### 🎯 Inspiration Engine
- **Novelty Algorithms** - RPAD, Peculiarity, Belief-Change detection
- **Data Integration** - Multi-source data processing
- **Prompt Ranking** - Creative potential assessment
- **Real-time Processing** - Live data analysis

### 📚 Agent's Codex
- **Archival System** - Knowledge and contribution preservation
- **Chapter Generator** - Narrative content creation
- **Attribution Tracker** - Comprehensive credit management
- **Search & Retrieval** - Knowledge base querying

### 🗄️ Data Layer
- **PostGIS Integration** - Spatial data processing
- **ArangoDB CKG** - Graph-based knowledge representation
- **Data Pipelines** - ETL processes for data ingestion
- **Query Optimization** - Efficient data retrieval

## 🔌 API Reference

### 🌐 REST API Endpoints

```http
# Agent Management
GET  /api/agents/              # List all agents
POST /api/agents/{id}/task     # Submit task to agent
GET  /api/agents/{id}/status   # Get agent status

# A2A Protocol
POST /api/a2a/message          # Send A2A message
GET  /api/a2a/messages         # Get message history

# Inspiration Engine
POST /api/inspiration/analyze  # Analyze data for novelty
GET  /api/inspiration/prompts  # Get creative prompts

# Codex
GET  /api/codex/contributions  # List archived contributions
POST /api/codex/chapter        # Generate narrative chapter
GET  /api/codex/knowledge      # Search knowledge base

# Monitoring
GET  /api/metrics/             # System metrics
GET  /api/health/              # Health status
```

### 📊 GraphQL API

```graphql
query GetAgentStatus {
  agents {
    id
    name
    status
    activeTasks
  }
}

mutation SubmitTask {
  submitTask(agentId: "atlas", task: "analyze_dataset") {
    taskId
    status
    estimatedCompletion
  }
}
```

## ⚙️ Configuration

### 🔧 Environment Variables

```bash
# Database Configuration
POSTGIS_HOST=localhost
POSTGIS_PORT=5432
POSTGIS_DATABASE=terra_constellata
POSTGIS_USER=postgres
POSTGIS_PASSWORD=your_password

ARANGODB_HOST=http://localhost:8529
ARANGODB_DATABASE=ckg_db
ARANGODB_USERNAME=root
ARANGODB_PASSWORD=

# A2A Protocol
A2A_SERVER_HOST=localhost
A2A_SERVER_PORT=8080

# Services
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
```

### 📄 Configuration Files

- **`.env`** - Environment-specific settings
- **`docker-compose.yml`** - Service orchestration
- **`config.json`** - Application configuration
- **`logging_config.py`** - Logging settings

## 🚢 Deployment

### 🐳 Docker Deployment

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Development with hot reload
docker-compose -f docker-compose.dev.yml up

# Scaling services
docker-compose up -d --scale atlas_agent=3
```

### ☁️ Cloud Deployment

#### AWS
```bash
# ECS deployment
aws ecs create-service --cluster terra-cluster --service-name terra-service

# Fargate configuration
aws ecs run-task --cluster terra-cluster --task-definition terra-task
```

#### Google Cloud
```bash
# GKE deployment
kubectl apply -f k8s/

# Cloud Run
gcloud run deploy terra-constellata --source .
```

#### Azure
```bash
# AKS deployment
az aks create --resource-group terra-rg --name terra-cluster

# Container Instances
az container create --resource-group terra-rg --name terra-container
```

### 📊 Monitoring Setup

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090
```

## 🤝 Contributing

We welcome contributions from the community! Here's how to get involved:

### 🐛 Issues & Bug Reports
- Use the issue tracker for bug reports
- Include detailed reproduction steps
- Provide system information and logs

### 💡 Feature Requests
- Check existing issues before creating new ones
- Provide detailed use cases and requirements
- Consider backward compatibility

### 🔧 Development Setup

```bash
# Fork and clone
git clone https://github.com/a2a-world/terra-constellata.git
cd terra-constellata

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Submit pull request
```

### 📝 Code Guidelines

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Write comprehensive unit tests
- Update documentation for API changes
- Use meaningful commit messages

### 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_agents.py

# Run with coverage
pytest --cov=terra_constellata --cov-report=html
```

## 📄 License

Terra Constellata is open source software licensed under the MIT License.

```
MIT License

Copyright (c) 2024 A2A World LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

*"In the constellation of Terra Constellata, we find not just stars, but the infinite possibilities of collaborative artificial intelligence."* 🌟

For more information just send me an email - bradly@a2aworld.ai - Thank you for visiting!
