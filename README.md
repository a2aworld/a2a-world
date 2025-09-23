# **Terra Constellata**

Terra Constellata is a comprehensive, multi-service platform designed for advanced geospatial and knowledge graph analysis. It integrates AI agents (LangChain), a geospatial database (PostGIS), and a graph database (ArangoDB) to create a powerful insight engine.

## **Core Components**

* **PostGIS:** The primary database for storing and querying geospatial "ground truth" data.
* **ArangoDB:** A multi-model database used to house the Cognitive Knowledge Graph (CKG), connecting disparate pieces of information.
* **A2A Protocol Server:** A JSON-RPC server that facilitates communication between different AI agents.
* **Data Gateway Agents:** 50 specialized agents providing authenticated access to authoritative global datasets (GEBCO, NASA, DPLA, etc.).
* **Backend API:** The main FastAPI server that orchestrates the services and exposes endpoints for the frontend.
* **React App:** The primary user interface for interacting with the platform.
* **Monitoring Stack:** Includes Prometheus for collecting metrics and Grafana for visualization and dashboards.

## **Getting Started**

This project is fully containerized using Docker and Docker Compose.

### **Prerequisites**

* Docker  
* Docker Compose

### **Running the Application**

1. **Clone the repository:**

```
git clone <your-repo-url>
cd terra-constellata
```

2.   
   **Start all services:**

```
docker-compose up -d --build
```

## **Service Endpoints**

Once running, the services are available at the following default ports:

* **React App:** `http://localhost:3000`  
* **Backend API:** `http://localhost:8000`  
* **A2A Server:** `http://localhost:8080`  
* **ArangoDB UI:** `http://localhost:8529`  
* **Grafana:** `http://localhost:3001`  
* **Prometheus:** `http://localhost:9090`  
* **PostgreSQL:** `localhost:5432`

## **Data Gateway Agents**

Terra Constellata includes 50 specialized data gateway agents that provide authenticated access to authoritative global datasets. These agents serve as the "Library of Alexandria for AI Wisdom," enabling AI systems to access critical geospatial, cultural, scientific, and infrastructural data.

### **Agent Categories**
- **Geospatial (10 agents):** GEBCO bathymetry, NASA Landsat, ESA Sentinel, USGS seismic data, etc.
- **Cultural Heritage (10 agents):** Wikidata, DPLA, Europeana, Internet Archive, Project Gutenberg, etc.
- **Linguistic (10 agents):** Glottolog, WordNet, Getty vocabularies, Wiktionary, etc.
- **Scientific (5 agents):** NASA ADS, GBIF biodiversity, CDC health data, World Bank, PubChem
- **Infrastructure (7 agents):** Registry, orchestrator, validator, reputation, ontology services

### **Key Features**
- **Secure Authentication:** API key and OAuth2 support with secrets management
- **A2A Protocol Integration:** Inter-agent communication and data sharing
- **Containerized Deployment:** Docker-based deployment with health monitoring
- **Schema Compliance:** Conforms to adapted A2A_World_VIA_Agent_Schema_v1.0

For detailed information, see: `agents/data_gateway/GENESIS_REPORT.md`

