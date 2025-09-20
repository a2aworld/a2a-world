# Tool Shed - Dynamic Tool Registry System

The Tool Shed is a comprehensive system for agent self-improvement in the Terra Constellata project. It enables agents to propose, validate, register, and discover new tools autonomously, creating a dynamic ecosystem of capabilities that evolves over time.

## 🏗️ Architecture

The Tool Shed consists of several key components:

### Core Components

- **ToolRegistry**: Dynamic registry for managing tool lifecycle
- **ToolSmithAgent**: Gatekeeper for tool validation and approval
- **ToolVectorStore**: Vector database for semantic tool discovery
- **SemanticSearchEngine**: Advanced search with ranking and filtering
- **ToolEvolutionManager**: Versioning and evolution support

### Data Models

- **Tool**: Complete tool definition with metadata, capabilities, and code
- **ToolProposal**: Proposal for new tools from agents
- **ToolVersion**: Version history and change tracking
- **ToolEvolutionRequest**: Requests for tool improvements and updates

## 🚀 Quick Start

### Installation

The Tool Shed uses existing dependencies from the Terra Constellata project:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `chromadb` for vector storage
- `sentence-transformers` for embeddings
- `pydantic` for data models
- `faiss-cpu` for similarity search

### Basic Usage

```python
from tool_shed import ToolRegistry, ToolVectorStore, ToolSmithAgent
from tool_shed.models import Tool, ToolMetadata, ToolCapabilities

# Initialize components
vector_store = ToolVectorStore()
registry = ToolRegistry(vector_store)

# Create a tool
tool = Tool(
    metadata=ToolMetadata(
        name="DataAnalyzer",
        description="Advanced data analysis tool",
        author="DataAgent",
        category="data_science",
        tags=["analysis", "statistics"]
    ),
    capabilities=ToolCapabilities(
        functions=["analyze", "visualize"],
        input_types=["csv", "json"],
        output_types=["charts", "reports"]
    ),
    code="def analyze(data): return {'result': 'analyzed'}",
    documentation="Analyzes data and generates insights."
)

# Register the tool
await registry.register_tool(tool)

# Search for tools
from tool_shed.search import SemanticSearchEngine
search_engine = SemanticSearchEngine(registry, vector_store)

results = await search_engine.search("data analysis")
print(f"Found {results.total_count} tools")
```

## 🔧 Components

### ToolRegistry

The central registry manages tool lifecycle:

```python
# Register a tool
await registry.register_tool(tool)

# Update a tool
await registry.update_tool(tool_id, updated_tool, "author")

# Search tools
tools = registry.list_tools(category="nlp")

# Get tool statistics
stats = registry.get_registry_stats()
```

### ToolSmithAgent

The gatekeeper validates and approves tools:

```python
# Initialize ToolSmith
tool_smith = ToolSmithAgent(llm=your_llm, registry=registry)

# Validate a proposal
result = await tool_smith._validate_proposal(proposal_id)

# Approve a tool
await tool_smith._approve_proposal(proposal_id, "Approved by ToolSmith")
```

### Semantic Search

Advanced search capabilities:

```python
# Basic search
query = SearchQuery(query="machine learning", limit=10)
results = await search_engine.search(query)

# Advanced search with filters
results = search_engine.advanced_search(
    query="text processing",
    filters={"category": "nlp", "min_rating": 4.0},
    sort_by="rating"
)

# Find similar tools
similar = registry.get_similar_tools(tool_id, limit=5)
```

### Tool Evolution

Manage tool versions and improvements:

```python
# Create evolution request
evolution_id = await evolution_manager.create_evolution_request(
    tool_id=tool.id,
    evolution_type="enhancement",
    description="Add ML capabilities",
    proposed_changes={"functions": ["predict", "classify"]},
    requester_agent="MLAgent"
)

# Approve evolution
await evolution_manager.approve_evolution_request(
    evolution_id, "ToolSmith", "2.0.0"
)

# Get evolution history
history = evolution_manager.get_evolution_history(tool_id)
```

## 🤖 Agent Autonomy

The Tool Shed enables autonomous agent behavior:

### Tool Discovery

Agents can discover tools based on their needs:

```python
# Agent searches for capabilities
results = await search_engine.search("image processing")

if results.total_count == 0:
    # No existing tools - propose new one
    proposal = ToolProposal(
        proposer_agent="VisionAgent",
        tool_name="ImageProcessor",
        description="Process and analyze images",
        capabilities=["detect_objects", "classify_images"],
        use_case="Computer vision tasks"
    )

    proposal_id = await registry.submit_proposal(proposal)
```

### Tool Evolution

Agents can request improvements to existing tools:

```python
# Request enhancement
await evolution_manager.create_evolution_request(
    tool_id=existing_tool.id,
    evolution_type="enhancement",
    description="Add GPU acceleration",
    proposed_changes={"performance": "gpu_accelerated"},
    requester_agent="PerformanceAgent"
)
```

## 🔒 Security & Validation

The ToolSmithAgent provides comprehensive validation:

### Security Scanning

- Detects dangerous patterns (exec, eval, etc.)
- Checks for network and file system access
- Validates security levels

### Code Quality

- Syntax checking
- PEP 8 compliance
- Linting for style issues

### Testing

- Unit test execution
- Coverage analysis
- Integration testing

## 📊 Vector Database

Uses ChromaDB for efficient tool discovery:

### Features

- Semantic similarity search
- Metadata filtering
- Hybrid search combining text and metadata
- Persistent storage

### Configuration

```python
vector_store = ToolVectorStore(
    persist_directory="./tool_shed_db",
    model_name="all-MiniLM-L6-v2",  # Embedding model
    collection_name="tool_shed"
)
```

## 🔄 Tool Evolution

Comprehensive versioning and evolution support:

### Version Management

- Semantic versioning (major.minor.patch)
- Breaking change detection
- Backward compatibility checking

### Evolution Types

- **Enhancement**: New features
- **Bug Fix**: Issue resolution
- **Optimization**: Performance improvements
- **Security**: Security updates
- **Deprecation**: Feature removal

## 📈 Monitoring & Analytics

Track tool usage and performance:

```python
# Get registry statistics
stats = registry.get_registry_stats()

# Get trending tools
trending = search_engine.get_trending_tools(days=7)

# Get tools by category
nlp_tools = search_engine.get_tools_by_category("nlp", sort_by="rating")
```

## 🧪 Testing

Run the example usage:

```bash
python -m tool_shed.example_usage
```

This demonstrates:
- Tool registration
- Search functionality
- Evolution requests
- Agent autonomy

## 🔧 Configuration

### Environment Variables

```bash
# Vector database
TOOL_SHED_DB_PATH=./tool_shed_db

# Embedding model
TOOL_SHED_EMBEDDING_MODEL=all-MiniLM-L6-v2

# A2A Protocol
A2A_SERVER_URL=http://localhost:8080
```

### Advanced Configuration

```python
# Custom vector store
vector_store = ToolVectorStore(
    persist_directory=custom_path,
    model_name="custom-model"
)

# Custom registry
registry = ToolRegistry(
    vector_store=vector_store,
    auto_validate=True  # Auto-validate proposals
)
```

## 🤝 Integration with A2A Protocol

The Tool Shed integrates with the existing A2A protocol:

- Uses `ToolProposal` messages for agent communication
- Supports distributed tool validation
- Enables cross-agent tool sharing

## 📚 API Reference

### ToolRegistry

- `register_tool(tool)`: Register a new tool
- `update_tool(tool_id, updated_tool, author)`: Update existing tool
- `get_tool(tool_id)`: Retrieve tool by ID
- `list_tools(filters)`: List tools with filtering
- `search_tools(query)`: Semantic search

### ToolSmithAgent

- `validate_proposal(proposal_id)`: Validate tool proposal
- `approve_proposal(proposal_id, comments)`: Approve proposal
- `reject_proposal(proposal_id, reason)`: Reject proposal

### SemanticSearchEngine

- `search(query)`: Basic semantic search
- `advanced_search(query, filters, sort_by)`: Advanced search
- `find_similar_tools(tool_id)`: Find similar tools

### ToolEvolutionManager

- `create_evolution_request(...)`: Create evolution request
- `approve_evolution_request(request_id, reviewer)`: Approve evolution
- `get_evolution_history(tool_id)`: Get version history

## 🚀 Future Enhancements

- **Federated Tool Sheds**: Cross-system tool sharing
- **Tool Marketplaces**: Community tool exchange
- **Automated Tool Generation**: AI-powered tool creation
- **Performance Profiling**: Tool benchmarking
- **Dependency Management**: Automatic dependency resolution

## 📄 License

Part of the Terra Constellata project.

---

The Tool Shed transforms static agent capabilities into a dynamic, evolving ecosystem where agents can continuously improve themselves through tool discovery, creation, and evolution.