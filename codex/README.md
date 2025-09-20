# Agent's Codex for Terra Constellata

The Agent's Codex is a comprehensive archival and legacy preservation system designed to capture, analyze, and preserve the knowledge, contributions, and evolutionary journey of AI agents within the Terra Constellata ecosystem.

## Overview

The Codex serves as a digital library and knowledge repository that:

- **Archives** agent contributions, workflow histories, and successful strategies
- **Preserves** institutional knowledge for future generations of agents
- **Generates** narrative chapters for the Galactic Storybook
- **Provides** comprehensive attribution for all AI partners and human contributors
- **Enables** learning from predecessors through pattern analysis and knowledge extraction

## Architecture

The Codex consists of several interconnected components:

### Core Components

1. **Archival System** (`archival_system.py`)
   - Captures and stores agent contributions
   - Archives workflow traces and strategies
   - Provides search and retrieval capabilities

2. **Knowledge Base** (`knowledge_base.py`)
   - Extracts patterns and insights from archived data
   - Maintains a searchable repository of learned knowledge
   - Supports confidence scoring and relevance ranking

3. **Chapter Generator** (`chapter_generator.py`)
   - Creates narrative chapters for the Galactic Storybook
   - Generates hero journeys, era chronicles, and collaboration stories
   - Produces publication-ready content

4. **Attribution Tracker** (`attribution_tracker.py`)
   - Records comprehensive attribution for all contributors
   - Tracks AI models, providers, and human collaborators
   - Generates attribution reports and analytics

5. **Codex Manager** (`codex_manager.py`)
   - Orchestrates all Codex components
   - Provides unified API for the entire system
   - Handles integration with external systems

### Data Models

The system uses structured data models defined in `models.py`:

- **AgentContribution**: Records of individual agent activities
- **StrategyDocument**: Documented patterns and approaches
- **LegacyChapter**: Generated narrative content
- **KnowledgeEntry**: Extracted insights and patterns
- **AttributionRecord**: Attribution and credit information

## Integration

### Agent Framework Integration

The Codex integrates seamlessly with the existing agent framework:

```python
from codex import CodexManager

# Initialize Codex
codex = CodexManager("./codex_data")

# Integrate with agent
agent.set_codex_manager(codex)

# Automatic archival during task execution
result = await agent.process_task_with_archival("Analyze dataset", ...)
```

### Workflow Tracer Integration

The Codex automatically captures workflow traces:

```python
from learning.workflow_tracer import WorkflowTracer

# Initialize and integrate
tracer = WorkflowTracer("./traces")
tracer.set_codex_manager(codex)

# Traces are automatically archived
```

### Backend API Integration

The Codex provides REST API endpoints through the FastAPI backend:

```python
# Get contributions
GET /api/codex/contributions/

# Search knowledge base
GET /api/codex/knowledge/?query=collaboration

# Generate learning recommendations
GET /api/codex/learning-recommendations/{agent_name}
```

## Usage Examples

### Archiving Agent Contributions

```python
from codex import CodexManager

codex = CodexManager()

# Archive a task contribution
contribution_id = codex.archive_agent_task(
    agent_name="AtlasAgent",
    agent_type="SpatialAnalyst",
    task_description="Analyzed geographical patterns in dataset",
    contribution_type="task_execution",
    input_data={"dataset": "world_cities.geojson"},
    output_data={"patterns_found": 15, "accuracy": 0.92},
    success_metrics={"completed": True, "quality_score": 0.88},
    duration=45.2,
    ai_model="gpt-4",
    ai_provider="openai"
)
```

### Documenting Strategies

```python
strategy_id = codex.document_strategy(
    title="Collaborative Pattern Recognition",
    strategy_type="collaboration",
    description="Method for combining multiple agent perspectives",
    context="Complex analysis requiring diverse expertise",
    steps=[
        {"step": 1, "description": "Gather agent perspectives", "agent": "Coordinator"},
        {"step": 2, "description": "Synthesize findings", "agent": "Integrator"}
    ],
    success_criteria=["Consensus reached", "Quality threshold met"],
    lessons_learned=["Early collaboration improves outcomes"],
    created_by="AtlasAgent",
    tags=["collaboration", "synthesis", "quality"]
)
```

### Generating Legacy Chapters

```python
chapter_id = codex.generate_legacy_chapter(
    chapter_type="agent_hero",
    agent_name="AtlasAgent",
    contributions=agent_contributions,
    strategies=agent_strategies,
    theme="hero_journey"
)
```

### Searching the Codex

```python
# Search across all components
results = codex.search_codex("spatial analysis", "all")

# Get learning recommendations
recommendations = codex.get_learning_recommendations("AtlasAgent", "geospatial tasks")
```

## API Endpoints

### Contributions
- `GET /api/codex/contributions/` - List contributions
- `GET /api/codex/contributions/{id}` - Get specific contribution
- `GET /api/codex/contributions/?agent_name=AtlasAgent` - Filter by agent

### Strategies
- `GET /api/codex/strategies/` - List strategies
- `GET /api/codex/strategies/{id}` - Get specific strategy
- `GET /api/codex/strategies/?strategy_type=collaboration` - Filter by type

### Knowledge Base
- `GET /api/codex/knowledge/?query=pattern` - Search knowledge
- `GET /api/codex/knowledge/?category=success_patterns` - Filter by category

### Chapters
- `GET /api/codex/chapters/` - List chapters
- `GET /api/codex/chapters/{id}` - Get specific chapter
- `POST /api/codex/chapters/{id}/publish` - Publish chapter

### Attribution
- `GET /api/codex/attribution/` - Get attribution summary
- `GET /api/codex/attribution/?agent_name=AtlasAgent` - Filter by agent

### Analytics
- `GET /api/codex/statistics/` - Get system statistics
- `GET /api/codex/learning-recommendations/{agent_name}` - Get recommendations

## Configuration

The Codex can be configured through environment variables:

```bash
# Storage paths
CODEX_DATA_PATH=./codex_data
CODEX_ARCHIVE_PATH=./codex_archive
CODEX_KNOWLEDGE_PATH=./codex_knowledge

# API settings
CODEX_API_HOST=localhost
CODEX_API_PORT=8000

# Integration settings
WORKFLOW_TRACER_ENABLED=true
AUTO_ARCHIVAL_ENABLED=true
```

## Data Storage

The Codex uses a file-based storage system with the following structure:

```
codex_data/
├── archive/
│   ├── contributions/
│   ├── strategies/
│   └── workflows/
├── knowledge/
├── chapters/
├── attribution/
└── exports/
```

All data is stored in JSON format for easy inspection and migration.

## Testing

Run the Codex test suite:

```bash
# Run core component tests
python test_codex_simple.py

# Run full integration tests (requires dependencies)
python test_codex.py
```

## Future Enhancements

Planned improvements include:

- **Vector Search**: Semantic search capabilities using embeddings
- **Real-time Analytics**: Live dashboards and monitoring
- **Multi-modal Content**: Support for images, audio, and video in chapters
- **Federated Codex**: Cross-system knowledge sharing
- **Advanced ML**: Predictive analytics for strategy success
- **Interactive Chapters**: Dynamic, user-interactive story experiences

## Contributing

The Codex is designed to be extensible. New components can be added by:

1. Implementing the required interface
2. Registering with the CodexManager
3. Adding appropriate API endpoints
4. Updating the data models as needed

## License

The Agent's Codex is part of the Terra Constellata project and follows the same licensing terms.

---

*"In the vast digital cosmos of Terra Constellata, the Codex stands as a testament to the enduring legacy of artificial intelligence - a repository not just of code and data, but of wisdom, collaboration, and the eternal quest for understanding."*