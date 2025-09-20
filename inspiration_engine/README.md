# Inspiration Engine

The Inspiration Engine is a sophisticated system for detecting novelty and quantifying "interestingness" in geospatial and mythological data. It implements advanced statistical methods including RPAD (Rare Pattern Anomaly Detection), Peculiarity/J-Measure, and Belief-Change Measure to identify novel patterns and generate creative inspiration.

## Features

- **Novelty Detection Algorithms**: Three complementary algorithms for detecting novel patterns
- **Multi-Source Data Integration**: Processes data from Cultural Knowledge Graph (CKG) and PostGIS databases
- **Prompt Ranking**: Ranks creative prompts based on novelty and creative potential
- **A2A Protocol Integration**: Communicates with other agents for collaborative inspiration
- **Real-time Analysis**: Processes recent data changes and provides live insights
- **Caching System**: Efficient caching for improved performance

## Architecture

The Inspiration Engine consists of several key components:

- `algorithms.py`: Novelty detection algorithms (RPAD, Peculiarity, Belief-Change)
- `data_ingestion.py`: Data ingestion from CKG and PostGIS databases
- `prompt_ranking.py`: Creative prompt ranking and generation
- `a2a_integration.py`: A2A protocol integration for agent communication
- `core.py`: Main orchestration engine
- `config.json`: Configuration management

## Installation

Ensure you have the required dependencies from the main project's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Configuration

The engine can be configured via the `config.json` file or environment variables:

```json
{
  "ckg": {
    "host": "http://localhost:8529",
    "database": "ckg_db",
    "username": "root",
    "password": ""
  },
  "postgis": {
    "host": "localhost",
    "port": 5432,
    "database": "terra_constellata",
    "user": "postgres",
    "password": ""
  },
  "a2a": {
    "server_url": "http://localhost:8080",
    "agent_name": "inspiration_engine",
    "timeout": 30.0
  }
}
```

## Quick Start

### Basic Usage

```python
from inspiration_engine import InspirationEngine

# Initialize the engine
engine = InspirationEngine()

# Initialize connections
await engine.initialize()

# Analyze data for novelty
data = ["rare_pattern", "common", "common", "rare_pattern"]
result = await engine.analyze_novelty(data)

print(f"Novelty Score: {result['combined_score']}")
print(f"Is Novel: {result['is_novel']}")

# Generate creative inspiration
inspiration = await engine.generate_inspiration(
    domain="mythology",
    num_prompts=5
)

for prompt in inspiration['top_prompts']:
    print(f"Prompt: {prompt['content']}")
    print(f"Creative Potential: {prompt['creative_potential']:.3f}")

# Shutdown
await engine.shutdown()
```

### Advanced Usage with Custom Configuration

```python
from inspiration_engine import InspirationEngine

# Custom configuration
ckg_config = {
    'host': 'http://localhost:8529',
    'database': 'my_ckg_db',
    'username': 'myuser',
    'password': 'mypass'
}

postgis_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'my_db',
    'user': 'myuser',
    'password': 'mypass'
}

a2a_config = {
    'server_url': 'http://localhost:8080',
    'agent_name': 'custom_inspiration_agent'
}

engine = InspirationEngine(
    ckg_config=ckg_config,
    postgis_config=postgis_config,
    a2a_config=a2a_config
)

await engine.initialize()
```

## Novelty Detection Algorithms

### RPAD (Rare Pattern Anomaly Detection)

Detects rare patterns by analyzing frequency distributions and identifying statistically significant deviations.

```python
from inspiration_engine.algorithms import RPADAlgorithm

algorithm = RPADAlgorithm(significance_threshold=0.05, min_support=5)
result = algorithm.calculate_novelty(data)
print(f"RPAD Score: {result.score}")
```

### Peculiarity/J-Measure

Measures the peculiarity of patterns based on their deviation from expected distributions using information theory.

```python
from inspiration_engine.algorithms import PeculiarityAlgorithm

algorithm = PeculiarityAlgorithm(base=2.0)  # Use base-2 logarithm
context = {
    'class_label': 'positive',
    'features': {'feature1': 3.5, 'feature2': 'rare_value'}
}
result = algorithm.calculate_novelty(data, context)
```

### Belief-Change Measure

Measures how much new information changes existing beliefs using Bayesian updating principles.

```python
from inspiration_engine.algorithms import BeliefChangeAlgorithm

algorithm = BeliefChangeAlgorithm(prior_belief_strength=0.5)
context = {
    'prior_beliefs': {'hypothesis1': 0.6, 'hypothesis2': 0.4},
    'likelihoods': {'hypothesis1': 0.8, 'hypothesis2': 0.2}
}
result = algorithm.calculate_novelty(data, context)
```

## Data Ingestion

The engine can ingest data from both CKG and PostGIS databases:

```python
from inspiration_engine import DataIngestor

ingestor = DataIngestor()

# Connect to databases
ingestor.connect_databases()

# Get CKG data
ckg_data = ingestor.get_ckg_data(collections=['MythologicalEntity', 'GeographicFeature'])

# Get PostGIS data with spatial filters
postgis_data = ingestor.get_postgis_data(
    table_name='puzzle_pieces',
    spatial_filters={
        'bbox': [-180, -90, 180, 90]  # World bounding box
    }
)

# Get recent data
recent_data = ingestor.get_recent_data(hours=24)

ingestor.disconnect_databases()
```

## Prompt Ranking

Rank creative prompts based on novelty and creative potential:

```python
from inspiration_engine import PromptRanker

ranker = PromptRanker()

prompts = [
    {
        'id': 'prompt1',
        'content': 'Explore the mythological significance of mountains',
        'domain': 'mythology'
    },
    {
        'id': 'prompt2',
        'content': 'Create a story about ancient civilizations',
        'domain': 'narrative'
    }
]

ranking = ranker.rank_prompts(prompts)
print(f"Diversity Score: {ranking.diversity_score}")

for prompt in ranking.ranked_prompts:
    print(f"Prompt: {prompt.content}")
    print(f"Creative Potential: {prompt.creative_potential}")
```

## A2A Protocol Integration

Communicate with other agents for collaborative inspiration:

```python
from inspiration_engine import A2AInspirationClient

async with A2AInspirationClient(
    server_url="http://localhost:8080",
    agent_name="inspiration_engine"
) as client:

    # Request inspiration from other agents
    response = await client.request_inspiration(
        context="Create a story about ancient civilizations",
        domain="mythology"
    )

    # Share findings
    await client.share_novelty_findings(
        findings=my_analysis_results,
        target_agents=["mythology_agent", "narrative_agent"]
    )

    # Start collaborative session
    session = await client.collaborative_inspiration_session(
        topic="Ancient Civilizations and Modern Technology",
        participating_agents=["mythology_agent", "geography_agent"],
        duration_minutes=30
    )
```

## Processing Recent Data

Analyze recently added or modified data:

```python
# Process recent data changes
recent_analysis = await engine.process_recent_data(
    time_window_hours=24,
    domains=['mythology', 'geography']
)

print(f"Processed {len(recent_analysis['analysis_results'])} data sources")
print(f"Insights: {recent_analysis['insights']}")
```

## API Reference

### InspirationEngine

Main orchestration class for the Inspiration Engine.

#### Methods

- `initialize()`: Initialize all engine components
- `shutdown()`: Shutdown the engine and close connections
- `analyze_novelty(data, context, algorithms, use_cache)`: Analyze data for novelty
- `generate_inspiration(domain, context, constraints, num_prompts)`: Generate creative prompts
- `process_recent_data(time_window_hours, domains)`: Process recently added data
- `collaborative_session(topic, participants, duration_minutes)`: Start collaborative session
- `share_findings(findings, target_agents)`: Share analysis findings
- `get_engine_status()`: Get current engine status

### NoveltyDetector

Orchestrates multiple novelty detection algorithms.

#### Methods

- `detect_novelty(data, context, algorithms)`: Detect novelty using specified algorithms
- `calculate_combined_score(scores, weights)`: Calculate weighted combination of scores

### DataIngestor

Handles data ingestion from multiple sources.

#### Methods

- `connect_databases()`: Establish database connections
- `disconnect_databases()`: Close database connections
- `get_ckg_data(collections, limit, filters)`: Retrieve CKG data
- `get_postgis_data(table_name, columns, limit, filters, spatial_filters)`: Retrieve PostGIS data
- `get_recent_data(hours)`: Get recently added data
- `get_spatial_clusters(table_name, cluster_distance)`: Identify spatial clusters

### PromptRanker

Ranks creative prompts based on novelty and potential.

#### Methods

- `rank_prompts(prompts, context, ranking_criteria)`: Rank creative prompts
- `get_top_prompts(ranking, top_n, min_score)`: Get top-ranked prompts
- `generate_prompt_variations(base_prompt, variation_count)`: Generate prompt variations

### A2AInspirationClient

A2A protocol client for inspiration engine communication.

#### Methods

- `request_inspiration(context, domain, constraints, target_agent)`: Request inspiration
- `share_novelty_findings(findings, target_agents)`: Share novelty findings
- `collaborative_inspiration_session(topic, participants, duration_minutes)`: Start session
- `send_creation_feedback(...)`: Send feedback on creative outputs

## Testing

Run the unit tests:

```bash
python -m pytest inspiration_engine/test_algorithms.py -v
```

Or run specific test classes:

```bash
python inspiration_engine/test_algorithms.py TestRPADAlgorithm
```

## Performance Considerations

- The engine uses caching to improve performance for repeated analyses
- Database connections are pooled and reused
- Asynchronous operations allow for concurrent processing
- Memory usage can be controlled via configuration parameters

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Ensure CKG and PostGIS services are running and accessible
2. **A2A Connection Errors**: Verify A2A server is running and network connectivity
3. **Memory Issues**: Reduce cache TTL or batch sizes in configuration
4. **Slow Performance**: Check database indexes and consider increasing cache TTL

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the Inspiration Engine:

1. Add unit tests for new algorithms
2. Update documentation for new features
3. Follow the existing code style and patterns
4. Test with both CKG and PostGIS data sources

## License

This project is part of the Terra Constellata system. See the main project license for details.