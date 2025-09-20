# Terra Constellata Specialist Agents

This directory contains the implementation of the Specialist Agent Army for the Terra Constellata project. The system uses LangChain and open-source LLMs to create autonomous agents that collaborate via the A2A (Agent-to-Agent) Protocol.

## Architecture Overview

### Base Agent Class
- **File**: `base_agent.py`
- **Purpose**: Provides the foundation for all specialist agents
- **Features**:
  - LangChain integration
  - A2A protocol communication
  - Memory management
  - Autonomous operation capabilities
  - Tool management

### Agent Registry
- **Purpose**: Manages all registered agents
- **Features**:
  - Agent registration and discovery
  - Status monitoring
  - Inter-agent communication

## Implemented Agents

### 1. Atlas Relational Analyst (`atlas/atlas_relational_analyst.py`)
**Specialization**: Relational data analysis and pattern discovery
**Tools**:
- Relational Analysis Tool: Analyzes relationships in data
- Data Pattern Discovery Tool: Identifies patterns and anomalies
**Databases**: CKG (Cognitive Knowledge Graph) and PostGIS

### 2. Comparative Mythology Agent (`myth/comparative_mythology_agent.py`)
**Specialization**: Cross-cultural mythological analysis
**Tools**:
- Myth Comparison Tool: Compares myths across cultures
- Cultural Context Tool: Analyzes historical and symbolic contexts
**Capabilities**: Archetype identification, cultural pattern recognition

### 3. Linguist Agent (`lang/linguist_agent.py`)
**Specialization**: Language processing and linguistic analysis
**Tools**:
- Text Analysis Tool: Analyzes linguistic structure and patterns
- Translation Tool: Handles language translation and challenges
- Linguistic Pattern Tool: Identifies phonological, morphological, and syntactic patterns

### 4. Sentinel Orchestrator (`sentinel/sentinel_orchestrator.py`)
**Specialization**: System coordination and workflow management
**Tools**:
- Agent Coordination Tool: Manages inter-agent activities
- Workflow Management Tool: Handles complex multi-agent workflows
- System Monitoring Tool: Monitors overall system health

## Key Features

### Autonomous Operation
- Each agent can operate autonomously
- Continuous monitoring and task processing
- Self-optimization based on performance metrics

### Inter-Agent Communication
- A2A Protocol for structured communication
- Message types: InspirationRequest, CreationFeedback, ToolProposal, etc.
- JSON-RPC 2.0 based communication

### Tool Integration
- Custom tools for each agent's specialization
- LangChain tool framework integration
- Extensible tool architecture

### Memory and State Management
- Conversation memory for context awareness
- Analysis history tracking
- Performance metric collection

## Usage

### Basic Agent Creation
```python
from agents import AtlasRelationalAnalyst, MockLLM

# Create LLM (would be Llama/GPT-J in production)
llm = MockLLM()

# Create agent
atlas = AtlasRelationalAnalyst(llm)

# Process task
result = await atlas.process_task("Analyze geospatial relationships")
```

### Agent Coordination
```python
from agents import SentinelOrchestrator, agent_registry

# Create orchestrator
sentinel = SentinelOrchestrator(llm)

# Register agents
agent_registry.register_agent(atlas)
agent_registry.register_agent(myth_agent)

# Coordinate complex task
result = await sentinel.coordinate_agents("Create comprehensive cultural analysis")
```

### Autonomous Operation
```python
# Start autonomous operation
await atlas.start_autonomous_operation()

# Agents will continuously monitor and process tasks
# Stop when needed
await atlas.stop_autonomous_operation()
```

## Configuration

### Environment Variables
- `A2A_SERVER_URL`: A2A protocol server URL (default: http://localhost:8080)
- `CKG_CONNECTION_STRING`: Cognitive Knowledge Graph connection
- `POSTGIS_CONNECTION_STRING`: PostGIS database connection

### Agent Parameters
- `memory_size`: Conversation memory buffer size
- `a2a_server_url`: Custom A2A server URL
- Custom parameters per agent type

## Testing

Run the test script to verify agent functionality:
```bash
cd terra-constellata
python test_agents.py
```

## Integration with A2A Protocol

### Message Types
- `GeospatialAnomalyIdentified`: Report spatial anomalies
- `InspirationRequest`: Request creative inspiration
- `CreationFeedback`: Provide feedback on outputs
- `ToolProposal`: Propose new tools/capabilities
- `NarrativePrompt`: Request narrative generation
- `CertificationRequest`: Request validation/certification

### Communication Flow
1. Agent sends request/notification via A2A client
2. A2A server routes message to target agent
3. Target agent processes message and responds
4. Response sent back through A2A protocol

## Future Enhancements

### Additional Agents
- InspirationEngine: Creative content generation
- LoreWeaverAgent: Narrative construction
- AestheticCognitionAgent: Beauty and art analysis
- ToolSmithAgent: Dynamic tool creation

### Advanced Features
- Multi-agent workflow orchestration
- Performance-based agent optimization
- Dynamic tool creation and management
- Enhanced learning and adaptation capabilities

## Dependencies

- LangChain: Agent framework and tool integration
- Transformers: Open-source LLM support (Llama, GPT-J)
- A2A Protocol: Inter-agent communication
- Database connectors: ArangoDB (CKG), PostgreSQL (PostGIS)
- Async libraries: aiohttp, asyncio

## Performance Considerations

- Memory management for long-running operations
- Rate limiting for API calls
- Caching for frequently accessed data
- Parallel processing for independent tasks
- Monitoring and logging for debugging

## Security

- Input validation for all agent interactions
- Secure communication via A2A protocol
- Access control for sensitive operations
- Audit logging for all agent activities