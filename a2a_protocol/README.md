# A2A Protocol v2.1 Implementation

This directory contains the implementation of the Agent-to-Agent (A2A) Protocol v2.1 for the Terra Constellata project. The protocol enables asynchronous communication between agents using JSON-RPC 2.0 over HTTP/WebSocket with enhanced message types for geospatial anomalies, inspiration requests, creation feedback, tool proposals, narrative prompts, and certification requests.

## Features

- **JSON-RPC 2.0 Compliant**: Full implementation of JSON-RPC 2.0 specification
- **Asynchronous Communication**: Built with asyncio for high-performance async operations
- **Modular Architecture**: Clean separation of concerns with extensible components
- **Message Validation**: Comprehensive validation using Pydantic schemas
- **Error Handling**: Robust error handling and logging throughout
- **Extensibility**: Plugin system for adding new message types and handlers
- **Integration Ready**: Easy integration with existing agent frameworks

## Architecture

```
a2a_protocol/
├── __init__.py           # Package initialization
├── schemas.py            # Message schemas and JSON-RPC structures
├── validation.py         # Message validation and business rules
├── server.py             # Asynchronous JSON-RPC server
├── client.py             # Asynchronous JSON-RPC client
├── integrated_agent.py   # Example integration with existing agents
├── extensibility.py      # Plugin system for extensions
├── test_a2a_protocol.py  # Comprehensive unit tests
└── README.md            # This documentation
```

## Message Types

### Core Message Types

1. **GEOSPATIAL_ANOMALY_IDENTIFIED**
   - Identifies geospatial anomalies with location, confidence, and metadata
   - Used for spatial pattern detection and anomaly reporting

2. **INSPIRATION_REQUEST**
   - Requests creative inspiration from other agents
   - Includes context, domain, and optional constraints

3. **CREATION_FEEDBACK**
   - Provides feedback on creative outputs
   - Supports ratings, suggestions, and detailed feedback

4. **TOOL_PROPOSAL**
   - Proposes new tools or capabilities
   - Includes tool specifications and use cases

5. **NARRATIVE_PROMPT**
   - Prompts for narrative generation
   - Specifies themes, elements, and style requirements

6. **CERTIFICATION_REQUEST**
   - Requests certification/validation of content
   - Includes evidence and certification criteria

## Quick Start

### Basic Server Setup

```python
from a2a_protocol.server import A2AServer

# Create server
server = A2AServer(host="localhost", port=8080)

# Register message handlers
async def handle_anomaly(message):
    print(f"Received anomaly: {message.description}")
    return {"status": "processed"}

server.register_method("GEOSPATIAL_ANOMALY_IDENTIFIED", handle_anomaly)

# Start server
server.run_forever()
```

### Basic Client Usage

```python
import asyncio
from a2a_protocol.client import A2AClient
from a2a_protocol.schemas import GeospatialAnomalyIdentified

async def main():
    async with A2AClient("http://localhost:8080", "my_agent") as client:
        # Create and send anomaly message
        anomaly = GeospatialAnomalyIdentified(
            sender_agent="my_agent",
            anomaly_type="cultural_pattern",
            location={"lat": 40.7128, "lon": -74.0060},
            confidence=0.85,
            description="Unusual concentration of mythological references",
            data_source="text_analysis"
        )

        response = await client.send_request("GEOSPATIAL_ANOMALY_IDENTIFIED", anomaly)
        print(f"Response: {response}")

asyncio.run(main())
```

### Integration with Existing Agents

```python
from a2a_protocol.integrated_agent import IntegratedCKGAgent

# Create integrated agent
agent = IntegratedCKGAgent("ckg_agent", "http://localhost:8080")

# The agent automatically handles A2A messages and integrates with CKG
await agent.run_ingestion_process(text_sources)
```

## API Reference

### Server API

#### A2AServer

- `A2AServer(host, port)`: Create server instance
- `register_method(method_name, handler)`: Register message handler
- `start()`: Start the server asynchronously
- `run_forever()`: Run server indefinitely

#### Message Handlers

Handlers should be async functions that accept an A2AMessage and return a response:

```python
async def my_handler(message: A2AMessage) -> dict:
    # Process message
    return {"result": "processed"}
```

### Client API

#### A2AClient

- `A2AClient(server_url, agent_name)`: Create client instance
- `send_request(method, message)`: Send request and wait for response
- `send_notification(method, message)`: Send notification (no response)
- `batch_request(requests)`: Send multiple requests in batch
- `health_check()`: Check server health

### Message Schemas

All messages inherit from `A2AMessage` and include:

- `message_id`: Unique message identifier
- `timestamp`: Message creation time
- `sender_agent`: Agent that sent the message
- `target_agent`: Optional target agent

## Extensibility

### Adding New Message Types

1. Create a new Pydantic model inheriting from `A2AMessage`:

```python
from a2a_protocol.schemas import A2AMessage

class MyCustomMessage(A2AMessage):
    custom_field: str
    another_field: int
```

2. Register the message type:

```python
from a2a_protocol.extensibility import MessageTypeRegistry

registry = MessageTypeRegistry()
registry.register_message_type("MY_CUSTOM_MESSAGE", MyCustomMessage)
```

3. Register a handler:

```python
async def handle_my_message(message: MyCustomMessage):
    return {"processed": True}

registry.register_handler("MY_CUSTOM_MESSAGE", handle_my_message)
```

### Plugin System

Create plugins for reusable message types:

```python
# my_plugin.py
from a2a_protocol.schemas import A2AMessage

class PluginMessage(A2AMessage):
    _message_type_name = "PLUGIN_MESSAGE"
    plugin_data: str

async def handle_plugin_message(message: PluginMessage):
    return {"plugin_response": message.plugin_data}
```

Load the plugin:

```python
from a2a_protocol.extensibility import PluginManager, MessageTypeRegistry

registry = MessageTypeRegistry()
plugin_manager = PluginManager(registry)
plugin_manager.load_plugin("my_plugin", "my_plugin")
```

## Configuration

### Environment Variables

- `A2A_SERVER_HOST`: Server host (default: localhost)
- `A2A_SERVER_PORT`: Server port (default: 8080)
- `A2A_CLIENT_TIMEOUT`: Client timeout in seconds (default: 30)

### Logging

The implementation uses Python's logging module. Configure logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Error Handling

### JSON-RPC Error Codes

- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

### Custom Exceptions

- `ValidationError`: Raised when message validation fails

## Testing

Run the test suite:

```bash
python -m pytest a2a_protocol/test_a2a_protocol.py -v
```

### Test Coverage

- Message schema validation
- JSON-RPC protocol compliance
- Client-server communication
- Error handling scenarios
- Extensibility features

## Performance Considerations

- Uses asyncio for asynchronous I/O
- Connection pooling in aiohttp client
- Efficient JSON serialization with built-in json module
- Minimal memory footprint with Pydantic models

## Security

- Input validation on all messages
- Business rule validation
- CORS support for web clients
- No authentication built-in (implement at application level)

## Dependencies

- `pydantic`: Data validation and serialization
- `aiohttp`: Asynchronous HTTP client/server
- `asyncio`: Asynchronous programming support

## Contributing

1. Follow the existing code style
2. Add unit tests for new features
3. Update documentation
4. Use type hints
5. Handle errors appropriately

## License

This implementation is part of the Terra Constellata project.

## Version History

- **v2.1.0**: Initial implementation with enhanced message types
  - JSON-RPC 2.0 compliance
  - Asynchronous architecture
  - Extensibility framework
  - Comprehensive testing