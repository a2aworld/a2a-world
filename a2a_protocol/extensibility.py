"""
A2A Protocol Extensibility Module

This module provides mechanisms for extending the A2A protocol with new message types,
handlers, and plugins at runtime.
"""

import importlib
import inspect
import logging
from typing import Any, Callable, Dict, Type, Optional
from pathlib import Path

from schemas import A2AMessage, MESSAGE_TYPES, create_message
from server import A2AServer

logger = logging.getLogger(__name__)


class MessageTypeRegistry:
    """Registry for dynamic message type registration"""

    def __init__(self):
        self._message_types: Dict[str, Type[A2AMessage]] = MESSAGE_TYPES.copy()
        self._handlers: Dict[str, Callable] = {}

    def register_message_type(self, name: str, message_class: Type[A2AMessage]):
        """
        Register a new message type.

        Args:
            name: Message type name (e.g., "GEOSPATIAL_ANOMALY_IDENTIFIED")
            message_class: Pydantic model class inheriting from A2AMessage
        """
        if not issubclass(message_class, A2AMessage):
            raise TypeError(f"Message class must inherit from A2AMessage")

        self._message_types[name] = message_class
        logger.info(f"Registered message type: {name}")

    def get_message_type(self, name: str) -> Optional[Type[A2AMessage]]:
        """Get message type class by name"""
        return self._message_types.get(name)

    def list_message_types(self) -> list:
        """List all registered message types"""
        return list(self._message_types.keys())

    def create_message(self, message_type: str, **kwargs) -> A2AMessage:
        """Create a message instance of the specified type"""
        message_class = self.get_message_type(message_type)
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")
        return message_class(**kwargs)

    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a message type.

        Args:
            message_type: Message type name
            handler: Async callable that handles the message
        """
        self._handlers[message_type] = handler
        logger.info(f"Registered handler for: {message_type}")

    def get_handler(self, message_type: str) -> Optional[Callable]:
        """Get handler for message type"""
        return self._handlers.get(message_type)


class PluginManager:
    """Manager for loading and managing A2A plugins"""

    def __init__(self, registry: MessageTypeRegistry):
        self.registry = registry
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_dirs: list = []

    def add_plugin_directory(self, directory: str):
        """Add a directory to search for plugins"""
        self._plugin_dirs.append(Path(directory))

    def load_plugin(self, plugin_name: str, plugin_module: str):
        """
        Load a plugin module.

        Args:
            plugin_name: Name for the plugin
            plugin_module: Module path (e.g., "my_plugins.custom_messages")
        """
        try:
            module = importlib.import_module(plugin_module)
            self._loaded_plugins[plugin_name] = module

            # Auto-register message types and handlers from plugin
            self._register_plugin_components(module)

            logger.info(f"Loaded plugin: {plugin_name}")

        except ImportError as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def _register_plugin_components(self, module):
        """Register message types and handlers from plugin module"""
        # Look for message classes
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, A2AMessage)
                and obj != A2AMessage
            ):
                # Register as message type
                message_type_name = getattr(obj, "_message_type_name", name.upper())
                self.registry.register_message_type(message_type_name, obj)

        # Look for handler functions
        for name, obj in inspect.getmembers(module):
            if inspect.iscoroutinefunction(obj) and name.startswith("handle_"):
                # Extract message type from function name
                message_type = name[7:].upper()  # Remove 'handle_' prefix
                self.registry.register_handler(message_type, obj)

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        if plugin_name in self._loaded_plugins:
            # Note: Python doesn't support true module unloading
            # This is just for tracking
            del self._loaded_plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")

    def list_plugins(self) -> list:
        """List loaded plugins"""
        return list(self._loaded_plugins.keys())


class ExtensibleA2AServer(A2AServer):
    """Extended A2A server with plugin support"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = MessageTypeRegistry()
        self.plugin_manager = PluginManager(self.registry)

    def register_method(self, method_name: str, handler: Callable):
        """Override to use registry"""
        self.registry.register_handler(method_name, handler)
        super().register_method(method_name, handler)

    async def _handle_request(self, rpc_request):
        """Override to use extensible handlers"""
        method = rpc_request.method
        params = rpc_request.params
        msg_id = rpc_request.id

        handler = self.registry.get_handler(method)
        if not handler:
            return super()._handle_request(rpc_request)

        # Use extensible handler
        try:
            # Validate A2A message if it's a known type
            message_class = self.registry.get_message_type(method)
            if message_class:
                from validation import MessageValidator

                a2a_message = MessageValidator.validate_a2a_message(method, params)
                if a2a_message is None:
                    return MessageValidator._create_error_response(
                        -32602,
                        "Invalid params",
                        "Invalid A2A message parameters",
                        msg_id,
                    )
                result = await handler(a2a_message)
            else:
                # Generic handler
                result = await handler(params)

            from validation import MessageValidator

            return MessageValidator.create_success_response(result, msg_id)

        except Exception as e:
            logger.error(f"Error executing extensible method {method}: {e}")
            from validation import MessageValidator

            return MessageValidator._create_error_response(
                -32603, "Internal error", str(e), msg_id
            )


# Example plugin
def create_example_plugin():
    """Create an example plugin module content"""
    plugin_content = '''
"""
Example A2A Plugin

This plugin demonstrates how to extend the A2A protocol with custom message types.
"""

from pydantic import BaseModel
from typing import List, Optional
from a2a_protocol.schemas import A2AMessage

class CustomAnalysisRequest(A2AMessage):
    """Custom analysis request message"""
    _message_type_name = "CUSTOM_ANALYSIS_REQUEST"

    analysis_type: str
    data: dict
    priority: str = "normal"
    parameters: Optional[dict] = None

class CustomAnalysisResult(A2AMessage):
    """Custom analysis result message"""
    _message_type_name = "CUSTOM_ANALYSIS_RESULT"

    request_id: str
    results: List[dict]
    confidence: float
    processing_time: float

async def handle_custom_analysis_request(message: CustomAnalysisRequest):
    """Handle custom analysis requests"""
    # Implementation here
    return {
        "status": "processed",
        "analysis_type": message.analysis_type,
        "result_id": f"result_{message.message_id}"
    }

async def handle_custom_analysis_result(message: CustomAnalysisResult):
    """Handle custom analysis results"""
    # Implementation here
    return {"status": "result_stored", "result_id": message.message_id}
'''

    return plugin_content


# Utility functions
def save_plugin_to_file(plugin_content: str, filename: str):
    """Save plugin content to a file"""
    with open(filename, "w") as f:
        f.write(plugin_content)


def load_plugin_from_file(
    filepath: str, plugin_name: str, registry: MessageTypeRegistry
):
    """Load plugin from file"""
    import sys
    import os

    # Add directory to Python path
    plugin_dir = os.path.dirname(filepath)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    # Import the module
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    plugin_module = f"{module_name}"

    try:
        module = importlib.import_module(plugin_module)

        # Register components
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, A2AMessage)
                and obj != A2AMessage
            ):
                message_type_name = getattr(obj, "_message_type_name", name.upper())
                registry.register_message_type(message_type_name, obj)

        for name, obj in inspect.getmembers(module):
            if inspect.iscoroutinefunction(obj) and name.startswith("handle_"):
                message_type = name[7:].upper()
                registry.register_handler(message_type, obj)

        logger.info(f"Loaded plugin from file: {plugin_name}")

    except Exception as e:
        logger.error(f"Failed to load plugin from file: {e}")


# Example usage
if __name__ == "__main__":
    # Create example plugin
    plugin_content = create_example_plugin()
    save_plugin_to_file(plugin_content, "example_plugin.py")

    # Setup registry
    registry = MessageTypeRegistry()

    # Load plugin
    load_plugin_from_file("example_plugin.py", "example", registry)

    # List registered types
    print("Registered message types:", registry.list_message_types())
