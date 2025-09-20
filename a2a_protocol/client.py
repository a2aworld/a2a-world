"""
A2A Protocol JSON-RPC Client

This module implements an asynchronous JSON-RPC 2.0 client for A2A communication,
supporting requests, notifications, and batch operations.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union
import aiohttp

from schemas import (
    A2AMessage,
    JSONRPCRequest,
    JSONRPCNotification,
    JSONRPCResponse,
    JSONRPCErrorResponse,
)
from validation import MessageValidator

logger = logging.getLogger(__name__)


class A2AClient:
    """Asynchronous JSON-RPC client for A2A protocol"""

    def __init__(self, server_url: str, agent_name: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.agent_name = agent_name
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Establish connection to the server"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            logger.info(f"A2A client connected to {self.server_url}")

    async def disconnect(self):
        """Close connection to the server"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("A2A client disconnected")

    def _get_next_id(self) -> str:
        """Generate next request ID"""
        self._request_id += 1
        return f"{self.agent_name}_{self._request_id}"

    async def _send_request(
        self, rpc_message: Union[JSONRPCRequest, JSONRPCNotification]
    ) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC message to server"""
        if not self.session:
            raise RuntimeError(
                "Client not connected. Use connect() or async context manager."
            )

        try:
            async with self.session.post(
                f"{self.server_url}/jsonrpc",
                json=rpc_message.dict(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 204:  # No Content for notifications
                    return None
                response_data = await response.json()
                return response_data

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise

    async def send_request(self, method: str, message: A2AMessage) -> Any:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: JSON-RPC method name
            message: A2A message to send

        Returns:
            Response result

        Raises:
            Exception: If request fails or returns error
        """
        # Set sender if not set
        if not hasattr(message, "sender_agent") or not message.sender_agent:
            message.sender_agent = self.agent_name

        request = JSONRPCRequest(
            method=method, params=message.dict(), id=self._get_next_id()
        )

        logger.debug(f"Sending request: {method}")
        response_data = await self._send_request(request)

        if response_data is None:
            raise RuntimeError("No response received")

        # Parse response
        if "error" in response_data:
            error = response_data["error"]
            raise RuntimeError(f"JSON-RPC Error {error['code']}: {error['message']}")

        return response_data.get("result")

    async def send_notification(self, method: str, message: A2AMessage):
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            method: JSON-RPC method name
            message: A2A message to send
        """
        # Set sender if not set
        if not hasattr(message, "sender_agent") or not message.sender_agent:
            message.sender_agent = self.agent_name

        notification = JSONRPCNotification(method=method, params=message.dict())

        logger.debug(f"Sending notification: {method}")
        await self._send_request(notification)

    async def batch_request(self, requests: list) -> list:
        """
        Send multiple requests in a batch.

        Args:
            requests: List of (method, message) tuples

        Returns:
            List of results or error dicts in same order
        """
        if not self.session:
            raise RuntimeError("Client not connected")

        batch = []
        for method, message in requests:
            if not hasattr(message, "sender_agent") or not message.sender_agent:
                message.sender_agent = self.agent_name

            batch.append(
                JSONRPCRequest(
                    method=method, params=message.dict(), id=self._get_next_id()
                ).dict()
            )

        try:
            async with self.session.post(
                f"{self.server_url}/jsonrpc",
                json=batch,
                headers={"Content-Type": "application/json"},
            ) as response:
                responses = await response.json()

                results = []
                for resp in responses:
                    if "error" in resp:
                        results.append(resp["error"])
                    else:
                        results.append(resp.get("result"))
                return results

        except aiohttp.ClientError as e:
            logger.error(f"Batch request failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        if not self.session:
            raise RuntimeError("Client not connected")

        try:
            async with self.session.get(f"{self.server_url}/health") as response:
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Health check failed: {e}")
            raise


# Convenience functions for common operations
async def send_geospatial_anomaly(
    client: A2AClient,
    anomaly_type: str,
    location: Dict[str, float],
    confidence: float,
    description: str,
    data_source: str,
    target_agent: Optional[str] = None,
):
    """Send geospatial anomaly message"""
    from schemas import GeospatialAnomalyIdentified

    message = GeospatialAnomalyIdentified(
        anomaly_type=anomaly_type,
        location=location,
        confidence=confidence,
        description=description,
        data_source=data_source,
        target_agent=target_agent,
    )

    return await client.send_request("GEOSPATIAL_ANOMALY_IDENTIFIED", message)


async def send_inspiration_request(
    client: A2AClient, context: str, domain: str, target_agent: Optional[str] = None, inspiration_type: str = "narrative"
):
    """Send inspiration request message"""
    from schemas import InspirationRequest

    message = InspirationRequest(
        context=context, domain=domain, target_agent=target_agent, inspiration_type=inspiration_type
    )

    return await client.send_request("INSPIRATION_REQUEST", message)


# Example usage
if __name__ == "__main__":

    async def main():
        async with A2AClient("http://localhost:8080", "test_agent") as client:
            # Send a test request
            result = await send_inspiration_request(
                client,
                context="Create a story about ancient civilizations",
                domain="mythology",
                inspiration_type="narrative",
            )
            print(f"Response: {result}")

    asyncio.run(main())
