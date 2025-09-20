"""
Integrated A2A Agent Example

This module demonstrates how to integrate the A2A protocol with existing agents
in the Terra Constellata framework, using the CKG ingestion agent as an example.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from client import A2AClient
from server import A2AServer
from schemas import GeospatialAnomalyIdentified, InspirationRequest, CreationFeedback

# Import existing agent components
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.ckg import CulturalKnowledgeGraph
from data.ckg.operations import insert_text_source, insert_mythological_entity

logger = logging.getLogger(__name__)


class IntegratedCKGAgent:
    """CKG Ingestion Agent integrated with A2A protocol"""

    def __init__(self, agent_name: str, server_url: str = "http://localhost:8080"):
        self.agent_name = agent_name
        self.ckg = CulturalKnowledgeGraph()
        self.a2a_client = A2AClient(server_url, agent_name)
        self.a2a_server = A2AServer()
        self._setup_a2a_handlers()

    def _setup_a2a_handlers(self):
        """Setup A2A message handlers"""
        self.a2a_server.register_method(
            "GEOSPATIAL_ANOMALY_IDENTIFIED", self._handle_geospatial_anomaly
        )
        self.a2a_server.register_method(
            "INSPIRATION_REQUEST", self._handle_inspiration_request
        )
        self.a2a_server.register_method(
            "CREATION_FEEDBACK", self._handle_creation_feedback
        )

    async def _handle_geospatial_anomaly(
        self, message: GeospatialAnomalyIdentified
    ) -> Dict[str, Any]:
        """Handle incoming geospatial anomaly messages"""
        logger.info(f"Received geospatial anomaly: {message.description}")

        # Process the anomaly - perhaps query CKG for related information
        try:
            # Example: Search for related geographic entities
            related_entities = self.ckg.query_geographic_entities_near(
                message.location["lat"], message.location["lon"], radius=100
            )

            # Generate response with related cultural knowledge
            response = {
                "status": "processed",
                "anomaly_id": message.message_id,
                "related_entities": related_entities,
                "cultural_context": self._get_cultural_context(message.location),
            }

            return response

        except Exception as e:
            logger.error(f"Error processing geospatial anomaly: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_inspiration_request(
        self, message: InspirationRequest
    ) -> Dict[str, Any]:
        """Handle inspiration requests"""
        logger.info(f"Received inspiration request: {message.context}")

        try:
            # Query CKG for inspiration based on domain
            inspiration_data = self.ckg.query_inspiration_by_domain(message.domain)

            # Generate creative response
            response = {
                "status": "inspired",
                "inspiration": self._generate_inspiration(message, inspiration_data),
                "sources": inspiration_data,
            }

            return response

        except Exception as e:
            logger.error(f"Error processing inspiration request: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_creation_feedback(
        self, message: CreationFeedback
    ) -> Dict[str, Any]:
        """Handle creation feedback"""
        logger.info(f"Received creation feedback: {message.feedback_type}")

        try:
            # Store feedback in CKG or process it
            feedback_record = {
                "original_request_id": message.original_request_id,
                "feedback_type": message.feedback_type,
                "content": message.content,
                "rating": message.rating,
                "timestamp": message.timestamp,
            }

            # Ingest feedback into CKG
            self.ckg.insert_feedback(feedback_record)

            return {"status": "feedback_processed", "feedback_id": message.message_id}

        except Exception as e:
            logger.error(f"Error processing creation feedback: {e}")
            return {"status": "error", "message": str(e)}

    def _get_cultural_context(self, location: Dict[str, float]) -> Dict[str, Any]:
        """Get cultural context for a location"""
        # Placeholder implementation
        return {
            "nearby_myths": ["Example myth 1", "Example myth 2"],
            "cultural_significance": "High cultural significance area",
            "historical_events": ["Event 1", "Event 2"],
        }

    def _generate_inspiration(self, request: InspirationRequest, data: list) -> str:
        """Generate inspiration based on request and data"""
        # Placeholder implementation
        return f"Based on {request.domain} and context '{request.context}', here's some inspiration: {len(data)} related items found."

    async def send_geospatial_anomaly(self, anomaly_data: Dict[str, Any]):
        """Send a geospatial anomaly message"""
        anomaly = GeospatialAnomalyIdentified(
            anomaly_type=anomaly_data["type"],
            location=anomaly_data["location"],
            confidence=anomaly_data["confidence"],
            description=anomaly_data["description"],
            data_source=anomaly_data["source"],
        )

        try:
            response = await self.a2a_client.send_request(
                "GEOSPATIAL_ANOMALY_IDENTIFIED", anomaly
            )
            logger.info(f"Anomaly sent successfully: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to send anomaly: {e}")
            return None

    async def request_inspiration(self, context: str, domain: str):
        """Request inspiration from other agents"""
        inspiration_req = InspirationRequest(context=context, domain=domain)

        try:
            response = await self.a2a_client.send_request(
                "INSPIRATION_REQUEST", inspiration_req
            )
            logger.info(f"Inspiration received: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to request inspiration: {e}")
            return None

    async def send_feedback(
        self, original_request_id: str, feedback: str, rating: int = None
    ):
        """Send creation feedback"""
        feedback_msg = CreationFeedback(
            original_request_id=original_request_id,
            feedback_type="constructive",
            content=feedback,
            rating=rating,
        )

        try:
            response = await self.a2a_client.send_notification(
                "CREATION_FEEDBACK", feedback_msg
            )
            logger.info("Feedback sent successfully")
        except Exception as e:
            logger.error(f"Failed to send feedback: {e}")

    async def start_server(self):
        """Start the A2A server"""
        await self.a2a_server.start()

    async def run_ingestion_process(self, text_sources: list):
        """Run the traditional ingestion process with A2A enhancements"""
        logger.info("Starting enhanced CKG ingestion process")

        async with self.a2a_client:
            for url in text_sources:
                try:
                    # Traditional processing
                    text = await self._download_and_process_text(url)

                    # Check for anomalies or interesting content
                    anomalies = self._detect_anomalies(text)
                    for anomaly in anomalies:
                        await self.send_geospatial_anomaly(anomaly)

                    # Request inspiration if needed
                    if self._needs_inspiration(text):
                        inspiration = await self.request_inspiration(
                            f"Process text from {url}", "cultural_knowledge"
                        )

                    # Ingest to CKG
                    await self._ingest_to_ckg(text, url)

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")

        logger.info("Enhanced CKG ingestion process completed")

    async def _download_and_process_text(self, url: str) -> str:
        """Download and preprocess text (placeholder)"""
        # Implementation would include actual downloading
        return f"Sample text from {url}"

    def _detect_anomalies(self, text: str) -> list:
        """Detect anomalies in text (placeholder)"""
        # Implementation would use NLP to detect anomalies
        return []  # Return list of anomaly dictionaries

    def _needs_inspiration(self, text: str) -> bool:
        """Determine if inspiration is needed (placeholder)"""
        return False

    async def _ingest_to_ckg(self, text: str, source: str):
        """Ingest processed text to CKG (placeholder)"""
        # Implementation would use existing CKG operations
        pass


# Example usage
if __name__ == "__main__":

    async def main():
        agent = IntegratedCKGAgent("ckg_agent")

        # Start server in background
        asyncio.create_task(agent.start_server())

        # Example: Send a test anomaly
        anomaly_data = {
            "type": "cultural_pattern",
            "location": {"lat": 40.7128, "lon": -74.0060},
            "confidence": 0.85,
            "description": "Unusual concentration of mythological references",
            "source": "text_analysis",
        }

        async with agent.a2a_client:
            await agent.send_geospatial_anomaly(anomaly_data)

        # Run ingestion process
        text_sources = ["http://example.com/text1.txt"]
        await agent.run_ingestion_process(text_sources)

    asyncio.run(main())
