"""
Tool Shed Example Usage

Demonstrates the complete Tool Shed system for agent self-improvement.
"""

import asyncio
import logging
from datetime import datetime

from .models import Tool, ToolMetadata, ToolCapabilities, SearchQuery
from .registry import ToolRegistry
from .vector_store import ToolVectorStore
from .search import SemanticSearchEngine
from .evolution import ToolEvolutionManager
from .tool_smith_agent import ToolSmithAgent


# Mock LLM for demonstration
class MockLLM:
    def __init__(self):
        pass

    async def arun(self, prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}..."


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main demonstration of Tool Shed functionality."""

    print("🚀 Tool Shed Demonstration")
    print("=" * 50)

    # Initialize components
    print("\n1. Initializing Tool Shed Components...")

    vector_store = ToolVectorStore(persist_directory="./tool_shed_demo_db")
    registry = ToolRegistry(vector_store=vector_store)
    search_engine = SemanticSearchEngine(registry, vector_store)
    evolution_manager = ToolEvolutionManager(registry)

    # Mock LLM for ToolSmith
    mock_llm = MockLLM()

    # Create ToolSmith Agent
    tool_smith = ToolSmithAgent(
        llm=mock_llm, registry=registry, a2a_server_url="http://localhost:8080"
    )

    print("✓ Components initialized")

    # Create sample tools
    print("\n2. Creating Sample Tools...")

    # Tool 1: Data Analysis Tool
    data_tool = Tool(
        metadata=ToolMetadata(
            name="DataAnalyzer",
            description="Advanced data analysis and visualization tool",
            author="DataScienceAgent",
            category="data_science",
            tags=["analysis", "visualization", "statistics"],
        ),
        capabilities=ToolCapabilities(
            functions=["analyze_data", "create_charts", "statistical_tests"],
            input_types=["csv", "json", "dataframe"],
            output_types=["charts", "reports", "insights"],
            security_level="medium",
        ),
        code="""
def analyze_data(data):
    # Mock implementation
    return {"summary": "Data analyzed", "insights": []}

def create_charts(data):
    # Mock implementation
    return {"chart_type": "bar", "data": data}
""",
        documentation="A comprehensive tool for data analysis and visualization.",
    )

    # Tool 2: Text Processing Tool
    text_tool = Tool(
        metadata=ToolMetadata(
            name="TextProcessor",
            description="Natural language processing and text analysis",
            author="NLPAgent",
            category="nlp",
            tags=["nlp", "text", "language"],
        ),
        capabilities=ToolCapabilities(
            functions=["tokenize", "sentiment_analysis", "summarize"],
            input_types=["text", "documents"],
            output_types=["tokens", "sentiment", "summary"],
            security_level="high",
        ),
        code="""
def tokenize(text):
    return text.split()

def sentiment_analysis(text):
    # Mock implementation
    return {"sentiment": "positive", "confidence": 0.85}
""",
        documentation="Advanced text processing and NLP capabilities.",
    )

    # Register tools
    await registry.register_tool(data_tool)
    await registry.register_tool(text_tool)

    print("✓ Sample tools registered")

    # Demonstrate search functionality
    print("\n3. Demonstrating Search Functionality...")

    # Basic search
    query = SearchQuery(query="data analysis", limit=5)
    results = await search_engine.search(query)

    print(f"Search for 'data analysis': Found {results.total_count} tools")
    for tool in results.tools[:3]:
        print(f"  - {tool.metadata.name}: {tool.metadata.description[:50]}...")

    # Advanced search
    advanced_results = search_engine.advanced_search(
        query="text processing",
        filters={"category": "nlp", "min_rating": 3.0},
        sort_by="rating",
    )

    print(f"Advanced search: Found {advanced_results.total_count} NLP tools")

    # Demonstrate tool evolution
    print("\n4. Demonstrating Tool Evolution...")

    # Create evolution request
    evolution_id = await evolution_manager.create_evolution_request(
        tool_id=data_tool.id,
        evolution_type="enhancement",
        description="Add machine learning capabilities",
        proposed_changes={
            "capabilities": {
                "functions": [
                    "analyze_data",
                    "create_charts",
                    "statistical_tests",
                    "predict",
                ]
            }
        },
        requester_agent="MLAgent",
    )

    print(f"✓ Evolution request created: {evolution_id}")

    # Approve evolution
    success = await evolution_manager.approve_evolution_request(
        evolution_id, reviewer_agent="ToolSmith"
    )

    if success:
        print("✓ Evolution request approved")
    else:
        print("✗ Evolution request failed")

    # Demonstrate ToolSmith validation
    print("\n5. Demonstrating ToolSmith Validation...")

    # Simulate tool proposal validation
    validation_result = await tool_smith._validate_proposal("sample_proposal_id")
    print("ToolSmith validation result:")
    print(validation_result[:200] + "...")

    # Show registry statistics
    print("\n6. Registry Statistics...")

    stats = registry.get_registry_stats()
    print(f"Total tools: {stats['total_tools']}")
    print(f"Active tools: {stats['active_tools']}")
    print(f"Categories: {list(stats['categories'].keys())}")
    print(f"Authors: {list(stats['authors'].keys())}")

    # Demonstrate similar tools
    print("\n7. Finding Similar Tools...")

    similar = registry.get_similar_tools(data_tool.id, limit=3)
    print(f"Tools similar to {data_tool.metadata.name}:")
    for sim in similar:
        print(
            f"  - {sim['metadata']['name']} (similarity: {sim['similarity_score']:.2f})"
        )

    # Clean up
    print("\n8. Cleaning up...")
    vector_store.client.delete_collection("tool_shed")
    tool_smith.cleanup()

    print("\n🎉 Tool Shed demonstration completed!")


async def agent_autonomy_demo():
    """Demonstrate agent autonomy in tool discovery and proposal."""

    print("\n🤖 Agent Autonomy Demonstration")
    print("=" * 40)

    # Initialize components
    vector_store = ToolVectorStore(persist_directory="./autonomy_demo_db")
    registry = ToolRegistry(vector_store=vector_store)

    # Simulate agent discovering need for new tool
    print("\nAgent identifies need for new capability...")

    # Agent searches for existing tools
    search_engine = SemanticSearchEngine(registry, vector_store)
    query = SearchQuery(query="image processing computer vision", limit=5)
    results = await search_engine.search(query)

    if results.total_count == 0:
        print("No existing image processing tools found.")
        print("Agent proposes new tool...")

        # Create tool proposal
        from .models import ToolProposal

        proposal = ToolProposal(
            proposer_agent="VisionAgent",
            tool_name="ImageAnalyzer",
            description="Computer vision and image analysis tool",
            capabilities=[
                "object_detection",
                "image_classification",
                "feature_extraction",
            ],
            use_case="Analyze images for patterns and objects",
            priority="high",
        )

        proposal_id = await registry.submit_proposal(proposal)
        print(f"✓ Tool proposal submitted: {proposal_id}")

        # ToolSmith reviews proposal
        print("ToolSmith reviewing proposal...")

        # Simulate approval
        success = await registry.approve_proposal(proposal_id, "ToolSmith")
        if success:
            print("✓ Proposal approved!")

            # Create the actual tool
            image_tool = Tool(
                metadata=ToolMetadata(
                    name="ImageAnalyzer",
                    description="Computer vision and image analysis",
                    author="VisionAgent",
                    category="computer_vision",
                    tags=["vision", "image", "analysis"],
                ),
                capabilities=ToolCapabilities(
                    functions=["object_detection", "image_classification"],
                    input_types=["images", "video"],
                    output_types=["detections", "classifications"],
                    security_level="medium",
                ),
                code="""
def object_detection(image):
    # Mock implementation
    return {"objects": ["person", "car"], "confidence": 0.95}
""",
                documentation="Advanced computer vision capabilities.",
            )

            await registry.register_tool(image_tool)
            print("✓ New tool registered in Tool Shed!")

    else:
        print(f"Found {results.total_count} existing tools. Using best match.")

    # Clean up
    vector_store.client.delete_collection("tool_shed")


if __name__ == "__main__":
    # Run main demonstration
    asyncio.run(main())

    # Run autonomy demonstration
    asyncio.run(agent_autonomy_demo())
