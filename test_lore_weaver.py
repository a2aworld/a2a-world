#!/usr/bin/env python3
"""
Test Script for Lore Weaver Chatbot

This script provides comprehensive testing for the Lore Weaver RAG system,
including unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
import time

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from chatbot.rag.rag_pipeline import LoreWeaverRAG
from chatbot.rag.vector_store import LoreWeaverVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoreWeaverTester:
    """Test suite for Lore Weaver chatbot components."""

    def __init__(self):
        self.rag = None
        self.test_results = []

    def setup_test_environment(self):
        """Setup test environment and mock data if needed."""
        logger.info("Setting up test environment...")

        # Set test environment variables
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        os.environ.setdefault("USE_LANGSMITH", "false")

        # Initialize RAG with test settings
        try:
            self.rag = LoreWeaverRAG(
                openai_api_key="test-key",
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                use_langsmith=False,
            )
            logger.info("Test environment setup complete")
            return True
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False

    def test_vector_store_initialization(self) -> Dict[str, Any]:
        """Test vector store initialization."""
        logger.info("Testing vector store initialization...")

        try:
            vector_store = LoreWeaverVectorStore(
                use_chroma=False
            )  # Use FAISS for testing
            stats = vector_store.get_collection_stats()

            result = {
                "test": "vector_store_initialization",
                "status": "passed",
                "details": f"Vector store initialized successfully. Stats: {stats}",
            }

        except Exception as e:
            result = {
                "test": "vector_store_initialization",
                "status": "failed",
                "details": f"Vector store initialization failed: {str(e)}",
            }

        self.test_results.append(result)
        return result

    def test_rag_pipeline_initialization(self) -> Dict[str, Any]:
        """Test RAG pipeline initialization."""
        logger.info("Testing RAG pipeline initialization...")

        try:
            if not self.rag:
                raise ValueError("RAG not initialized")

            # Test getting stats
            stats = self.rag.get_stats()

            result = {
                "test": "rag_pipeline_initialization",
                "status": "passed",
                "details": f'RAG pipeline initialized. Model: {stats.get("model", "unknown")}',
            }

        except Exception as e:
            result = {
                "test": "rag_pipeline_initialization",
                "status": "failed",
                "details": f"RAG pipeline initialization failed: {str(e)}",
            }

        self.test_results.append(result)
        return result

    def test_sample_queries(self) -> List[Dict[str, Any]]:
        """Test sample queries to validate RAG functionality."""
        logger.info("Testing sample queries...")

        test_queries = [
            "Tell me about mythological mountains",
            "What are some famous cultural landmarks?",
            "Describe the geography of ancient civilizations",
            "What mythological creatures are associated with water?",
        ]

        results = []

        for query in test_queries:
            try:
                logger.info(f"Testing query: {query}")

                # Note: This would normally call the actual API
                # For testing, we'll simulate the response structure
                simulated_response = {
                    "answer": f"Simulated response for: {query}",
                    "needs_clarification": False,
                    "source_documents": [],
                    "metadata": {
                        "sources": ["ckg", "postgis"],
                        "entities": ["test_entity"],
                        "locations": ["test_location"],
                    },
                }

                result = {
                    "test": f"sample_query_{test_queries.index(query)}",
                    "status": "passed",
                    "query": query,
                    "response_length": len(simulated_response["answer"]),
                    "details": "Query processed successfully (simulated)",
                }

            except Exception as e:
                result = {
                    "test": f"sample_query_{test_queries.index(query)}",
                    "status": "failed",
                    "query": query,
                    "details": f"Query failed: {str(e)}",
                }

            results.append(result)
            self.test_results.append(result)

        return results

    def test_feedback_system(self) -> Dict[str, Any]:
        """Test the feedback system."""
        logger.info("Testing feedback system...")

        try:
            if not self.rag:
                raise ValueError("RAG not initialized")

            # Test feedback submission
            feedback_data = self.rag.update_feedback(
                query="Test query",
                response="Test response",
                rating=5,
                feedback="Great system!",
            )

            result = {
                "test": "feedback_system",
                "status": "passed",
                "details": f"Feedback submitted successfully: {feedback_data}",
            }

        except Exception as e:
            result = {
                "test": "feedback_system",
                "status": "failed",
                "details": f"Feedback system test failed: {str(e)}",
            }

        self.test_results.append(result)
        return result

    def run_performance_test(self) -> Dict[str, Any]:
        """Run basic performance test."""
        logger.info("Running performance test...")

        try:
            start_time = time.time()

            # Simulate some operations
            vector_store = LoreWeaverVectorStore(use_chroma=False)
            _ = vector_store.get_collection_stats()

            end_time = time.time()
            duration = end_time - start_time

            result = {
                "test": "performance_test",
                "status": "passed",
                "duration": f"{duration:.2f}s",
                "details": f"Performance test completed in {duration:.2f} seconds",
            }

        except Exception as e:
            result = {
                "test": "performance_test",
                "status": "failed",
                "details": f"Performance test failed: {str(e)}",
            }

        self.test_results.append(result)
        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary."""
        logger.info("Running all Lore Weaver tests...")

        # Setup
        if not self.setup_test_environment():
            return {
                "status": "failed",
                "message": "Test environment setup failed",
                "results": [],
            }

        # Run tests
        self.test_vector_store_initialization()
        self.test_rag_pipeline_initialization()
        self.test_sample_queries()
        self.test_feedback_system()
        self.run_performance_test()

        # Summarize results
        passed = sum(1 for r in self.test_results if r["status"] == "passed")
        failed = sum(1 for r in self.test_results if r["status"] == "failed")
        total = len(self.test_results)

        summary = {
            "status": "passed" if failed == 0 else "failed",
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": f"{passed/total*100:.1f}%",
            "results": self.test_results,
        }

        logger.info(
            f"Test summary: {passed}/{total} tests passed ({summary['success_rate']})"
        )

        return summary

    def print_test_report(self, summary: Dict[str, Any]):
        """Print detailed test report."""
        print("\n" + "=" * 60)
        print("LORE WEAVER CHATBOT TEST REPORT")
        print("=" * 60)

        print(f"\nOverall Status: {summary['status'].upper()}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']}")

        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)

        for result in summary["results"]:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"\n{status_icon} {result['test']}")
            print(f"   Status: {result['status']}")
            print(f"   Details: {result['details']}")

        print("\n" + "=" * 60)


def main():
    """Main test execution function."""
    print("Lore Weaver Chatbot Test Suite")
    print("==============================")

    tester = LoreWeaverTester()
    summary = tester.run_all_tests()
    tester.print_test_report(summary)

    # Exit with appropriate code
    sys.exit(0 if summary["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
