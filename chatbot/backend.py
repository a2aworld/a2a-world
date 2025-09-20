"""
Lore Weaver Chatbot Backend Service

FastAPI-based backend service for the Lore Weaver chatbot, providing REST API
endpoints for RAG-based geomythological storytelling.
"""

import os
import logging
import uvicorn
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .rag.rag_pipeline import LoreWeaverRAG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lore Weaver Chatbot API",
    description="RAG-based chatbot for geomythological storytelling",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance: Optional[LoreWeaverRAG] = None


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for chat queries."""

    question: str = Field(..., description="User's question or query")
    max_results: int = Field(5, description="Maximum number of results to retrieve")


class QueryResponse(BaseModel):
    """Response model for chat queries."""

    answer: str = Field(..., description="Generated narrative response")
    needs_clarification: bool = Field(
        ..., description="Whether clarification is needed"
    )
    source_documents: list = Field(..., description="Source documents used")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""

    query: str = Field(..., description="Original query")
    response: str = Field(..., description="System response")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    feedback: Optional[str] = Field(None, description="Optional text feedback")


class StatsResponse(BaseModel):
    """Response model for system statistics."""

    vector_store_stats: Dict[str, Any]
    model_info: str
    temperature: float
    langsmith_enabled: bool
    uptime: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_instance

    try:
        logger.info("Initializing Lore Weaver RAG system...")

        # Initialize RAG
        rag_instance = LoreWeaverRAG(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            use_langsmith=os.getenv("USE_LANGSMITH", "true").lower() == "true",
        )

        # Load data into vector store
        logger.info("Loading data into vector store...")
        load_stats = rag_instance.load_data()
        logger.info(f"Data loading completed: {load_stats}")

        # Setup QA chain
        rag_instance.setup_qa_chain()

        logger.info("Lore Weaver backend initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Lore Weaver backend")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Lore Weaver Chatbot API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    return {"status": "healthy", "rag_initialized": True, "timestamp": datetime.now()}


@app.post("/chat", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """
    Process a chat query through the RAG pipeline.

    Args:
        request: Query request with question and parameters

    Returns:
        Query response with generated narrative
    """
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        logger.info(f"Processing query: {request.question[:100]}...")

        # Process the query
        result = rag_instance.query(request.question, k=request.max_results)

        # Add timestamp
        result["timestamp"] = datetime.now()

        logger.info("Query processed successfully")
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit user feedback for reinforcement learning.

    Args:
        request: Feedback request
        background_tasks: FastAPI background tasks
    """
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Process feedback in background to avoid blocking
        background_tasks.add_task(
            rag_instance.update_feedback,
            request.query,
            request.response,
            request.rating,
            request.feedback,
        )

        return {
            "message": "Feedback submitted successfully",
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Feedback submission failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        stats = rag_instance.get_stats()

        return StatsResponse(
            vector_store_stats=stats["vector_store"],
            model_info=stats["model"],
            temperature=stats["temperature"],
            langsmith_enabled=stats["langsmith_enabled"],
            uptime="System uptime tracking not implemented",  # Could be enhanced
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.post("/reload-data")
async def reload_data(background_tasks: BackgroundTasks):
    """Reload data from databases into vector store."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Reload data in background
        background_tasks.add_task(rag_instance.load_data)

        return {"message": "Data reload initiated", "timestamp": datetime.now()}

    except Exception as e:
        logger.error(f"Error initiating data reload: {e}")
        raise HTTPException(status_code=500, detail=f"Data reload failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {"error": True, "message": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": True, "message": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run("backend:app", host=host, port=port, reload=True, log_level="info")
