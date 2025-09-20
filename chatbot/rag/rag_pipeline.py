"""
RAG Pipeline for Lore Weaver Chatbot

This module implements the Retrieval-Augmented Generation pipeline using LangChain,
integrating with the Cultural Knowledge Graph and PostGIS databases for contextual
narrative generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseRetriever, Document
from langchain.callbacks import LangChainTracer
from langsmith import Client

from .vector_store import LoreWeaverVectorStore

logger = logging.getLogger(__name__)


class LoreWeaverRAG:
    """
    RAG pipeline for the Lore Weaver chatbot.
    Combines retrieval from CKG and PostGIS with generative AI for storytelling.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        use_langsmith: bool = True,
        langsmith_project: str = "lore-weaver-chatbot",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            openai_api_key: OpenAI API key
            model_name: OpenAI model to use
            temperature: Model temperature for creativity
            use_langsmith: Whether to use LangSmith for tracing
            langsmith_project: LangSmith project name
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.use_langsmith = use_langsmith
        self.langsmith_project = langsmith_project

        # Initialize components
        self.vector_store = LoreWeaverVectorStore()
        self.llm = None
        self.qa_chain = None
        self.langsmith_client = None

        # Initialize LangSmith if enabled
        if use_langsmith:
            self._setup_langsmith()

        # Storytelling prompts
        self._setup_prompts()

        logger.info("Initialized LoreWeaverRAG pipeline")

    def _setup_langsmith(self):
        """Setup LangSmith for explainable AI tracing."""
        try:
            self.langsmith_client = Client()
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            logger.info("LangSmith tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to setup LangSmith: {e}")
            self.use_langsmith = False

    def _setup_prompts(self):
        """Setup custom prompts for storytelling."""
        # Base storytelling prompt
        self.storytelling_prompt = PromptTemplate(
            template="""
You are Lore Weaver, an expert storyteller specializing in geomythological narratives.
You have access to a rich Cultural Knowledge Graph and geospatial data about mythological
entities, geographic features, and cultural concepts.

Based on the following retrieved information, create an engaging, evocative narrative
that weaves together the mythological, cultural, and geographical elements.

Retrieved Information:
{context}

User Query: {question}

Instructions:
1. Create a compelling narrative that connects the retrieved information
2. Use poetic and evocative language appropriate for storytelling
3. Highlight the cultural and mythological significance
4. Include geographical context where relevant
5. Maintain historical and cultural accuracy
6. If information is incomplete, focus on what is available and suggest areas for exploration

Your response should be engaging, informative, and immersive:
""",
            input_variables=["context", "question"],
        )

        # Follow-up question prompt for clarification
        self.clarification_prompt = PromptTemplate(
            template="""
Based on the user's query and the available information, determine if more details
are needed to provide a complete narrative.

User Query: {question}
Available Context: {context}

If clarification is needed, ask specific questions about:
- Specific mythological figures or entities
- Geographic locations or features
- Cultural contexts or time periods
- Types of narratives (creation myths, hero stories, etc.)

Response format:
- If clarification needed: "CLARIFY: [specific questions]"
- If ready to respond: "READY: [brief summary of understanding]"
""",
            input_variables=["question", "context"],
        )

    def initialize_llm(self):
        """Initialize the language model."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        callbacks = []
        if self.use_langsmith:
            callbacks.append(LangChainTracer())

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            callbacks=callbacks,
        )

        logger.info(f"Initialized LLM: {self.model_name}")

    def setup_qa_chain(self):
        """Setup the QA chain with custom retriever."""
        if not self.llm:
            self.initialize_llm()

        # Create custom retriever
        retriever = LoreWeaverRetriever(self.vector_store)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.storytelling_prompt,
                "document_variable_name": "context",
            },
            return_source_documents=True,
        )

        logger.info("QA chain initialized")

    def load_data(self):
        """Load data from CKG and PostGIS into vector store."""
        logger.info("Loading data into vector store...")

        # Load CKG data
        ckg_stats = self.vector_store.load_ckg_data()
        logger.info(f"CKG loading stats: {ckg_stats}")

        # Load PostGIS data
        postgis_stats = self.vector_store.load_postgis_data()
        logger.info(f"PostGIS loading stats: {postgis_stats}")

        return {"ckg": ckg_stats, "postgis": postgis_stats}

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        if not self.qa_chain:
            self.setup_qa_chain()

        try:
            # First, check if clarification is needed
            clarification_result = self._check_clarification_needed(question)
            if clarification_result.startswith("CLARIFY:"):
                return {
                    "answer": clarification_result.replace("CLARIFY:", "").strip(),
                    "needs_clarification": True,
                    "source_documents": [],
                    "metadata": {},
                }

            # Process the query
            result = self.qa_chain({"query": question})

            # Extract and enhance metadata
            metadata = self._extract_metadata(result.get("source_documents", []))

            return {
                "answer": result["result"],
                "needs_clarification": False,
                "source_documents": result.get("source_documents", []),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "needs_clarification": False,
                "source_documents": [],
                "metadata": {"error": str(e)},
            }

    def _check_clarification_needed(self, question: str) -> str:
        """Check if the query needs clarification."""
        try:
            # Get some context for clarification check
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            # Use LLM to check if clarification is needed
            chain = self.clarification_prompt | self.llm
            result = chain.invoke({"question": question, "context": context})

            return result.content.strip()

        except Exception as e:
            logger.warning(f"Error checking clarification: {e}")
            return "READY: Proceeding with available information"

    def _extract_metadata(self, source_documents: List[Document]) -> Dict[str, Any]:
        """Extract metadata from source documents."""
        metadata = {
            "sources": [],
            "entities": [],
            "locations": [],
            "collections": set(),
        }

        for doc in source_documents:
            meta = doc.metadata

            # Track sources
            source = meta.get("source", "unknown")
            metadata["sources"].append(source)

            # Track collections
            if "collection" in meta:
                metadata["collections"].add(meta["collection"])

            # Extract entities and locations from content
            content = doc.page_content.lower()

            # Simple entity extraction (can be enhanced with NER)
            if "name:" in content:
                name_part = content.split("name:")[1].split("|")[0].strip()
                if name_part and len(name_part) > 2:
                    metadata["entities"].append(name_part)

            # Extract location info
            if "location:" in content:
                loc_part = content.split("location:")[1].split("|")[0].strip()
                if loc_part:
                    metadata["locations"].append(loc_part)

        # Convert sets to lists
        metadata["collections"] = list(metadata["collections"])

        return metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "model": self.model_name,
            "temperature": self.temperature,
            "langsmith_enabled": self.use_langsmith,
        }

    def update_feedback(
        self, query: str, response: str, rating: int, feedback: Optional[str] = None
    ):
        """
        Update the system with user feedback for reinforcement learning.

        Args:
            query: Original query
            response: System response
            rating: User rating (1-5)
            feedback: Optional text feedback
        """
        # This is a placeholder for reinforcement learning implementation
        # In a full implementation, this would update the model's behavior
        # based on user feedback

        feedback_data = {
            "query": query,
            "response": response,
            "rating": rating,
            "feedback": feedback,
            "timestamp": None,  # Would add timestamp
        }

        logger.info(f"Feedback received: {feedback_data}")

        # Store feedback for analysis (could be saved to database)
        # This would be used to fine-tune prompts, retriever, or model

        return feedback_data


class LoreWeaverRetriever(BaseRetriever):
    """
    Custom retriever for Lore Weaver that combines CKG and PostGIS search.
    """

    def __init__(self, vector_store: LoreWeaverVectorStore):
        super().__init__()
        self.vector_store = vector_store

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for the query."""
        return self.vector_store.similarity_search(query, k=5)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        # For now, just call the sync version
        return self.get_relevant_documents(query)
