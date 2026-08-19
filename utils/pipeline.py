"""
TruthLens AI pipeline initialization.

Creates the real retrieval and reasoning components used by
the application.
"""

from typing import Dict, Any

from agents.evidence_agent import EvidenceAgent
from agents.fact_check_agent import FactCheckAgent
from agents.search_agent import SearchAgent
from rag.reasoner import RAGReasoner
from retriever.embedder import Embedder
from retriever.faiss_index import EvidenceIndex
from utils.config import settings
from utils.dataset_loader import load_dataset


def build_pipeline() -> FactCheckAgent:
    """
    Build and connect all TruthLens components.
    """

    # ---------------------------------------------------------
    # 1. Embedding model
    # ---------------------------------------------------------

    embedder = Embedder()

    # ---------------------------------------------------------
    # 2. FAISS evidence index
    # ---------------------------------------------------------

    evidence_index = EvidenceIndex(
        embedder=embedder
    )

    dataset = load_dataset()

    evidence_index.add_documents(
        dataset
    )

    # ---------------------------------------------------------
    # 3. Web search agent
    # ---------------------------------------------------------

    search_agent = SearchAgent()

    # ---------------------------------------------------------
    # 4. Hybrid evidence agent
    # ---------------------------------------------------------

    evidence_agent = EvidenceAgent(
        evidence_index=evidence_index,
        search_agent=search_agent,
    )

    # ---------------------------------------------------------
    # 5. Groq RAG reasoner
    # ---------------------------------------------------------

    reasoner = RAGReasoner()

    # ---------------------------------------------------------
    # 6. Agentic fact-checking loop
    # ---------------------------------------------------------

    fact_check_agent = FactCheckAgent(
        evidence_agent=evidence_agent,
        reasoner=reasoner,
        max_iterations=settings.MAX_AGENT_ITERATIONS,
        confidence_threshold=0.75,
    )

    return fact_check_agent