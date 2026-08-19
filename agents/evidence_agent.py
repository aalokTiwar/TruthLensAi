"""
Hybrid evidence retrieval for TruthLens AI.

Combines:

1. Local FAISS semantic retrieval
2. Multi-query web retrieval
3. Evidence deduplication
4. Source quality scoring

The result is a unified, enriched evidence collection
that can be passed to the semantic evidence filter
and RAG reasoner.
"""

from typing import List, Dict, Any

from retriever.faiss_index import EvidenceIndex
from agents.search_agent import SearchAgent
from utils.source_quality import SourceQuality


class EvidenceAgent:
    """Combine local and web evidence."""

    def __init__(
        self,
        evidence_index: EvidenceIndex,
        search_agent: SearchAgent,
    ):
        self.evidence_index = evidence_index
        self.search_agent = search_agent

    # =====================================================
    # LOCAL FAISS RETRIEVAL
    # =====================================================

    def retrieve_local(
        self,
        claim: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence from the local FAISS index."""

        if not claim or not claim.strip():
            return []

        return self.evidence_index.search(
            claim,
            top_k=top_k,
        )

    # =====================================================
    # WEB RETRIEVAL
    # =====================================================

    def retrieve_web(
        self,
        claim: str,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence from the public web using
        multiple claim-aware queries.
        """

        if not claim or not claim.strip():
            return []

        return self.search_agent.gather_claim_evidence(
            claim,
            max_results_per_query=max_results,
            max_queries=4,
        )

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    @staticmethod
    def _deduplicate_evidence(
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate evidence.

        Web evidence is deduplicated using URLs.

        Local evidence is deduplicated using evidence
        or text content.
        """

        unique = []

        seen_urls = set()
        seen_content = set()

        for item in evidence:

            url = str(
                item.get(
                    "url",
                    "",
                )
            ).strip().lower()

            content = str(
                item.get(
                    "evidence",
                    "",
                )
            ).strip().lower()

            if not content:

                content = str(
                    item.get(
                        "text",
                        "",
                    )
                ).strip().lower()

            # -------------------------------------------------
            # Web evidence
            # -------------------------------------------------

            if url:

                if url in seen_urls:
                    continue

                seen_urls.add(url)

            # -------------------------------------------------
            # Local evidence
            # -------------------------------------------------

            elif content:

                if content in seen_content:
                    continue

                seen_content.add(content)

            unique.append(item)

        return unique

    # =====================================================
    # SOURCE QUALITY
    # =====================================================

    @staticmethod
    def _add_source_quality(
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add source quality information to each evidence item.

        Each item receives:

            source_quality_score
            source_quality
        """

        return SourceQuality.score_many(
            evidence
        )

    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        claim: str,
        top_k: int = 5,
        max_web_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and enrich evidence from local FAISS
        and web search.
        """

        if not claim or not claim.strip():
            return []

        # -------------------------------------------------
        # Local evidence
        # -------------------------------------------------

        local_evidence = self.retrieve_local(
            claim,
            top_k=top_k,
        )

        # -------------------------------------------------
        # Web evidence
        # -------------------------------------------------

        web_evidence = self.retrieve_web(
            claim,
            max_results=max_web_results,
        )

        combined = []

        # -------------------------------------------------
        # Mark local evidence
        # -------------------------------------------------

        for item in local_evidence:

            document = item.copy()

            document[
                "source_type"
            ] = "local"

            combined.append(
                document
            )

        # -------------------------------------------------
        # Mark web evidence
        # -------------------------------------------------

        for item in web_evidence:

            document = item.copy()

            document[
                "source_type"
            ] = "web"

            combined.append(
                document
            )

        # -------------------------------------------------
        # Deduplicate
        # -------------------------------------------------

        combined = self._deduplicate_evidence(
            combined
        )

        # -------------------------------------------------
        # Add source quality
        # -------------------------------------------------

        combined = self._add_source_quality(
            combined
        )

        return combined