"""
Evidence relevance filtering for TruthLens AI.

Uses the existing Sentence Transformer embedder to measure
semantic similarity between a claim and retrieved evidence.

Supports both:
- Web evidence
- Local dataset evidence
"""

from typing import List, Dict, Any

import numpy as np

from retriever.embedder import Embedder


class EvidenceFilter:
    """
    Filter retrieved evidence using semantic similarity.
    """

    def __init__(
        self,
        threshold: float = 0.30,
    ):
        self.threshold = threshold
        self.embedder = Embedder()

    @staticmethod
    def _build_evidence_text(
        item: Dict[str, Any],
    ) -> str:
        """
        Build the text representation used for comparison.

        Supports:
        - title
        - snippet
        - text
        - evidence

        The 'evidence' field is important for local
        dataset records.
        """

        title = str(
            item.get(
                "title",
                "",
            )
        )

        snippet = str(
            item.get(
                "snippet",
                "",
            )
        )

        text = str(
            item.get(
                "text",
                "",
            )
        )

        evidence = str(
            item.get(
                "evidence",
                "",
            )
        )

        # Prevent very large webpages from dominating
        # the embedding.
        text = text[:3000]

        return " ".join(
            part
            for part in (
                title,
                snippet,
                text,
                evidence,
            )
            if part.strip()
        )

    @staticmethod
    def _cosine_similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        The Embedder already normalizes embeddings, but
        normalization is kept here for safety.
        """

        first_norm = np.linalg.norm(
            first
        )

        second_norm = np.linalg.norm(
            second
        )

        if (
            first_norm == 0
            or second_norm == 0
        ):
            return 0.0

        return float(
            np.dot(
                first,
                second,
            )
            / (
                first_norm
                * second_norm
            )
        )

    def filter(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return evidence that is sufficiently relevant
        to the claim.

        Each retained evidence item receives:

            semantic_relevance

        containing its semantic similarity score.
        """

        if not claim or not claim.strip():
            return []

        if not evidence:
            return []

        # -------------------------------------------------
        # Build evidence text
        # -------------------------------------------------

        evidence_texts = [
            self._build_evidence_text(
                item
            )
            for item in evidence
        ]

        # -------------------------------------------------
        # Generate claim embedding
        # -------------------------------------------------

        claim_embedding = (
            self.embedder.encode_single(
                claim
            )
        )

        # -------------------------------------------------
        # Generate evidence embeddings
        # -------------------------------------------------

        evidence_embeddings = (
            self.embedder.encode(
                evidence_texts
            )
        )

        # -------------------------------------------------
        # Calculate similarity and filter
        # -------------------------------------------------

        filtered = []

        for item, embedding in zip(
            evidence,
            evidence_embeddings,
        ):

            similarity = (
                self._cosine_similarity(
                    claim_embedding,
                    embedding,
                )
            )

            updated_item = dict(item)

            updated_item[
                "semantic_relevance"
            ] = similarity

            if similarity >= self.threshold:

                filtered.append(
                    updated_item
                )

        # -------------------------------------------------
        # Highest relevance first
        # -------------------------------------------------

        filtered.sort(
            key=lambda item: item.get(
                "semantic_relevance",
                0.0,
            ),
            reverse=True,
        )

        return filtered