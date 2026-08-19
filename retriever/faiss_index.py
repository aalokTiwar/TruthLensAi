"""
FAISS vector index for TruthLens AI.

Stores evidence embeddings and performs semantic similarity search.
"""

from typing import List, Dict, Any

import faiss
import numpy as np

from retriever.embedder import Embedder


class EvidenceIndex:
    """FAISS-based semantic search index for evidence."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
        self.dimension = 384

        # Inner product works as cosine similarity because
        # our embeddings are normalized.
        self.index = faiss.IndexFlatIP(self.dimension)

        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add evidence documents to the FAISS index.

        Each document must contain an 'evidence' field.
        """

        if not documents:
            return

        texts = [
            document["evidence"]
            for document in documents
        ]

        embeddings = self.embedder.encode(texts)

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the evidence index for the most relevant documents.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedder.encode_single(query)

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            document = self.documents[index].copy()

            document["relevance_score"] = float(score)

            results.append(document)

        return results

    def count(self) -> int:
        """Return the number of indexed documents."""

        return self.index.ntotal