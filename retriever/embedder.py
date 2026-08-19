"""
Embedding utilities for TruthLens AI.

Uses a lightweight Sentence Transformer model to convert
claims and evidence into numerical vectors.
"""

from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Generate semantic embeddings using Sentence Transformers."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        """
        Convert a list of texts into embedding vectors.

        Args:
            texts: List of text strings.

        Returns:
            Numpy array containing embedding vectors.
        """

        if not texts:
            return []

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_single(self, text: str):
        """
        Convert one text string into an embedding vector.
        """

        return self.encode([text])[0]