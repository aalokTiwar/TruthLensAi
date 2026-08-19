"""
Claim extraction utilities for TruthLens AI.

The first version uses lightweight text processing instead of an LLM.
This keeps claim extraction fast, deterministic, and free.
"""

import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Clean unnecessary whitespace from input text.
    """

    if not text:
        return ""

    text = text.strip()

    # Replace multiple whitespace characters with a single space.
    text = re.sub(r"\s+", " ", text)

    return text


def extract_claims(text: str) -> List[str]:
    """
    Extract individual claims from user-provided text.

    The current implementation:
    1. Normalizes whitespace.
    2. Splits text into sentences.
    3. Removes empty fragments.
    4. Returns a list of claims.
    """

    text = normalize_text(text)

    if not text:
        return []

    # Split on sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            claims.append(sentence)

    return claims