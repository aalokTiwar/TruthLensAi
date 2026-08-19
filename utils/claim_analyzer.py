"""
Claim analysis utilities for TruthLens AI.

The analyzer performs lightweight classification before retrieval.
It does not decide whether a claim is true or false.

Its purpose is to understand:
- claim type
- whether the claim is time-sensitive
- whether it appears subjective
- whether it is ambiguous
- useful search terms
"""

import re
from typing import List, Dict


SUBJECTIVE_PATTERNS = [
    "god of",
    "greatest",
    "best",
    "worst",
    "most beautiful",
    "most handsome",
    "most talented",
    "legendary",
    "better than",
    "the greatest",
    "the best",
    "the worst",
    "in my opinion",
    "i think",
    "arguably",
    "is amazing",
    "is overrated",
    "is underrated",
]


TEMPORAL_PATTERNS = [
    "today",
    "yesterday",
    "tomorrow",
    "currently",
    "right now",
    "this year",
    "this month",
    "this week",
    "recently",
    "latest",
    "new",
    "2024",
    "2025",
    "2026",
    "2027",
    "2028",
    "2029",
    "2030",
]


QUESTION_WORDS = [
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
]


def normalize_claim(claim: str) -> str:
    """
    Normalize claim text for analysis.
    """

    return re.sub(
        r"\s+",
        " ",
        claim.strip(),
    )


def detect_subjective(claim: str) -> bool:
    """
    Detect whether the claim contains language
    commonly associated with subjective opinions.
    """

    lowered = claim.lower()

    return any(
        pattern in lowered
        for pattern in SUBJECTIVE_PATTERNS
    )


def detect_temporal(claim: str) -> bool:
    """
    Detect whether the claim appears time-sensitive.
    """

    lowered = claim.lower()

    if any(
        pattern in lowered
        for pattern in TEMPORAL_PATTERNS
    ):
        return True

    # Detect four-digit years.
    return bool(
        re.search(
            r"\b(19|20)\d{2}\b",
            lowered,
        )
    )


def detect_question(claim: str) -> bool:
    """
    Detect whether the input is phrased as a question.
    """

    lowered = claim.lower()

    if "?" in lowered:
        return True

    first_word = lowered.split(
        maxsplit=1
    )[0] if lowered else ""

    return first_word in QUESTION_WORDS


def extract_search_terms(claim: str) -> List[str]:
    """
    Extract useful terms for search generation.

    This intentionally remains lightweight.
    The actual search engine will perform semantic retrieval.
    """

    normalized = normalize_claim(claim)

    words = re.findall(
        r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b",
        normalized,
    )

    stop_words = {
        "the",
        "is",
        "are",
        "was",
        "were",
        "a",
        "an",
        "of",
        "on",
        "in",
        "to",
        "and",
        "or",
        "for",
        "with",
        "that",
        "this",
        "has",
        "have",
        "had",
        "been",
        "be",
        "by",
        "from",
        "as",
    }

    terms = [
        word
        for word in words
        if word.lower() not in stop_words
    ]

    return terms


def analyze_claim(
    claim: str,
) -> Dict[str, object]:
    """
    Analyze a claim before retrieval.
    """

    normalized = normalize_claim(
        claim
    )

    if not normalized:
        raise ValueError(
            "Claim cannot be empty."
        )

    subjective = detect_subjective(
        normalized
    )

    temporal = detect_temporal(
        normalized
    )

    question = detect_question(
        normalized
    )

    if subjective:
        claim_type = "SUBJECTIVE"

    elif question:
        claim_type = "QUESTION"

    elif temporal:
        claim_type = "TIME_SENSITIVE"

    else:
        claim_type = "FACTUAL"

    return {
        "claim": normalized,
        "claim_type": claim_type,
        "is_subjective": subjective,
        "is_time_sensitive": temporal,
        "is_question": question,
        "search_terms": extract_search_terms(
            normalized
        ),
    }