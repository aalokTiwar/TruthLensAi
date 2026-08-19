"""
Confidence utilities for TruthLens AI.

Provides:
1. Human-readable confidence classification.
2. Evidence-aware confidence calculation.
3. Evidence agreement scoring.

The goal is to avoid relying only on the LLM's
self-reported confidence.
"""

from typing import List, Dict, Any


# =========================================================
# Confidence Classification
# =========================================================

def get_confidence_level(
    confidence: float,
) -> str:
    """
    Convert a numerical confidence score into
    a human-readable confidence level.
    """

    confidence = max(
        0.0,
        min(
            1.0,
            float(confidence),
        ),
    )

    if confidence >= 0.90:
        return "VERY_HIGH"

    if confidence >= 0.75:
        return "HIGH"

    if confidence >= 0.50:
        return "MEDIUM"

    if confidence >= 0.25:
        return "LOW"

    return "VERY_LOW"


# =========================================================
# Evidence Relevance
# =========================================================

def calculate_relevance_score(
    evidence: List[Dict[str, Any]],
) -> float:
    """
    Calculate the average semantic relevance of evidence.

    Uses:
        semantic_relevance

    Falls back to:
        relevance_score
    """

    scores = []

    for item in evidence:

        score = item.get(
            "semantic_relevance"
        )

        if score is None:
            score = item.get(
                "relevance_score"
            )

        if score is None:
            continue

        try:
            score = float(score)

        except (
            TypeError,
            ValueError,
        ):
            continue

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        scores.append(score)

    if not scores:
        return 0.0

    return round(
        sum(scores) / len(scores),
        4,
    )


# =========================================================
# Source Quality
# =========================================================

def calculate_source_quality(
    evidence: List[Dict[str, Any]],
) -> float:
    """
    Calculate the average source quality.

    Uses:
        source_quality_score

    If no source-quality information exists,
    return a neutral score of 0.50.
    """

    scores = []

    for item in evidence:

        score = item.get(
            "source_quality_score"
        )

        if score is None:
            continue

        try:
            score = float(score)

        except (
            TypeError,
            ValueError,
        ):
            continue

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        scores.append(score)

    if not scores:
        return 0.50

    return round(
        sum(scores) / len(scores),
        4,
    )


# =========================================================
# Evidence Agreement
# =========================================================

def calculate_evidence_agreement(
    evidence: List[Dict[str, Any]],
) -> float:
    """
    Estimate how strongly the available evidence
    provides sufficient evidence coverage.

    Evidence with relevance >= 0.70 is considered
    strongly relevant.

    This function does NOT determine whether the
    claim is TRUE or FALSE.

    The LLM reasoner remains responsible for the
    actual verdict.
    """

    if not evidence:
        return 0.0

    relevance_scores = []

    for item in evidence:

        score = item.get(
            "semantic_relevance"
        )

        if score is None:
            score = item.get(
                "relevance_score"
            )

        if score is None:
            continue

        try:
            score = float(score)

        except (
            TypeError,
            ValueError,
        ):
            continue

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        relevance_scores.append(score)

    if not relevance_scores:
        return 0.50

    strong = sum(
        1
        for score in relevance_scores
        if score >= 0.70
    )

    agreement = (
        strong / len(relevance_scores)
    )

    return agreement


# =========================================================
# Final Confidence
# =========================================================

def calculate_final_confidence(
    llm_confidence: float,
    evidence: List[Dict[str, Any]],
) -> float:
    """
    Calculate evidence-aware final confidence.

    Weighted components:

        LLM confidence      50%
        Semantic relevance 20%
        Source quality      20%
        Evidence agreement  10%

    Formula:

        final_confidence =
            LLM confidence * 0.50
            + relevance * 0.20
            + source quality * 0.20
            + agreement * 0.10

    Returns a value between 0.0 and 1.0.
    """

    try:
        llm_confidence = float(
            llm_confidence
        )

    except (
        TypeError,
        ValueError,
    ):
        llm_confidence = 0.0

    llm_confidence = max(
        0.0,
        min(
            1.0,
            llm_confidence,
        ),
    )

    relevance = calculate_relevance_score(
        evidence
    )

    source_quality = calculate_source_quality(
        evidence
    )

    agreement = calculate_evidence_agreement(
        evidence
    )

    final_confidence = (
        (llm_confidence * 0.50)
        + (relevance * 0.20)
        + (source_quality * 0.20)
        + (agreement * 0.10)
    )

    final_confidence = max(
        0.0,
        min(
            1.0,
            final_confidence,
        ),
    )

    return round(
        final_confidence,
        4,
    )