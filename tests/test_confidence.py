import pytest

from utils.confidence import (
    get_confidence_level,
    calculate_relevance_score,
    calculate_source_quality,
    calculate_evidence_agreement,
    calculate_final_confidence,
)


# =========================================================
# Confidence Level Tests
# =========================================================


def test_very_high_confidence():

    assert (
        get_confidence_level(0.99)
        == "VERY_HIGH"
    )


def test_high_confidence():

    assert (
        get_confidence_level(0.80)
        == "HIGH"
    )


def test_medium_confidence():

    assert (
        get_confidence_level(0.60)
        == "MEDIUM"
    )


def test_low_confidence():

    assert (
        get_confidence_level(0.30)
        == "LOW"
    )


def test_very_low_confidence():

    assert (
        get_confidence_level(0.10)
        == "VERY_LOW"
    )


# =========================================================
# Relevance Score Tests
# =========================================================


def test_relevance_score():

    evidence = [
        {
            "semantic_relevance": 0.90,
        },
        {
            "semantic_relevance": 0.80,
        },
    ]

    score = calculate_relevance_score(
        evidence
    )

    assert score == 0.85


def test_empty_evidence_relevance():

    score = calculate_relevance_score(
        []
    )

    assert score == 0.0


# =========================================================
# Source Quality Tests
# =========================================================


def test_source_quality_score():

    evidence = [
        {
            "source_quality_score": 1.00,
        },
        {
            "source_quality_score": 0.80,
        },
    ]

    score = calculate_source_quality(
        evidence
    )

    assert score == 0.90


def test_missing_source_quality_uses_neutral_score():

    evidence = [
        {
            "semantic_relevance": 0.90,
        }
    ]

    score = calculate_source_quality(
        evidence
    )

    assert score == 0.50


# =========================================================
# Evidence Agreement Tests
# =========================================================


def test_evidence_agreement():

    evidence = [
        {
            "semantic_relevance": 0.95,
        },
        {
            "semantic_relevance": 0.85,
        },
        {
            "semantic_relevance": 0.40,
        },
    ]

    score = calculate_evidence_agreement(
        evidence
    )

    assert score == pytest.approx(
        2 / 3,
        abs=0.0001,
    )


def test_empty_evidence_agreement():

    score = calculate_evidence_agreement(
        []
    )

    assert score == 0.0


# =========================================================
# Final Confidence Tests
# =========================================================


def test_final_confidence_is_between_zero_and_one():

    evidence = [
        {
            "semantic_relevance": 0.90,
            "source_quality_score": 1.00,
        },
        {
            "semantic_relevance": 0.85,
            "source_quality_score": 0.85,
        },
    ]

    confidence = calculate_final_confidence(
        llm_confidence=0.95,
        evidence=evidence,
    )

    assert 0.0 <= confidence <= 1.0


def test_strong_evidence_produces_high_confidence():

    evidence = [
        {
            "semantic_relevance": 0.95,
            "source_quality_score": 1.00,
        },
        {
            "semantic_relevance": 0.90,
            "source_quality_score": 0.85,
        },
    ]

    confidence = calculate_final_confidence(
        llm_confidence=0.95,
        evidence=evidence,
    )

    assert confidence >= 0.85


# =========================================================
# Edge Case Tests
# =========================================================


def test_confidence_is_clamped():

    assert (
        get_confidence_level(2.0)
        == "VERY_HIGH"
    )

    assert (
        get_confidence_level(-1.0)
        == "VERY_LOW"
    )


def test_relevance_score_ignores_invalid_values():

    evidence = [
        {
            "semantic_relevance": "invalid",
        },
        {
            "semantic_relevance": 0.80,
        },
        {
            "semantic_relevance": None,
        },
    ]

    score = calculate_relevance_score(
        evidence
    )

    assert score == 0.80


def test_final_confidence_handles_invalid_llm_value():

    evidence = [
        {
            "semantic_relevance": 0.90,
            "source_quality_score": 1.00,
        }
    ]

    confidence = calculate_final_confidence(
        llm_confidence="invalid",
        evidence=evidence,
    )

    assert 0.0 <= confidence <= 1.0