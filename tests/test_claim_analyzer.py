from utils.claim_analyzer import (
    analyze_claim,
    detect_subjective,
    detect_temporal,
    extract_search_terms,
)


def test_subjective_claim():

    result = analyze_claim(
        "Virat Kohli is god of cricket."
    )

    assert result["claim_type"] == "SUBJECTIVE"
    assert result["is_subjective"] is True


def test_temporal_claim():

    result = analyze_claim(
        "Humans landed on Mars in 2025."
    )

    assert result["is_time_sensitive"] is True


def test_factual_claim():

    result = analyze_claim(
        "The Earth revolves around the Sun."
    )

    assert result["claim_type"] == "FACTUAL"
    assert result["is_subjective"] is False


def test_question():

    result = analyze_claim(
        "Who discovered penicillin?"
    )

    assert result["is_question"] is True


def test_search_terms():

    terms = extract_search_terms(
        "The Earth revolves around the Sun."
    )

    assert "Earth" in terms
    assert "Sun" in terms


def test_empty_claim():

    try:
        analyze_claim("")
        assert False

    except ValueError:
        assert True