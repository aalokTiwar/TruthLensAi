from utils.claim_analyzer import analyze_claim
from utils.query_planner import plan_queries


def test_empty_claim():

    try:
        plan_queries("")

        assert False

    except ValueError:
        assert True


def test_factual_claim_queries():

    claim = (
        "The Earth revolves around the Sun."
    )

    analysis = analyze_claim(
        claim
    )

    queries = plan_queries(
        claim,
        analysis,
    )

    assert len(queries) >= 3

    assert any(
        "Earth" in query
        for query in queries
    )


def test_subjective_claim_queries():

    claim = (
        "Virat Kohli is god of cricket."
    )

    analysis = analyze_claim(
        claim
    )

    queries = plan_queries(
        claim,
        analysis,
    )

    assert analysis["claim_type"] == (
        "SUBJECTIVE"
    )

    assert any(
        "expert opinion" in query
        for query in queries
    )


def test_temporal_claim_queries():

    claim = (
        "Humans landed on Mars in 2025."
    )

    analysis = analyze_claim(
        claim
    )

    queries = plan_queries(
        claim,
        analysis,
    )

    assert analysis[
        "is_time_sensitive"
    ] is True

    assert any(
        "official source" in query
        for query in queries
    )


def test_query_deduplication():

    claim = (
        "The Earth revolves around the Sun."
    )

    analysis = analyze_claim(
        claim
    )

    queries = plan_queries(
        claim,
        analysis,
    )

    normalized = [
        query.lower()
        for query in queries
    ]

    assert len(normalized) == len(
        set(normalized)
    )