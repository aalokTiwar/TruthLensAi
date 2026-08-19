from rag.reasoner import RAGReasoner


def test_reasoner_with_mars_evidence():

    reasoner = RAGReasoner()

    claim = "Humans landed on Mars in 2025."

    evidence = [
        {
            "title": "Space Exploration Reference",
            "url": "https://example.com/mars",
            "snippet": (
                "No human has landed on Mars. "
                "Mars has only been explored by robotic missions."
            ),
            "text": (
                "No human has landed on Mars. "
                "Mars has only been explored by robotic missions."
            ),
            "source_type": "local",
            "relevance_score": 0.95,
            "semantic_relevance": 0.95,
            "source_quality_score": 0.90,
        }
    ]

    verdict = reasoner.reason(
        claim,
        evidence,
    )

    # -----------------------------------------------------
    # Verdict validation
    # -----------------------------------------------------

    assert verdict.label in {
        "TRUE",
        "FALSE",
        "NOT_ENOUGH_EVIDENCE",
    }

    # -----------------------------------------------------
    # Confidence validation
    # -----------------------------------------------------

    assert 0.0 <= verdict.confidence <= 1.0

    # -----------------------------------------------------
    # Explanation validation
    # -----------------------------------------------------

    assert isinstance(
        verdict.explanation,
        str,
    )

    assert len(
        verdict.explanation
    ) > 0

    # -----------------------------------------------------
    # Evidence validation
    # -----------------------------------------------------

    assert isinstance(
        verdict.evidence,
        list,
    )

    assert len(
        verdict.evidence
    ) >= 1


def test_reasoner_preserves_evidence_metadata():

    reasoner = RAGReasoner()

    claim = "Humans landed on Mars in 2025."

    evidence = [
        {
            "title": "Space Exploration Reference",
            "url": "https://example.com/mars",
            "snippet": "No human has landed on Mars.",
            "text": "No human has landed on Mars.",
            "source_type": "local",
            "relevance_score": 0.95,
            "semantic_relevance": 0.95,
            "source_quality_score": 0.90,
        }
    ]

    verdict = reasoner.reason(
        claim,
        evidence,
    )

    assert len(verdict.evidence) >= 1

    selected = verdict.evidence[0]

    # -----------------------------------------------------
    # Basic metadata
    # -----------------------------------------------------

    assert (
        selected.title
        == "Space Exploration Reference"
    )

    assert (
        selected.url
        == "https://example.com/mars"
    )

    assert (
        selected.source_type
        == "local"
    )

    # -----------------------------------------------------
    # Existing relevance score
    # -----------------------------------------------------

    assert (
        selected.relevance_score
        == 0.95
    )

    # -----------------------------------------------------
    # New semantic relevance
    # -----------------------------------------------------

    assert (
        selected.semantic_relevance
        == 0.95
    )

    # -----------------------------------------------------
    # New source quality score
    # -----------------------------------------------------

    assert (
        selected.source_quality_score
        == 0.90
    )


def test_reasoner_handles_empty_evidence():

    reasoner = RAGReasoner()

    claim = "Humans landed on Mars in 2025."

    verdict = reasoner.reason(
        claim,
        [],
    )

    assert verdict.label in {
        "TRUE",
        "FALSE",
        "NOT_ENOUGH_EVIDENCE",
    }

    # No evidence means confidence must remain low.
    assert 0.0 <= verdict.confidence <= 0.25

    assert isinstance(
        verdict.explanation,
        str,
    )


def test_reasoner_rejects_empty_claim():

    reasoner = RAGReasoner()

    try:

        reasoner.reason(
            "",
            [],
        )

        assert False, (
            "Expected ValueError for empty claim"
        )

    except ValueError as exc:

        assert (
            str(exc)
            == "Claim cannot be empty."
        )