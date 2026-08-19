from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from utils.schemas import Verdict, Evidence


client = TestClient(app)


# =========================================================
# Mock fact-check result
# =========================================================

def mock_verify(claim):

    evidence = [
        Evidence(
            title="Space Exploration Reference",
            url="https://example.com/mars",
            snippet=(
                "No human has landed on Mars. "
                "Mars has been explored by robotic missions."
            ),
            source_type="local",
            relevance_score=0.95,
            semantic_relevance=0.95,
            source_quality_score=0.90,
        ),
        Evidence(
            title="Mars Exploration Source",
            url="https://example.com/space",
            snippet=(
                "Human missions to Mars have not yet occurred."
            ),
            source_type="web",
            relevance_score=0.90,
            semantic_relevance=0.90,
            source_quality_score=0.85,
        ),
    ]

    verdict = Verdict(
        label="FALSE",
        explanation=(
            "The available evidence contradicts the claim. "
            "The evidence states that humans have not landed "
            "on Mars and that Mars has been explored by "
            "robotic missions."
        ),
        confidence=0.95,
        evidence=evidence,
        missing_evidence=[],
    )

    return {
        "claim": claim,
        "verdict": verdict,
        "agent_iterations": 1,
        "searches_performed": 1,
        "evidence_count": len(evidence),
    }


# =========================================================
# Root endpoint
# =========================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "TruthLens AI"
    assert data["version"] == "1.0.0"


# =========================================================
# Health endpoint
# =========================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# =========================================================
# Empty claim validation
# =========================================================

def test_verify_empty_claim():

    response = client.post(
        "/verify",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


# =========================================================
# Verify endpoint response structure
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_verify_claim_response(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    # -----------------------------------------------------
    # Top-level response
    # -----------------------------------------------------

    assert "claim" in data
    assert "verdict" in data
    assert "metadata" in data

    assert (
        data["claim"]
        == "Humans landed on Mars in 2025."
    )

    # -----------------------------------------------------
    # Verify mock was called
    # -----------------------------------------------------

    mock_agent.assert_called_once_with(
        "Humans landed on Mars in 2025."
    )

    # -----------------------------------------------------
    # Verdict
    # -----------------------------------------------------

    verdict = data["verdict"]

    assert "label" in verdict
    assert "explanation" in verdict
    assert "confidence" in verdict
    assert "evidence" in verdict
    assert "missing_evidence" in verdict

    assert verdict["label"] in {
        "TRUE",
        "FALSE",
        "NOT_ENOUGH_EVIDENCE",
    }

    assert (
        0.0
        <= verdict["confidence"]
        <= 1.0
    )

    assert isinstance(
        verdict["explanation"],
        str,
    )

    assert len(
        verdict["explanation"]
    ) > 0

    assert isinstance(
        verdict["evidence"],
        list,
    )


# =========================================================
# Evidence structure
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_evidence_metadata_structure(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    evidence = data["verdict"]["evidence"]

    assert len(evidence) >= 1

    for item in evidence:

        assert "title" in item
        assert "url" in item
        assert "snippet" in item
        assert "source_type" in item
        assert "relevance_score" in item
        assert "semantic_relevance" in item
        assert "source_quality_score" in item

        assert item["source_type"] in {
            "local",
            "web",
        }

        if item["relevance_score"] is not None:

            assert (
                0.0
                <= item["relevance_score"]
                <= 1.0
            )

        if item["semantic_relevance"] is not None:

            assert (
                0.0
                <= item["semantic_relevance"]
                <= 1.0
            )

        if item["source_quality_score"] is not None:

            assert (
                0.0
                <= item["source_quality_score"]
                <= 1.0
            )


# =========================================================
# Metadata
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_metadata_structure(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    metadata = data["metadata"]

    assert "confidence_level" in metadata
    assert "evidence_count" in metadata
    assert "local_evidence_count" in metadata
    assert "web_evidence_count" in metadata
    assert "agent_iterations" in metadata
    assert "searches_performed" in metadata
    assert "confidence_breakdown" in metadata

    assert metadata[
        "confidence_level"
    ] in {
        "VERY_HIGH",
        "HIGH",
        "MEDIUM",
        "LOW",
        "VERY_LOW",
    }

    assert metadata["evidence_count"] >= 0
    assert metadata["local_evidence_count"] >= 0
    assert metadata["web_evidence_count"] >= 0
    assert metadata["agent_iterations"] >= 1
    assert metadata["searches_performed"] >= 0

    assert (
        metadata["local_evidence_count"]
        == 1
    )

    assert (
        metadata["web_evidence_count"]
        == 1
    )


# =========================================================
# Confidence breakdown
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_confidence_breakdown(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    breakdown = (
        data["metadata"]
        ["confidence_breakdown"]
    )

    assert breakdown is not None

    assert "llm_confidence" in breakdown
    assert "evidence_relevance" in breakdown
    assert "source_quality" in breakdown
    assert "evidence_agreement" in breakdown
    assert "final_confidence" in breakdown
    assert "confidence_level" in breakdown

    assert (
        0.0
        <= breakdown["llm_confidence"]
        <= 1.0
    )

    assert (
        0.0
        <= breakdown["evidence_relevance"]
        <= 1.0
    )

    assert (
        0.0
        <= breakdown["source_quality"]
        <= 1.0
    )

    assert (
        0.0
        <= breakdown["evidence_agreement"]
        <= 1.0
    )

    assert (
        0.0
        <= breakdown["final_confidence"]
        <= 1.0
    )

    assert (
        breakdown["confidence_level"]
        in {
            "VERY_HIGH",
            "HIGH",
            "MEDIUM",
            "LOW",
            "VERY_LOW",
        }
    )


# =========================================================
# Confidence consistency
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_confidence_consistency(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    verdict_confidence = (
        data["verdict"]["confidence"]
    )

    final_confidence = (
        data["metadata"]
        ["confidence_breakdown"]
        ["final_confidence"]
    )

    assert (
        verdict_confidence
        == final_confidence
    )


# =========================================================
# Confidence level consistency
# =========================================================

@patch(
    "backend.main.fact_check_agent.verify",
    side_effect=mock_verify,
)
def test_confidence_level_consistency(mock_agent):

    response = client.post(
        "/verify",
        json={
            "text": "Humans landed on Mars in 2025."
        },
    )

    assert response.status_code == 200

    data = response.json()

    verdict_confidence = (
        data["verdict"]["confidence"]
    )

    confidence_level = (
        data["metadata"]
        ["confidence_breakdown"]
        ["confidence_level"]
    )

    expected_levels = {
        "VERY_HIGH": (
            verdict_confidence >= 0.90
        ),
        "HIGH": (
            0.75
            <= verdict_confidence
            < 0.90
        ),
        "MEDIUM": (
            0.50
            <= verdict_confidence
            < 0.75
        ),
        "LOW": (
            0.25
            <= verdict_confidence
            < 0.50
        ),
        "VERY_LOW": (
            verdict_confidence < 0.25
        ),
    }

    assert expected_levels[
        confidence_level
    ] is True