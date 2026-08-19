from utils.schemas import (
    VerifyRequest,
    Evidence,
    Verdict,
    ConfidenceBreakdown,
    VerificationMetadata,
    VerificationResponse,
)


# =========================================================
# VerifyRequest
# =========================================================

def test_verify_request():

    request = VerifyRequest(
        text="The Earth is round."
    )

    assert request.text == "The Earth is round."


# =========================================================
# Evidence
# =========================================================

def test_evidence():

    evidence = Evidence(
        title="Example Source",
        url="https://example.com",
        snippet="Example evidence.",
        source_type="web",
        relevance_score=0.92,
        semantic_relevance=0.91,
        source_quality_score=0.88,
    )

    assert evidence.title == "Example Source"

    assert evidence.relevance_score == 0.92

    assert evidence.semantic_relevance == 0.91

    assert evidence.source_quality_score == 0.88


# =========================================================
# Verdict
# =========================================================

def test_verdict():

    verdict = Verdict(
        label="TRUE",
        explanation="The evidence supports the claim.",
        confidence=0.95,
    )

    assert verdict.label == "TRUE"

    assert verdict.confidence == 0.95


# =========================================================
# Confidence Breakdown
# =========================================================

def test_confidence_breakdown():

    breakdown = ConfidenceBreakdown(
        llm_confidence=0.95,
        evidence_relevance=0.90,
        source_quality=0.85,
        evidence_agreement=0.80,
        final_confidence=0.90,
        confidence_level="VERY_HIGH",
    )

    assert breakdown.llm_confidence == 0.95

    assert breakdown.evidence_relevance == 0.90

    assert breakdown.source_quality == 0.85

    assert breakdown.evidence_agreement == 0.80

    assert breakdown.final_confidence == 0.90

    assert breakdown.confidence_level == "VERY_HIGH"


# =========================================================
# Verification Metadata
# =========================================================

def test_verification_metadata():

    metadata = VerificationMetadata(
        confidence_level="VERY_HIGH",
        evidence_count=2,
        local_evidence_count=1,
        web_evidence_count=1,
        agent_iterations=2,
        searches_performed=3,
    )

    assert metadata.confidence_level == "VERY_HIGH"

    assert metadata.evidence_count == 2

    assert metadata.local_evidence_count == 1

    assert metadata.web_evidence_count == 1

    assert metadata.agent_iterations == 2

    assert metadata.searches_performed == 3


# =========================================================
# Verification Response
# =========================================================

def test_verification_response():

    verdict = Verdict(
        label="FALSE",
        explanation="The evidence contradicts the claim.",
        confidence=0.90,
    )

    breakdown = ConfidenceBreakdown(
        llm_confidence=0.95,
        evidence_relevance=0.90,
        source_quality=0.85,
        evidence_agreement=0.80,
        final_confidence=0.90,
        confidence_level="VERY_HIGH",
    )

    metadata = VerificationMetadata(
        confidence_level="VERY_HIGH",
        evidence_count=2,
        local_evidence_count=1,
        web_evidence_count=1,
        agent_iterations=2,
        searches_performed=3,
        confidence_breakdown=breakdown,
    )

    response = VerificationResponse(
        claim="Example claim",
        verdict=verdict,
        metadata=metadata,
    )

    assert response.claim == "Example claim"

    assert response.verdict.label == "FALSE"

    assert response.verdict.confidence == 0.90

    assert response.metadata.confidence_level == "VERY_HIGH"

    assert response.metadata.evidence_count == 2

    assert response.metadata.local_evidence_count == 1

    assert response.metadata.web_evidence_count == 1

    assert response.metadata.agent_iterations == 2

    assert response.metadata.searches_performed == 3

    assert response.metadata.confidence_breakdown is not None

    assert (
        response.metadata
        .confidence_breakdown
        .llm_confidence
        == 0.95
    )

    assert (
        response.metadata
        .confidence_breakdown
        .evidence_relevance
        == 0.90
    )

    assert (
        response.metadata
        .confidence_breakdown
        .source_quality
        == 0.85
    )

    assert (
        response.metadata
        .confidence_breakdown
        .evidence_agreement
        == 0.80
    )

    assert (
        response.metadata
        .confidence_breakdown
        .final_confidence
        == 0.90
    )

    assert (
        response.metadata
        .confidence_breakdown
        .confidence_level
        == "VERY_HIGH"
    )