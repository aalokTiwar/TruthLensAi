"""
Pydantic schemas used throughout TruthLens AI.

These models define the structure of:
- API requests
- Extracted claims
- Evidence
- Fact-check verdicts
- Confidence analysis
- Verification metadata
- Final API responses
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# =========================================================
# API Request
# =========================================================

class VerifyRequest(BaseModel):
    """Request received by the /verify endpoint."""

    text: str = Field(
        ...,
        min_length=3,
        description="Claim or text that needs to be fact checked.",
    )


# =========================================================
# Evidence
# =========================================================

class Evidence(BaseModel):
    """A single piece of evidence used for fact checking."""

    title: str

    url: Optional[str] = None

    snippet: str

    source_type: str = "web"

    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    semantic_relevance: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    source_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# =========================================================
# Extracted Claim
# =========================================================

class Claim(BaseModel):
    """Represents an extracted claim."""

    text: str

    claim_id: Optional[str] = None


# =========================================================
# Verdict
# =========================================================

class Verdict(BaseModel):
    """Structured reasoning result for a claim."""

    label: str

    explanation: str

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    evidence: List[Evidence] = Field(
        default_factory=list,
    )

    missing_evidence: List[str] = Field(
        default_factory=list,
    )


# =========================================================
# Confidence Breakdown
# =========================================================

class ConfidenceBreakdown(BaseModel):
    """
    Explain how TruthLens AI calculated final confidence.

    Weighting:

        LLM confidence       50%
        Evidence relevance   20%
        Source quality       20%
        Evidence agreement   10%
    """

    llm_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    evidence_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    source_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    evidence_agreement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    final_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    confidence_level: str


# =========================================================
# Verification Metadata
# =========================================================

class VerificationMetadata(BaseModel):
    """Metadata describing how the fact-check was performed."""

    confidence_level: str

    evidence_count: int = Field(
        default=0,
        ge=0,
    )

    local_evidence_count: int = Field(
        default=0,
        ge=0,
    )

    web_evidence_count: int = Field(
        default=0,
        ge=0,
    )

    agent_iterations: int = Field(
        default=1,
        ge=1,
    )

    searches_performed: int = Field(
        default=0,
        ge=0,
    )

    confidence_breakdown: Optional[
        ConfidenceBreakdown
    ] = None


# =========================================================
# Final API Response
# =========================================================

class VerificationResponse(BaseModel):
    """Final response returned by TruthLens AI."""

    claim: str

    verdict: Verdict

    metadata: VerificationMetadata