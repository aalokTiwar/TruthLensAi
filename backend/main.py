"""
TruthLens AI FastAPI backend.

Exposes the agentic fact-checking pipeline through HTTP.
"""

from fastapi import FastAPI, HTTPException

from utils.confidence import (
    get_confidence_level,
    calculate_relevance_score,
    calculate_source_quality,
    calculate_evidence_agreement,
)

from utils.pipeline import build_pipeline

from utils.schemas import (
    VerifyRequest,
    ConfidenceBreakdown,
    VerificationMetadata,
    VerificationResponse,
)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="TruthLens AI",
    description=(
        "Explainable agentic fact-checking assistant "
        "using hybrid retrieval and RAG."
    ),
    version="1.0.0",
)


# =========================================================
# Build Pipeline
# =========================================================

fact_check_agent = build_pipeline()


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def root():
    """Basic service information."""

    return {
        "status": "ok",
        "service": "TruthLens AI",
        "version": "1.0.0",
    }


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/health")
def health():
    """Backend health check."""

    return {
        "status": "healthy",
    }


# =========================================================
# Verify Endpoint
# =========================================================

@app.post(
    "/verify",
    response_model=VerificationResponse,
)
def verify_claim(
    request: VerifyRequest,
):
    """
    Verify a user-provided claim using the
    agentic fact-checking pipeline.
    """

    try:

        # -------------------------------------------------
        # Step 1 — Run fact-checking pipeline
        # -------------------------------------------------

        result = fact_check_agent.verify(
            request.text
        )

        verdict = result["verdict"]

        # -------------------------------------------------
        # Step 2 — Convert evidence to dictionaries
        # -------------------------------------------------

        evidence = []

        for item in verdict.evidence:

            if hasattr(
                item,
                "model_dump",
            ):

                evidence.append(
                    item.model_dump()
                )

            else:

                evidence.append(
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet,
                        "source_type": item.source_type,
                        "relevance_score": (
                            item.relevance_score
                        ),
                        "semantic_relevance": (
                            item.semantic_relevance
                        ),
                        "source_quality_score": (
                            item.source_quality_score
                        ),
                    }
                )

        # -------------------------------------------------
        # Step 3 — Evidence counts
        # -------------------------------------------------

        evidence_count = len(
            evidence
        )

        local_evidence_count = sum(
            1
            for item in evidence
            if item.get(
                "source_type"
            ) == "local"
        )

        web_evidence_count = sum(
            1
            for item in evidence
            if item.get(
                "source_type"
            ) == "web"
        )

        # -------------------------------------------------
        # Step 4 — Evidence quality analysis
        # -------------------------------------------------

        evidence_relevance = (
            calculate_relevance_score(
                evidence
            )
        )

        source_quality = (
            calculate_source_quality(
                evidence
            )
        )

        evidence_agreement = (
            calculate_evidence_agreement(
                evidence
            )
        )

        # -------------------------------------------------
        # Step 5 — Use confidence from the agent
        # -------------------------------------------------

        final_confidence = float(
            verdict.confidence
        )

        confidence_level = (
            get_confidence_level(
                final_confidence
            )
        )

        # -------------------------------------------------
        # Step 6 — Build confidence breakdown
        # -------------------------------------------------

        # IMPORTANT:
        # Until raw LLM confidence is exposed separately
        # by RAGReasoner, we do not pretend that the final
        # confidence is the raw LLM confidence.
        #
        # For now this field represents the confidence
        # available from the reasoning pipeline.

        llm_confidence = final_confidence

        confidence_breakdown = (
            ConfidenceBreakdown(
                llm_confidence=(
                    llm_confidence
                ),

                evidence_relevance=(
                    evidence_relevance
                ),

                source_quality=(
                    source_quality
                ),

                evidence_agreement=(
                    evidence_agreement
                ),

                final_confidence=(
                    final_confidence
                ),

                confidence_level=(
                    confidence_level
                ),
            )
        )

        # -------------------------------------------------
        # Step 7 — Build verification metadata
        # -------------------------------------------------

        metadata = VerificationMetadata(

            confidence_level=(
                confidence_level
            ),

            evidence_count=(
                evidence_count
            ),

            local_evidence_count=(
                local_evidence_count
            ),

            web_evidence_count=(
                web_evidence_count
            ),

            agent_iterations=(
                result[
                    "agent_iterations"
                ]
            ),

            searches_performed=(
                result[
                    "searches_performed"
                ]
            ),

            confidence_breakdown=(
                confidence_breakdown
            ),
        )

        # -------------------------------------------------
        # Step 8 — Return API response
        # -------------------------------------------------

        return VerificationResponse(
            claim=result["claim"],
            verdict=verdict,
            metadata=metadata,
        )

    # -----------------------------------------------------
    # Validation errors
    # -----------------------------------------------------

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # -----------------------------------------------------
    # Unexpected errors
    # -----------------------------------------------------

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "TruthLens failed to verify "
                "the claim."
            ),
        ) from exc