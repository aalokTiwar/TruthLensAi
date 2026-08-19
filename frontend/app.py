"""
TruthLens AI Streamlit Frontend.

Provides a user-friendly interface for submitting claims
to the FastAPI fact-checking backend.
"""

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

REQUEST_TIMEOUT = 180


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# Session State
# =========================================================

if "claim_input" not in st.session_state:
    st.session_state.claim_input = ""


def set_example_claim(claim: str):
    """Set the claim selected by an example button."""

    st.session_state.claim_input = claim


# =========================================================
# Header
# =========================================================

st.title("🔎 TruthLens AI")

st.markdown(
    """
### Explainable Agentic Fact-Checking Assistant

TruthLens AI analyzes claims using:

- 🔍 Claim analysis
- 📚 Local evidence retrieval
- 🌐 Web search
- 🔬 Semantic evidence filtering
- 🏆 Source quality analysis
- 🤖 LLM-based reasoning
- 🔁 Agentic retry loop
- 📊 Trust-aware confidence
"""
)


# =========================================================
# Claim Input
# =========================================================

st.subheader("🔍 Verify a Claim")

st.write(
    "Enter the statement you want TruthLens AI to verify:"
)

claim = st.text_area(
    "Claim",
    placeholder=(
        "Example:\n"
        "Humans landed on Mars in 2025."
    ),
    height=130,
    key="claim_input",
    label_visibility="collapsed",
)


# =========================================================
# Example Claims
# =========================================================

st.caption("Try an example:")

col1, col2, col3 = st.columns(3)


with col1:

    st.button(
        "🌍 Earth orbits the Sun",
        use_container_width=True,
        on_click=set_example_claim,
        args=(
            "The Earth revolves around the Sun.",
        ),
    )


with col2:

    st.button(
        "🚀 Humans landed on Mars",
        use_container_width=True,
        on_click=set_example_claim,
        args=(
            "Humans landed on Mars in 2025.",
        ),
    )


with col3:

    st.button(
        "🏏 Virat Kohli is a cricketer",
        use_container_width=True,
        on_click=set_example_claim,
        args=(
            "Virat Kohli is an Indian cricketer.",
        ),
    )


# =========================================================
# Verify Button
# =========================================================

verify_clicked = st.button(
    "🔍 VERIFY CLAIM",
    type="primary",
    use_container_width=True,
)


# =========================================================
# Verify Claim
# =========================================================

if verify_clicked:

    claim = st.session_state.claim_input.strip()

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if not claim:

        st.warning(
            "Please enter a claim before verifying."
        )

        st.stop()


    # -----------------------------------------------------
    # Backend Request
    # -----------------------------------------------------

    with st.spinner(
        "🔎 TruthLens is researching the claim..."
    ):

        try:

            response = requests.post(
                f"{BACKEND_URL}/verify",
                json={
                    "text": claim,
                },
                timeout=REQUEST_TIMEOUT,
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to the TruthLens backend."
            )

            st.info(
                "Make sure FastAPI is running:"
            )

            st.code(
                "uvicorn backend.main:app --reload --port 8000",
                language="bash",
            )

            st.stop()


        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Verification request timed out."
            )

            st.info(
                "The backend may still be processing "
                "web retrieval and LLM reasoning."
            )

            st.stop()


        except requests.RequestException as exc:

            st.error(
                f"❌ Request failed: {exc}"
            )

            st.stop()


    # =====================================================
    # Backend Error
    # =====================================================

    if response.status_code != 200:

        try:

            error_detail = response.json().get(
                "detail",
                "Unknown backend error.",
            )

        except Exception:

            error_detail = response.text

        st.error(
            f"❌ Verification failed: {error_detail}"
        )

        st.stop()


    # =====================================================
    # Parse Response
    # =====================================================

    try:

        result = response.json()

        verdict = result["verdict"]

        metadata = result["metadata"]

    except (ValueError, KeyError) as exc:

        st.error(
            "❌ Invalid response received from backend."
        )

        st.exception(exc)

        st.stop()


    # =====================================================
    # Result
    # =====================================================

    st.divider()

    st.subheader("📋 Verification Result")


    label = verdict.get(
        "label",
        "NOT_ENOUGH_EVIDENCE",
    )

    confidence = float(
        verdict.get(
            "confidence",
            0.0,
        )
    )

    confidence_level = metadata.get(
        "confidence_level",
        "UNKNOWN",
    )


    # -----------------------------------------------------
    # Verdict
    # -----------------------------------------------------

    if label == "TRUE":

        verdict_icon = "✅"
        verdict_text = "TRUE"

    elif label == "FALSE":

        verdict_icon = "❌"
        verdict_text = "FALSE"

    else:

        verdict_icon = "⚠️"
        verdict_text = "NOT ENOUGH EVIDENCE"


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Verdict",
            f"{verdict_icon} {verdict_text}",
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%",
        )


    with col3:

        st.metric(
            "Confidence Level",
            confidence_level,
        )


    st.progress(
        max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )
    )


    # =====================================================
    # Explanation
    # =====================================================

    st.subheader("🧠 Explanation")

    st.info(
        verdict.get(
            "explanation",
            "No explanation was provided.",
        )
    )


    # =====================================================
    # Confidence Breakdown
    # =====================================================

    breakdown = metadata.get(
        "confidence_breakdown"
    )


    if breakdown:

        st.subheader(
            "📊 Confidence Analysis"
        )

        col1, col2 = st.columns(2)


        with col1:

            llm_confidence = breakdown.get(
                "llm_confidence",
                0.0,
            )

            st.metric(
                "LLM Confidence",
                f"{llm_confidence * 100:.1f}%",
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        llm_confidence,
                    ),
                )
            )


            evidence_relevance = breakdown.get(
                "evidence_relevance",
                0.0,
            )

            st.metric(
                "Evidence Relevance",
                f"{evidence_relevance * 100:.1f}%",
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        evidence_relevance,
                    ),
                )
            )


        with col2:

            source_quality = breakdown.get(
                "source_quality",
                0.0,
            )

            st.metric(
                "Source Quality",
                f"{source_quality * 100:.1f}%",
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        source_quality,
                    ),
                )
            )


            evidence_agreement = breakdown.get(
                "evidence_agreement",
                0.0,
            )

            st.metric(
                "Evidence Agreement",
                f"{evidence_agreement * 100:.1f}%",
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        evidence_agreement,
                    ),
                )
            )


        st.divider()

        final_confidence = breakdown.get(
            "final_confidence",
            confidence,
        )

        st.metric(
            "🎯 Final Trust-Aware Confidence",
            f"{final_confidence * 100:.1f}%",
        )


    # =====================================================
    # Evidence
    # =====================================================

    st.subheader("📚 Evidence")

    evidence = verdict.get(
        "evidence",
        [],
    )


    if not evidence:

        st.warning(
            "No evidence was selected for the final verdict."
        )

    else:

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            title = item.get(
                "title",
                "Unknown source",
            )

            with st.expander(
                f"Evidence {index}: {title}"
            ):

                st.write(
                    item.get(
                        "snippet",
                        "",
                    )
                )


                source_type = item.get(
                    "source_type",
                    "unknown",
                )

                st.caption(
                    f"Source type: {source_type}"
                )


                relevance = item.get(
                    "relevance_score"
                )

                if relevance is not None:

                    st.caption(
                        f"Retrieval relevance: "
                        f"{relevance:.4f}"
                    )


                semantic_relevance = item.get(
                    "semantic_relevance"
                )

                if semantic_relevance is not None:

                    st.caption(
                        f"Semantic relevance: "
                        f"{semantic_relevance:.4f}"
                    )


                source_quality = item.get(
                    "source_quality_score"
                )

                if source_quality is not None:

                    st.caption(
                        f"Source quality: "
                        f"{source_quality:.4f}"
                    )


                url = item.get(
                    "url"
                )

                if url:

                    st.markdown(
                        f"[🔗 Open source]({url})"
                    )


    # =====================================================
    # Missing Evidence
    # =====================================================

    missing = verdict.get(
        "missing_evidence",
        [],
    )


    if missing:

        st.subheader(
            "🔎 Missing Evidence"
        )

        for item in missing:

            st.warning(
                item
            )


    # =====================================================
    # Agent Activity
    # =====================================================

    st.subheader(
        "🤖 Agent Activity"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Evidence",
            metadata.get(
                "evidence_count",
                0,
            ),
        )


    with col2:

        st.metric(
            "Local Evidence",
            metadata.get(
                "local_evidence_count",
                0,
            ),
        )


    with col3:

        st.metric(
            "Web Evidence",
            metadata.get(
                "web_evidence_count",
                0,
            ),
        )


    with col4:

        st.metric(
            "Agent Iterations",
            metadata.get(
                "agent_iterations",
                1,
            ),
        )


    st.caption(
        "🔍 Searches performed: "
        f"{metadata.get('searches_performed', 0)}"
    )


    # =====================================================
    # Pipeline Explanation
    # =====================================================

    with st.expander(
        "🔬 How TruthLens verified this claim"
    ):

        st.write(
            """
            **1. Claim Analysis**

            The submitted statement is analyzed.

            **2. Query Planning**

            Search queries are generated based on
            the claim.

            **3. Hybrid Retrieval**

            Evidence is retrieved from local FAISS
            data and web sources.

            **4. Evidence Filtering**

            Evidence is ranked using semantic similarity.

            **5. Source Quality**

            Sources are evaluated for reliability.

            **6. RAG Reasoning**

            The LLM evaluates the claim using the
            retrieved evidence.

            **7. Confidence Analysis**

            Multiple signals are combined into a
            trust-aware confidence score.

            **8. Agentic Loop**

            If evidence is insufficient, another
            retrieval and reasoning cycle can occur.
            """
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "TruthLens AI • Explainable Agentic Fact-Checking Assistant"
)