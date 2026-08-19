"""
LLM-based RAG reasoner for TruthLens AI.

The reasoner evaluates a claim using retrieved evidence and returns
a structured, explainable verdict.

Features:
1. Groq LLM-based reasoning.
2. Strict evidence-based fact checking.
3. Structured JSON output.
4. TRUE / FALSE / NOT_ENOUGH_EVIDENCE labels.
5. LLM confidence extraction.
6. Trust-aware confidence calculation.
7. Evidence metadata preservation.
8. LLM evidence selection.
9. Fallback evidence selection when the LLM selects nothing.
10. Safety limit for confidence when evidence is empty.
"""

import json
from typing import List, Dict, Any

from groq import Groq

from utils.config import settings
from utils.schemas import Verdict, Evidence
from utils.confidence import calculate_final_confidence


# =========================================================
# Allowed verdict labels
# =========================================================

ALLOWED_LABELS = {
    "TRUE",
    "FALSE",
    "NOT_ENOUGH_EVIDENCE",
}


# =========================================================
# RAG Reasoner
# =========================================================

class RAGReasoner:
    """
    Use Groq to reason over retrieved evidence.
    """

    def __init__(self):

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured in the .env file."
            )

        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

        self.model = settings.GROQ_MODEL

    # =====================================================
    # Prompt Construction
    # =====================================================

    def _build_prompt(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> str:
        """
        Build the fact-checking prompt.
        """

        evidence_text = []

        for i, item in enumerate(
            evidence,
            start=1,
        ):

            title = (
                item.get("title")
                or item.get("source")
                or "Unknown source"
            )

            url = item.get(
                "url",
                "",
            )

            snippet = item.get(
                "snippet",
                "",
            )

            text = item.get(
                "text",
                "",
            )

            source_type = item.get(
                "source_type",
                "local",
            )

            content = (
                text
                or item.get(
                    "evidence",
                    "",
                )
                or snippet
            )

            semantic_relevance = item.get(
                "semantic_relevance"
            )

            relevance_score = item.get(
                "relevance_score"
            )

            source_quality = item.get(
                "source_quality_score"
            )

            evidence_text.append(
                f"""
Evidence {i}
Source type: {source_type}
Title: {title}
URL: {url}
Semantic relevance: {semantic_relevance}
Relevance score: {relevance_score}
Source quality: {source_quality}
Content: {content[:5000]}
"""
            )

        joined_evidence = "\n".join(
            evidence_text
        )

        return f"""
You are TruthLens AI, an evidence-based fact-checking assistant.

Your task is to evaluate the CLAIM using ONLY the provided evidence.

CLAIM:
{claim}

EVIDENCE:
{joined_evidence}

Rules:

1. Do not use unsupported background knowledge.
2. Do not invent sources or evidence.
3. If the evidence clearly supports the claim, use TRUE.
4. If the evidence clearly contradicts the claim, use FALSE.
5. If the evidence is insufficient or ambiguous, use NOT_ENOUGH_EVIDENCE.
6. Confidence must be between 0 and 1.
7. Confidence represents your confidence in the reasoning based ONLY on the supplied evidence.
8. Explain the verdict using the supplied evidence.
9. List important evidence that is still missing.
10. Select evidence that directly contributes to the verdict.
11. Evidence indices must refer to the Evidence numbers provided above.
12. If evidence supports the claim, select the supporting evidence.
13. If evidence contradicts the claim, select the contradicting evidence.
14. If evidence is insufficient, select the most relevant evidence explaining why.
15. Return ONLY valid JSON.

Required JSON format:

{{
    "label": "TRUE | FALSE | NOT_ENOUGH_EVIDENCE",
    "explanation": "Evidence-based explanation.",
    "confidence": 0.0,
    "missing_evidence": [
        "Important missing evidence"
    ],
    "evidence_indices": [1]
}}
"""

    # =====================================================
    # Evidence conversion helper
    # =====================================================

    @staticmethod
    def _convert_to_evidence(
        item: Dict[str, Any],
    ) -> Evidence:
        """
        Convert an evidence dictionary into the
        Pydantic Evidence model.
        """

        title = (
            item.get("title")
            or item.get("source")
            or "Unknown source"
        )

        snippet = (
            item.get("snippet")
            or item.get("evidence")
            or item.get(
                "text",
                "",
            )
        )

        return Evidence(
            title=title,

            url=item.get(
                "url"
            ),

            snippet=str(
                snippet
            )[:500],

            source_type=item.get(
                "source_type",
                "local",
            ),

            relevance_score=item.get(
                "relevance_score"
            ),

            semantic_relevance=item.get(
                "semantic_relevance"
            ),

            source_quality_score=item.get(
                "source_quality_score"
            ),
        )

    # =====================================================
    # Evidence selection
    # =====================================================

    def _select_evidence(
        self,
        evidence: List[Dict[str, Any]],
        evidence_indices: Any,
    ) -> List[Evidence]:
        """
        Select evidence using LLM-provided indices.

        If the LLM does not provide valid indices,
        automatically select the strongest available
        evidence using semantic relevance/relevance score.
        """

        selected_evidence: List[
            Evidence
        ] = []

        # -------------------------------------------------
        # Validate LLM evidence indices
        # -------------------------------------------------

        valid_indices = []

        if isinstance(
            evidence_indices,
            list,
        ):

            for index in evidence_indices:

                if not isinstance(
                    index,
                    int,
                ):
                    continue

                if (
                    index < 1
                    or index > len(evidence)
                ):
                    continue

                if index not in valid_indices:
                    valid_indices.append(
                        index
                    )

        # -------------------------------------------------
        # Use LLM-selected evidence
        # -------------------------------------------------

        for index in valid_indices:

            item = evidence[
                index - 1
            ]

            selected_evidence.append(
                self._convert_to_evidence(
                    item
                )
            )

        # -------------------------------------------------
        # FALLBACK
        #
        # If Groq selected no valid evidence,
        # preserve the strongest retrieved evidence.
        # -------------------------------------------------

        if (
            not selected_evidence
            and evidence
        ):

            def evidence_score(
                item: Dict[str, Any],
            ) -> float:

                semantic = item.get(
                    "semantic_relevance"
                )

                relevance = item.get(
                    "relevance_score"
                )

                try:

                    if semantic is not None:
                        return float(
                            semantic
                        )

                    if relevance is not None:
                        return float(
                            relevance
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

                return 0.0

            ranked_evidence = sorted(
                evidence,
                key=evidence_score,
                reverse=True,
            )

            # Keep a small number of strong
            # evidence items for the final response.
            for item in ranked_evidence[:5]:

                selected_evidence.append(
                    self._convert_to_evidence(
                        item
                    )
                )

        return selected_evidence

    # =====================================================
    # Reason
    # =====================================================

    def reason(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> Verdict:
        """
        Generate an evidence-based verdict.

        The LLM produces the initial confidence.

        TruthLens then calculates final trust-aware
        confidence using evidence quality signals.
        """

        # -------------------------------------------------
        # Validate claim
        # -------------------------------------------------

        if not claim or not claim.strip():

            raise ValueError(
                "Claim cannot be empty."
            )

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------

        prompt = self._build_prompt(
            claim,
            evidence,
        )

        # -------------------------------------------------
        # Call Groq
        # -------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise fact-checking "
                        "system. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,

            response_format={
                "type": "json_object"
            },
        )

        # -------------------------------------------------
        # Extract response
        # -------------------------------------------------

        raw_content = (
            response
            .choices[0]
            .message
            .content
        )

        if not raw_content:

            raise ValueError(
                "Groq returned an empty response."
            )

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        try:

            result = json.loads(
                raw_content
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Groq returned invalid JSON."
            ) from exc

        # -------------------------------------------------
        # Validate label
        # -------------------------------------------------

        label = str(
            result.get(
                "label",
                "NOT_ENOUGH_EVIDENCE",
            )
        ).upper().strip()

        if label not in ALLOWED_LABELS:

            label = (
                "NOT_ENOUGH_EVIDENCE"
            )

        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------

        explanation = str(
            result.get(
                "explanation",
                "No explanation was provided.",
            )
        ).strip()

        if not explanation:

            explanation = (
                "No explanation was provided."
            )

        # -------------------------------------------------
        # LLM confidence
        # -------------------------------------------------

        try:

            llm_confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
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

        # -------------------------------------------------
        # Missing evidence
        # -------------------------------------------------

        missing_evidence = result.get(
            "missing_evidence",
            [],
        )

        if not isinstance(
            missing_evidence,
            list,
        ):

            missing_evidence = []

        missing_evidence = [
            str(item)
            for item in missing_evidence
        ]

        # -------------------------------------------------
        # Evidence indices
        # -------------------------------------------------

        evidence_indices = result.get(
            "evidence_indices",
            [],
        )

        if not isinstance(
            evidence_indices,
            list,
        ):

            evidence_indices = []

        # -------------------------------------------------
        # Select final evidence
        # -------------------------------------------------

        selected_evidence = (
            self._select_evidence(
                evidence,
                evidence_indices,
            )
        )

        # -------------------------------------------------
        # Calculate trust-aware confidence
        # -------------------------------------------------

        final_confidence = (
            calculate_final_confidence(
                llm_confidence=llm_confidence,
                evidence=evidence,
            )
        )

        # -------------------------------------------------
        # Safety rule
        #
        # Never allow an LLM to produce artificially
        # high confidence when there is no evidence.
        # -------------------------------------------------

        if not evidence:

            final_confidence = min(
                final_confidence,
                0.25,
            )

        # -------------------------------------------------
        # Additional safety rule
        #
        # If the final verdict contains no selected
        # evidence, confidence should not be extremely high.
        #
        # This protects against cases where the LLM
        # forgets to return evidence_indices.
        # -------------------------------------------------

        if (
            evidence
            and not selected_evidence
        ):

            final_confidence = min(
                final_confidence,
                0.50,
            )

        # -------------------------------------------------
        # Return structured verdict
        # -------------------------------------------------

        return Verdict(
            label=label,

            explanation=explanation,

            confidence=final_confidence,

            evidence=selected_evidence,

            missing_evidence=missing_evidence,
        )