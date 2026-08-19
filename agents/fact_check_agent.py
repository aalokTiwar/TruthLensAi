"""
Agentic fact-checking loop for TruthLens AI.

Pipeline:

1. Retrieve hybrid evidence.
2. Filter evidence using semantic relevance.
3. Reason over relevant evidence.
4. Evaluate verdict and confidence.
5. Retry only when evidence is genuinely insufficient.
6. Refine the search query.
7. Search again.
8. Return the best available verdict.

The agent is designed to avoid unnecessary retries when:
- the verdict is already TRUE or FALSE,
- relevant evidence is available,
- and the evidence provides reasonable support.
"""

from typing import Dict, Any, List

from agents.evidence_agent import EvidenceAgent
from rag.reasoner import RAGReasoner
from retriever.evidence_filter import EvidenceFilter
from utils.schemas import Verdict


class FactCheckAgent:
    """Agentic fact-checking orchestrator."""

    def __init__(
        self,
        evidence_agent: EvidenceAgent,
        reasoner: RAGReasoner,
        max_iterations: int = 2,
        confidence_threshold: float = 0.75,
        evidence_filter: EvidenceFilter | None = None,
    ):
        self.evidence_agent = evidence_agent
        self.reasoner = reasoner

        self.max_iterations = max(
            1,
            int(max_iterations),
        )

        self.confidence_threshold = max(
            0.0,
            min(
                1.0,
                float(confidence_threshold),
            ),
        )

        # Allow dependency injection for tests.
        # Otherwise create the normal semantic evidence filter.
        self.evidence_filter = (
            evidence_filter
            if evidence_filter is not None
            else EvidenceFilter()
        )

    # =====================================================
    # Evidence strength
    # =====================================================

    @staticmethod
    def _calculate_evidence_strength(
        evidence: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate a lightweight evidence-strength score.

        Preference order:

        1. semantic_relevance
        2. relevance_score

        The score represents how strongly the retrieved
        evidence is related to the original claim.
        """

        if not evidence:
            return 0.0

        scores = []

        for item in evidence:

            score = item.get(
                "semantic_relevance"
            )

            if score is None:
                score = item.get(
                    "relevance_score"
                )

            if score is None:
                continue

            try:
                score = float(score)
            except (
                TypeError,
                ValueError,
            ):
                continue

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            scores.append(score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # =====================================================
    # Retry decision
    # =====================================================

    def _should_search_again(
        self,
        verdict: Verdict,
        evidence: List[Dict[str, Any]],
        iteration: int,
    ) -> bool:
        """
        Decide whether another retrieval/reasoning cycle
        is actually necessary.

        Retry when:

        1. We have reached max_iterations -> stop.
        2. There is no relevant evidence -> retry.
        3. Verdict is NOT_ENOUGH_EVIDENCE -> retry.
        4. Evidence exists but is weak and confidence is low.

        A clear TRUE/FALSE verdict with reasonably strong
        evidence is allowed to stop even when confidence
        is slightly below the normal threshold.
        """

        # -------------------------------------------------
        # Never exceed configured iteration limit.
        # -------------------------------------------------

        if iteration >= self.max_iterations:
            return False

        # -------------------------------------------------
        # No relevant evidence.
        # -------------------------------------------------

        if not evidence:
            return True

        # -------------------------------------------------
        # Explicit uncertainty.
        # -------------------------------------------------

        if verdict.label == "NOT_ENOUGH_EVIDENCE":
            return True

        # -------------------------------------------------
        # Calculate evidence strength.
        # -------------------------------------------------

        evidence_strength = (
            self._calculate_evidence_strength(
                evidence
            )
        )

        # -------------------------------------------------
        # Strong evidence + clear verdict.
        #
        # Do not waste another web search merely because
        # confidence is slightly below 0.75.
        # -------------------------------------------------

        if (
            verdict.label in {
                "TRUE",
                "FALSE",
            }
            and evidence_strength >= 0.70
            and verdict.confidence >= 0.60
        ):
            return False

        # -------------------------------------------------
        # Weak evidence + low confidence.
        #
        # More research is useful.
        # -------------------------------------------------

        if (
            verdict.confidence
            < self.confidence_threshold
            and evidence_strength < 0.70
        ):
            return True

        # -------------------------------------------------
        # Very low confidence should trigger another search.
        # -------------------------------------------------

        if verdict.confidence < 0.50:
            return True

        # -------------------------------------------------
        # Otherwise stop.
        # -------------------------------------------------

        return False

    # =====================================================
    # Query refinement
    # =====================================================

    def _refine_query(
        self,
        claim: str,
        verdict: Verdict,
    ) -> str:
        """
        Create a more focused search query
        for the next iteration.
        """

        missing = " ".join(
            str(item)
            for item in verdict.missing_evidence
        ).strip()

        if missing:

            return (
                f"{claim} "
                f"{missing}"
            ).strip()

        return (
            f"{claim} "
            f"evidence facts verification"
        ).strip()

    # =====================================================
    # Evidence filtering
    # =====================================================

    def _filter_evidence(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove evidence that is semantically unrelated
        to the original claim.
        """

        if not evidence:
            return []

        return self.evidence_filter.filter(
            claim,
            evidence,
        )

    # =====================================================
    # Main verification loop
    # =====================================================

    def verify(
        self,
        claim: str,
        top_k: int = 5,
        max_web_results: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute the complete agentic fact-checking process.
        """

        if not claim or not claim.strip():

            raise ValueError(
                "Claim cannot be empty."
            )

        claim = claim.strip()

        current_query = claim

        all_evidence: List[
            Dict[str, Any]
        ] = []

        final_verdict = None

        searches_performed = 0

        filtered_evidence_count = 0

        iteration = 0

        # =================================================
        # Agent loop
        # =================================================

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            # ---------------------------------------------
            # STEP 1 — Hybrid evidence retrieval
            # ---------------------------------------------

            evidence = (
                self.evidence_agent.retrieve(
                    current_query,
                    top_k=top_k,
                    max_web_results=max_web_results,
                )
            )

            searches_performed += 1

            # ---------------------------------------------
            # Keep raw evidence for statistics.
            # ---------------------------------------------

            if evidence:

                all_evidence.extend(
                    evidence
                )

            # ---------------------------------------------
            # STEP 2 — Semantic filtering
            # ---------------------------------------------

            relevant_evidence = (
                self._filter_evidence(
                    claim,
                    evidence,
                )
            )

            filtered_evidence_count += len(
                relevant_evidence
            )

            # ---------------------------------------------
            # STEP 3 — LLM reasoning
            # ---------------------------------------------

            verdict = self.reasoner.reason(
                claim,
                relevant_evidence,
            )

            final_verdict = verdict

            # ---------------------------------------------
            # STEP 4 — Retry decision
            # ---------------------------------------------

            should_continue = (
                self._should_search_again(
                    verdict,
                    relevant_evidence,
                    iteration,
                )
            )

            if not should_continue:
                break

            # ---------------------------------------------
            # STEP 5 — Refine query
            # ---------------------------------------------

            current_query = (
                self._refine_query(
                    claim,
                    verdict,
                )
            )

        # =================================================
        # Safety fallback
        # =================================================

        if final_verdict is None:

            final_verdict = Verdict(
                label="NOT_ENOUGH_EVIDENCE",
                explanation=(
                    "TruthLens could not obtain a "
                    "usable verdict."
                ),
                confidence=0.0,
                evidence=[],
                missing_evidence=[
                    "Additional reliable evidence."
                ],
            )

        # =================================================
        # Final result
        # =================================================

        return {
            "claim": claim,
            "verdict": final_verdict,
            "agent_iterations": iteration,
            "searches_performed": searches_performed,
            "evidence_count": len(
                all_evidence
            ),
            "filtered_evidence_count": (
                filtered_evidence_count
            ),
        }