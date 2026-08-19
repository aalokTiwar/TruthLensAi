"""
Search query planner for TruthLens AI.

Converts a user claim into multiple focused search queries.

The planner does NOT determine whether a claim is true or false.
Its purpose is to improve evidence retrieval.
"""

from typing import Dict, List


def _clean_query(query: str) -> str:
    """Clean and normalize a search query."""

    return " ".join(query.split()).strip()


def plan_queries(
    claim: str,
    claim_analysis: Dict[str, object] | None = None,
) -> List[str]:
    """
    Generate multiple search queries for a claim.

    Parameters
    ----------
    claim:
        Original user claim.

    claim_analysis:
        Optional output from utils.claim_analyzer.analyze_claim().

    Returns
    -------
    List[str]
        Unique search queries ordered by usefulness.
    """

    if not claim or not claim.strip():
        raise ValueError(
            "Claim cannot be empty."
        )

    claim = _clean_query(claim)

    if claim_analysis is None:
        claim_analysis = {}

    claim_type = str(
        claim_analysis.get(
            "claim_type",
            "FACTUAL",
        )
    )

    search_terms = claim_analysis.get(
        "search_terms",
        [],
    )

    if not isinstance(
        search_terms,
        list,
    ):
        search_terms = []

    queries: List[str] = []

    # -----------------------------------------------------
    # Query 1 — Original claim
    # -----------------------------------------------------

    queries.append(claim)

    # -----------------------------------------------------
    # Query 2 — Important search terms
    # -----------------------------------------------------

    if search_terms:

        terms_query = " ".join(
            str(term)
            for term in search_terms
        )

        queries.append(
            terms_query
        )

    # -----------------------------------------------------
    # Subjective claims
    # -----------------------------------------------------

    if claim_type == "SUBJECTIVE":

        queries.extend(
            [
                f'"{claim}"',
                f"{claim} evidence",
                f"{claim} expert opinion",
            ]
        )

    # -----------------------------------------------------
    # Time-sensitive claims
    # -----------------------------------------------------

    elif claim_type == "TIME_SENSITIVE":

        queries.extend(
            [
                f"{claim} latest evidence",
                f"{claim} official source",
                f"{claim} recent reports",
            ]
        )

    # -----------------------------------------------------
    # Questions
    # -----------------------------------------------------

    elif claim_type == "QUESTION":

        queries.extend(
            [
                f"{claim} reliable source",
                f"{claim} official source",
            ]
        )

    # -----------------------------------------------------
    # Normal factual claims
    # -----------------------------------------------------

    else:

        queries.extend(
            [
                f"{claim} evidence",
                f"{claim} reliable source",
                f"{claim} official source",
            ]
        )

    # -----------------------------------------------------
    # Remove duplicates while preserving order
    # -----------------------------------------------------

    unique_queries: List[str] = []

    seen = set()

    for query in queries:

        cleaned = _clean_query(query)

        if not cleaned:
            continue

        normalized = cleaned.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        unique_queries.append(
            cleaned
        )

    return unique_queries