"""
Source quality scoring for TruthLens AI.

Assigns a simple reliability score to evidence sources.

The score is only a supporting signal. It does NOT determine
whether a claim is true or false by itself.
"""

from typing import Dict, Any
from urllib.parse import urlparse


# ---------------------------------------------------------
# Source categories
# ---------------------------------------------------------

HIGH_TRUST_DOMAINS = {
    "nasa.gov",
    "isro.gov.in",
    "who.int",
    "un.org",
    "gov.in",
    "gov.uk",
    "usa.gov",
    "europa.eu",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "noaa.gov",
}

MEDIUM_HIGH_TRUST_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "nature.com",
    "sciencemag.org",
    "scientificamerican.com",
    "espn.com",
    "cricbuzz.com",
}

LOW_TRUST_INDICATORS = {
    "blogspot.com",
    "wordpress.com",
    "medium.com",
    "substack.com",
}


class SourceQuality:
    """
    Estimate the reliability of an evidence source.

    Scores:

        1.00 -> HIGH
        0.80 -> MEDIUM_HIGH
        0.60 -> MEDIUM
        0.35 -> LOW
    """

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        """
        Normalize a domain name.
        """

        domain = (
            domain
            .lower()
            .strip()
            .removeprefix("www.")
        )

        return domain

    @classmethod
    def _domain_matches(
        cls,
        domain: str,
        trusted_domain: str,
    ) -> bool:
        """
        Check whether a domain matches a trusted domain.

        Example:

            news.nasa.gov

        matches:

            nasa.gov
        """

        domain = cls._normalize_domain(domain)
        trusted_domain = cls._normalize_domain(
            trusted_domain
        )

        return (
            domain == trusted_domain
            or domain.endswith(
                "." + trusted_domain
            )
        )

    @classmethod
    def score(
        cls,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate source quality for an evidence item.

        Returns the original evidence dictionary with:

            source_quality_score
            source_quality
        """

        result = dict(evidence)

        url = str(
            evidence.get(
                "url",
                "",
            )
        ).strip()

        source_type = str(
            evidence.get(
                "source_type",
                "",
            )
        ).lower()

        # -------------------------------------------------
        # Local evidence
        # -------------------------------------------------

        if source_type == "local":

            result[
                "source_quality_score"
            ] = 0.70

            result[
                "source_quality"
            ] = "MEDIUM"

            return result

        # -------------------------------------------------
        # Missing URL
        # -------------------------------------------------

        if not url:

            result[
                "source_quality_score"
            ] = 0.35

            result[
                "source_quality"
            ] = "LOW"

            return result

        # -------------------------------------------------
        # Parse domain
        # -------------------------------------------------

        try:

            parsed = urlparse(url)

            domain = cls._normalize_domain(
                parsed.netloc
            )

        except Exception:

            domain = ""

        # -------------------------------------------------
        # Official / government sources
        # -------------------------------------------------

        for trusted_domain in HIGH_TRUST_DOMAINS:

            if cls._domain_matches(
                domain,
                trusted_domain,
            ):

                result[
                    "source_quality_score"
                ] = 1.00

                result[
                    "source_quality"
                ] = "HIGH"

                return result

        # -------------------------------------------------
        # Established sources
        # -------------------------------------------------

        for trusted_domain in MEDIUM_HIGH_TRUST_DOMAINS:

            if cls._domain_matches(
                domain,
                trusted_domain,
            ):

                result[
                    "source_quality_score"
                ] = 0.85

                result[
                    "source_quality"
                ] = "MEDIUM_HIGH"

                return result

        # -------------------------------------------------
        # Known user-generated publishing platforms
        # -------------------------------------------------

        for low_domain in LOW_TRUST_INDICATORS:

            if cls._domain_matches(
                domain,
                low_domain,
            ):

                result[
                    "source_quality_score"
                ] = 0.35

                result[
                    "source_quality"
                ] = "LOW"

                return result

        # -------------------------------------------------
        # Generic web source
        # -------------------------------------------------

        result[
            "source_quality_score"
        ] = 0.60

        result[
            "source_quality"
        ] = "MEDIUM"

        return result

    @classmethod
    def score_many(
        cls,
        evidence: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """
        Score multiple evidence items.
        """

        return [
            cls.score(item)
            for item in evidence
        ]