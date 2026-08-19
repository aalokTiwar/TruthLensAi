"""
Web search and evidence extraction agent for TruthLens AI.

Features:
1. DuckDuckGo search through the ddgs package.
2. Fast search-snippet evidence extraction.
3. Optional webpage text extraction through fetch_page().
4. Multi-query search using Query Planner.
5. Duplicate evidence removal.
6. Backward-compatible search methods.

The main evidence pipeline intentionally uses search snippets
instead of downloading every webpage. This significantly
reduces fact-checking latency.
"""

from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from utils.claim_analyzer import analyze_claim
from utils.query_planner import plan_queries


class SearchAgent:
    """Retrieve evidence from public web pages."""

    def __init__(
        self,
        timeout: int = 10,
    ):
        self.timeout = timeout

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            )
        }

    # =====================================================
    # Web search
    # =====================================================

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Search the web using the ddgs package.

        Returns:
            List containing:
            - title
            - url
            - snippet
        """

        if not query or not query.strip():
            return []

        try:

            results = []

            with DDGS() as ddgs:

                search_results = ddgs.text(
                    query.strip(),
                    max_results=max_results,
                )

                for item in search_results:

                    title = str(
                        item.get(
                            "title",
                            "Untitled",
                        )
                    ).strip()

                    url = str(
                        item.get(
                            "href",
                            item.get(
                                "url",
                                "",
                            ),
                        )
                    ).strip()

                    snippet = str(
                        item.get(
                            "body",
                            item.get(
                                "snippet",
                                "",
                            ),
                        )
                    ).strip()

                    if not url:
                        continue

                    results.append(
                        {
                            "title": (
                                title
                                or "Untitled"
                            ),
                            "url": url,
                            "snippet": snippet,
                        }
                    )

            return results

        except Exception:
            return []

    # =====================================================
    # Web page extraction
    # =====================================================

    def fetch_page(
        self,
        url: str,
    ) -> str:
        """
        Download a webpage and extract readable text.

        This method is kept available for cases where
        full webpage extraction is explicitly required.

        It is NOT called for every search result during
        normal evidence gathering, which keeps the system fast.
        """

        if not url:
            return ""

        try:

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException:
            return ""

        except Exception:
            return ""

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Remove elements that usually do not contain
        # useful article information.
        for element in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "noscript",
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        return text[:10000]

    # =====================================================
    # Single-query evidence gathering
    # =====================================================

    def gather_evidence(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the web and create evidence directly
        from search results.

        Search snippets are used immediately as evidence.

        We intentionally DO NOT download every webpage here.
        Downloading multiple webpages can add several seconds
        or even tens of seconds to every fact-check request.

        The search result already provides:

            title
            URL
            snippet

        The snippet is stored as "text" so the downstream
        evidence filter and RAG reasoner can use it.
        """

        if not query or not query.strip():
            return []

        # -------------------------------------------------
        # Search web
        # -------------------------------------------------

        search_results = self.search(
            query,
            max_results=max_results,
        )

        if not search_results:
            return []

        # -------------------------------------------------
        # Convert search results into evidence
        # -------------------------------------------------

        evidence: List[
            Dict[str, Any]
        ] = []

        for result in search_results:

            title = result.get(
                "title",
                "Untitled",
            )

            url = result.get(
                "url",
                "",
            )

            snippet = result.get(
                "snippet",
                "",
            )

            evidence.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "text": snippet,
                    "source_type": "web",
                }
            )

        return evidence

    # =====================================================
    # Duplicate detection
    # =====================================================

    @staticmethod
    def _deduplicate_evidence(
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate evidence.

        URLs are preferred as the unique identifier.
        Titles are used as a fallback.
        """

        unique = []

        seen_urls = set()
        seen_titles = set()

        for item in evidence:

            url = str(
                item.get(
                    "url",
                    "",
                )
            ).strip()

            title = str(
                item.get(
                    "title",
                    "",
                )
            ).strip().lower()

            # ---------------------------------------------
            # URL-based duplicate detection
            # ---------------------------------------------

            if url:

                normalized_url = (
                    url.lower().rstrip("/")
                )

                if normalized_url in seen_urls:
                    continue

                seen_urls.add(
                    normalized_url
                )

            # ---------------------------------------------
            # Title fallback
            # ---------------------------------------------

            elif title:

                if title in seen_titles:
                    continue

                seen_titles.add(
                    title
                )

            unique.append(
                item
            )

        return unique

    # =====================================================
    # Multi-query evidence gathering
    # =====================================================

    def gather_claim_evidence(
        self,
        claim: str,
        max_results_per_query: int = 3,
        max_queries: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a claim, generate targeted search queries,
        and gather evidence from multiple searches.

        This is the main general-purpose retrieval method.
        """

        if not claim or not claim.strip():
            return []

        # -------------------------------------------------
        # Analyze claim
        # -------------------------------------------------

        analysis = analyze_claim(
            claim
        )

        # -------------------------------------------------
        # Generate targeted queries
        # -------------------------------------------------

        queries = plan_queries(
            claim,
            analysis,
        )

        # Fallback to the original claim if the planner
        # does not generate any queries.
        if not queries:

            queries = [
                claim.strip()
            ]

        # Respect configured query limit.
        queries = queries[
            :max_queries
        ]

        # -------------------------------------------------
        # Search all planned queries
        # -------------------------------------------------

        all_evidence: List[
            Dict[str, Any]
        ] = []

        for query in queries:

            evidence = self.gather_evidence(
                query,
                max_results=max_results_per_query,
            )

            for item in evidence:

                # Record the exact query that produced
                # this evidence.
                item["search_query"] = query

                # Preserve claim analysis metadata.
                item["claim_type"] = analysis.get(
                    "claim_type",
                    "unknown",
                )

                item["is_time_sensitive"] = analysis.get(
                    "is_time_sensitive",
                    False,
                )

                all_evidence.append(
                    item
                )

        # -------------------------------------------------
        # Remove duplicate pages
        # -------------------------------------------------

        return self._deduplicate_evidence(
            all_evidence
        )