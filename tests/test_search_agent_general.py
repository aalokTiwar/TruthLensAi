from agents.search_agent import SearchAgent


def test_empty_claim_evidence():

    agent = SearchAgent()

    result = agent.gather_claim_evidence(
        ""
    )

    assert result == []


def test_deduplicate_evidence():

    evidence = [
        {
            "title": "Example",
            "url": "https://example.com",
            "snippet": "First",
        },
        {
            "title": "Example duplicate",
            "url": "https://example.com",
            "snippet": "Second",
        },
    ]

    result = SearchAgent._deduplicate_evidence(
        evidence
    )

    assert len(result) == 1


def test_search_agent_has_general_method():

    agent = SearchAgent()

    assert hasattr(
        agent,
        "gather_claim_evidence",
    )