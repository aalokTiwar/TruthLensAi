from unittest.mock import Mock

from agents.evidence_agent import EvidenceAgent


def test_retrieve_combines_local_and_web_evidence():

    mock_index = Mock()
    mock_search_agent = Mock()

    mock_index.search.return_value = [
        {
            "id": "local-1",
            "claim": "Humans landed on Mars.",
            "evidence": "No human has landed on Mars.",
            "label": "FALSE",
            "source": "Local dataset",
            "relevance_score": 0.95,
        }
    ]

    mock_search_agent.gather_claim_evidence.return_value = [
        {
            "title": "Mars Exploration",
            "url": "https://example.com/mars",
            "snippet": "No humans have landed on Mars.",
            "text": "Mars has been explored by robotic missions.",
        }
    ]

    agent = EvidenceAgent(
        evidence_index=mock_index,
        search_agent=mock_search_agent,
    )

    results = agent.retrieve(
        "Have humans landed on Mars?",
        top_k=5,
        max_web_results=3,
    )

    assert len(results) == 2

    assert results[0]["source_type"] == "local"
    assert results[1]["source_type"] == "web"

    mock_index.search.assert_called_once_with(
        "Have humans landed on Mars?",
        top_k=5,
    )

    mock_search_agent.gather_claim_evidence.assert_called_once_with(
        "Have humans landed on Mars?",
        max_results_per_query=3,
        max_queries=4,
    )