from agents.search_agent import SearchAgent


def test_empty_query():
    agent = SearchAgent()

    results = agent.search("")

    assert results == []


def test_search_returns_list():
    agent = SearchAgent()

    results = agent.search(
        "Earth is approximately 4.54 billion years old",
        max_results=3,
    )

    assert isinstance(results, list)


def test_fetch_invalid_url():
    agent = SearchAgent()

    result = agent.fetch_page(
        "https://this-domain-does-not-exist-example-12345.com"
    )

    assert result == ""