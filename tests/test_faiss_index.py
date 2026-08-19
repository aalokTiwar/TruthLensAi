from retriever.embedder import Embedder
from retriever.faiss_index import EvidenceIndex


def test_add_documents():
    embedder = Embedder()

    index = EvidenceIndex(embedder)

    documents = [
        {
            "id": "1",
            "claim": "The Sun is a star.",
            "evidence": (
                "The Sun is the star at the center of our Solar System."
            ),
            "label": "TRUE",
            "source": "Astronomy reference",
        },
        {
            "id": "2",
            "claim": "Humans landed on Mars in 2025.",
            "evidence": (
                "No human has landed on Mars. "
                "Mars has only been explored by robotic missions."
            ),
            "label": "FALSE",
            "source": "Space exploration reference",
        },
    ]

    index.add_documents(documents)

    assert index.count() == 2


def test_search_returns_relevant_document():
    embedder = Embedder()

    index = EvidenceIndex(embedder)

    documents = [
        {
            "id": "1",
            "claim": "The Sun is a star.",
            "evidence": (
                "The Sun is the star at the center of our Solar System."
            ),
            "label": "TRUE",
            "source": "Astronomy reference",
        },
        {
            "id": "2",
            "claim": "Humans landed on Mars in 2025.",
            "evidence": (
                "No human has landed on Mars. "
                "Mars has only been explored by robotic missions."
            ),
            "label": "FALSE",
            "source": "Space exploration reference",
        },
    ]

    index.add_documents(documents)

    results = index.search(
        "Have humans landed on Mars?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["id"] == "2"
    assert "relevance_score" in results[0]