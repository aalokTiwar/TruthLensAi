from retriever.evidence_filter import (
    EvidenceFilter,
)


def test_empty_claim():

    evidence_filter = EvidenceFilter()

    result = evidence_filter.filter(
        "",
        [],
    )

    assert result == []


def test_empty_evidence():

    evidence_filter = EvidenceFilter()

    result = evidence_filter.filter(
        "The Earth revolves around the Sun.",
        [],
    )

    assert result == []


def test_relevance_field_is_added():

    evidence_filter = EvidenceFilter(
        threshold=0.0
    )

    evidence = [
        {
            "title": "Earth and Sun",
            "snippet": (
                "The Earth revolves around the Sun."
            ),
            "text": (
                "The Earth orbits the Sun."
            ),
            "source_type": "local",
        }
    ]

    result = evidence_filter.filter(
        "The Earth revolves around the Sun.",
        evidence,
    )

    assert len(result) == 1

    assert (
        "semantic_relevance"
        in result[0]
    )


def test_irrelevant_evidence_can_be_removed():

    evidence_filter = EvidenceFilter(
        threshold=0.90
    )

    evidence = [
        {
            "title": "Pacific Ocean",
            "snippet": (
                "The Pacific Ocean is the largest ocean."
            ),
            "text": (
                "The Pacific Ocean covers a large "
                "area of the Earth."
            ),
            "source_type": "web",
        }
    ]

    result = evidence_filter.filter(
        "Virat Kohli is an Indian cricketer.",
        evidence,
    )

    assert len(result) == 0