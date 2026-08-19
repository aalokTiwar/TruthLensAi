from utils.source_quality import SourceQuality


def test_nasa_is_high_trust():

    evidence = {
        "title": "NASA Mars Exploration",
        "url": "https://www.nasa.gov/mars/",
        "source_type": "web",
    }

    result = SourceQuality.score(
        evidence
    )

    assert result["source_quality"] == "HIGH"
    assert result["source_quality_score"] == 1.00


def test_reuters_is_medium_high():

    evidence = {
        "title": "Reuters News",
        "url": "https://www.reuters.com/world/",
        "source_type": "web",
    }

    result = SourceQuality.score(
        evidence
    )

    assert result["source_quality"] == "MEDIUM_HIGH"
    assert result["source_quality_score"] == 0.85


def test_generic_website_is_medium():

    evidence = {
        "title": "Example Website",
        "url": "https://example.com/article",
        "source_type": "web",
    }

    result = SourceQuality.score(
        evidence
    )

    assert result["source_quality"] == "MEDIUM"
    assert result["source_quality_score"] == 0.60


def test_missing_url_is_low():

    evidence = {
        "title": "Unknown Source",
        "source_type": "web",
    }

    result = SourceQuality.score(
        evidence
    )

    assert result["source_quality"] == "LOW"
    assert result["source_quality_score"] == 0.35


def test_local_evidence_is_medium():

    evidence = {
        "title": "Local Dataset",
        "source_type": "local",
        "evidence": "Example evidence.",
    }

    result = SourceQuality.score(
        evidence
    )

    assert result["source_quality"] == "MEDIUM"
    assert result["source_quality_score"] == 0.70


def test_score_many():

    evidence = [
        {
            "title": "NASA",
            "url": "https://nasa.gov",
            "source_type": "web",
        },
        {
            "title": "Example",
            "url": "https://example.com",
            "source_type": "web",
        },
    ]

    results = SourceQuality.score_many(
        evidence
    )

    assert len(results) == 2
    assert results[0]["source_quality"] == "HIGH"
    assert results[1]["source_quality"] == "MEDIUM"