from utils.claim_extractor import (
    normalize_text,
    extract_claims,
)


def test_normalize_text():
    text = "   The   Earth    is   round.   "

    result = normalize_text(text)

    assert result == "The Earth is round."


def test_empty_text():
    assert normalize_text("") == ""


def test_extract_single_claim():
    text = "The Earth is round."

    claims = extract_claims(text)

    assert len(claims) == 1
    assert claims[0] == "The Earth is round."


def test_extract_multiple_claims():
    text = (
        "The Earth is round. "
        "Water freezes at 0 degrees Celsius. "
        "The Sun is a star."
    )

    claims = extract_claims(text)

    assert len(claims) == 3


def test_extract_empty_claims():
    assert extract_claims("") == []