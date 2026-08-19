from unittest.mock import Mock

from agents.fact_check_agent import FactCheckAgent
from utils.schemas import Verdict


def create_verdict(
    label="FALSE",
    confidence=0.95,
    missing_evidence=None,
):
    return Verdict(
        label=label,
        explanation="The evidence contradicts the claim.",
        confidence=confidence,
        missing_evidence=missing_evidence or [],
    )


def test_agent_stops_when_confidence_is_high():

    evidence_agent = Mock()
    reasoner = Mock()

    evidence_agent.retrieve.return_value = [
        {
            "id": "1",
            "evidence": "No humans have landed on Mars.",
            "source_type": "local",
        }
    ]

    reasoner.reason.return_value = create_verdict(
        label="FALSE",
        confidence=0.95,
    )

    agent = FactCheckAgent(
        evidence_agent=evidence_agent,
        reasoner=reasoner,
        max_iterations=2,
        confidence_threshold=0.75,
    )

    result = agent.verify(
        "Humans landed on Mars in 2025."
    )

    assert result["verdict"].label == "FALSE"

    assert result["agent_iterations"] == 1

    assert result["searches_performed"] == 1

    evidence_agent.retrieve.assert_called_once()

    reasoner.reason.assert_called_once()


def test_agent_retries_when_confidence_is_low():

    evidence_agent = Mock()
    reasoner = Mock()

    evidence_agent.retrieve.return_value = [
        {
            "id": "1",
            "evidence": "Evidence is incomplete.",
            "source_type": "web",
        }
    ]

    reasoner.reason.side_effect = [
        create_verdict(
            label="NOT_ENOUGH_EVIDENCE",
            confidence=0.40,
            missing_evidence=[
                "Need confirmation from an authoritative source."
            ],
        ),
        create_verdict(
            label="FALSE",
            confidence=0.92,
        ),
    ]

    agent = FactCheckAgent(
        evidence_agent=evidence_agent,
        reasoner=reasoner,
        max_iterations=2,
        confidence_threshold=0.75,
    )

    result = agent.verify(
        "Humans landed on Mars in 2025."
    )

    assert result["verdict"].label == "FALSE"

    assert result["agent_iterations"] == 2

    assert result["searches_performed"] == 2

    assert evidence_agent.retrieve.call_count == 2

    assert reasoner.reason.call_count == 2