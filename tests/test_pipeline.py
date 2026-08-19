from utils.pipeline import build_pipeline


def test_pipeline_builds():

    agent = build_pipeline()

    assert agent is not None
    assert agent.evidence_agent is not None
    assert agent.reasoner is not None