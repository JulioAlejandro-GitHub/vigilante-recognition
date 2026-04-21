from app.consumer import load_fixture_message
from app.domain.entities import PresenceDecision
from app.services.presence_service import PresenceService


class DummyTrack:
    def __init__(self, frame_count: int, person_presence_score: float) -> None:
        self.frame_count = frame_count
        self.person_presence_score = person_presence_score


def test_presence_detected_after_threshold():
    service = PresenceService()
    decision = service.decide(DummyTrack(frame_count=3, person_presence_score=0.9))
    assert decision is not None
    assert decision.event_type == "human_presence_detected"
    assert decision.severity == "medium"
    assert decision.confidence == 0.9


def test_presence_no_face_before_threshold():
    service = PresenceService()
    decision = service.decide(DummyTrack(frame_count=1, person_presence_score=0.5))
    assert decision.event_type == "human_presence_no_face"
    assert decision.severity == "low"
    assert decision.confidence == 0.5


def test_fixture_contract_shape():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert "correlation_id" in msg.context
    assert msg.payload.captured_at is not None
