from uuid import uuid4

from app.consumer import load_fixture_message
from app.domain.events import build_recognition_event
from app.services.presence_service import PresenceService

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


class DummyTrack:
    def __init__(self, person_presence_score: float) -> None:
        self.person_presence_score = person_presence_score


def test_face_detected_unidentified_for_usable_face_fixture():
    service = PresenceService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_example.json")

    decision, face_detection = service.decide(
        track=DummyTrack(person_presence_score=0.9),
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )

    assert decision is not None
    assert decision.event_type == "face_detected_unidentified"
    assert decision.severity == "medium"
    assert decision.confidence >= 0.75
    assert face_detection.detected is True
    assert face_detection.usable is True
    assert decision.payload["face_detection"]["usable"] is True


def test_presence_no_face_for_low_quality_face_fixture():
    service = PresenceService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")

    decision, face_detection = service.decide(
        track=DummyTrack(person_presence_score=0.9),
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )

    assert decision.event_type == "human_presence_no_face"
    assert decision.severity == "low"
    assert decision.confidence == 0.9
    assert face_detection.detected is True
    assert face_detection.usable is False
    assert "face_quality_threshold_failed" in decision.decision_reason
    assert decision.payload["face_detection"]["usable"] is False


def test_fixture_contract_shape():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert "correlation_id" in msg.context
    assert msg.payload.captured_at is not None
    assert msg.payload.external_camera_key == "cam_101"
    assert msg.frame_ref.endswith("face_detectable.jpg")


def test_recognition_event_keeps_canonical_camera_uuid_and_serializes_internal_ids():
    track_id = uuid4()
    subject_id = uuid4()

    event = build_recognition_event(
        event_type="face_detected_unidentified",
        camera_id=CAMERA_ID,
        track_id=track_id,
        subject_id=subject_id,
        severity="medium",
        confidence=0.9,
        decision_reason=["presence_score_threshold_reached"],
        frame_ref="tests/fixtures/images/face_detectable.jpg",
        payload_details={"face_detection": {"usable": True}},
    )

    assert event["context"]["camera_id"] == CAMERA_ID
    assert event["context"]["track_id"] == str(track_id)
    assert event["context"]["subject_id"] == str(subject_id)
    assert event["payload"]["face_detection"]["usable"] is True
