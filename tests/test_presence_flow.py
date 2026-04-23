from uuid import uuid4

from app.consumer import load_fixture_message
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult, FaceMatchCandidate, FaceMatchResult
from app.domain.events import build_recognition_event
from app.services.presence_service import PresenceService

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


class DummyTrack:
    def __init__(self, person_presence_score: float) -> None:
        self.person_presence_score = person_presence_score


def test_face_detected_unidentified_for_usable_face_fixture():
    service = PresenceService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_example.json")

    face_detection = service.inspect_face(
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )
    decision = service.decide(track=DummyTrack(person_presence_score=0.9), face_detection=face_detection)

    assert decision is not None
    assert decision.event_type == "face_detected_unidentified"
    assert decision.severity == "medium"
    assert decision.confidence >= 0.75
    assert face_detection.detected is True
    assert face_detection.usable is True
    assert decision.payload["face_detection"]["usable"] is True
    assert decision.payload["identified"] is False


def test_face_detected_identified_when_match_is_confident():
    service = PresenceService()
    face_detection = FaceDetectionResult(
        detected=True,
        usable=True,
        quality_score=0.95,
        bbox={"x": 10, "y": 20, "width": 100, "height": 120},
    )
    embedding_result = FaceEmbeddingResult(
        generated=True,
        backend="simple_face_crop_512",
        dimensions=512,
        vector=[0.1, 0.2],
    )
    match_result = FaceMatchResult(
        identified=True,
        match_confidence=0.97,
        matching_strategy="cosine_similarity:simple_face_crop_512",
        threshold=0.82,
        second_best_margin_threshold=0.05,
        evaluated_candidates=2,
        gallery_source="local_dev_fixture",
        best_match=FaceMatchCandidate(
            person_profile_id="22222222-2222-2222-2222-222222222222",
            full_name="Barack Obama (dev fixture)",
            person_type="employee",
            risk_level="low",
            external_person_key="dev_obama_001",
            similarity=0.97,
            gallery_source="local_dev_fixture",
        ),
        best_similarity=0.97,
        second_best_similarity=0.21,
        second_best_margin=0.76,
    )

    decision = service.decide(
        track=DummyTrack(person_presence_score=0.9),
        face_detection=face_detection,
        embedding_result=embedding_result,
        match_result=match_result,
    )

    assert decision.event_type == "face_detected_identified"
    assert decision.confidence == 0.97
    assert decision.payload["identified"] is True
    assert decision.payload["person_profile_id"] == "22222222-2222-2222-2222-222222222222"
    assert decision.payload["matched_person"]["full_name"] == "Barack Obama (dev fixture)"


def test_presence_no_face_for_low_quality_face_fixture():
    service = PresenceService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")

    face_detection = service.inspect_face(
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )
    decision = service.decide(track=DummyTrack(person_presence_score=0.9), face_detection=face_detection)

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

    identified_msg = load_fixture_message("tests/fixtures/frame_ingested_identified.json")
    assert identified_msg.frame_ref.endswith("face_identified.jpg")

    cross_camera_msg = load_fixture_message("tests/fixtures/frame_cross_camera_positive.json")
    assert cross_camera_msg.camera_id == "22222222-1111-1111-1111-111111111111"

    conflict_msg = load_fixture_message("tests/fixtures/frame_identity_conflict.json")
    assert conflict_msg.context["dev_known_face_gallery_path"].endswith("dev_known_face_gallery_conflict.json")

    manual_review_msg = load_fixture_message("tests/fixtures/frame_manual_review_required.json")
    assert manual_review_msg.context["dev_known_face_gallery_path"].endswith("dev_known_face_gallery_obama_only.json")


def test_recognition_event_keeps_canonical_camera_uuid_and_serializes_internal_ids():
    track_id = uuid4()
    subject_id = uuid4()

    event = build_recognition_event(
        event_type="face_detected_identified",
        camera_id=CAMERA_ID,
        track_id=track_id,
        subject_id=subject_id,
        severity="medium",
        confidence=0.9,
        decision_reason=["presence_score_threshold_reached"],
        frame_ref="tests/fixtures/images/face_identified.jpg",
        payload_details={
            "face_detection": {"usable": True},
            "identified": True,
            "person_profile_id": "22222222-2222-2222-2222-222222222222",
        },
    )

    assert event["context"]["camera_id"] == CAMERA_ID
    assert event["context"]["track_id"] == str(track_id)
    assert event["context"]["subject_id"] == str(subject_id)
    assert event["payload"]["face_detection"]["usable"] is True
    assert event["payload"]["person_profile_id"] == "22222222-2222-2222-2222-222222222222"


def test_recognition_event_can_carry_manual_review_payload():
    track_id = uuid4()
    subject_id = uuid4()

    event = build_recognition_event(
        event_type="manual_review_required",
        camera_id=CAMERA_ID,
        track_id=track_id,
        subject_id=subject_id,
        severity="medium",
        confidence=0.42,
        decision_reason=["cross_camera_candidate_needs_review"],
        frame_ref="tests/fixtures/images/gallery_known_biden.jpg",
        payload_details={
            "requires_human_review": True,
            "review_type": "cross_camera_correlation",
        },
    )

    assert event["event_type"] == "manual_review_required"
    assert event["payload"]["requires_human_review"] is True
    assert event["payload"]["review_type"] == "cross_camera_correlation"
