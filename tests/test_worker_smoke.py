from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch

import pytest

from app.consumer import load_fixture_message
from app.domain.entities import (
    ContinuityResolution,
    CrossCameraAssessment,
    CrossCameraCandidate,
    FrameIngestedMessage,
    InvalidCameraIdError,
    RecurrentSubjectAssessment,
    RecurrentSubjectCandidate,
    RecurrentSubjectResolution,
    SupplementalRecognitionDecision,
)
from app.models import CrossCameraCorrelation, EventOutbox, HumanTrack, RecognitionEvent
from app.worker import process_fixture

CAMERA_ID = "11111111-1111-1111-1111-111111111111"
CAMERA_B_ID = "22222222-1111-1111-1111-111111111111"


def _make_recognition_event(event_id=None):
    recognition_event = MagicMock()
    recognition_event.recognition_event_id = event_id or uuid4()
    return recognition_event


def _configure_repo_for_slice3_flow(mock_repo_instance, *, subject, track, recognition_event, existing_track: bool = False):
    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    if existing_track:
        mock_repo_instance.find_track_by_camera_and_external_key.return_value = track
        mock_repo_instance.get_subject.return_value = subject
    else:
        mock_repo_instance.find_track_by_camera_and_external_key.return_value = None
        mock_repo_instance.create_subject.return_value = subject
        mock_repo_instance.create_track.return_value = track

    mock_repo_instance.add_recognition_event.return_value = recognition_event
    mock_repo_instance.update_track_presence.side_effect = update_track_presence
    mock_repo_instance.update_track_face_observation.return_value = track
    mock_repo_instance.update_track_match_result.return_value = track
    mock_repo_instance.update_subject_face_profile.return_value = subject
    mock_repo_instance.update_track_semantic_descriptor.return_value = track
    mock_repo_instance.load_known_face_gallery_entries.return_value = []
    mock_repo_instance.load_recent_subject_candidates.return_value = []
    mock_repo_instance.load_recent_unresolved_subject_candidates.return_value = []
    mock_repo_instance.find_latest_cross_camera_correlation_for_source_track.return_value = None
    mock_repo_instance.mark_subject_continuity.return_value = subject
    mock_repo_instance.add_cross_camera_correlation.return_value = MagicMock(spec=CrossCameraCorrelation)
    mock_repo_instance.attach_subject_to_track.return_value = track
    mock_repo_instance.touch_subject.return_value = subject
    mock_repo_instance.update_track_continuity_resolution.return_value = track
    mock_repo_instance.update_track_recurrent_resolution.return_value = track


def test_fixture_loads():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert msg.event_type == "frame.ingested"
    assert msg.camera_id == CAMERA_ID
    assert msg.payload.external_camera_key == "cam_101"
    assert msg.frame_ref.endswith("face_detectable.jpg")

    identified = load_fixture_message("tests/fixtures/frame_ingested_identified.json")
    assert identified.frame_ref.endswith("face_identified.jpg")

    cross_camera = load_fixture_message("tests/fixtures/frame_cross_camera_positive.json")
    assert cross_camera.camera_id == CAMERA_B_ID


def test_orm_alignment_for_uuid_and_frame_count_dependency():
    assert RecognitionEvent.__table__.c.camera_id.type.as_uuid is True
    assert HumanTrack.__table__.c.camera_id.type.as_uuid is True
    assert EventOutbox.__table__.c.aggregate_id.type.as_uuid is True
    assert CrossCameraCorrelation.__table__.c.source_subject_id.type.as_uuid is True
    assert "frame_count" not in HumanTrack.__table__.c


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_smoke(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    resolved_camera_id = UUID(CAMERA_ID)
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = resolved_camera_id
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
    )
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event is not None
    assert event["event_type"] == "face_detected_unidentified"
    assert event["context"]["camera_id"] == CAMERA_ID
    assert event["context"]["track_id"] == str(track_id)
    assert event["context"]["subject_id"] == str(subject_id)
    assert event["payload"]["face_detection"]["usable"] is True
    assert event["payload"]["identified"] is False
    assert event["payload"]["match_confidence"] < 0.82
    assert event["payload"]["semantic_descriptor"]["backend"] == "simple_color_signature_v1"
    assert event["payload"]["semantic_descriptor"]["descriptor_backend"] == "simple_color_signature_v1"

    mock_repo_instance.create_subject.assert_called_once()
    assert mock_repo_instance.create_subject.call_args.kwargs["camera_id"] == resolved_camera_id
    assert mock_repo_instance.create_track.call_args.kwargs["camera_id"] == resolved_camera_id

    recognition_event_call = mock_repo_instance.add_recognition_event.call_args.kwargs
    assert recognition_event_call["camera_id"] == resolved_camera_id
    assert "human_track_confirmed" in recognition_event_call["decision_reason"]
    assert "embedding_generated" in recognition_event_call["decision_reason"]
    assert "match_below_threshold" in recognition_event_call["decision_reason"]
    assert recognition_event_call["evidence_refs"] == ["tests/fixtures/images/face_detectable.jpg"]
    assert recognition_event_call["payload"]["face_detection"]["usable"] is True
    assert recognition_event_call["payload"]["identified"] is False
    assert recognition_event_call["payload"]["semantic_descriptor"]["backend"] == "simple_color_signature_v1"
    assert recognition_event_call["payload"]["semantic_descriptor"]["descriptor_backend"] == "simple_color_signature_v1"

    outbox_call = mock_repo_instance.add_outbox_event.call_args.kwargs
    assert outbox_call["aggregate_id"] == recognition_event_id
    assert outbox_call["event_type"] == "face_detected_unidentified"
    assert outbox_call["payload"]["payload"]["face_detection"]["usable"] is True
    assert outbox_call["payload"]["payload"]["identified"] is False

    mock_repo_instance.update_subject_face_profile.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_reuses_existing_track(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    resolved_camera_id = UUID(CAMERA_ID)
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = resolved_camera_id
    track.observed_subject_id = subject_id
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
        existing_track=True,
    )
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event["context"]["camera_id"] == CAMERA_ID
    assert event["event_type"] == "face_detected_unidentified"
    mock_repo_instance.find_track_by_camera_and_external_key.assert_called_once()
    mock_repo_instance.get_subject.assert_called_once_with(subject_id)
    mock_repo_instance.create_track.assert_not_called()
    mock_repo_instance.create_subject.assert_not_called()
    mock_repo_instance.touch_subject.assert_called_once()
    mock_repo_instance.update_track_face_observation.assert_called_once()
    mock_repo_instance.update_track_match_result.assert_called_once()
    mock_repo_instance.add_cross_camera_correlation.assert_not_called()
    mock_repo_instance.mark_subject_continuity.assert_not_called()
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["aggregate_id"] == recognition_event_id
    mock_session.commit.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_existing_identified_track_skips_continuity_re_evaluation(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    resolved_camera_id = UUID(CAMERA_ID)
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = resolved_camera_id
    track.observed_subject_id = subject_id
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
        existing_track=True,
    )
    mock_repo_class.return_value = mock_repo_instance

    with patch("app.worker.CrossCameraCorrelationService") as mock_cross_service_class, patch(
        "app.worker.ConflictResolutionService"
    ) as mock_conflict_service_class:
        event = process_fixture("tests/fixtures/frame_ingested_identified.json")

    assert event["event_type"] == "face_detected_identified"
    assert event["payload"]["identified"] is True
    assert "continuity_status" not in event["payload"]
    mock_cross_service_class.return_value.evaluate.assert_not_called()
    mock_conflict_service_class.return_value.resolve.assert_not_called()
    mock_repo_instance.add_cross_camera_correlation.assert_not_called()
    mock_repo_instance.mark_subject_continuity.assert_not_called()
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["aggregate_id"] == recognition_event_id


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_existing_cross_camera_track_reuses_persisted_continuity(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()
    target_subject_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID(CAMERA_B_ID)
    track.observed_subject_id = subject_id
    track.person_presence_score = 0.0
    track.track_metadata = {}

    fixture = load_fixture_message("tests/fixtures/frame_cross_camera_positive.json")
    correlation = MagicMock()
    correlation.correlation_status = "auto"
    correlation.aggregate_score = 0.985
    correlation.target_subject_id = target_subject_id
    correlation.target_track_id = uuid4()
    correlation.created_at = fixture.captured_at
    correlation.signals_json = {
        "source_subject_id": str(subject_id),
        "target_subject_id": str(target_subject_id),
        "cross_camera_assessment": {
            "current_subject_id": str(subject_id),
            "current_track_id": str(track_id),
            "current_camera_id": CAMERA_B_ID,
            "threshold": 0.85,
            "manual_review_threshold": 0.35,
            "second_best_margin_threshold": 0.05,
            "evaluated_candidates": 1,
            "decision_reason": ["cross_camera_candidate_above_threshold"],
        },
    }

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
        existing_track=True,
    )
    mock_repo_instance.find_latest_cross_camera_correlation_for_source_track.return_value = correlation
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(recognition_event_id),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    with patch("app.worker.CrossCameraCorrelationService") as mock_cross_service_class, patch(
        "app.worker.ConflictResolutionService"
    ) as mock_conflict_service_class:
        event = process_fixture("tests/fixtures/frame_cross_camera_positive.json")

    assert event["event_type"] == "face_detected_identified"
    assert event["payload"]["continuity_status"] == "correlated"
    mock_cross_service_class.return_value.evaluate.assert_not_called()
    mock_conflict_service_class.return_value.resolve.assert_not_called()
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "face_detected_identified",
        "cross_camera_subject_correlated",
    ]
    mock_repo_instance.add_cross_camera_correlation.assert_not_called()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_low_quality_face_persists_no_face_event(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    resolved_camera_id = UUID(CAMERA_ID)
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = resolved_camera_id
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
    )
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_no_face.json")

    assert event["event_type"] == "human_presence_no_face"
    assert event["payload"]["face_detection"]["detected"] is True
    assert event["payload"]["face_detection"]["usable"] is False
    assert event["payload"]["semantic_descriptor"]["backend"] == "simple_color_signature_v1"
    assert event["payload"]["semantic_descriptor"]["descriptor_backend"] == "simple_color_signature_v1"
    assert mock_repo_instance.add_recognition_event.call_args.kwargs["payload"]["face_detection"]["usable"] is False
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["payload"]["event_type"] == "human_presence_no_face"
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["aggregate_id"] == recognition_event_id
    mock_repo_instance.update_track_match_result.assert_not_called()
    mock_repo_instance.update_subject_face_profile.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_identified_face_persists_identified_event(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    resolved_camera_id = UUID(CAMERA_ID)
    subject_id = uuid4()
    track_id = uuid4()
    recognition_event_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = resolved_camera_id
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(recognition_event_id),
    )
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_identified.json")

    assert event["event_type"] == "face_detected_identified"
    assert event["payload"]["identified"] is True
    assert event["payload"]["person_profile_id"] == "22222222-2222-2222-2222-222222222222"
    assert event["payload"]["matching_strategy"] == "cosine_similarity:simple_face_crop_512"
    assert mock_repo_instance.add_recognition_event.call_args.kwargs["payload"]["identified"] is True
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["event_type"] == "face_detected_identified"
    mock_repo_instance.update_track_match_result.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_cross_camera_positive_persists_correlation_event(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    current_subject_id = uuid4()
    target_subject_id = uuid4()
    track_id = uuid4()
    target_track_id = uuid4()

    current_subject = MagicMock()
    current_subject.observed_subject_id = current_subject_id
    target_subject = MagicMock()
    target_subject.observed_subject_id = target_subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID(CAMERA_B_ID)
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=current_subject,
        track=track,
        recognition_event=_make_recognition_event(),
    )
    mock_repo_instance.get_subject.return_value = target_subject
    mock_repo_instance.touch_subject.return_value = target_subject
    mock_repo_instance.update_subject_face_profile.return_value = target_subject
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    fixture = load_fixture_message("tests/fixtures/frame_cross_camera_positive.json")
    assessment = CrossCameraAssessment(
        current_subject_id=str(current_subject_id),
        current_track_id=str(track_id),
        current_camera_id=fixture.camera_id,
        threshold=0.85,
        manual_review_threshold=0.35,
        second_best_margin_threshold=0.05,
        evaluated_candidates=1,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(target_subject_id),
            latest_track_id=str(target_track_id),
            last_camera_id=CAMERA_ID,
            last_seen_at=fixture.captured_at,
            recurrence_count=2,
            face_similarity_score=1.0,
            temporal_coherence_score=0.9,
            camera_switch_score=1.0,
            aggregate_score=0.985,
            resolved_identity={"person_profile_id": "22222222-2222-2222-2222-222222222222"},
        ),
        decision_reason=["cross_camera_candidate_above_threshold"],
    )
    resolution = ContinuityResolution(
        outcome="correlated",
        subject_id_to_use=str(target_subject_id),
        target_track_id=str(target_track_id),
        correlation_status="auto",
        decision_reason=["cross_camera_auto_resolved"],
        payload={
            "source_subject_id": str(current_subject_id),
            "target_subject_id": str(target_subject_id),
            "cross_camera_assessment": {"aggregate_score": 0.985},
        },
        assessment=assessment,
        supplemental_decisions=[
            SupplementalRecognitionDecision(
                event_type="cross_camera_subject_correlated",
                severity="medium",
                confidence=0.985,
                decision_reason=["cross_camera_auto_resolved"],
                payload={"target_subject_id": str(target_subject_id)},
                subject_id=str(target_subject_id),
            )
        ],
    )

    with patch("app.worker.CrossCameraCorrelationService") as mock_cross_service_class, patch(
        "app.worker.ConflictResolutionService"
    ) as mock_conflict_service_class:
        mock_cross_service_class.return_value.evaluate.return_value = assessment
        mock_conflict_service_class.return_value.resolve.return_value = resolution

        event = process_fixture("tests/fixtures/frame_cross_camera_positive.json")

    assert event["event_type"] == "face_detected_identified"
    assert event["context"]["subject_id"] == str(target_subject_id)
    assert event["payload"]["continuity_status"] == "correlated"
    mock_repo_instance.add_cross_camera_correlation.assert_called_once()
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "face_detected_identified",
        "cross_camera_subject_correlated",
    ]
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_outbox_event.call_args_list] == [
        "face_detected_identified",
        "cross_camera_subject_correlated",
    ]


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_identity_conflict_emits_conflict_and_manual_review(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    subject_id = uuid4()
    candidate_subject_id = uuid4()
    candidate_track_id = uuid4()
    track_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID("33333333-1111-1111-1111-111111111111")
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(),
    )
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(),
        _make_recognition_event(),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    fixture = load_fixture_message("tests/fixtures/frame_identity_conflict.json")
    assessment = CrossCameraAssessment(
        current_subject_id=str(subject_id),
        current_track_id=str(track_id),
        current_camera_id=fixture.camera_id,
        threshold=0.85,
        manual_review_threshold=0.35,
        second_best_margin_threshold=0.05,
        evaluated_candidates=1,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(candidate_subject_id),
            latest_track_id=str(candidate_track_id),
            last_camera_id=CAMERA_ID,
            last_seen_at=fixture.captured_at,
            recurrence_count=2,
            face_similarity_score=1.0,
            temporal_coherence_score=0.9,
            camera_switch_score=1.0,
            aggregate_score=1.0,
            resolved_identity={"person_profile_id": "22222222-2222-2222-2222-222222222222"},
        ),
        decision_reason=["cross_camera_candidate_above_threshold"],
    )
    resolution = ContinuityResolution(
        outcome="identity_conflict",
        subject_id_to_use=str(subject_id),
        correlation_status="pending_review",
        requires_human_review=True,
        decision_reason=["identity_signals_incompatible"],
        payload={
            "requires_human_review": True,
            "cross_camera_assessment": {"aggregate_score": 1.0},
        },
        assessment=assessment,
        supplemental_decisions=[
            SupplementalRecognitionDecision(
                event_type="identity_conflict",
                severity="high",
                confidence=1.0,
                decision_reason=["identity_signals_incompatible"],
                payload={"requires_human_review": True},
                subject_id=str(subject_id),
            ),
            SupplementalRecognitionDecision(
                event_type="manual_review_required",
                severity="medium",
                confidence=1.0,
                decision_reason=["identity_conflict_detected"],
                payload={"requires_human_review": True},
                subject_id=str(subject_id),
            ),
        ],
    )

    with patch("app.worker.CrossCameraCorrelationService") as mock_cross_service_class, patch(
        "app.worker.ConflictResolutionService"
    ) as mock_conflict_service_class:
        mock_cross_service_class.return_value.evaluate.return_value = assessment
        mock_conflict_service_class.return_value.resolve.return_value = resolution

        event = process_fixture("tests/fixtures/frame_identity_conflict.json")

    assert event["event_type"] == "face_detected_identified"
    assert event["payload"]["requires_human_review"] is True
    assert event["payload"]["continuity_status"] == "identity_conflict"
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "face_detected_identified",
        "identity_conflict",
        "manual_review_required",
    ]
    mock_repo_instance.add_cross_camera_correlation.assert_called_once()
    mock_repo_instance.mark_subject_continuity.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_manual_review_emits_review_event(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    subject_id = uuid4()
    candidate_subject_id = uuid4()
    candidate_track_id = uuid4()
    track_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID("44444444-1111-1111-1111-111111111111")
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=subject,
        track=track,
        recognition_event=_make_recognition_event(),
    )
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    fixture = load_fixture_message("tests/fixtures/frame_manual_review_required.json")
    assessment = CrossCameraAssessment(
        current_subject_id=str(subject_id),
        current_track_id=str(track_id),
        current_camera_id=fixture.camera_id,
        threshold=0.85,
        manual_review_threshold=0.35,
        second_best_margin_threshold=0.05,
        evaluated_candidates=2,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(candidate_subject_id),
            latest_track_id=str(candidate_track_id),
            last_camera_id=CAMERA_ID,
            last_seen_at=fixture.captured_at,
            recurrence_count=1,
            face_similarity_score=0.2673,
            temporal_coherence_score=0.75,
            camera_switch_score=1.0,
            aggregate_score=0.4125,
            resolved_identity={},
        ),
        second_best_margin=0.01,
        decision_reason=["cross_camera_candidate_needs_review"],
    )
    resolution = ContinuityResolution(
        outcome="manual_review_required",
        subject_id_to_use=str(subject_id),
        correlation_status="pending_review",
        requires_human_review=True,
        decision_reason=["cross_camera_candidate_needs_review"],
        payload={
            "requires_human_review": True,
            "cross_camera_assessment": {"aggregate_score": 0.4125},
        },
        assessment=assessment,
        supplemental_decisions=[
            SupplementalRecognitionDecision(
                event_type="manual_review_required",
                severity="medium",
                confidence=0.4125,
                decision_reason=["cross_camera_candidate_needs_review"],
                payload={"requires_human_review": True},
                subject_id=str(subject_id),
            )
        ],
    )

    with patch("app.worker.CrossCameraCorrelationService") as mock_cross_service_class, patch(
        "app.worker.ConflictResolutionService"
    ) as mock_conflict_service_class:
        mock_cross_service_class.return_value.evaluate.return_value = assessment
        mock_conflict_service_class.return_value.resolve.return_value = resolution

        event = process_fixture("tests/fixtures/frame_manual_review_required.json")

    assert event["event_type"] == "face_detected_unidentified"
    assert event["payload"]["requires_human_review"] is True
    assert event["payload"]["continuity_status"] == "manual_review_required"
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "face_detected_unidentified",
        "manual_review_required",
    ]
    mock_repo_instance.add_cross_camera_correlation.assert_called_once()
    mock_repo_instance.mark_subject_continuity.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_recurrent_unresolved_emits_recurrence_and_review(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    current_subject_id = uuid4()
    target_subject_id = uuid4()
    track_id = uuid4()
    target_track_id = uuid4()

    current_subject = MagicMock()
    current_subject.observed_subject_id = current_subject_id
    target_subject = MagicMock()
    target_subject.observed_subject_id = target_subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID("55555555-1111-1111-1111-111111111111")
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=current_subject,
        track=track,
        recognition_event=_make_recognition_event(),
    )
    mock_repo_instance.get_subject.return_value = target_subject
    mock_repo_instance.touch_subject.return_value = target_subject
    mock_repo_instance.update_subject_face_profile.return_value = target_subject
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(),
        _make_recognition_event(),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    fixture = load_fixture_message("tests/fixtures/frame_recurrent_unresolved.json")
    assessment = RecurrentSubjectAssessment(
        current_subject_id=str(current_subject_id),
        current_track_id=str(track_id),
        current_camera_id=fixture.camera_id,
        semantic_similarity_threshold=0.72,
        recurrent_subject_threshold=0.78,
        case_suggestion_threshold=0.9,
        manual_review_threshold=0.35,
        evaluated_candidates=1,
        best_candidate=RecurrentSubjectCandidate(
            observed_subject_id=str(target_subject_id),
            latest_track_id=str(target_track_id),
            last_camera_id=CAMERA_ID,
            last_seen_at=fixture.captured_at,
            recurrence_count=1,
            semantic_similarity_score=1.0,
            visual_similarity_score=0.0,
            temporal_coherence_score=0.8,
            camera_relation_score=1.0,
            aggregate_score=0.96,
            descriptor_summary={"dominant_palette": ["gray", "blue"]},
        ),
        decision_reason=["unresolved_recurrence_threshold_passed"],
    )
    resolution = RecurrentSubjectResolution(
        outcome="recurrent_unresolved",
        subject_id_to_use=str(target_subject_id),
        target_track_id=str(target_track_id),
        requires_human_review=True,
        decision_reason=["semantic_subject_recurrence_detected"],
        payload={"evidence_count": 2},
        assessment=assessment,
        supplemental_decisions=[
            SupplementalRecognitionDecision(
                event_type="recurrent_unresolved_subject",
                severity="medium",
                confidence=0.96,
                decision_reason=["semantic_subject_recurrence_detected"],
                payload={"evidence_count": 2},
                subject_id=str(target_subject_id),
            ),
            SupplementalRecognitionDecision(
                event_type="manual_review_required",
                severity="medium",
                confidence=0.96,
                decision_reason=["recurrent_unresolved_subject_detected"],
                payload={"requires_human_review": True},
                subject_id=str(target_subject_id),
            ),
        ],
    )

    with patch("app.worker.RecurrentSubjectService") as mock_recurrent_service_class:
        mock_recurrent_service_class.return_value.evaluate.return_value = assessment
        mock_recurrent_service_class.return_value.resolve.return_value = resolution

        event = process_fixture("tests/fixtures/frame_recurrent_unresolved.json")

    assert event["event_type"] == "human_presence_no_face"
    assert event["payload"]["unresolved_recurrence_status"] == "recurrent_unresolved"
    assert event["payload"]["requires_human_review"] is True
    assert event["payload"]["semantic_descriptor"]["descriptor_backend"] == "simple_color_signature_v1"
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "human_presence_no_face",
        "recurrent_unresolved_subject",
        "manual_review_required",
    ]


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_case_suggestion_emits_case_event(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    current_subject_id = uuid4()
    target_subject_id = uuid4()
    track_id = uuid4()
    target_track_id = uuid4()

    current_subject = MagicMock()
    current_subject.observed_subject_id = current_subject_id
    target_subject = MagicMock()
    target_subject.observed_subject_id = target_subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID("66666666-1111-1111-1111-111111111111")
    track.person_presence_score = 0.0

    _configure_repo_for_slice3_flow(
        mock_repo_instance,
        subject=current_subject,
        track=track,
        recognition_event=_make_recognition_event(),
    )
    mock_repo_instance.get_subject.return_value = target_subject
    mock_repo_instance.touch_subject.return_value = target_subject
    mock_repo_instance.update_subject_face_profile.return_value = target_subject
    mock_repo_instance.add_recognition_event.side_effect = [
        _make_recognition_event(),
        _make_recognition_event(),
        _make_recognition_event(),
        _make_recognition_event(),
    ]
    mock_repo_class.return_value = mock_repo_instance

    fixture = load_fixture_message("tests/fixtures/frame_case_suggestion_created.json")
    assessment = RecurrentSubjectAssessment(
        current_subject_id=str(current_subject_id),
        current_track_id=str(track_id),
        current_camera_id=fixture.camera_id,
        semantic_similarity_threshold=0.72,
        recurrent_subject_threshold=0.78,
        case_suggestion_threshold=0.9,
        manual_review_threshold=0.35,
        evaluated_candidates=1,
        best_candidate=RecurrentSubjectCandidate(
            observed_subject_id=str(target_subject_id),
            latest_track_id=str(target_track_id),
            last_camera_id="55555555-1111-1111-1111-111111111111",
            last_seen_at=fixture.captured_at,
            recurrence_count=2,
            semantic_similarity_score=1.0,
            visual_similarity_score=0.0,
            temporal_coherence_score=0.8,
            camera_relation_score=1.0,
            aggregate_score=0.96,
            descriptor_summary={"dominant_palette": ["gray", "blue"]},
        ),
        decision_reason=["unresolved_recurrence_threshold_passed"],
    )
    resolution = RecurrentSubjectResolution(
        outcome="recurrent_unresolved",
        subject_id_to_use=str(target_subject_id),
        target_track_id=str(target_track_id),
        requires_human_review=True,
        requires_case_evaluation=True,
        decision_reason=["semantic_subject_recurrence_detected"],
        payload={"evidence_count": 3, "requires_case_evaluation": True},
        assessment=assessment,
        supplemental_decisions=[
            SupplementalRecognitionDecision(
                event_type="recurrent_unresolved_subject",
                severity="medium",
                confidence=0.96,
                decision_reason=["semantic_subject_recurrence_detected"],
                payload={"evidence_count": 3},
                subject_id=str(target_subject_id),
            ),
            SupplementalRecognitionDecision(
                event_type="manual_review_required",
                severity="medium",
                confidence=0.96,
                decision_reason=["recurrent_unresolved_subject_detected"],
                payload={"requires_human_review": True},
                subject_id=str(target_subject_id),
            ),
            SupplementalRecognitionDecision(
                event_type="case_suggestion_created",
                severity="medium",
                confidence=0.96,
                decision_reason=["case_suggestion_threshold_passed"],
                payload={"requires_case_evaluation": True},
                subject_id=str(target_subject_id),
            ),
        ],
    )

    with patch("app.worker.RecurrentSubjectService") as mock_recurrent_service_class:
        mock_recurrent_service_class.return_value.evaluate.return_value = assessment
        mock_recurrent_service_class.return_value.resolve.return_value = resolution

        event = process_fixture("tests/fixtures/frame_case_suggestion_created.json")

    assert event["event_type"] == "human_presence_no_face"
    assert event["payload"]["unresolved_recurrence_status"] == "recurrent_unresolved"
    assert event["payload"]["requires_case_evaluation"] is True
    assert event["payload"]["semantic_descriptor"]["descriptor_backend"] == "simple_color_signature_v1"
    assert [call.kwargs["event_type"] for call in mock_repo_instance.add_recognition_event.call_args_list] == [
        "human_presence_no_face",
        "recurrent_unresolved_subject",
        "manual_review_required",
        "case_suggestion_created",
    ]


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
@patch("app.worker.load_fixture_message")
def test_process_fixture_fails_with_clear_invalid_uuid_error(mock_load_fixture_message, mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    mock_repo_class.return_value = mock_repo_instance
    mock_load_fixture_message.return_value = FrameIngestedMessage(
        event_id="evt_ing_invalid",
        event_type="frame.ingested",
        event_version="1.0",
        occurred_at="2026-03-17T15:30:45.123Z",
        payload={
            "frame_ref": "frames/demo/person_001.jpg",
            "camera_id": "cam_101",
            "external_camera_key": "cam_101",
            "captured_at": "2026-03-17T15:30:45.123Z",
            "quality_metadata": {},
        },
        context={"correlation_id": "corr_invalid"},
    )

    with pytest.raises(InvalidCameraIdError, match="expected canonical UUID from api.camera.camera_id"):
        process_fixture("tests/fixtures/frame_ingested_example.json")

    mock_repo_instance.create_subject.assert_not_called()
    mock_session.commit.assert_not_called()
