from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch

import pytest

from app.consumer import load_fixture_message
from app.domain.entities import FrameIngestedMessage, InvalidCameraIdError
from app.models import EventOutbox, HumanTrack, RecognitionEvent
from app.worker import process_fixture

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_fixture_loads():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert msg.event_type == "frame.ingested"
    assert msg.camera_id == CAMERA_ID
    assert msg.payload.external_camera_key == "cam_101"
    assert msg.frame_ref.endswith("face_detectable.jpg")


def test_orm_alignment_for_uuid_and_frame_count_dependency():
    assert RecognitionEvent.__table__.c.camera_id.type.as_uuid is True
    assert HumanTrack.__table__.c.camera_id.type.as_uuid is True
    assert EventOutbox.__table__.c.aggregate_id.type.as_uuid is True
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

    recognition_event = MagicMock()
    recognition_event.recognition_event_id = recognition_event_id

    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    mock_repo_instance.find_track_by_camera_and_external_key.return_value = None
    mock_repo_instance.create_subject.return_value = subject
    mock_repo_instance.create_track.return_value = track
    mock_repo_instance.add_recognition_event.return_value = recognition_event
    mock_repo_instance.update_track_presence.side_effect = update_track_presence
    mock_repo_instance.update_track_face_observation.return_value = track
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event is not None
    assert event["event_type"] == "face_detected_unidentified"
    assert event["context"]["camera_id"] == CAMERA_ID
    assert event["context"]["track_id"] == str(track_id)
    assert event["context"]["subject_id"] == str(subject_id)
    assert event["payload"]["face_detection"]["usable"] is True

    mock_repo_instance.create_subject.assert_called_once()
    assert mock_repo_instance.create_subject.call_args.kwargs["camera_id"] == resolved_camera_id
    assert mock_repo_instance.create_track.call_args.kwargs["camera_id"] == resolved_camera_id

    recognition_event_call = mock_repo_instance.add_recognition_event.call_args.kwargs
    assert recognition_event_call["camera_id"] == resolved_camera_id
    assert recognition_event_call["decision_reason"] == [
        "human_track_confirmed",
        "face_detected",
        "face_quality_threshold_passed",
    ]
    assert recognition_event_call["evidence_refs"] == ["tests/fixtures/images/face_detectable.jpg"]
    assert recognition_event_call["payload"]["face_detection"]["usable"] is True

    outbox_call = mock_repo_instance.add_outbox_event.call_args.kwargs
    assert outbox_call["aggregate_id"] == recognition_event_id
    assert outbox_call["event_type"] == "face_detected_unidentified"
    assert outbox_call["payload"]["payload"]["face_detection"]["usable"] is True

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

    recognition_event = MagicMock()
    recognition_event.recognition_event_id = recognition_event_id

    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    mock_repo_instance.find_track_by_camera_and_external_key.return_value = track
    mock_repo_instance.get_subject.return_value = subject
    mock_repo_instance.add_recognition_event.return_value = recognition_event
    mock_repo_instance.update_track_presence.side_effect = update_track_presence
    mock_repo_instance.update_track_face_observation.return_value = track
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
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["aggregate_id"] == recognition_event_id
    mock_session.commit.assert_called_once()


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

    recognition_event = MagicMock()
    recognition_event.recognition_event_id = recognition_event_id

    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    mock_repo_instance.find_track_by_camera_and_external_key.return_value = None
    mock_repo_instance.create_subject.return_value = subject
    mock_repo_instance.create_track.return_value = track
    mock_repo_instance.add_recognition_event.return_value = recognition_event
    mock_repo_instance.update_track_presence.side_effect = update_track_presence
    mock_repo_instance.update_track_face_observation.return_value = track
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_no_face.json")

    assert event["event_type"] == "human_presence_no_face"
    assert event["payload"]["face_detection"]["detected"] is True
    assert event["payload"]["face_detection"]["usable"] is False
    assert mock_repo_instance.add_recognition_event.call_args.kwargs["payload"]["face_detection"]["usable"] is False
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["payload"]["event_type"] == "human_presence_no_face"
    assert mock_repo_instance.add_outbox_event.call_args.kwargs["aggregate_id"] == recognition_event_id
    mock_session.commit.assert_called_once()


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
