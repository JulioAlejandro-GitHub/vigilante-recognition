from __future__ import annotations

from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch

from app.config import settings
from app.services.semantic_backends import (
    SemanticBackendContext,
    SemanticBackendError,
    SemanticBackendOutput,
    SemanticDescriptorBackend,
    SimpleSemanticDescriptorBackend,
)
from app.services.semantic_descriptor_service import SemanticDescriptorService
from app.worker import process_fixture

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


class FailingSemanticBackend(SemanticDescriptorBackend):
    def __init__(self, *, key: str, backend_name: str, error: Exception) -> None:
        self.key = key
        self.backend_name = backend_name
        self.error = error

    def generate_descriptor(
        self,
        *,
        image_path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        raise self.error


def _make_recognition_event():
    recognition_event = MagicMock()
    recognition_event.recognition_event_id = uuid4()
    return recognition_event


def _configure_repo(mock_repo_instance, *, subject, track):
    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    mock_repo_instance.find_track_by_camera_and_external_key.return_value = None
    mock_repo_instance.create_subject.return_value = subject
    mock_repo_instance.create_track.return_value = track
    mock_repo_instance.add_recognition_event.return_value = _make_recognition_event()
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
    mock_repo_instance.attach_subject_to_track.return_value = track
    mock_repo_instance.touch_subject.return_value = subject
    mock_repo_instance.update_track_continuity_resolution.return_value = track
    mock_repo_instance.update_track_recurrent_resolution.return_value = track


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_worker_degrades_to_simple_backend_when_real_vlm_backends_fail(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    subject_id = uuid4()
    track_id = uuid4()

    subject = MagicMock()
    subject.observed_subject_id = subject_id

    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID(CAMERA_ID)
    track.person_presence_score = 0.0

    _configure_repo(mock_repo_instance, subject=subject, track=track)
    mock_repo_class.return_value = mock_repo_instance

    service = SemanticDescriptorService(
        backends={
            "qwen": FailingSemanticBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                error=SemanticBackendError(
                    "backend_timeout",
                    details={"stage": "runtime", "timeout_seconds": 5},
                ),
            ),
            "smolvlm": FailingSemanticBackend(
                key="smolvlm",
                backend_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                error=SemanticBackendError("model_load_failed:RuntimeError"),
            ),
            "simple": SimpleSemanticDescriptorBackend(),
        }
    )

    with patch("app.worker.SemanticDescriptorService", return_value=service), patch.object(
        settings,
        "semantic_use_real_vlm",
        True,
    ), patch.object(settings, "semantic_descriptor_backend", "auto"):
        event = process_fixture("tests/fixtures/frame_ingested_no_face.json")

    assert event["event_type"] == "human_presence_no_face"
    semantic_payload = event["payload"]["semantic_descriptor"]
    assert semantic_payload["descriptor_backend"] == "simple_color_signature_v1"
    attempts = semantic_payload["generation_trace"]["attempts"]
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen", "smolvlm", "simple"]
    assert attempts[0]["reason"] == "backend_timeout"
    assert attempts[0]["stage"] == "runtime"
    assert attempts[-1]["status"] == "success"
    assert semantic_payload["semantic_backend_requested"] == "auto"
    assert semantic_payload["semantic_backend_fallback_used"] is True
    mock_session.commit.assert_called_once()
