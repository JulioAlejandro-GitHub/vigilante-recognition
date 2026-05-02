from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult
from app.ingestion import FileEventDeduper, RejectedEventStore
from app.ingestion.rabbitmq_event_source import RabbitMqDelivery
from app.messaging.topology import FrameIngestedTopology
from app.runner.process_rabbitmq_frames import process_rabbitmq_frames
from app.services.face_backend_selector import FaceBackendSelector
from app.services.face_backend_service import FaceBackendError
from app.worker import process_fixture, process_message

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


class _FakeFaceBackend:
    def __init__(self, *, backend_key: str, fail_stage: str | None = None, reason: str = "backend_failed") -> None:
        self.backend_key = backend_key
        self.backend_name = f"{backend_key}:test"
        self.provider_name = f"{backend_key}_provider"
        self.fail_stage = fail_stage
        self.reason = reason
        self.inspect_calls = 0
        self.generate_calls = 0

    def inspect_face(self, *, frame_ref: str, quality_metadata: dict[str, float] | None = None) -> FaceDetectionResult:
        self.inspect_calls += 1
        if self.fail_stage == "detect":
            raise FaceBackendError(self.reason, backend_key=self.backend_key, stage="detect")
        return FaceDetectionResult(
            detected=True,
            usable=True,
            quality_score=0.91,
            bbox={"x": 1, "y": 2, "width": 50, "height": 60},
            image_size={"width": 100, "height": 120},
            quality_metrics={"detection_confidence": 0.95},
            frame_quality_metadata=quality_metadata or {},
        )

    def generate(self, *, frame_ref: str, face_detection: FaceDetectionResult | None = None) -> FaceEmbeddingResult:
        self.generate_calls += 1
        if self.fail_stage == "embedding":
            raise FaceBackendError(self.reason, backend_key=self.backend_key, stage="embedding")
        return FaceEmbeddingResult(
            generated=True,
            backend=f"{self.backend_key}:embedding",
            dimensions=2,
            vector=[0.6, 0.8],
            bbox=face_detection.bbox if face_detection else None,
            source_frame_ref=frame_ref,
        )


def test_face_backend_selector_uses_insightface_in_auto_when_available():
    simple = _FakeFaceBackend(backend_key="simple")
    insightface = _FakeFaceBackend(backend_key="insightface")

    with patch.object(settings, "face_backend", "auto"), patch.object(settings, "insightface_enabled", True):
        selector = FaceBackendSelector(simple_backend=simple, insightface_backend=insightface)
        detection = selector.inspect_face(frame_ref="frame.jpg")
        embedding = selector.generate(frame_ref="frame.jpg", face_detection=detection)

    assert detection.face_backend_requested == "auto"
    assert detection.face_backend_selected == "insightface"
    assert detection.face_backend_fallback_used is False
    assert detection.face_backend_trace["provider"] == "insightface_provider"
    assert detection.face_backend_trace["detected"] is True
    assert detection.face_backend_trace["usable"] is True
    assert detection.face_backend_trace["elapsed_ms"] >= 0.0
    assert embedding.embedding_backend_selected == "insightface"
    assert embedding.embedding_backend_trace["provider"] == "insightface_provider"
    assert embedding.embedding_backend_trace["generated"] is True
    assert insightface.inspect_calls == 1
    assert insightface.generate_calls == 1
    assert simple.inspect_calls == 0
    assert simple.generate_calls == 0


def test_face_backend_selector_does_not_attempt_insightface_when_simple_forced():
    simple = _FakeFaceBackend(backend_key="simple")
    insightface = _FakeFaceBackend(backend_key="insightface", fail_stage="detect")

    with patch.object(settings, "face_backend", "simple"), patch.object(settings, "insightface_enabled", True):
        selector = FaceBackendSelector(simple_backend=simple, insightface_backend=insightface)
        detection = selector.inspect_face(frame_ref="frame.jpg")

    assert detection.face_backend_requested == "simple"
    assert detection.face_backend_selected == "simple"
    assert detection.face_backend_fallback_used is False
    assert simple.inspect_calls == 1
    assert insightface.inspect_calls == 0


def test_face_backend_selector_falls_back_to_simple_in_auto_when_insightface_fails():
    simple = _FakeFaceBackend(backend_key="simple")
    insightface = _FakeFaceBackend(
        backend_key="insightface",
        fail_stage="detect",
        reason="insightface_load_failed:RuntimeError",
    )

    with patch.object(settings, "face_backend", "auto"), patch.object(settings, "insightface_enabled", True):
        selector = FaceBackendSelector(simple_backend=simple, insightface_backend=insightface)
        detection = selector.inspect_face(frame_ref="frame.jpg")

    assert detection.face_backend_requested == "auto"
    assert detection.face_backend_selected == "simple"
    assert detection.face_backend_fallback_used is True
    assert detection.face_backend_error == "insightface_load_failed:RuntimeError"
    assert [attempt["status"] for attempt in detection.face_backend_trace["attempts"]] == ["failed", "success"]


def test_face_backend_selector_does_not_fallback_when_insightface_forced():
    simple = _FakeFaceBackend(backend_key="simple")
    insightface = _FakeFaceBackend(backend_key="insightface", fail_stage="detect", reason="insightface_not_installed")

    with patch.object(settings, "face_backend", "insightface"), patch.object(settings, "insightface_enabled", True):
        selector = FaceBackendSelector(simple_backend=simple, insightface_backend=insightface)
        with pytest.raises(FaceBackendError) as exc_info:
            selector.inspect_face(frame_ref="frame.jpg")

    assert exc_info.value.reason == "insightface_not_installed"
    assert simple.inspect_calls == 0


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_worker_event_contains_face_backend_trace_and_preserves_evidence_refs(mock_get_session, mock_repo_class):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    mock_repo_instance = _repo()
    mock_repo_class.return_value = mock_repo_instance

    with patch.object(settings, "face_backend", "auto"), patch.object(settings, "insightface_enabled", False):
        event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event["event_type"] == "face_detected_unidentified"
    assert event["payload"]["face_backend_requested"] == "auto"
    assert event["payload"]["face_backend_selected"] == "simple"
    assert event["payload"]["face_backend_fallback_used"] is True
    assert event["payload"]["face_backend_error"] == "insightface_disabled"
    assert event["payload"]["face_detection"]["face_backend_trace"]["attempts"][0]["status"] == "skipped"
    assert event["payload"]["evidence_refs"] == ["tests/fixtures/images/face_detectable.jpg"]
    assert event["payload"]["semantic_descriptor"]["source_frame_ref"] == "tests/fixtures/images/face_detectable.jpg"
    assert mock_repo_instance.add_recognition_event.call_args.kwargs["payload"]["evidence_refs"] == [
        "tests/fixtures/images/face_detectable.jpg"
    ]
    mock_session.commit.assert_called_once()


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_rabbitmq_processing_with_auto_fallback_acks_and_emits_trace(mock_get_session, mock_repo_class, tmp_path):
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    mock_repo_class.return_value = _repo()
    source = _FakeRabbitMqEventSource([_delivery(_rabbitmq_event())])

    with patch.object(settings, "face_backend", "auto"), patch.object(settings, "insightface_enabled", False):
        result = process_rabbitmq_frames(
            processor=process_message,
            event_source=source,
            event_deduper=FileEventDeduper(tmp_path / "processed_events.json"),
            rejected_event_store=RejectedEventStore(tmp_path / "rejected_events.jsonl"),
            topology=FrameIngestedTopology(),
            max_messages=1,
        )

    assert result.processed == 1
    assert result.acked == 1
    assert result.rejected_to_dlq == 0
    assert source.acked == [1]
    assert result.emitted_events[0]["payload"]["face_backend_selected"] == "simple"
    assert result.emitted_events[0]["payload"]["face_backend_fallback_used"] is True
    assert result.emitted_events[0]["payload"]["evidence_refs"] == ["tests/fixtures/images/face_detectable.jpg"]


def _repo():
    repo = MagicMock()
    subject_id = uuid4()
    track_id = uuid4()
    subject = MagicMock()
    subject.observed_subject_id = subject_id
    track = MagicMock()
    track.human_track_id = track_id
    track.camera_id = UUID(CAMERA_ID)
    track.person_presence_score = 0.0
    recognition_event = MagicMock()
    recognition_event.recognition_event_id = uuid4()

    def update_track_presence(track_obj, *, score_increment: float = 0.34):
        current_score = track_obj.person_presence_score or 0.0
        track_obj.person_presence_score = min(1.0, current_score + score_increment)
        track_obj.track_status = "confirmed_human" if track_obj.person_presence_score >= 0.6 else "probable_human"
        return track_obj

    repo.find_track_by_camera_and_external_key.return_value = None
    repo.create_subject.return_value = subject
    repo.create_track.return_value = track
    repo.add_recognition_event.return_value = recognition_event
    repo.update_track_presence.side_effect = update_track_presence
    repo.update_track_face_observation.return_value = track
    repo.update_track_match_result.return_value = track
    repo.update_subject_face_profile.return_value = subject
    repo.update_track_semantic_descriptor.return_value = track
    repo.load_known_face_gallery_entries.return_value = []
    repo.load_recent_subject_candidates.return_value = []
    repo.load_recent_unresolved_subject_candidates.return_value = []
    repo.find_latest_cross_camera_correlation_for_source_track.return_value = None
    repo.mark_subject_continuity.return_value = subject
    repo.add_cross_camera_correlation.return_value = MagicMock()
    repo.attach_subject_to_track.return_value = track
    repo.touch_subject.return_value = subject
    repo.update_track_continuity_resolution.return_value = track
    repo.update_track_recurrent_resolution.return_value = track
    return repo


def _delivery(event: dict) -> RabbitMqDelivery:
    return RabbitMqDelivery(
        body=json.dumps(event).encode("utf-8"),
        delivery_tag=1,
        headers={},
    )


def _rabbitmq_event() -> dict:
    captured_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "event_id": f"evt_face_backend_{uuid4().hex}",
        "event_type": "frame.ingested",
        "event_version": "1.0",
        "occurred_at": captured_at,
        "payload": {
            "camera_id": CAMERA_ID,
            "captured_at": captured_at,
            "content_type": "image/jpeg",
            "frame_ref": "tests/fixtures/images/face_detectable.jpg",
            "frame_uri": "tests/fixtures/images/face_detectable.jpg",
            "height": 90,
            "metadata": {
                "capture_fps": 1.0,
                "sample_index": 0,
                "source_frame_index": 0,
                "source_timestamp_seconds": 0.0,
                "source_uri": "rtsp://camera.local/live",
            },
            "quality_metadata": {
                "capture_fps": 1.0,
                "source_timestamp_seconds": 0.0,
            },
            "source_type": "rtsp",
            "width": 160,
        },
        "context": {
            "correlation_id": "corr_face_backend",
            "idempotency_key": "frame:face_backend",
        },
    }


class _FakeRabbitMqEventSource:
    def __init__(self, deliveries: list[RabbitMqDelivery]) -> None:
        self.deliveries = deliveries
        self.acked: list[int] = []
        self.rejected: list[int] = []
        self.retries: list[RabbitMqDelivery] = []
        self.closed = False

    def iter_deliveries(self, *, max_messages=None):
        for delivery in self.deliveries[:max_messages]:
            yield delivery

    def ack(self, delivery: RabbitMqDelivery) -> None:
        self.acked.append(delivery.delivery_tag)

    def reject_to_dlq(self, delivery: RabbitMqDelivery) -> None:
        self.rejected.append(delivery.delivery_tag)

    def retry(self, delivery: RabbitMqDelivery, *, retry_count: int) -> None:
        self.retries.append(delivery)

    def close(self) -> None:
        self.closed = True
