from __future__ import annotations

from datetime import datetime, timezone

from app.correlation import source_correlation_payload
from app.domain.entities import FrameIngestedMessage
from app.domain.events import build_recognition_event


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_source_correlation_payload_extracts_run_id_and_source_event() -> None:
    message = _message(run_id="pipeline-run-1")

    correlation = source_correlation_payload(message)

    assert correlation["run_id"] == "pipeline-run-1"
    assert correlation["source_event_id"] == "evt_frame_1"
    assert correlation["source_frame_event_id"] == "evt_frame_1"
    assert correlation["source_frame_ref"] == "s3://vigilante-frames/frame.jpg"


def test_build_recognition_event_propagates_run_id_and_source_event_id() -> None:
    correlation = source_correlation_payload(_message(run_id="pipeline-run-1"))

    event = build_recognition_event(
        event_type="human_presence_no_face",
        camera_id=CAMERA_ID,
        track_id="22222222-2222-2222-2222-222222222222",
        subject_id="33333333-3333-3333-3333-333333333333",
        severity="low",
        confidence=0.6,
        decision_reason=["human_track_confirmed"],
        frame_ref="s3://vigilante-frames/frame.jpg",
        correlation=correlation,
    )

    assert event["context"]["run_id"] == "pipeline-run-1"
    assert event["context"]["source_event_id"] == "evt_frame_1"
    assert event["payload"]["source_event_id"] == "evt_frame_1"
    assert event["payload"]["source_frame_event_id"] == "evt_frame_1"
    assert event["payload"]["correlation"]["run_id"] == "pipeline-run-1"


def test_build_recognition_event_keeps_source_event_id_without_run_id() -> None:
    correlation = source_correlation_payload(_message(run_id=None))

    event = build_recognition_event(
        event_type="human_presence_no_face",
        camera_id=CAMERA_ID,
        track_id="22222222-2222-2222-2222-222222222222",
        subject_id="33333333-3333-3333-3333-333333333333",
        severity="low",
        confidence=0.6,
        decision_reason=["human_track_confirmed"],
        frame_ref="s3://vigilante-frames/frame.jpg",
        correlation=correlation,
    )

    assert "run_id" not in event["context"]
    assert event["payload"]["source_event_id"] == "evt_frame_1"
    assert event["payload"]["source_frame_event_id"] == "evt_frame_1"


def _message(*, run_id: str | None) -> FrameIngestedMessage:
    captured_at = datetime(2026, 5, 11, tzinfo=timezone.utc)
    context = {
        "correlation_id": "corr_frame_1",
        "idempotency_key": "frame:test",
    }
    metadata = {
        "pipeline": {"run_id": run_id},
    } if run_id else {}
    return FrameIngestedMessage(
        event_id="evt_frame_1",
        event_type="frame.ingested",
        event_version="1.0",
        occurred_at=captured_at,
        payload={
            "camera_id": CAMERA_ID,
            "captured_at": captured_at,
            "frame_ref": "s3://vigilante-frames/frame.jpg",
            "frame_uri": "s3://vigilante-frames/frame.jpg",
            "metadata": metadata,
        },
        context=context,
    )
