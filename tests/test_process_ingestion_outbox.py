from __future__ import annotations

import json
from pathlib import Path

from app.runner.process_ingestion_outbox import process_ingestion_outbox


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_process_ingestion_outbox_passes_resolved_messages_to_processor(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(_event(frame_ref="logical-ref.jpg", frame_uri=str(frame_path))) + "\n", encoding="utf-8")
    seen_frame_refs: list[str] = []

    def processor(message):
        seen_frame_refs.append(message.frame_ref)
        assert Path(message.frame_ref).is_file()
        assert str(message.camera_uuid) == CAMERA_ID
        return {"event_type": "human_presence_no_face", "payload": {"frame_ref": message.frame_ref}}

    result = process_ingestion_outbox(jsonl_path, processor=processor)

    assert result.processed == 1
    assert seen_frame_refs == [str(frame_path.resolve())]
    assert result.emitted_events[0]["payload"]["frame_ref"] == str(frame_path.resolve())


def _event(*, frame_ref: str, frame_uri: str) -> dict:
    return {
        "event_id": "evt_ingestion_test",
        "event_type": "frame.ingested",
        "event_version": "1.0",
        "occurred_at": "2026-01-01T00:00:00.000Z",
        "payload": {
            "camera_id": CAMERA_ID,
            "captured_at": "2026-01-01T00:00:00.000Z",
            "content_type": "image/jpeg",
            "frame_ref": frame_ref,
            "frame_uri": frame_uri,
            "height": 90,
            "quality_metadata": {"capture_fps": 1.0},
            "source_type": "video_file",
            "width": 160,
        },
        "context": {
            "correlation_id": "corr_ingestion_test",
            "idempotency_key": "frame:test",
        },
    }

