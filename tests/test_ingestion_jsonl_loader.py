from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.frame_ingested_loader import (
    FrameIngestedJsonlLoader,
    InvalidFrameIngestedEventError,
)
from app.ingestion.jsonl_event_source import InvalidJsonlLineError
from app.storage.frame_resolver import FrameResolutionError, LocalFrameResolver


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_frame_ingested_loader_parses_jsonl_and_resolves_frame_uri(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(_event(frame_ref="missing.jpg", frame_uri=str(frame_path))) + "\n", encoding="utf-8")

    loaded = list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())

    assert len(loaded) == 1
    assert loaded[0].message.event_type == "frame.ingested"
    assert loaded[0].message.camera_uuid
    assert loaded[0].message.frame_ref == str(frame_path.resolve())
    assert loaded[0].message.payload.metadata["original_frame_ref"] == "missing.jpg"


def test_frame_ingested_loader_falls_back_to_frame_ref(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(_event(frame_ref=str(frame_path), frame_uri="s3://bucket/frame.jpg")) + "\n", encoding="utf-8")

    loaded = list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())

    assert loaded[0].message.frame_ref == str(frame_path.resolve())


def test_frame_ingested_loader_rejects_invalid_json(tmp_path) -> None:
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(InvalidJsonlLineError):
        list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())


def test_frame_ingested_loader_rejects_incomplete_event(tmp_path) -> None:
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps({"event_type": "frame.ingested"}) + "\n", encoding="utf-8")

    with pytest.raises(InvalidFrameIngestedEventError):
        list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())


def test_frame_ingested_loader_rejects_invalid_camera_uuid(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    event = _event(frame_ref=str(frame_path))
    event["payload"]["camera_id"] = "cam01"
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(InvalidFrameIngestedEventError):
        list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())


def test_frame_ingested_loader_raises_when_frame_is_missing(tmp_path) -> None:
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(_event(frame_ref="missing.jpg")) + "\n", encoding="utf-8")

    with pytest.raises(FrameResolutionError):
        list(FrameIngestedJsonlLoader(jsonl_path).iter_messages())


def test_frame_resolver_supports_search_roots(tmp_path) -> None:
    frame_path = tmp_path / "storage" / "frames" / "cam01.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(json.dumps(_event(frame_ref="frames/cam01.jpg")) + "\n", encoding="utf-8")
    resolver = LocalFrameResolver(search_roots=[tmp_path / "storage"])

    loaded = list(FrameIngestedJsonlLoader(jsonl_path, frame_resolver=resolver).iter_messages())

    assert loaded[0].message.frame_ref == str(frame_path.resolve())


def _event(*, frame_ref: str, frame_uri: str | None = None) -> dict:
    payload = {
        "camera_id": CAMERA_ID,
        "captured_at": "2026-01-01T00:00:00.000Z",
        "content_type": "image/jpeg",
        "frame_ref": frame_ref,
        "height": 90,
        "quality_metadata": {"capture_fps": 1.0},
        "source_type": "video_file",
        "width": 160,
    }
    if frame_uri is not None:
        payload["frame_uri"] = frame_uri
    return {
        "event_id": "evt_ingestion_test",
        "event_type": "frame.ingested",
        "event_version": "1.0",
        "occurred_at": "2026-01-01T00:00:00.000Z",
        "payload": payload,
        "context": {
            "correlation_id": "corr_ingestion_test",
            "idempotency_key": "frame:test",
        },
    }

