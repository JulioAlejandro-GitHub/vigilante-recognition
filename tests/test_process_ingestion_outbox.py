from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.ingestion import FileCheckpointStore, FileEventDeduper, RejectedEventStore
from app.runner.process_ingestion_outbox import process_ingestion_outbox
from app.services.track_continuity_service import TrackContinuityService
from app.storage.frame_resolver import FrameResolver
from app.storage.s3_frame_resolver import S3FrameResolver


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

    result = process_ingestion_outbox(jsonl_path, processor=processor, **_stores(tmp_path))

    assert result.processed == 1
    assert seen_frame_refs == [str(frame_path.resolve())]
    assert result.emitted_events[0]["payload"]["frame_ref"] == str(frame_path.resolve())
    assert result.rejected == 0


def test_process_ingestion_outbox_checkpoint_skips_already_consumed_lines_and_force_replays(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(_event(event_id="evt_1", frame_ref=str(frame_path), sample_index=0)),
                json.dumps(_event(event_id="evt_2", frame_ref=str(frame_path), sample_index=1)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stores = _stores(tmp_path)
    seen_event_ids: list[str] = []

    def processor(message):
        seen_event_ids.append(message.event_id)
        return {"event_type": "processed", "payload": {"event_id": message.event_id}}

    first = process_ingestion_outbox(jsonl_path, processor=processor, **stores)
    second = process_ingestion_outbox(jsonl_path, processor=processor, **stores)
    forced = process_ingestion_outbox(jsonl_path, processor=processor, force_replay=True, **stores)

    assert first.processed == 2
    assert first.read == 2
    assert second.processed == 0
    assert second.read == 0
    assert second.skipped_checkpoint == 2
    assert forced.processed == 2
    assert forced.read == 2
    assert seen_event_ids == ["evt_1", "evt_2", "evt_1", "evt_2"]


def test_process_ingestion_outbox_skips_duplicate_event_id_without_reprocessing(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(_event(event_id="evt_same", frame_ref=str(frame_path), sample_index=0)),
                json.dumps(_event(event_id="evt_same", frame_ref=str(frame_path), sample_index=1)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    processed: list[str] = []

    def processor(message):
        processed.append(message.event_id)
        return {"event_type": "processed", "payload": {"event_id": message.event_id}}

    result = process_ingestion_outbox(jsonl_path, processor=processor, **_stores(tmp_path))

    assert result.read == 2
    assert result.processed == 1
    assert result.skipped_duplicate == 1
    assert processed == ["evt_same"]


def test_process_ingestion_outbox_writes_rejected_events_without_aborting(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    invalid_camera = _event(event_id="evt_invalid_camera", frame_ref=str(frame_path))
    invalid_camera["payload"]["camera_id"] = "cam01"
    missing_ref = _event(event_id="evt_missing_ref", frame_ref=str(frame_path))
    missing_ref["payload"].pop("frame_ref")
    missing_ref["payload"].pop("frame_uri")
    unsupported = _event(event_id="evt_unsupported", frame_ref=str(frame_path))
    unsupported["event_type"] = "camera.created"
    missing_file = _event(event_id="evt_missing_file", frame_ref=str(tmp_path / "missing.jpg"))
    stores = _stores(tmp_path)
    jsonl_path.write_text(
        "\n".join(
            [
                "{not-json}",
                json.dumps({"event_id": "evt_missing_type", "payload": {}}),
                json.dumps(unsupported),
                json.dumps(invalid_camera),
                json.dumps(missing_ref),
                json.dumps(missing_file),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = process_ingestion_outbox(
        jsonl_path,
        processor=lambda message: {"event_type": "processed", "payload": {"event_id": message.event_id}},
        **stores,
    )
    rejected_records = _read_jsonl(stores["rejected_event_store"].path)
    reasons = {record["reason"] for record in rejected_records}

    assert result.read == 6
    assert result.processed == 0
    assert result.rejected == 6
    assert result.frame_resolution_errors == 1
    assert reasons == {
        "invalid_json",
        "missing_event_type",
        "unsupported_event_type",
        "invalid_camera_id",
        "missing_frame_reference",
        "frame_resolution_failed",
    }
    assert all(record["source_path"] == str(jsonl_path.resolve()) for record in rejected_records)
    assert all(record["rejected_at"] for record in rejected_records)
    assert rejected_records[0]["line_number"] == 1


def test_process_ingestion_outbox_applies_basic_temporal_track_continuity(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    _event(
                        event_id="evt_cont_1",
                        frame_ref=str(frame_path),
                        captured_at=base_time,
                        sample_index=0,
                        source_timestamp_seconds=0.0,
                    )
                ),
                json.dumps(
                    _event(
                        event_id="evt_cont_2",
                        frame_ref=str(frame_path),
                        captured_at=base_time + timedelta(seconds=1),
                        sample_index=1,
                        source_timestamp_seconds=1.0,
                    )
                ),
                json.dumps(
                    _event(
                        event_id="evt_cont_3",
                        frame_ref=str(frame_path),
                        captured_at=base_time + timedelta(seconds=30),
                        sample_index=30,
                        source_timestamp_seconds=30.0,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    correlation_ids: list[str] = []
    statuses: list[str] = []

    def processor(message):
        correlation_ids.append(message.context["correlation_id"])
        statuses.append(message.context["track_continuity"]["status"])
        return {"event_type": "processed", "payload": {"track_key": message.context["correlation_id"]}}

    result = process_ingestion_outbox(
        jsonl_path,
        processor=processor,
        track_continuity_service=TrackContinuityService(window_seconds=5),
        **_stores(tmp_path),
    )

    assert result.processed == 3
    assert correlation_ids[0] == correlation_ids[1]
    assert correlation_ids[2] != correlation_ids[0]
    assert statuses == ["opened_local_track", "reused_recent_track", "opened_local_track"]


def test_process_ingestion_outbox_processes_remote_s3_frame(tmp_path) -> None:
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(
        json.dumps(_event(frame_ref="s3://vigilante-frames/frames/cam01/frame.jpg"))
        + "\n",
        encoding="utf-8",
    )
    resolver = FrameResolver(
        remote_resolver=S3FrameResolver(
            endpoint="localhost:9000",
            access_key="minio",
            secret_key="minio123",
            cache_dir=tmp_path / "cache",
            client=_FakeS3Client({("vigilante-frames", "frames/cam01/frame.jpg"): b"\xff\xd8test\xff\xd9"}),
        )
    )
    seen_frame_refs: list[str] = []

    def processor(message):
        seen_frame_refs.append(message.frame_ref)
        assert Path(message.frame_ref).is_file()
        assert message.payload.frame_uri == "s3://vigilante-frames/frames/cam01/frame.jpg"
        return {"event_type": "processed", "payload": {"frame_ref": message.frame_ref}}

    result = process_ingestion_outbox(
        jsonl_path,
        processor=processor,
        frame_resolver=resolver,
        **_stores(tmp_path),
    )

    assert result.processed == 1
    assert result.rejected == 0
    assert seen_frame_refs == [str((tmp_path / "cache" / "vigilante-frames" / "frames" / "cam01" / "frame.jpg").resolve())]


def test_process_ingestion_outbox_rejects_invalid_remote_uri_without_aborting(tmp_path) -> None:
    jsonl_path = tmp_path / "frame_ingested.jsonl"
    jsonl_path.write_text(
        json.dumps(_event(event_id="evt_invalid_s3", frame_ref="s3:///missing-bucket.jpg"))
        + "\n",
        encoding="utf-8",
    )
    stores = _stores(tmp_path)

    result = process_ingestion_outbox(
        jsonl_path,
        processor=lambda message: {"event_type": "should_not_run"},
        frame_resolver=FrameResolver(
            remote_resolver=S3FrameResolver(
                endpoint="localhost:9000",
                access_key="minio",
                secret_key="minio123",
                cache_dir=tmp_path / "cache",
                client=_FakeS3Client({}),
            )
        ),
        **stores,
    )
    rejected = _read_jsonl(stores["rejected_event_store"].path)

    assert result.processed == 0
    assert result.rejected == 1
    assert result.frame_resolution_errors == 1
    assert rejected[0]["reason"] == "frame_resolution_failed"
    assert rejected[0]["details"]["reason"] == "invalid_remote_frame_uri"


def _stores(tmp_path) -> dict:
    return {
        "checkpoint_store": FileCheckpointStore(tmp_path / "state" / "checkpoint.json"),
        "event_deduper": FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        "rejected_event_store": RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl"),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event(
    *,
    frame_ref: str,
    frame_uri: str | None = None,
    event_id: str = "evt_ingestion_test",
    captured_at: datetime | None = None,
    sample_index: int = 0,
    source_timestamp_seconds: float = 0.0,
) -> dict:
    captured_at = captured_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    captured_at_value = captured_at.isoformat().replace("+00:00", "Z")
    if frame_uri is None:
        frame_uri = frame_ref
    return {
        "event_id": event_id,
        "event_type": "frame.ingested",
        "event_version": "1.0",
        "occurred_at": captured_at_value,
        "payload": {
            "camera_id": CAMERA_ID,
            "captured_at": captured_at_value,
            "content_type": "image/jpeg",
            "frame_ref": frame_ref,
            "frame_uri": frame_uri,
            "height": 90,
            "metadata": {
                "capture_fps": 1.0,
                "sample_index": sample_index,
                "source_frame_index": sample_index * 10,
                "source_timestamp_seconds": source_timestamp_seconds,
                "source_uri": "samples/cam01.mp4",
            },
            "quality_metadata": {
                "capture_fps": 1.0,
                "source_timestamp_seconds": source_timestamp_seconds,
            },
            "source_type": "video_file",
            "width": 160,
        },
        "context": {
            "correlation_id": "corr_ingestion_test",
            "idempotency_key": "frame:test",
        },
    }


class _FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def fget_object(self, bucket: str, object_key: str, file_path: str) -> None:
        Path(file_path).write_bytes(self.objects[(bucket, object_key)])
