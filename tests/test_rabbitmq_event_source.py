from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion import FileEventDeduper, RejectedEventStore
from app.ingestion.rabbitmq_event_source import RabbitMqDelivery, RabbitMqEventSource
from app.messaging.topology import FrameIngestedTopology
from app.runner.process_rabbitmq_frames import process_rabbitmq_frames
from app.storage.frame_resolver import FrameResolver
from app.storage.s3_frame_resolver import S3FrameResolver


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_process_rabbitmq_frames_acks_valid_message_after_processing(tmp_path) -> None:
    frame_path = _frame(tmp_path)
    source = _FakeRabbitMqEventSource([_delivery(_event(frame_ref=str(frame_path)))])
    processed: list[str] = []

    def processor(message):
        processed.append(message.event_id)
        assert Path(message.frame_ref).is_file()
        assert message.context["track_continuity"]["status"] == "opened_local_track"
        return {"event_type": "processed", "payload": {"event_id": message.event_id}}

    result = process_rabbitmq_frames(
        processor=processor,
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl"),
        topology=FrameIngestedTopology(),
        max_messages=1,
    )

    assert result.consumed == 1
    assert result.processed == 1
    assert result.acked == 1
    assert result.rejected_to_dlq == 0
    assert processed == ["evt_rabbitmq_test"]
    assert source.acked == [1]


def test_process_rabbitmq_frames_skips_duplicate_event_id_with_ack(tmp_path) -> None:
    frame_path = _frame(tmp_path)
    deduper = FileEventDeduper(tmp_path / "state" / "processed_events.json")
    deduper.mark_processed("evt_duplicate", source_path="rabbitmq:vigilante.frames:frame.ingested")
    source = _FakeRabbitMqEventSource([_delivery(_event(event_id="evt_duplicate", frame_ref=str(frame_path)))])

    result = process_rabbitmq_frames(
        processor=lambda message: {"event_type": "should_not_run"},
        event_source=source,
        event_deduper=deduper,
        rejected_event_store=RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl"),
        topology=FrameIngestedTopology(),
        max_messages=1,
    )

    assert result.processed == 0
    assert result.skipped_duplicate == 1
    assert result.acked == 1
    assert source.acked == [1]


def test_process_rabbitmq_frames_rejects_invalid_contract_to_broker_dlq(tmp_path) -> None:
    invalid = _event(frame_ref=str(_frame(tmp_path)))
    invalid["payload"]["camera_id"] = "cam01"
    source = _FakeRabbitMqEventSource([_delivery(invalid)])
    rejected_store = RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl")

    result = process_rabbitmq_frames(
        processor=lambda message: {"event_type": "should_not_run"},
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=rejected_store,
        topology=FrameIngestedTopology(),
        max_messages=1,
    )
    rejected = _read_jsonl(rejected_store.path)

    assert result.invalid_messages == 1
    assert result.rejected_to_dlq == 1
    assert source.rejected == [1]
    assert rejected[0]["reason"] == "invalid_camera_id"
    assert rejected[0]["details"]["broker_dlq"] == "vigilante.recognition.frame_ingested.dlq"


def test_process_rabbitmq_frames_rejects_unresolvable_frame_to_broker_dlq(tmp_path) -> None:
    source = _FakeRabbitMqEventSource([_delivery(_event(frame_ref=str(tmp_path / "missing.jpg")))])
    rejected_store = RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl")

    result = process_rabbitmq_frames(
        processor=lambda message: {"event_type": "should_not_run"},
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=rejected_store,
        topology=FrameIngestedTopology(),
        max_messages=1,
    )

    assert result.frame_resolution_errors == 1
    assert result.rejected_to_dlq == 1
    assert source.rejected == [1]
    assert _read_jsonl(rejected_store.path)[0]["reason"] == "frame_resolution_failed"


def test_process_rabbitmq_frames_processes_remote_s3_frame(tmp_path) -> None:
    event = _event(frame_ref="s3://vigilante-frames/frames/cam01/frame.jpg")
    source = _FakeRabbitMqEventSource([_delivery(event)])
    resolver = FrameResolver(
        remote_resolver=S3FrameResolver(
            endpoint="localhost:9000",
            access_key="minio",
            secret_key="minio123",
            cache_dir=tmp_path / "cache",
            client=_FakeS3Client({("vigilante-frames", "frames/cam01/frame.jpg"): b"\xff\xd8test\xff\xd9"}),
        )
    )
    processed: list[str] = []

    def processor(message):
        processed.append(message.frame_ref)
        assert Path(message.frame_ref).is_file()
        assert message.canonical_frame_ref == "s3://vigilante-frames/frames/cam01/frame.jpg"
        assert message.cached_path == message.frame_ref
        return {"event_type": "processed", "payload": {"frame_ref": message.frame_ref}}

    result = process_rabbitmq_frames(
        processor=processor,
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl"),
        frame_resolver=resolver,
        topology=FrameIngestedTopology(),
        max_messages=1,
    )

    assert result.processed == 1
    assert result.rejected_to_dlq == 0
    assert source.acked == [1]
    assert processed == [str((tmp_path / "cache" / "vigilante-frames" / "frames" / "cam01" / "frame.jpg").resolve())]


def test_process_rabbitmq_frames_retries_processing_failure_then_acks_original(tmp_path) -> None:
    frame_path = _frame(tmp_path)
    source = _FakeRabbitMqEventSource([_delivery(_event(frame_ref=str(frame_path)))])

    def processor(message):
        raise RuntimeError("database unavailable")

    result = process_rabbitmq_frames(
        processor=processor,
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl"),
        topology=FrameIngestedTopology(),
        max_messages=1,
        retry_limit=2,
    )

    assert result.processing_errors == 1
    assert result.retried == 1
    assert result.acked == 1
    assert result.rejected_to_dlq == 0
    assert source.retries[0].headers["x-retry-count"] == 1
    assert source.acked == [1]


def test_process_rabbitmq_frames_rejects_after_retry_limit(tmp_path) -> None:
    frame_path = _frame(tmp_path)
    delivery = _delivery(_event(frame_ref=str(frame_path)), headers={"x-retry-count": 2})
    source = _FakeRabbitMqEventSource([delivery])
    rejected_store = RejectedEventStore(tmp_path / "state" / "rejected_events.jsonl")

    def processor(message):
        raise RuntimeError("still failing")

    result = process_rabbitmq_frames(
        processor=processor,
        event_source=source,
        event_deduper=FileEventDeduper(tmp_path / "state" / "processed_events.json"),
        rejected_event_store=rejected_store,
        topology=FrameIngestedTopology(),
        max_messages=1,
        retry_limit=2,
    )

    assert result.processing_errors == 1
    assert result.retried == 0
    assert result.rejected_to_dlq == 1
    assert source.rejected == [1]
    assert _read_jsonl(rejected_store.path)[0]["reason"] == "processing_failed_retries_exhausted"


def test_rabbitmq_event_source_logs_ready_after_declaring_topology(caplog) -> None:
    channel = _FakeRabbitMqChannel()
    source = RabbitMqEventSource(
        host="localhost",
        port=5672,
        username="guest",
        password="guest",
        virtual_host="/",
        topology=FrameIngestedTopology(),
        prefetch_count=7,
        connection_factory=lambda: _FakeRabbitMqConnection(channel),
    )

    with caplog.at_level(logging.INFO, logger="app.ingestion.rabbitmq_event_source"):
        source._ensure_channel()

    assert channel.prefetch_count == 7
    assert "rabbitmq_consumer_ready" in caplog.text
    assert "queue=vigilante.recognition.frame_ingested" in caplog.text


def _frame(tmp_path: Path) -> Path:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"\xff\xd8test\xff\xd9")
    return frame_path


def _delivery(event: dict, *, headers: dict | None = None) -> RabbitMqDelivery:
    return RabbitMqDelivery(
        body=json.dumps(event).encode("utf-8"),
        delivery_tag=1,
        headers=headers or {},
    )


def _event(
    *,
    frame_ref: str,
    event_id: str = "evt_rabbitmq_test",
) -> dict:
    captured_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "event_id": event_id,
        "event_type": "frame.ingested",
        "event_version": "1.0",
        "occurred_at": captured_at,
        "payload": {
            "camera_id": CAMERA_ID,
            "captured_at": captured_at,
            "content_type": "image/jpeg",
            "frame_ref": frame_ref,
            "frame_uri": frame_ref,
            "height": 90,
            "metadata": {
                "capture_fps": 1.0,
                "sample_index": 0,
                "source_frame_index": 0,
                "source_timestamp_seconds": 0.0,
                "source_uri": "samples/cam01.mp4",
            },
            "quality_metadata": {
                "capture_fps": 1.0,
                "source_timestamp_seconds": 0.0,
            },
            "source_type": "video_file",
            "width": 160,
        },
        "context": {
            "correlation_id": "corr_rabbitmq_test",
            "idempotency_key": "frame:test",
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _RetriedMessage:
    def __init__(self, delivery: RabbitMqDelivery, retry_count: int) -> None:
        self.body = delivery.body
        self.headers = {**delivery.headers, "x-retry-count": retry_count}


class _FakeRabbitMqEventSource:
    def __init__(self, deliveries: list[RabbitMqDelivery]) -> None:
        self.deliveries = deliveries
        self.acked: list[int] = []
        self.rejected: list[int] = []
        self.nacked: list[tuple[int, bool]] = []
        self.retries: list[_RetriedMessage] = []
        self.closed = False

    def iter_deliveries(self, *, max_messages=None):
        for delivery in self.deliveries[:max_messages]:
            yield delivery

    def ack(self, delivery: RabbitMqDelivery) -> None:
        self.acked.append(delivery.delivery_tag)

    def reject_to_dlq(self, delivery: RabbitMqDelivery) -> None:
        self.rejected.append(delivery.delivery_tag)

    def nack(self, delivery: RabbitMqDelivery, *, requeue: bool) -> None:
        self.nacked.append((delivery.delivery_tag, requeue))

    def retry(self, delivery: RabbitMqDelivery, *, retry_count: int) -> None:
        self.retries.append(_RetriedMessage(delivery, retry_count))

    def close(self) -> None:
        self.closed = True


class _FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def fget_object(self, bucket: str, object_key: str, file_path: str) -> None:
        Path(file_path).write_bytes(self.objects[(bucket, object_key)])


class _FakeRabbitMqConnection:
    is_closed = False

    def __init__(self, channel: "_FakeRabbitMqChannel") -> None:
        self._channel = channel

    def channel(self) -> "_FakeRabbitMqChannel":
        return self._channel

    def close(self) -> None:
        self.is_closed = True


class _FakeRabbitMqChannel:
    is_open = True

    def __init__(self) -> None:
        self.prefetch_count: int | None = None

    def exchange_declare(self, **kwargs) -> None:
        pass

    def queue_declare(self, **kwargs) -> None:
        pass

    def queue_bind(self, **kwargs) -> None:
        pass

    def basic_qos(self, *, prefetch_count: int) -> None:
        self.prefetch_count = prefetch_count
