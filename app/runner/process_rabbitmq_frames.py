from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.correlation import extract_run_id
from app.domain.entities import FrameIngestedMessage
from app.ingestion.event_deduper import FileEventDeduper
from app.ingestion.frame_ingested_validator import RejectedFrameIngestedEvent, string_or_none, validate_frame_ingested_event
from app.ingestion.rabbitmq_event_source import RabbitMqDelivery, RabbitMqEventSource
from app.ingestion.rejected_event_store import RejectedEventStore
from app.messaging.topology import FrameIngestedTopology
from app.services.track_continuity_service import TrackContinuityService
from app.storage.factory import build_frame_resolver
from app.storage.frame_resolver import FrameResolutionError, FrameResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessRabbitMqFramesResult:
    queue_name: str
    consumed: int
    processed: int
    acked: int
    retried: int
    rejected_to_dlq: int
    skipped_duplicate: int
    invalid_messages: int
    frame_resolution_errors: int
    processing_errors: int
    emitted_events: list[dict] = field(default_factory=list)
    deduper_path: Path | None = None
    rejected_events_path: Path | None = None


def process_rabbitmq_frames(
    *,
    processor: Callable[[FrameIngestedMessage], dict],
    event_source: RabbitMqEventSource | None = None,
    frame_search_roots: list[Path] | None = None,
    event_deduper: FileEventDeduper | None = None,
    rejected_event_store: RejectedEventStore | None = None,
    frame_resolver: FrameResolver | None = None,
    track_continuity_service: TrackContinuityService | None = None,
    topology: FrameIngestedTopology | None = None,
    max_messages: int | None = None,
    retry_limit: int | None = None,
) -> ProcessRabbitMqFramesResult:
    topology = topology or _topology_from_settings()
    source = event_source or _event_source_from_settings(topology)
    owns_source = event_source is None
    event_deduper = event_deduper or FileEventDeduper(settings.ingestion_deduper_path)
    rejected_event_store = rejected_event_store or RejectedEventStore(settings.ingestion_rejected_events_path)
    frame_resolver = frame_resolver or build_frame_resolver(frame_search_roots=frame_search_roots)
    track_continuity_service = track_continuity_service or TrackContinuityService(
        window_seconds=settings.ingestion_track_continuity_window_seconds,
    )
    retry_limit = settings.rabbitmq_retry_limit if retry_limit is None else max(0, int(retry_limit))

    emitted_events: list[dict] = []
    consumed = 0
    processed = 0
    acked = 0
    retried = 0
    rejected_to_dlq = 0
    skipped_duplicate = 0
    invalid_messages = 0
    frame_resolution_errors = 0
    processing_errors = 0

    try:
        for delivery in source.iter_deliveries(max_messages=max_messages):
            consumed += 1
            raw_body = _raw_body(delivery.body)
            event_id: str | None = None
            event_type: str | None = None
            try:
                payload = _decode_payload(delivery.body)
                event_id = string_or_none(payload.get("event_id"))
                event_type = string_or_none(payload.get("event_type"))
                run_id = extract_run_id(payload) or ""
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "rabbitmq_frame_payload queue=%s delivery_tag=%s event_id=%s payload=%s",
                        topology.recognition_queue,
                        delivery.delivery_tag,
                        event_id,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    )

                if event_id and event_deduper.has_processed(event_id):
                    skipped_duplicate += 1
                    source.ack(delivery)
                    acked += 1
                    logger.info(
                        "rabbitmq_frame_skipped_duplicate queue=%s delivery_tag=%s event_id=%s run_id=%s",
                        topology.recognition_queue,
                        delivery.delivery_tag,
                        event_id,
                        run_id,
                    )
                    continue

                message = validate_frame_ingested_event(payload)
                resolved_message = frame_resolver.with_resolved_frame_ref(message)
                continuity_message = track_continuity_service.apply(resolved_message)
                emitted_events.append(processor(continuity_message))
                if event_id:
                    event_deduper.mark_processed(
                        event_id,
                        source_path=f"rabbitmq:{topology.exchange}:{topology.routing_key}",
                    )
                source.ack(delivery)
                consumed_event_id = event_id or "<missing>"
                processed += 1
                acked += 1
                logger.info(
                    "rabbitmq_frame_acked queue=%s delivery_tag=%s event_id=%s run_id=%s processed=%s",
                    topology.recognition_queue,
                    delivery.delivery_tag,
                    consumed_event_id,
                    run_id,
                    processed,
                )
            except UnicodeDecodeError as exc:
                invalid_messages += 1
                rejected_to_dlq += 1
                _reject_local(
                    rejected_event_store,
                    reason="invalid_utf8",
                    event_id=event_id,
                    event_type=event_type,
                    delivery=delivery,
                    details={"error": str(exc)},
                    raw_body=raw_body,
                    topology=topology,
                )
                source.reject_to_dlq(delivery)
            except json.JSONDecodeError as exc:
                invalid_messages += 1
                rejected_to_dlq += 1
                _reject_local(
                    rejected_event_store,
                    reason="invalid_json",
                    event_id=event_id,
                    event_type=event_type,
                    delivery=delivery,
                    details={"error": exc.msg},
                    raw_body=raw_body,
                    topology=topology,
                )
                source.reject_to_dlq(delivery)
            except RejectedFrameIngestedEvent as exc:
                invalid_messages += 1
                rejected_to_dlq += 1
                _reject_local(
                    rejected_event_store,
                    reason=exc.reason,
                    event_id=exc.event_id,
                    event_type=exc.event_type,
                    delivery=delivery,
                    details=exc.details,
                    raw_body=raw_body,
                    topology=topology,
                )
                source.reject_to_dlq(delivery)
            except FrameResolutionError as exc:
                frame_resolution_errors += 1
                rejected_to_dlq += 1
                _reject_local(
                    rejected_event_store,
                    reason="frame_resolution_failed",
                    event_id=event_id,
                    event_type=event_type,
                    delivery=delivery,
                    details={
                        "error": str(exc),
                        "reason": exc.reason,
                        "frame_refs": exc.frame_refs,
                        "attempted_paths": [str(path) for path in exc.attempted_paths],
                        "attempted_locations": exc.attempted_locations,
                        "resolver_details": exc.details,
                    },
                    raw_body=raw_body,
                    topology=topology,
                )
                source.reject_to_dlq(delivery)
            except Exception as exc:  # pragma: no cover - covered by targeted retry tests with fake source
                processing_errors += 1
                current_retry_count = _retry_count(delivery)
                if current_retry_count < retry_limit:
                    next_retry_count = current_retry_count + 1
                    source.retry(delivery, retry_count=next_retry_count)
                    source.ack(delivery)
                    retried += 1
                    acked += 1
                    logger.warning(
                        "rabbitmq_frame_retried queue=%s delivery_tag=%s event_id=%s retry_count=%s error=%s",
                        topology.recognition_queue,
                        delivery.delivery_tag,
                        event_id,
                        next_retry_count,
                        type(exc).__name__,
                    )
                else:
                    rejected_to_dlq += 1
                    logger.exception(
                        "rabbitmq_frame_processing_failed_retries_exhausted queue=%s delivery_tag=%s event_id=%s",
                        topology.recognition_queue,
                        delivery.delivery_tag,
                        event_id,
                    )
                    _reject_local(
                        rejected_event_store,
                        reason="processing_failed_retries_exhausted",
                        event_id=event_id,
                        event_type=event_type,
                        delivery=delivery,
                        details={"error": str(exc), "error_type": type(exc).__name__, "retry_count": current_retry_count},
                        raw_body=raw_body,
                        topology=topology,
                    )
                    source.reject_to_dlq(delivery)
    finally:
        if owns_source:
            source.close()

    return ProcessRabbitMqFramesResult(
        queue_name=topology.recognition_queue,
        consumed=consumed,
        processed=processed,
        acked=acked,
        retried=retried,
        rejected_to_dlq=rejected_to_dlq,
        skipped_duplicate=skipped_duplicate,
        invalid_messages=invalid_messages,
        frame_resolution_errors=frame_resolution_errors,
        processing_errors=processing_errors,
        emitted_events=emitted_events,
        deduper_path=getattr(event_deduper, "path", None),
        rejected_events_path=getattr(rejected_event_store, "path", None),
    )


def _decode_payload(body: bytes) -> dict[str, Any]:
    decoded = body.decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise RejectedFrameIngestedEvent(reason="event_not_object")
    return payload


def _raw_body(body: bytes) -> str | None:
    try:
        return body.decode("utf-8")[:4000]
    except UnicodeDecodeError:
        return None


def _retry_count(delivery: RabbitMqDelivery) -> int:
    try:
        return max(0, int((delivery.headers or {}).get("x-retry-count", 0)))
    except (TypeError, ValueError):
        return 0


def _reject_local(
    rejected_event_store: RejectedEventStore,
    *,
    reason: str,
    event_id: str | None,
    event_type: str | None,
    delivery: RabbitMqDelivery,
    details: dict[str, Any] | None,
    raw_body: str | None,
    topology: FrameIngestedTopology,
) -> None:
    logger.warning(
        "rabbitmq_frame_rejected_to_dlq queue=%s delivery_tag=%s event_id=%s reason=%s",
        topology.recognition_queue,
        delivery.delivery_tag,
        event_id,
        reason,
    )
    logger.debug(
        "rabbitmq_frame_rejection_detail queue=%s delivery_tag=%s event_id=%s event_type=%s details=%s raw_body=%s",
        topology.recognition_queue,
        delivery.delivery_tag,
        event_id,
        event_type,
        details or {},
        raw_body or "",
    )
    rejected_event_store.append(
        reason=reason,
        source_path=f"rabbitmq:{topology.exchange}:{topology.routing_key}:{topology.recognition_queue}",
        event_id=event_id,
        event_type=event_type,
        details={
            **(details or {}),
            "delivery_tag": delivery.delivery_tag,
            "redelivered": delivery.redelivered,
            "retry_count": _retry_count(delivery),
            "broker_dlq": topology.dead_letter_queue,
        },
        raw_line=raw_body,
    )


def _event_source_from_settings(topology: FrameIngestedTopology) -> RabbitMqEventSource:
    return RabbitMqEventSource(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        username=settings.rabbitmq_user,
        password=settings.rabbitmq_password,
        virtual_host=settings.rabbitmq_vhost,
        topology=topology,
        prefetch_count=settings.rabbitmq_prefetch_count,
        idle_timeout_seconds=settings.rabbitmq_idle_timeout_seconds,
    )


def _topology_from_settings() -> FrameIngestedTopology:
    return FrameIngestedTopology(
        exchange=settings.rabbitmq_frame_exchange,
        routing_key=settings.rabbitmq_frame_routing_key,
        recognition_queue=settings.rabbitmq_frame_queue_name,
        dead_letter_exchange=settings.rabbitmq_frame_dlx,
        dead_letter_queue=settings.rabbitmq_frame_dlq,
        dead_letter_routing_key=settings.rabbitmq_frame_dlq_routing_key,
    )
