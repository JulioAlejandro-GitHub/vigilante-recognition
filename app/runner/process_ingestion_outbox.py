from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from app.config import settings
from app.domain.entities import FrameIngestedMessage, InvalidCameraIdError
from app.ingestion.checkpoint_store import FileCheckpointStore
from app.ingestion.event_deduper import FileEventDeduper
from app.ingestion.rejected_event_store import RejectedEventStore
from app.storage.frame_resolver import LocalFrameResolver
from app.services.track_continuity_service import TrackContinuityService
from app.storage.frame_resolver import FrameResolutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessIngestionOutboxResult:
    source_path: Path
    read: int
    processed: int
    emitted_events: list[dict] = field(default_factory=list)
    skipped_checkpoint: int = 0
    skipped_duplicate: int = 0
    rejected: int = 0
    frame_resolution_errors: int = 0
    processing_errors: int = 0
    start_offset: int = 0
    final_offset: int = 0
    checkpoint_path: Path | None = None
    deduper_path: Path | None = None
    rejected_events_path: Path | None = None


@dataclass(frozen=True)
class _JsonlLine:
    line_number: int
    offset: int
    next_offset: int
    raw: bytes


def process_ingestion_outbox(
    jsonl_path: Path | str,
    *,
    processor: Callable[[FrameIngestedMessage], dict],
    frame_search_roots: list[Path] | None = None,
    checkpoint_store: FileCheckpointStore | None = None,
    event_deduper: FileEventDeduper | None = None,
    rejected_event_store: RejectedEventStore | None = None,
    force_replay: bool = False,
    track_continuity_service: TrackContinuityService | None = None,
) -> ProcessIngestionOutboxResult:
    path = Path(jsonl_path).expanduser()
    resolved_source_path = path.resolve(strict=False)
    checkpoint_store = checkpoint_store or FileCheckpointStore(settings.ingestion_checkpoint_path)
    event_deduper = event_deduper or FileEventDeduper(settings.ingestion_deduper_path)
    rejected_event_store = rejected_event_store or RejectedEventStore(settings.ingestion_rejected_events_path)
    frame_resolver = LocalFrameResolver(search_roots=frame_search_roots)
    track_continuity_service = track_continuity_service or TrackContinuityService(
        window_seconds=settings.ingestion_track_continuity_window_seconds,
    )

    start_offset = 0 if force_replay else checkpoint_store.get_offset(resolved_source_path)
    file_size = path.stat().st_size
    if start_offset > file_size:
        logger.warning(
            "ingestion_checkpoint_past_eof source_path=%s checkpoint_offset=%s file_size=%s resetting_offset=0",
            resolved_source_path,
            start_offset,
            file_size,
        )
        start_offset = 0

    skipped_checkpoint = 0 if force_replay else _count_lines_before_offset(path, start_offset)
    run_seen_event_ids: set[str] = set()
    emitted_events: list[dict] = []
    read = 0
    skipped_duplicate = 0
    rejected = 0
    frame_resolution_errors = 0
    processing_errors = 0
    final_offset = start_offset

    for line in _iter_jsonl_lines(path, start_offset=start_offset):
        final_offset = line.next_offset
        raw_line_for_rejection: str | None = None
        event_id: str | None = None
        event_type: str | None = None
        try:
            decoded_line = line.raw.decode("utf-8")
            stripped_line = decoded_line.strip()
            if not stripped_line:
                continue
            raw_line_for_rejection = stripped_line
            read += 1

            try:
                payload = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                rejected += 1
                _reject(
                    rejected_event_store,
                    reason="invalid_json",
                    source_path=resolved_source_path,
                    line_number=line.line_number,
                    offset=line.offset,
                    details={"error": exc.msg},
                    raw_line=raw_line_for_rejection,
                )
                continue

            if not isinstance(payload, dict):
                rejected += 1
                _reject(
                    rejected_event_store,
                    reason="event_not_object",
                    source_path=resolved_source_path,
                    line_number=line.line_number,
                    offset=line.offset,
                    raw_line=raw_line_for_rejection,
                )
                continue

            event_id = _string_or_none(payload.get("event_id"))
            event_type = _string_or_none(payload.get("event_type"))
            if event_id and (event_id in run_seen_event_ids or (not force_replay and event_deduper.has_processed(event_id))):
                skipped_duplicate += 1
                logger.info(
                    "ingestion_event_skipped_duplicate source_path=%s line_number=%s event_id=%s",
                    resolved_source_path,
                    line.line_number,
                    event_id,
                )
                continue

            message = _validate_frame_ingested_message(
                payload,
                source_path=resolved_source_path,
                line_number=line.line_number,
            )
            resolved_message = frame_resolver.with_resolved_frame_ref(message)
            continuity_message = track_continuity_service.apply(resolved_message)
            emitted_events.append(processor(continuity_message))
            if event_id:
                run_seen_event_ids.add(event_id)
                event_deduper.mark_processed(
                    event_id,
                    source_path=resolved_source_path,
                    line_number=line.line_number,
                )
        except UnicodeDecodeError as exc:
            rejected += 1
            _reject(
                rejected_event_store,
                reason="invalid_utf8",
                source_path=resolved_source_path,
                line_number=line.line_number,
                offset=line.offset,
                details={"error": str(exc)},
            )
        except _RejectedFrameIngestedEvent as exc:
            rejected += 1
            _reject(
                rejected_event_store,
                reason=exc.reason,
                source_path=resolved_source_path,
                line_number=line.line_number,
                offset=line.offset,
                event_id=exc.event_id,
                event_type=exc.event_type,
                details=exc.details,
                raw_line=raw_line_for_rejection,
            )
        except FrameResolutionError as exc:
            rejected += 1
            frame_resolution_errors += 1
            _reject(
                rejected_event_store,
                reason="frame_resolution_failed",
                source_path=resolved_source_path,
                line_number=line.line_number,
                offset=line.offset,
                event_id=event_id,
                event_type=event_type,
                details={
                    "error": str(exc),
                    "frame_refs": exc.frame_refs,
                    "attempted_paths": [str(path) for path in exc.attempted_paths],
                },
                raw_line=raw_line_for_rejection,
            )
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            rejected += 1
            processing_errors += 1
            logger.exception(
                "ingestion_event_processing_failed source_path=%s line_number=%s error=%s",
                resolved_source_path,
                line.line_number,
                type(exc).__name__,
            )
            _reject(
                rejected_event_store,
                reason="processing_failed",
                source_path=resolved_source_path,
                line_number=line.line_number,
                offset=line.offset,
                event_id=event_id,
                event_type=event_type,
                details={"error": str(exc), "error_type": type(exc).__name__},
                raw_line=raw_line_for_rejection,
            )
        finally:
            checkpoint_store.mark_consumed(
                resolved_source_path,
                offset=line.next_offset,
                line_number=line.line_number,
            )

    return ProcessIngestionOutboxResult(
        source_path=resolved_source_path,
        read=read,
        processed=len(emitted_events),
        emitted_events=emitted_events,
        skipped_checkpoint=skipped_checkpoint,
        skipped_duplicate=skipped_duplicate,
        rejected=rejected,
        frame_resolution_errors=frame_resolution_errors,
        processing_errors=processing_errors,
        start_offset=start_offset,
        final_offset=final_offset,
        checkpoint_path=getattr(checkpoint_store, "path", None),
        deduper_path=getattr(event_deduper, "path", None),
        rejected_events_path=getattr(rejected_event_store, "path", None),
    )


class _RejectedFrameIngestedEvent(ValueError):
    def __init__(
        self,
        *,
        reason: str,
        event_id: str | None = None,
        event_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.event_id = event_id
        self.event_type = event_type
        self.details = details or {}


def _iter_jsonl_lines(path: Path, *, start_offset: int) -> Iterator[_JsonlLine]:
    line_number = _count_lines_before_offset(path, start_offset) + 1
    with path.open("rb") as event_file:
        event_file.seek(start_offset)
        while True:
            offset = event_file.tell()
            raw = event_file.readline()
            if not raw:
                break
            next_offset = event_file.tell()
            yield (
                _JsonlLine(
                    line_number=line_number,
                    offset=offset,
                    next_offset=next_offset,
                    raw=raw,
                )
            )
            line_number += 1


def _count_lines_before_offset(path: Path, offset: int) -> int:
    if offset <= 0:
        return 0
    count = 0
    remaining = offset
    with path.open("rb") as event_file:
        while remaining > 0:
            chunk = event_file.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            count += chunk.count(b"\n")
            remaining -= len(chunk)
    return count


def _validate_frame_ingested_message(
    payload: dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
) -> FrameIngestedMessage:
    event_id = _string_or_none(payload.get("event_id"))
    event_type = _string_or_none(payload.get("event_type"))
    if not event_type:
        raise _RejectedFrameIngestedEvent(
            reason="missing_event_type",
            event_id=event_id,
        )
    if event_type != "frame.ingested":
        raise _RejectedFrameIngestedEvent(
            reason="unsupported_event_type",
            event_id=event_id,
            event_type=event_type,
            details={"supported_event_types": ["frame.ingested"]},
        )

    event_payload = payload.get("payload")
    if not isinstance(event_payload, dict):
        raise _RejectedFrameIngestedEvent(
            reason="missing_payload",
            event_id=event_id,
            event_type=event_type,
        )
    if not event_payload.get("frame_ref") and not event_payload.get("frame_uri"):
        raise _RejectedFrameIngestedEvent(
            reason="missing_frame_reference",
            event_id=event_id,
            event_type=event_type,
        )

    try:
        message = FrameIngestedMessage.model_validate(payload)
        message.camera_uuid
    except InvalidCameraIdError as exc:
        raise _RejectedFrameIngestedEvent(
            reason="invalid_camera_id",
            event_id=event_id,
            event_type=event_type,
            details={"error": str(exc)},
        ) from exc
    except ValidationError as exc:
        raise _RejectedFrameIngestedEvent(
            reason="malformed_frame_ingested_event",
            event_id=event_id,
            event_type=event_type,
            details={"error": str(exc), "source_path": str(source_path), "line_number": line_number},
        ) from exc

    return message


def _reject(
    rejected_event_store: RejectedEventStore,
    *,
    reason: str,
    source_path: Path,
    line_number: int,
    offset: int,
    event_id: str | None = None,
    event_type: str | None = None,
    details: dict[str, Any] | None = None,
    raw_line: str | None = None,
) -> None:
    logger.warning(
        "ingestion_event_rejected source_path=%s line_number=%s event_id=%s reason=%s",
        source_path,
        line_number,
        event_id,
        reason,
    )
    rejected_event_store.append(
        reason=reason,
        source_path=source_path,
        line_number=line_number,
        offset=offset,
        event_id=event_id,
        event_type=event_type,
        details=details,
        raw_line=raw_line,
    )


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
