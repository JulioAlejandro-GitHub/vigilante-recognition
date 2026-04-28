from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.domain.entities import FrameIngestedMessage, InvalidCameraIdError


class RejectedFrameIngestedEvent(ValueError):
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


def validate_frame_ingested_event(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    line_number: int | None = None,
) -> FrameIngestedMessage:
    event_id = string_or_none(payload.get("event_id"))
    event_type = string_or_none(payload.get("event_type"))
    if not event_type:
        raise RejectedFrameIngestedEvent(
            reason="missing_event_type",
            event_id=event_id,
        )
    if event_type != "frame.ingested":
        raise RejectedFrameIngestedEvent(
            reason="unsupported_event_type",
            event_id=event_id,
            event_type=event_type,
            details={"supported_event_types": ["frame.ingested"]},
        )

    event_payload = payload.get("payload")
    if not isinstance(event_payload, dict):
        raise RejectedFrameIngestedEvent(
            reason="missing_payload",
            event_id=event_id,
            event_type=event_type,
        )
    if not event_payload.get("frame_ref") and not event_payload.get("frame_uri"):
        raise RejectedFrameIngestedEvent(
            reason="missing_frame_reference",
            event_id=event_id,
            event_type=event_type,
        )

    try:
        message = FrameIngestedMessage.model_validate(payload)
        message.camera_uuid
    except InvalidCameraIdError as exc:
        raise RejectedFrameIngestedEvent(
            reason="invalid_camera_id",
            event_id=event_id,
            event_type=event_type,
            details={"error": str(exc)},
        ) from exc
    except ValidationError as exc:
        details: dict[str, Any] = {"error": str(exc)}
        if source_path is not None:
            details["source_path"] = str(source_path)
        if line_number is not None:
            details["line_number"] = line_number
        raise RejectedFrameIngestedEvent(
            reason="malformed_frame_ingested_event",
            event_id=event_id,
            event_type=event_type,
            details=details,
        ) from exc

    return message


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
