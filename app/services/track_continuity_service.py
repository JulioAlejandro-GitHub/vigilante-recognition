from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from typing import Any

from app.domain.entities import FrameIngestedMessage


@dataclass(frozen=True)
class _RecentTrackContext:
    track_key: str
    last_captured_at: datetime
    last_sample_index: int | None
    last_source_frame_index: int | None


class TrackContinuityService:
    """Assign a stable local track key to temporally adjacent frames."""

    def __init__(self, *, window_seconds: int) -> None:
        self.window_seconds = max(0, int(window_seconds))
        self._recent_by_source: dict[str, _RecentTrackContext] = {}

    def apply(self, message: FrameIngestedMessage) -> FrameIngestedMessage:
        source_key = self._source_key(message)
        sample_index = self._int_metadata(message, "sample_index")
        source_frame_index = self._int_metadata(message, "source_frame_index")
        source_timestamp_seconds = self._float_metadata(message, "source_timestamp_seconds")
        recent = self._recent_by_source.get(source_key)

        if recent and self._can_reuse_recent(
            message=message,
            recent=recent,
            sample_index=sample_index,
            source_frame_index=source_frame_index,
        ):
            track_key = recent.track_key
            continuity_status = "reused_recent_track"
        else:
            track_key = self._new_track_key(
                message=message,
                source_key=source_key,
                source_timestamp_seconds=source_timestamp_seconds,
            )
            continuity_status = "opened_local_track"

        self._recent_by_source[source_key] = _RecentTrackContext(
            track_key=track_key,
            last_captured_at=message.captured_at,
            last_sample_index=sample_index,
            last_source_frame_index=source_frame_index,
        )

        context = dict(message.context or {})
        original_correlation_id = context.get("correlation_id")
        if original_correlation_id != track_key:
            context.setdefault("original_correlation_id", original_correlation_id)
            context["correlation_id"] = track_key
        context["track_continuity"] = {
            "strategy": "same_camera_source_temporal_window_v1",
            "status": continuity_status,
            "window_seconds": self.window_seconds,
            "source_key": source_key,
            "sample_index": sample_index,
            "source_frame_index": source_frame_index,
            "source_timestamp_seconds": source_timestamp_seconds,
            "time_bucket": self._time_bucket(
                message=message,
                source_timestamp_seconds=source_timestamp_seconds,
            ),
        }
        return message.model_copy(update={"context": context})

    def _can_reuse_recent(
        self,
        *,
        message: FrameIngestedMessage,
        recent: _RecentTrackContext,
        sample_index: int | None,
        source_frame_index: int | None,
    ) -> bool:
        delta = abs((message.captured_at - recent.last_captured_at).total_seconds())
        if delta > self.window_seconds:
            return False
        if sample_index is not None and recent.last_sample_index is not None:
            if sample_index < recent.last_sample_index:
                return False
        if source_frame_index is not None and recent.last_source_frame_index is not None:
            if source_frame_index < recent.last_source_frame_index:
                return False
        return True

    def _source_key(self, message: FrameIngestedMessage) -> str:
        metadata = dict(message.payload.metadata or {})
        source_uri = metadata.get("source_uri") or message.payload.frame_uri or message.payload.frame_ref
        source_type = message.payload.source_type or "unknown_source"
        external_camera_key = message.payload.external_camera_key or ""
        return "|".join(
            [
                str(message.camera_uuid),
                str(source_type),
                str(source_uri),
                str(external_camera_key),
            ]
        )

    def _new_track_key(
        self,
        *,
        message: FrameIngestedMessage,
        source_key: str,
        source_timestamp_seconds: float | None,
    ) -> str:
        anchor = self._time_bucket(
            message=message,
            source_timestamp_seconds=source_timestamp_seconds,
        )
        digest = sha1(f"{source_key}|{anchor}".encode("utf-8")).hexdigest()[:20]
        return f"local_track_{digest}"

    def _time_bucket(
        self,
        *,
        message: FrameIngestedMessage,
        source_timestamp_seconds: float | None,
    ) -> str:
        window = max(1, self.window_seconds)
        if source_timestamp_seconds is not None:
            return f"source_ts:{int(source_timestamp_seconds // window)}"
        return f"captured_at:{int(message.captured_at.timestamp() // window)}"

    def _int_metadata(self, message: FrameIngestedMessage, key: str) -> int | None:
        metadata: dict[str, Any] = dict(message.payload.metadata or {})
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _float_metadata(self, message: FrameIngestedMessage, key: str) -> float | None:
        metadata: dict[str, Any] = dict(message.payload.metadata or {})
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
