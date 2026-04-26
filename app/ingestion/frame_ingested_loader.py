from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from app.domain.entities import FrameIngestedMessage, InvalidCameraIdError
from app.ingestion.jsonl_event_source import JsonlEventSource
from app.storage.frame_resolver import LocalFrameResolver


class InvalidFrameIngestedEventError(ValueError):
    def __init__(self, *, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"Invalid frame.ingested event at {path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason


@dataclass(frozen=True)
class LoadedFrameIngestedEvent:
    line_number: int
    message: FrameIngestedMessage
    resolved_frame_path: Path


class FrameIngestedJsonlLoader:
    def __init__(
        self,
        path: Path | str,
        *,
        frame_resolver: LocalFrameResolver | None = None,
    ) -> None:
        self.path = Path(path)
        self.event_source = JsonlEventSource(self.path)
        self.frame_resolver = frame_resolver or LocalFrameResolver()

    def iter_messages(self) -> Iterator[LoadedFrameIngestedEvent]:
        for event in self.event_source.iter_events():
            try:
                message = FrameIngestedMessage.model_validate(event.payload)
                if message.event_type != "frame.ingested":
                    raise ValueError(f"unexpected event_type={message.event_type!r}")
                message.camera_uuid
            except (ValidationError, InvalidCameraIdError, ValueError) as exc:
                raise InvalidFrameIngestedEventError(
                    path=self.path,
                    line_number=event.line_number,
                    reason=str(exc),
                ) from exc

            resolved_message = self.frame_resolver.with_resolved_frame_ref(message)
            yield LoadedFrameIngestedEvent(
                line_number=event.line_number,
                message=resolved_message,
                resolved_frame_path=Path(resolved_message.frame_ref),
            )

