from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class FrameIngestedMessage:
    event_id: str
    event_type: str
    event_version: str
    occurred_at: str
    payload: dict[str, Any]
    context: dict[str, Any]

    @property
    def camera_id(self) -> str:
        return str(self.payload["camera_id"])

    @property
    def captured_at(self) -> datetime:
        return datetime.fromisoformat(self.payload["captured_at"].replace("Z", "+00:00"))

    @property
    def frame_ref(self) -> str:
        return str(self.payload["frame_ref"])


@dataclass(slots=True)
class PresenceDecision:
    event_type: str
    severity: str
    confidence: float
    decision_reason: list[str]
