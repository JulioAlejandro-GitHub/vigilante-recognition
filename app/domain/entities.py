from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FramePayload(BaseModel):
    camera_id: str
    captured_at: datetime
    frame_ref: str
    quality_metadata: dict[str, float] = Field(default_factory=dict)


class FrameIngestedMessage(BaseModel):
    event_id: str
    event_type: str
    event_version: str
    occurred_at: datetime
    payload: FramePayload
    context: dict[str, Any]

    @property
    def camera_id(self) -> str:
        return self.payload.camera_id

    @property
    def captured_at(self) -> datetime:
        return self.payload.captured_at

    @property
    def frame_ref(self) -> str:
        return self.payload.frame_ref


class PresenceDecision(BaseModel):
    event_type: str
    severity: str
    confidence: float
    decision_reason: list[str]
