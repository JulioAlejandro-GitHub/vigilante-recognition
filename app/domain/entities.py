from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class InvalidCameraIdError(ValueError):
    pass


class FramePayload(BaseModel):
    camera_id: str
    external_camera_key: Optional[str] = None
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
    def camera_uuid(self) -> UUID:
        try:
            return UUID(self.payload.camera_id)
        except ValueError as exc:
            raise InvalidCameraIdError(
                f"Invalid frame.ingested.payload.camera_id '{self.payload.camera_id}': expected canonical UUID from api.camera.camera_id."
            ) from exc

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
    payload: dict[str, Any] = Field(default_factory=dict)


class FaceDetectionResult(BaseModel):
    detected: bool = False
    usable: bool = False
    quality_score: float = 0.0
    bbox: Optional[dict[str, int]] = None
    image_size: Optional[dict[str, int]] = None
    rejection_reasons: list[str] = Field(default_factory=list)
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    frame_quality_metadata: dict[str, float] = Field(default_factory=dict)
