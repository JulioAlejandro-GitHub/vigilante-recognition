from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ObservedSubject(Base):
    __tablename__ = "observed_subject"
    __table_args__ = {"schema": "recognition"}

    observed_subject_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    subject_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    current_status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    recurrence_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class HumanTrack(Base):
    __tablename__ = "human_track"
    __table_args__ = {"schema": "recognition"}

    human_track_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    observed_subject_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("recognition.observed_subject.observed_subject_id"),
        nullable=False,
        index=True,
    )
    external_track_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    track_status: Mapped[str] = mapped_column(String(32), default="candidate", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    person_presence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    face_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class RecognitionEvent(Base):
    __tablename__ = "recognition_event"
    __table_args__ = {"schema": "recognition"}

    recognition_event_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    observed_subject_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)
    human_track_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    decision_reason: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    evidence_refs: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class EventOutbox(Base):
    __tablename__ = "event_outbox"
    __table_args__ = {"schema": "outbox"}

    outbox_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    publish_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
