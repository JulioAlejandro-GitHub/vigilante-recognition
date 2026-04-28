from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID as PyUUID

from sqlalchemy import Boolean, DateTime, Integer, Numeric, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ObservedSubject(Base):
    __tablename__ = "observed_subject"
    __table_args__ = {"schema": "recognition"}

    observed_subject_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    subject_code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    current_status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'unknown'::text"))
    risk_level: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'low'::text"))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_camera_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    last_camera_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    appearance_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    recurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    subject_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))


class HumanTrack(Base):
    __tablename__ = "human_track"
    __table_args__ = {"schema": "recognition"}

    human_track_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    camera_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    observed_subject_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    external_track_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    track_status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'candidate'::text"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trajectory_summary: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    person_presence_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    face_available: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    cumulative_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    evidence_window_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("15"))
    track_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))


class RecognitionEvent(Base):
    __tablename__ = "recognition_event"
    __table_args__ = {"schema": "recognition"}

    recognition_event_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    observed_subject_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    human_track_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    camera_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    event_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'low'::text"))
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    decision_reason: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    evidence_refs: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    requires_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    requires_case_evaluation: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class EventOutbox(Base):
    __tablename__ = "event_outbox"
    __table_args__ = {"schema": "outbox"}

    event_outbox_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    aggregate_type: Mapped[str] = mapped_column(Text, nullable=False)
    aggregate_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    headers: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    publish_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    publish_status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'::text"))
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class CrossCameraCorrelation(Base):
    __tablename__ = "cross_camera_correlation"
    __table_args__ = {"schema": "recognition"}

    cross_camera_correlation_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    source_subject_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    target_subject_id: Mapped[PyUUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    source_track_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    target_track_id: Mapped[Optional[PyUUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True, index=True)
    correlation_status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'auto'::text"))
    face_similarity_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    appearance_similarity_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    semantic_similarity_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    temporal_coherence_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    spatial_coherence_score: Mapped[Optional[float]] = mapped_column(Numeric(asdecimal=False), nullable=True)
    aggregate_score: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    signals_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    reviewed_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
