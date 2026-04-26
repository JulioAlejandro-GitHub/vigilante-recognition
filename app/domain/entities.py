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
    frame_uri: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    content_type: Optional[str] = None
    source_type: Optional[str] = None
    quality_metadata: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class FaceEmbeddingResult(BaseModel):
    generated: bool = False
    backend: str
    dimensions: int = 0
    vector: list[float] = Field(default_factory=list)
    bbox: Optional[dict[str, int]] = None
    source_frame_ref: Optional[str] = None
    rejection_reasons: list[str] = Field(default_factory=list)


class SemanticDescriptorResult(BaseModel):
    generated: bool = False
    backend: str
    descriptor: dict[str, Any] = Field(default_factory=dict)
    signature: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source_frame_ref: Optional[str] = None
    rejection_reasons: list[str] = Field(default_factory=list)


class KnownFaceGalleryEntry(BaseModel):
    person_profile_id: str
    full_name: str
    person_type: str
    risk_level: str
    external_person_key: Optional[str] = None
    embedding: list[float] = Field(default_factory=list)
    embedding_backend: Optional[str] = None
    source_image_ref: Optional[str] = None
    gallery_source: str = "database_projection"


class FaceMatchCandidate(BaseModel):
    person_profile_id: str
    full_name: str
    person_type: str
    risk_level: str
    similarity: float
    gallery_source: str
    external_person_key: Optional[str] = None


class FaceMatchResult(BaseModel):
    identified: bool = False
    match_confidence: float = 0.0
    matching_strategy: str
    threshold: float
    second_best_margin_threshold: float
    evaluated_candidates: int = 0
    gallery_source: str = "none"
    best_match: Optional[FaceMatchCandidate] = None
    second_best_match: Optional[FaceMatchCandidate] = None
    best_similarity: float = 0.0
    second_best_similarity: float = 0.0
    second_best_margin: float = 0.0
    rejection_reasons: list[str] = Field(default_factory=list)


class CrossCameraCandidate(BaseModel):
    observed_subject_id: str
    latest_track_id: Optional[str] = None
    last_camera_id: Optional[str] = None
    last_seen_at: datetime
    recurrence_count: int = 1
    face_similarity_score: float
    temporal_coherence_score: float
    camera_switch_score: float
    aggregate_score: float
    resolved_identity: dict[str, Any] = Field(default_factory=dict)


class CrossCameraAssessment(BaseModel):
    current_subject_id: str
    current_track_id: str
    current_camera_id: str
    threshold: float
    manual_review_threshold: float
    second_best_margin_threshold: float
    evaluated_candidates: int = 0
    best_candidate: Optional[CrossCameraCandidate] = None
    second_best_candidate: Optional[CrossCameraCandidate] = None
    second_best_margin: float = 0.0
    decision_reason: list[str] = Field(default_factory=list)


class SupplementalRecognitionDecision(BaseModel):
    event_type: str
    severity: str
    confidence: float
    decision_reason: list[str]
    payload: dict[str, Any] = Field(default_factory=dict)
    subject_id: Optional[str] = None


class ContinuityResolution(BaseModel):
    outcome: str = "none"
    subject_id_to_use: Optional[str] = None
    target_track_id: Optional[str] = None
    correlation_status: Optional[str] = None
    requires_human_review: bool = False
    decision_reason: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    assessment: Optional[CrossCameraAssessment] = None
    supplemental_decisions: list[SupplementalRecognitionDecision] = Field(default_factory=list)


class RecurrentSubjectCandidate(BaseModel):
    observed_subject_id: str
    latest_track_id: Optional[str] = None
    last_camera_id: Optional[str] = None
    last_seen_at: datetime
    recurrence_count: int = 1
    semantic_similarity_score: float
    visual_similarity_score: float = 0.0
    temporal_coherence_score: float
    camera_relation_score: float
    aggregate_score: float
    descriptor_summary: dict[str, Any] = Field(default_factory=dict)


class RecurrentSubjectAssessment(BaseModel):
    current_subject_id: str
    current_track_id: str
    current_camera_id: str
    semantic_similarity_threshold: float
    recurrent_subject_threshold: float
    case_suggestion_threshold: float
    manual_review_threshold: float
    evaluated_candidates: int = 0
    best_candidate: Optional[RecurrentSubjectCandidate] = None
    second_best_candidate: Optional[RecurrentSubjectCandidate] = None
    second_best_margin: float = 0.0
    decision_reason: list[str] = Field(default_factory=list)


class RecurrentSubjectResolution(BaseModel):
    outcome: str = "none"
    subject_id_to_use: Optional[str] = None
    target_track_id: Optional[str] = None
    requires_human_review: bool = False
    requires_case_evaluation: bool = False
    decision_reason: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    assessment: Optional[RecurrentSubjectAssessment] = None
    supplemental_decisions: list[SupplementalRecognitionDecision] = Field(default_factory=list)
