from __future__ import annotations

from datetime import datetime

from app.config import settings
from app.domain.entities import (
    ContinuityResolution,
    CrossCameraAssessment,
    FaceDetectionResult,
    FaceEmbeddingResult,
    FaceMatchResult,
    FrameIngestedMessage,
    SupplementalRecognitionDecision,
)
from app.infra.repository import RecognitionRepository


class TrackService:
    def __init__(self, repo: RecognitionRepository) -> None:
        self.repo = repo

    def open_track_from_frame(self, message: FrameIngestedMessage):
        persisted_camera_id = message.camera_uuid
        external_track_key = message.context.get("correlation_id")
        track = self.repo.find_track_by_camera_and_external_key(
            camera_id=persisted_camera_id,
            external_track_key=external_track_key,
        )
        if track is None:
            subject = self.repo.create_subject(first_seen_at=message.captured_at, camera_id=persisted_camera_id)
            track = self.repo.create_track(
                camera_id=persisted_camera_id,
                subject_id=subject.observed_subject_id,
                started_at=message.captured_at,
                external_track_key=external_track_key,
            )
            is_new_appearance = True
        else:
            subject = self.repo.get_subject(track.observed_subject_id)
            if subject is None:
                subject = self.repo.create_subject(first_seen_at=message.captured_at, camera_id=persisted_camera_id)
                self.repo.attach_subject_to_track(track, subject_id=subject.observed_subject_id)
                is_new_appearance = True
            else:
                self.repo.touch_subject(subject, seen_at=message.captured_at, camera_id=persisted_camera_id)
                is_new_appearance = False
        self.repo.update_track_presence(track)
        return subject, track, is_new_appearance

    def confirm_basic_presence(self, track):
        for _ in range(max(0, settings.presence_confirmation_frames - 1)):
            track = self.repo.update_track_presence(track)
        return track

    def register_face_observation(
        self,
        *,
        track,
        face_detection: FaceDetectionResult,
        frame_ref: str,
        detected_at: datetime,
    ):
        return self.repo.update_track_face_observation(
            track,
            face_detection=face_detection,
            frame_ref=frame_ref,
            detected_at=detected_at,
        )

    def register_face_match(
        self,
        *,
        track,
        match_result: FaceMatchResult,
        matched_at: datetime,
    ):
        return self.repo.update_track_match_result(
            track,
            match_result=match_result,
            matched_at=matched_at,
        )

    def update_subject_face_profile(
        self,
        *,
        subject,
        camera_id,
        observed_at: datetime,
        frame_ref: str,
        face_detection: FaceDetectionResult,
        embedding_result: FaceEmbeddingResult | None = None,
        match_result: FaceMatchResult | None = None,
    ):
        return self.repo.update_subject_face_profile(
            subject,
            camera_id=camera_id,
            observed_at=observed_at,
            frame_ref=frame_ref,
            face_detection=face_detection,
            embedding_result=embedding_result,
            match_result=match_result,
        )

    def link_track_to_subject(
        self,
        *,
        track,
        source_subject,
        target_subject,
        resolved_at: datetime,
        payload: dict,
    ):
        self.repo.mark_subject_continuity(
            source_subject,
            outcome="correlated",
            resolved_at=resolved_at,
            payload=payload,
        )
        self.repo.attach_subject_to_track(track, subject_id=target_subject.observed_subject_id)
        target_subject = self.repo.touch_subject(
            target_subject,
            seen_at=resolved_at,
            camera_id=track.camera_id,
            increment_recurrence=True,
        )
        return target_subject, track

    def record_continuity_resolution(
        self,
        *,
        track,
        continuity_resolution: ContinuityResolution,
        resolved_at: datetime,
    ):
        serialized = {
            "outcome": continuity_resolution.outcome,
            "subject_id_to_use": continuity_resolution.subject_id_to_use,
            "target_track_id": continuity_resolution.target_track_id,
            "correlation_status": continuity_resolution.correlation_status,
            "requires_human_review": continuity_resolution.requires_human_review,
            "decision_reason": continuity_resolution.decision_reason,
            "payload": continuity_resolution.payload,
            "assessment": continuity_resolution.assessment.model_dump(mode="json") if continuity_resolution.assessment else None,
            "supplemental_decisions": [
                supplemental.model_dump(mode="json") for supplemental in continuity_resolution.supplemental_decisions
            ],
            "resolved_at": resolved_at.isoformat(),
        }
        return self.repo.update_track_continuity_resolution(track, continuity_resolution=serialized)

    def load_continuity_resolution(self, *, track):
        metadata = dict(track.track_metadata or {})
        stored = metadata.get("continuity_resolution")
        if stored:
            return ContinuityResolution.model_validate(stored)

        correlation = self.repo.find_latest_cross_camera_correlation_for_source_track(track.human_track_id)
        if correlation is None:
            return None

        payload = dict(correlation.signals_json or {})
        assessment_payload = payload.get("cross_camera_assessment")
        assessment = None
        if isinstance(assessment_payload, dict):
            required_keys = {"current_subject_id", "current_track_id", "current_camera_id"}
            if required_keys.issubset(assessment_payload):
                assessment = CrossCameraAssessment.model_validate(assessment_payload)
        subject_id_to_use = str(track.observed_subject_id) if track.observed_subject_id else None
        outcome = "manual_review_required"
        supplemental_decisions = [
            SupplementalRecognitionDecision(
                event_type="manual_review_required",
                severity="medium",
                confidence=float(correlation.aggregate_score),
                decision_reason=["continuity_resolution_reused"],
                payload=payload,
                subject_id=subject_id_to_use,
            )
        ]

        if correlation.correlation_status == "auto":
            outcome = "correlated"
            subject_id_to_use = str(correlation.target_subject_id)
            supplemental_decisions = [
                SupplementalRecognitionDecision(
                    event_type="cross_camera_subject_correlated",
                    severity="medium",
                    confidence=float(correlation.aggregate_score),
                    decision_reason=["continuity_resolution_reused"],
                    payload=payload,
                    subject_id=str(correlation.target_subject_id),
                )
            ]
        elif payload.get("conflict_type"):
            outcome = "identity_conflict"
            supplemental_decisions = [
                SupplementalRecognitionDecision(
                    event_type="identity_conflict",
                    severity="high",
                    confidence=float(correlation.aggregate_score),
                    decision_reason=["continuity_resolution_reused"],
                    payload=payload,
                    subject_id=subject_id_to_use,
                ),
                SupplementalRecognitionDecision(
                    event_type="manual_review_required",
                    severity="medium",
                    confidence=float(correlation.aggregate_score),
                    decision_reason=["continuity_resolution_reused"],
                    payload=payload,
                    subject_id=subject_id_to_use,
                ),
            ]

        continuity_resolution = ContinuityResolution(
            outcome=outcome,
            subject_id_to_use=subject_id_to_use,
            target_track_id=str(correlation.target_track_id) if correlation.target_track_id else None,
            correlation_status=correlation.correlation_status,
            requires_human_review=outcome != "correlated",
            decision_reason=["continuity_resolution_reused"],
            payload=payload,
            assessment=assessment,
            supplemental_decisions=supplemental_decisions,
        )
        self.record_continuity_resolution(
            track=track,
            continuity_resolution=continuity_resolution,
            resolved_at=correlation.created_at,
        )
        return continuity_resolution
