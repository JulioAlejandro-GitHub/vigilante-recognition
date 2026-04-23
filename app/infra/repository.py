from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult, FaceMatchResult, KnownFaceGalleryEntry
from app.models import CrossCameraCorrelation, EventOutbox, HumanTrack, ObservedSubject, RecognitionEvent


class RecognitionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def find_track_by_camera_and_external_key(
        self,
        *,
        camera_id: UUID,
        external_track_key: Optional[str],
    ) -> Optional[HumanTrack]:
        if not external_track_key:
            return None
        statement = select(HumanTrack).where(
            HumanTrack.camera_id == camera_id,
            HumanTrack.external_track_key == external_track_key,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def get_subject(self, subject_id: Optional[UUID]) -> Optional[ObservedSubject]:
        if subject_id is None:
            return None
        return self.session.get(ObservedSubject, subject_id)

    def create_subject(self, *, first_seen_at: datetime, camera_id: UUID) -> ObservedSubject:
        subject = ObservedSubject(
            subject_code=f"obs_{uuid4().hex[:12]}",
            first_seen_at=first_seen_at,
            last_seen_at=first_seen_at,
            first_camera_id=camera_id,
            last_camera_id=camera_id,
        )
        self.session.add(subject)
        self.session.flush()
        return subject

    def touch_subject(
        self,
        subject: ObservedSubject,
        *,
        seen_at: datetime,
        camera_id: UUID,
        increment_recurrence: bool = False,
    ) -> ObservedSubject:
        subject.last_seen_at = seen_at
        subject.last_camera_id = camera_id
        if increment_recurrence:
            subject.recurrence_count = int(subject.recurrence_count or 0) + 1
        self.session.add(subject)
        self.session.flush()
        return subject

    def create_track(
        self,
        *,
        camera_id: UUID,
        subject_id: UUID,
        started_at: datetime,
        external_track_key: str | None = None,
    ) -> HumanTrack:
        track = HumanTrack(
            camera_id=camera_id,
            observed_subject_id=subject_id,
            external_track_key=external_track_key,
            started_at=started_at,
            person_presence_score=0.0,
            face_available=False,
        )
        self.session.add(track)
        self.session.flush()
        return track

    def attach_subject_to_track(self, track: HumanTrack, *, subject_id: UUID) -> HumanTrack:
        track.observed_subject_id = subject_id
        self.session.add(track)
        self.session.flush()
        return track

    def update_track_presence(self, track: HumanTrack, *, score_increment: float = 0.34) -> HumanTrack:
        current_score = track.person_presence_score or 0.0
        track.person_presence_score = min(1.0, current_score + score_increment)
        if track.person_presence_score > 0.0:
            track.track_status = "probable_human"
        if track.person_presence_score >= 0.6:
            track.track_status = "confirmed_human"
        self.session.add(track)
        self.session.flush()
        return track

    def update_track_face_observation(
        self,
        track: HumanTrack,
        *,
        face_detection: FaceDetectionResult,
        frame_ref: str,
        detected_at: datetime,
    ) -> HumanTrack:
        metadata = dict(track.track_metadata or {})
        metadata["last_face_observation"] = {
            "attempted_at": detected_at.isoformat(),
            "frame_ref": frame_ref,
            "detected": face_detection.detected,
            "usable": face_detection.usable,
            "quality_score": face_detection.quality_score,
            "bbox": face_detection.bbox,
            "image_size": face_detection.image_size,
            "rejection_reasons": face_detection.rejection_reasons,
            "quality_metrics": face_detection.quality_metrics,
        }

        if face_detection.usable:
            best_face = metadata.get("best_face_observation", {})
            best_quality = float(best_face.get("quality_score", 0.0))
            if face_detection.quality_score >= best_quality:
                metadata["best_face_observation"] = metadata["last_face_observation"]
            track.face_available = True
            track.track_status = "face_unknown"
        elif face_detection.detected:
            track.face_available = False
            track.track_status = "face_low_quality"
        else:
            track.face_available = False
            track.track_status = "face_attempted"

        track.track_metadata = metadata
        self.session.add(track)
        self.session.flush()
        return track

    def update_track_match_result(
        self,
        track: HumanTrack,
        *,
        match_result: FaceMatchResult,
        matched_at: datetime,
    ) -> HumanTrack:
        metadata = dict(track.track_metadata or {})
        metadata["last_face_match"] = {
            "matched_at": matched_at.isoformat(),
            "identified": match_result.identified,
            "match_confidence": match_result.match_confidence,
            "matching_strategy": match_result.matching_strategy,
            "threshold": match_result.threshold,
            "second_best_margin_threshold": match_result.second_best_margin_threshold,
            "best_similarity": match_result.best_similarity,
            "second_best_similarity": match_result.second_best_similarity,
            "second_best_margin": match_result.second_best_margin,
            "rejection_reasons": match_result.rejection_reasons,
            "evaluated_candidates": match_result.evaluated_candidates,
        }
        if match_result.best_match is not None:
            metadata["last_face_match"]["best_match"] = {
                "person_profile_id": match_result.best_match.person_profile_id,
                "full_name": match_result.best_match.full_name,
                "external_person_key": match_result.best_match.external_person_key,
                "person_type": match_result.best_match.person_type,
                "risk_level": match_result.best_match.risk_level,
                "similarity": match_result.best_match.similarity,
                "gallery_source": match_result.best_match.gallery_source,
            }

        if match_result.identified:
            track.track_status = "identified"

        track.track_metadata = metadata
        self.session.add(track)
        self.session.flush()
        return track

    def update_subject_face_profile(
        self,
        subject: ObservedSubject,
        *,
        camera_id: UUID,
        observed_at: datetime,
        frame_ref: str,
        face_detection: FaceDetectionResult,
        embedding_result: FaceEmbeddingResult | None = None,
        match_result: FaceMatchResult | None = None,
    ) -> ObservedSubject:
        metadata = dict(subject.subject_metadata or {})
        metadata["last_observation"] = {
            "observed_at": observed_at.isoformat(),
            "camera_id": str(camera_id),
            "frame_ref": frame_ref,
            "face_quality_score": face_detection.quality_score,
        }
        if embedding_result and embedding_result.generated:
            metadata["last_face_embedding"] = embedding_result.vector
            metadata["last_face_embedding_backend"] = embedding_result.backend
            metadata["last_face_embedding_frame_ref"] = frame_ref
            best_quality = float(metadata.get("representative_face_quality_score", 0.0))
            if face_detection.quality_score >= best_quality:
                metadata["representative_face_embedding"] = embedding_result.vector
                metadata["representative_face_embedding_backend"] = embedding_result.backend
                metadata["representative_face_quality_score"] = face_detection.quality_score
                metadata["representative_face_frame_ref"] = frame_ref

        if match_result and match_result.identified and match_result.best_match is not None:
            metadata["resolved_identity"] = {
                "person_profile_id": match_result.best_match.person_profile_id,
                "external_person_key": match_result.best_match.external_person_key,
                "full_name": match_result.best_match.full_name,
                "person_type": match_result.best_match.person_type,
                "risk_level": match_result.best_match.risk_level,
                "confidence": match_result.match_confidence,
                "matching_strategy": match_result.matching_strategy,
                "resolved_at": observed_at.isoformat(),
            }
        elif match_result:
            metadata["last_match_attempt"] = {
                "match_confidence": match_result.match_confidence,
                "matching_strategy": match_result.matching_strategy,
                "best_similarity": match_result.best_similarity,
                "second_best_similarity": match_result.second_best_similarity,
                "second_best_margin": match_result.second_best_margin,
                "rejection_reasons": match_result.rejection_reasons,
                "attempted_at": observed_at.isoformat(),
            }

        appearance_summary = dict(subject.appearance_summary or {})
        appearance_summary["last_face_quality_score"] = face_detection.quality_score
        appearance_summary["last_frame_ref"] = frame_ref
        appearance_summary["embedding_backend"] = embedding_result.backend if embedding_result else appearance_summary.get("embedding_backend")

        subject.subject_metadata = metadata
        subject.appearance_summary = appearance_summary
        self.session.add(subject)
        self.session.flush()
        return subject

    def find_latest_track_for_subject(self, subject_id: UUID) -> Optional[HumanTrack]:
        statement = (
            select(HumanTrack)
            .where(HumanTrack.observed_subject_id == subject_id)
            .order_by(HumanTrack.started_at.desc())
        )
        return self.session.execute(statement).scalars().first()

    def load_recent_subject_candidates(
        self,
        *,
        exclude_subject_id: UUID,
        observed_at: datetime,
        window_seconds: int,
    ) -> list[dict[str, Any]]:
        cutoff = observed_at - timedelta(seconds=window_seconds)
        statement = (
            select(ObservedSubject)
            .where(
                ObservedSubject.observed_subject_id != exclude_subject_id,
                ObservedSubject.last_seen_at >= cutoff,
            )
            .order_by(ObservedSubject.last_seen_at.desc())
        )
        subjects = self.session.execute(statement).scalars().all()
        candidates: list[dict[str, Any]] = []
        for subject in subjects:
            metadata = dict(subject.subject_metadata or {})
            continuity_resolution = metadata.get("continuity_resolution", {})
            if continuity_resolution.get("outcome") == "correlated" and continuity_resolution.get("target_subject_id"):
                continue

            embedding = metadata.get("representative_face_embedding") or metadata.get("last_face_embedding")
            if not embedding:
                continue

            latest_track = self.find_latest_track_for_subject(subject.observed_subject_id)
            candidates.append(
                {
                    "subject": subject,
                    "latest_track": latest_track,
                    "embedding": embedding,
                    "resolved_identity": metadata.get("resolved_identity", {}),
                }
            )
        return candidates

    def mark_subject_continuity(
        self,
        subject: ObservedSubject,
        *,
        outcome: str,
        resolved_at: datetime,
        payload: dict[str, Any],
    ) -> ObservedSubject:
        metadata = dict(subject.subject_metadata or {})
        metadata["continuity_resolution"] = {
            "outcome": outcome,
            "resolved_at": resolved_at.isoformat(),
            **payload,
        }
        subject.subject_metadata = metadata
        self.session.add(subject)
        self.session.flush()
        return subject

    def add_cross_camera_correlation(
        self,
        *,
        source_subject_id: UUID,
        target_subject_id: UUID,
        source_track_id: Optional[UUID],
        target_track_id: Optional[UUID],
        correlation_status: str,
        face_similarity_score: float,
        temporal_coherence_score: float,
        aggregate_score: float,
        signals_json: dict[str, Any],
    ) -> CrossCameraCorrelation:
        correlation = CrossCameraCorrelation(
            source_subject_id=source_subject_id,
            target_subject_id=target_subject_id,
            source_track_id=source_track_id,
            target_track_id=target_track_id,
            correlation_status=correlation_status,
            face_similarity_score=face_similarity_score,
            temporal_coherence_score=temporal_coherence_score,
            aggregate_score=aggregate_score,
            signals_json=signals_json,
        )
        self.session.add(correlation)
        self.session.flush()
        return correlation

    def load_known_face_gallery_entries(self, *, embedding_backend: str) -> list[KnownFaceGalleryEntry]:
        statement = text(
            """
            SELECT DISTINCT ON (p.person_profile_id)
                p.person_profile_id::text AS person_profile_id,
                p.external_person_key,
                p.full_name,
                p.person_type,
                p.risk_level,
                em.model_key AS embedding_backend,
                e.source_image_ref,
                e.embedding::text AS embedding_text
            FROM recognition.person_profile_embedding_projection e
            JOIN recognition.person_profile_projection p
              ON p.person_profile_id = e.person_profile_id
            JOIN recognition.embedding_model em
              ON em.embedding_model_id = e.embedding_model_id
            WHERE p.is_match_enabled IS TRUE
              AND e.is_active IS TRUE
              AND em.model_key = :embedding_backend
            ORDER BY p.person_profile_id, COALESCE(e.quality_score, 0) DESC, e.created_at DESC
            """
        )
        rows = self.session.execute(statement, {"embedding_backend": embedding_backend}).mappings().all()
        entries: list[KnownFaceGalleryEntry] = []
        for row in rows:
            embedding = self._parse_pgvector_text(row["embedding_text"])
            if not embedding:
                continue

            entries.append(
                KnownFaceGalleryEntry(
                    person_profile_id=row["person_profile_id"],
                    external_person_key=row["external_person_key"],
                    full_name=row["full_name"],
                    person_type=row["person_type"],
                    risk_level=row["risk_level"],
                    embedding_backend=row["embedding_backend"],
                    source_image_ref=row["source_image_ref"],
                    embedding=embedding,
                    gallery_source="database_projection",
                )
            )
        return entries

    def add_recognition_event(
        self,
        *,
        subject_id: UUID,
        track_id: UUID,
        camera_id: UUID,
        event_type: str,
        event_ts: datetime,
        severity: str,
        confidence: float,
        decision_reason: list[str],
        evidence_refs: list[str],
        payload: dict[str, Any],
    ) -> RecognitionEvent:
        event = RecognitionEvent(
            observed_subject_id=subject_id,
            human_track_id=track_id,
            camera_id=camera_id,
            event_type=event_type,
            event_ts=event_ts,
            severity=severity,
            confidence=confidence,
            decision_reason=decision_reason,
            evidence_refs=evidence_refs,
            payload=payload,
            requires_human_review=bool(payload.get("requires_human_review", False)),
            requires_case_evaluation=False,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def add_outbox_event(self, *, aggregate_type: str, aggregate_id: UUID, event_type: str, payload: dict[str, Any]) -> EventOutbox:
        statement = select(EventOutbox).where(
            EventOutbox.aggregate_type == aggregate_type,
            EventOutbox.aggregate_id == aggregate_id,
            EventOutbox.event_type == event_type,
        )
        outbox = self.session.execute(statement).scalars().first()
        if outbox is None:
            outbox = EventOutbox(
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                event_type=event_type,
                payload=payload,
            )
        else:
            outbox.payload = payload
        self.session.add(outbox)
        self.session.flush()
        return outbox

    def _parse_pgvector_text(self, vector_text: str | None) -> list[float]:
        if not vector_text:
            return []

        stripped = vector_text.strip().strip("[]")
        if not stripped:
            return []

        return [float(value) for value in stripped.split(",")]
