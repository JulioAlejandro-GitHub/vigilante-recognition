from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities import FaceDetectionResult
from app.models import EventOutbox, HumanTrack, ObservedSubject, RecognitionEvent


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

    def touch_subject(self, subject: ObservedSubject, *, seen_at: datetime, camera_id: UUID) -> ObservedSubject:
        subject.last_seen_at = seen_at
        subject.last_camera_id = camera_id
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
