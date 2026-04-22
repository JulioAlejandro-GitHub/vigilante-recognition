from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import CameraRef, EventOutbox, HumanTrack, ObservedSubject, RecognitionEvent


class RecognitionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def resolve_camera_id(self, external_camera_key: str) -> str:
        camera = self.session.query(CameraRef).filter_by(external_camera_key=external_camera_key).first()
        if not camera:
            raise ValueError(f"Camera mapping not found for {external_camera_key}")
        return camera.camera_id

    def create_subject(self, first_seen_at: datetime) -> ObservedSubject:
        subject = ObservedSubject(
            subject_code=f"obs_{uuid4().hex[:12]}",
            current_status="unknown",
            recurrence_count=1,
            first_seen_at=first_seen_at,
            last_seen_at=first_seen_at,
        )
        self.session.add(subject)
        self.session.flush()
        return subject

    def create_track(self, *, camera_id: str, subject_id: str, started_at: datetime, external_track_key: str | None = None) -> HumanTrack:
        track = HumanTrack(
            camera_id=camera_id,
            observed_subject_id=subject_id,
            external_track_key=external_track_key,
            track_status="candidate",
            started_at=started_at,
            person_presence_score=0.0,
            face_available=False,
        )
        self.session.add(track)
        self.session.flush()
        return track

    def update_track_presence(self, track: HumanTrack, *, score_increment: float = 0.34) -> HumanTrack:
        track.person_presence_score = min(1.0, track.person_presence_score + score_increment)
        if track.person_presence_score > 0.0:
            track.track_status = "probable_human"
        if track.person_presence_score >= 0.6:
            track.track_status = "confirmed_human"
        self.session.add(track)
        self.session.flush()
        return track

    def add_recognition_event(
        self,
        *,
        subject_id: str,
        track_id: str,
        camera_id: str,
        event_type: str,
        event_ts: datetime,
        severity: str,
        confidence: float,
        decision_reason: dict,
        evidence_refs: dict,
        payload: dict,
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
        )
        self.session.add(event)
        self.session.flush()
        return event

    def add_outbox_event(self, *, aggregate_type: str, aggregate_id: str, event_type: str, payload: dict) -> EventOutbox:
        outbox = EventOutbox(
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type=event_type,
            payload=payload,
            publish_status="pending",
            attempts=0,
        )
        self.session.add(outbox)
        self.session.flush()
        return outbox
