from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import EventOutbox, HumanTrack, ObservedSubject, RecognitionEvent


class RecognitionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

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
            frame_count=0,
        )
        self.session.add(track)
        self.session.flush()
        return track

    def update_track_presence(self, track: HumanTrack, *, increment_frames: int = 1, score_increment: float = 0.34) -> HumanTrack:
        track.frame_count += increment_frames
        track.person_presence_score = min(1.0, track.person_presence_score + score_increment)
        if track.frame_count >= 1:
            track.track_status = "probable_human"
        if track.frame_count >= 3:
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
