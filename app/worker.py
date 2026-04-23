from __future__ import annotations

import argparse
import logging

from app.config import settings
from app.consumer import load_fixture_message
from app.db import get_session, init_db
from app.domain.entities import InvalidCameraIdError
from app.domain.events import build_recognition_event
from app.infra.repository import RecognitionRepository
from app.logging import configure_logging
from app.publisher import EventPublisher
from app.services.presence_service import PresenceService
from app.services.track_service import TrackService

logger = logging.getLogger(__name__)


def process_fixture(fixture_path: str) -> dict:
    message = load_fixture_message(fixture_path)

    with get_session() as session:
        repo = RecognitionRepository(session)
        track_service = TrackService(repo)
        presence_service = PresenceService()
        publisher = EventPublisher()

        subject, track = track_service.open_track_from_frame(message)
        track = track_service.confirm_basic_presence(track)
        decision, face_detection = presence_service.decide(
            track=track,
            frame_ref=message.frame_ref,
            quality_metadata=message.payload.quality_metadata,
        )
        track_service.register_face_observation(
            track=track,
            face_detection=face_detection,
            frame_ref=message.frame_ref,
            detected_at=message.captured_at,
        )
        event = build_recognition_event(
            event_type=decision.event_type,
            camera_id=track.camera_id,
            track_id=track.human_track_id,
            subject_id=subject.observed_subject_id,
            severity=decision.severity,
            confidence=decision.confidence,
            decision_reason=decision.decision_reason,
            frame_ref=message.frame_ref,
            payload_details=decision.payload,
        )

        recognition_event = repo.add_recognition_event(
            subject_id=subject.observed_subject_id,
            track_id=track.human_track_id,
            camera_id=track.camera_id,
            event_type=decision.event_type,
            event_ts=message.captured_at,
            severity=decision.severity,
            confidence=decision.confidence,
            decision_reason=decision.decision_reason,
            evidence_refs=[message.frame_ref],
            payload=event["payload"],
        )
        repo.add_outbox_event(
            aggregate_type="recognition_event",
            aggregate_id=recognition_event.recognition_event_id,
            event_type=decision.event_type,
            payload=event,
        )
        session.commit()

        publisher.publish(event)
        return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap worker for vigilante-recognition slice 2")
    parser.add_argument("--fixture", required=True, help="Path to frame.ingested example JSON")
    args = parser.parse_args()

    configure_logging(settings.log_level)
    init_db()
    try:
        event = process_fixture(args.fixture)
    except InvalidCameraIdError as exc:
        logger.error("worker_failed reason=%s", exc)
        raise SystemExit(2) from exc

    logger.info("worker_finished event_type=%s track_id=%s", event["event_type"], event["context"]["track_id"])


if __name__ == "__main__":
    main()
