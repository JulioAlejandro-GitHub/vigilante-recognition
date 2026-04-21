from __future__ import annotations

import argparse
import logging

from app.config import settings
from app.consumer import load_fixture_message
from app.db import get_session
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

        repo.update_track_presence(track)
        repo.update_track_presence(track)

        decision = presence_service.decide(track)
        event = build_recognition_event(
            event_type=decision.event_type,
            camera_id=message.camera_id,
            track_id=track.human_track_id,
            subject_id=subject.observed_subject_id,
            severity=decision.severity,
            confidence=decision.confidence,
            decision_reason=decision.decision_reason,
            frame_ref=message.frame_ref,
        )

        repo.add_recognition_event(
            subject_id=subject.observed_subject_id,
            track_id=track.human_track_id,
            camera_id=message.camera_id,
            event_type=decision.event_type,
            event_ts=message.captured_at,
            severity=decision.severity,
            confidence=decision.confidence,
            decision_reason={"reasons": decision.decision_reason},
            evidence_refs={"frames": [message.frame_ref]},
            payload=event["payload"],
        )
        repo.add_outbox_event(
            aggregate_type="recognition_event",
            aggregate_id=track.human_track_id,
            event_type=decision.event_type,
            payload=event,
        )
        session.commit()

        publisher.publish(event)
        return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap worker for vigilante-recognition slice 1")
    parser.add_argument("--fixture", required=True, help="Path to frame.ingested example JSON")
    args = parser.parse_args()

    configure_logging(settings.log_level)
    event = process_fixture(args.fixture)
    logger.info("worker_finished event_type=%s track_id=%s", event["event_type"], event["context"]["track_id"])


if __name__ == "__main__":
    main()
