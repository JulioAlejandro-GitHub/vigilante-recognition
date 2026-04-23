from __future__ import annotations

import argparse
import logging
from uuid import UUID

from app.config import settings
from app.consumer import load_fixture_message
from app.db import get_session, init_db
from app.domain.entities import InvalidCameraIdError
from app.domain.events import build_recognition_event
from app.infra.repository import RecognitionRepository
from app.logging import configure_logging
from app.publisher import EventPublisher
from app.services.conflict_resolution_service import ConflictResolutionService
from app.services.cross_camera_correlation_service import CrossCameraCorrelationService
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.face_matching_service import FaceMatchingService
from app.services.presence_service import PresenceService
from app.services.track_service import TrackService

logger = logging.getLogger(__name__)


def process_fixture(fixture_path: str) -> dict:
    message = load_fixture_message(fixture_path)

    with get_session() as session:
        repo = RecognitionRepository(session)
        track_service = TrackService(repo)
        presence_service = PresenceService()
        embedding_service = FaceEmbeddingService()
        matching_service = FaceMatchingService(repo=repo, embedding_service=embedding_service)
        cross_camera_service = CrossCameraCorrelationService(repo=repo)
        conflict_service = ConflictResolutionService()
        publisher = EventPublisher()

        subject, track, is_new_appearance = track_service.open_track_from_frame(message)
        track = track_service.confirm_basic_presence(track)
        face_detection = presence_service.inspect_face(
            frame_ref=message.frame_ref,
            quality_metadata=message.payload.quality_metadata,
        )
        track_service.register_face_observation(
            track=track,
            face_detection=face_detection,
            frame_ref=message.frame_ref,
            detected_at=message.captured_at,
        )
        embedding_result = None
        match_result = None
        continuity_resolution = None
        if face_detection.usable:
            embedding_result = embedding_service.generate(
                frame_ref=message.frame_ref,
                face_detection=face_detection,
            )
            match_result = matching_service.match(
                embedding_result,
                gallery_override_path=message.context.get("dev_known_face_gallery_path"),
            )
            track_service.register_face_match(
                track=track,
                match_result=match_result,
                matched_at=message.captured_at,
            )
            if is_new_appearance:
                assessment = cross_camera_service.evaluate(
                    current_subject=subject,
                    current_track=track,
                    observed_at=message.captured_at,
                    current_camera_id=track.camera_id,
                    embedding_result=embedding_result,
                    match_result=match_result,
                )
                continuity_resolution = conflict_service.resolve(
                    current_subject=subject,
                    current_track=track,
                    assessment=assessment,
                    match_result=match_result,
                )

                best_candidate = assessment.best_candidate
                if continuity_resolution.outcome == "correlated" and best_candidate is not None:
                    repo.add_cross_camera_correlation(
                        source_subject_id=subject.observed_subject_id,
                        target_subject_id=UUID(best_candidate.observed_subject_id),
                        source_track_id=track.human_track_id,
                        target_track_id=UUID(best_candidate.latest_track_id) if best_candidate.latest_track_id else None,
                        correlation_status=continuity_resolution.correlation_status or "auto",
                        face_similarity_score=best_candidate.face_similarity_score,
                        temporal_coherence_score=best_candidate.temporal_coherence_score,
                        aggregate_score=best_candidate.aggregate_score,
                        signals_json=continuity_resolution.payload,
                    )
                    target_subject = repo.get_subject(UUID(best_candidate.observed_subject_id))
                    if target_subject is not None:
                        target_subject, track = track_service.link_track_to_subject(
                            track=track,
                            source_subject=subject,
                            target_subject=target_subject,
                            resolved_at=message.captured_at,
                            payload=continuity_resolution.payload,
                        )
                        subject = track_service.update_subject_face_profile(
                            subject=target_subject,
                            camera_id=track.camera_id,
                            observed_at=message.captured_at,
                            frame_ref=message.frame_ref,
                            face_detection=face_detection,
                            embedding_result=embedding_result,
                            match_result=match_result,
                        )
                else:
                    subject = track_service.update_subject_face_profile(
                        subject=subject,
                        camera_id=track.camera_id,
                        observed_at=message.captured_at,
                        frame_ref=message.frame_ref,
                        face_detection=face_detection,
                        embedding_result=embedding_result,
                        match_result=match_result,
                    )
                    if continuity_resolution and continuity_resolution.assessment and best_candidate is not None and continuity_resolution.outcome in {
                        "identity_conflict",
                        "manual_review_required",
                    }:
                        repo.add_cross_camera_correlation(
                            source_subject_id=subject.observed_subject_id,
                            target_subject_id=UUID(best_candidate.observed_subject_id),
                            source_track_id=track.human_track_id,
                            target_track_id=UUID(best_candidate.latest_track_id) if best_candidate.latest_track_id else None,
                            correlation_status=continuity_resolution.correlation_status or "pending_review",
                            face_similarity_score=best_candidate.face_similarity_score,
                            temporal_coherence_score=best_candidate.temporal_coherence_score,
                            aggregate_score=best_candidate.aggregate_score,
                            signals_json=continuity_resolution.payload,
                        )
                        repo.mark_subject_continuity(
                            subject,
                            outcome=continuity_resolution.outcome,
                            resolved_at=message.captured_at,
                            payload=continuity_resolution.payload,
                        )
                if continuity_resolution and continuity_resolution.outcome != "none":
                    track = track_service.record_continuity_resolution(
                        track=track,
                        continuity_resolution=continuity_resolution,
                        resolved_at=message.captured_at,
                    )
            else:
                continuity_resolution = track_service.load_continuity_resolution(track=track)
                subject = track_service.update_subject_face_profile(
                    subject=subject,
                    camera_id=track.camera_id,
                    observed_at=message.captured_at,
                    frame_ref=message.frame_ref,
                    face_detection=face_detection,
                    embedding_result=embedding_result,
                    match_result=match_result,
                )
        else:
            subject = track_service.update_subject_face_profile(
                subject=subject,
                camera_id=track.camera_id,
                observed_at=message.captured_at,
                frame_ref=message.frame_ref,
                face_detection=face_detection,
            )

        decision = presence_service.decide(
            track=track,
            face_detection=face_detection,
            embedding_result=embedding_result,
            match_result=match_result,
        )
        if continuity_resolution and continuity_resolution.outcome != "none":
            decision.payload["continuity_status"] = continuity_resolution.outcome
            decision.payload["subject_continuity"] = continuity_resolution.payload
            if continuity_resolution.requires_human_review:
                decision.payload["requires_human_review"] = True

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
        events_to_publish = [event]
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

        if continuity_resolution:
            for supplemental in continuity_resolution.supplemental_decisions:
                supplemental_subject_id = UUID(supplemental.subject_id) if supplemental.subject_id else subject.observed_subject_id
                supplemental_event = build_recognition_event(
                    event_type=supplemental.event_type,
                    camera_id=track.camera_id,
                    track_id=track.human_track_id,
                    subject_id=supplemental_subject_id,
                    severity=supplemental.severity,
                    confidence=supplemental.confidence,
                    decision_reason=supplemental.decision_reason,
                    frame_ref=message.frame_ref,
                    payload_details=supplemental.payload,
                )
                supplemental_recognition_event = repo.add_recognition_event(
                    subject_id=supplemental_subject_id,
                    track_id=track.human_track_id,
                    camera_id=track.camera_id,
                    event_type=supplemental.event_type,
                    event_ts=message.captured_at,
                    severity=supplemental.severity,
                    confidence=supplemental.confidence,
                    decision_reason=supplemental.decision_reason,
                    evidence_refs=[message.frame_ref],
                    payload=supplemental_event["payload"],
                )
                repo.add_outbox_event(
                    aggregate_type="recognition_event",
                    aggregate_id=supplemental_recognition_event.recognition_event_id,
                    event_type=supplemental.event_type,
                    payload=supplemental_event,
                )
                events_to_publish.append(supplemental_event)
        session.commit()

        for emitted_event in events_to_publish:
            publisher.publish(emitted_event)
        return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap worker for vigilante-recognition slice 4")
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
