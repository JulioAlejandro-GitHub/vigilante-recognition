from __future__ import annotations

import argparse
import logging
from uuid import UUID

from app.config import settings
from app.consumer import load_fixture_message
from app.db import get_session, init_db
from app.domain.entities import (
    FrameIngestedMessage,
    InvalidCameraIdError,
    RecurrentSubjectResolution,
    SemanticDescriptorResult,
)
from app.domain.events import build_recognition_event
from app.ingestion import InvalidFrameIngestedEventError, InvalidJsonlLineError
from app.infra.repository import RecognitionRepository
from app.logging import configure_logging
from app.publisher import EventPublisher
from app.runner.process_ingestion_outbox import (
    ProcessIngestionOutboxResult,
    process_ingestion_outbox,
)
from app.runner.process_rabbitmq_frames import (
    ProcessRabbitMqFramesResult,
    process_rabbitmq_frames,
)
from app.services.conflict_resolution_service import ConflictResolutionService
from app.services.cross_camera_correlation_service import CrossCameraCorrelationService
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.face_matching_service import FaceMatchingService
from app.services.presence_service import PresenceService
from app.services.recurrent_subject_service import RecurrentSubjectService
from app.services.semantic_descriptor_service import SemanticDescriptorService
from app.services.track_service import TrackService
from app.storage.frame_resolver import FrameResolutionError

logger = logging.getLogger(__name__)


def _safe_generate_semantic_descriptor(
    semantic_descriptor_service: SemanticDescriptorService,
    *,
    frame_ref: str,
    face_detection,
) -> SemanticDescriptorResult:
    try:
        return semantic_descriptor_service.generate(
            frame_ref=frame_ref,
            face_detection=face_detection,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception(
            "semantic_descriptor_generation_failed frame_ref=%s error=%s",
            frame_ref,
            type(exc).__name__,
        )
        return SemanticDescriptorResult(
            backend="simple_color_signature_v1",
            source_frame_ref=frame_ref,
            rejection_reasons=["semantic_descriptor_generation_unexpected_failure"],
            descriptor={
                "generation_trace": {
                    "requested_backend": settings.semantic_descriptor_backend,
                    "fallback_enabled": settings.semantic_enable_fallback,
                    "selected_backend": None,
                    "selected_backend_key": None,
                    "attempts": [
                        {
                            "backend_key": "service",
                            "backend_name": "semantic_descriptor_service",
                            "status": "failed",
                            "reason": f"unexpected_service_error:{type(exc).__name__}",
                        }
                    ],
                }
            },
        )


def process_fixture(fixture_path: str) -> dict:
    message = load_fixture_message(fixture_path)
    return process_message(message)


def process_message(message: FrameIngestedMessage) -> dict:
    with get_session() as session:
        repo = RecognitionRepository(session)
        track_service = TrackService(repo)
        presence_service = PresenceService()
        embedding_service = FaceEmbeddingService()
        matching_service = FaceMatchingService(repo=repo, embedding_service=embedding_service)
        cross_camera_service = CrossCameraCorrelationService(repo=repo)
        conflict_service = ConflictResolutionService()
        semantic_descriptor_service = SemanticDescriptorService()
        recurrent_subject_service = RecurrentSubjectService(
            repo=repo,
            semantic_descriptor_service=semantic_descriptor_service,
        )
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
        semantic_descriptor_result = None
        continuity_resolution = None
        recurrent_resolution = None
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
            if not match_result.identified:
                semantic_descriptor_result = _safe_generate_semantic_descriptor(
                    semantic_descriptor_service,
                    frame_ref=message.frame_ref,
                    face_detection=face_detection,
                )
                if semantic_descriptor_result.generated:
                    track = track_service.register_semantic_descriptor(
                        track=track,
                        semantic_descriptor_result=semantic_descriptor_result,
                        described_at=message.captured_at,
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
                            semantic_descriptor_result=semantic_descriptor_result,
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
                        semantic_descriptor_result=semantic_descriptor_result,
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
                    semantic_descriptor_result=semantic_descriptor_result,
                )
        else:
            semantic_descriptor_result = _safe_generate_semantic_descriptor(
                semantic_descriptor_service,
                frame_ref=message.frame_ref,
                face_detection=face_detection,
            )
            if semantic_descriptor_result.generated:
                track = track_service.register_semantic_descriptor(
                    track=track,
                    semantic_descriptor_result=semantic_descriptor_result,
                    described_at=message.captured_at,
                )
            subject = track_service.update_subject_face_profile(
                subject=subject,
                camera_id=track.camera_id,
                observed_at=message.captured_at,
                frame_ref=message.frame_ref,
                face_detection=face_detection,
                semantic_descriptor_result=semantic_descriptor_result,
            )

        unresolved_flow = semantic_descriptor_result is not None and semantic_descriptor_result.generated
        if unresolved_flow and (continuity_resolution is None or continuity_resolution.outcome == "none"):
            if is_new_appearance:
                recurrent_assessment = recurrent_subject_service.evaluate(
                    current_subject=subject,
                    current_track=track,
                    observed_at=message.captured_at,
                    current_camera_id=track.camera_id,
                    semantic_descriptor_result=semantic_descriptor_result,
                    embedding_result=embedding_result,
                )
                recurrent_resolution = recurrent_subject_service.resolve(
                    current_subject=subject,
                    current_track=track,
                    semantic_descriptor_result=semantic_descriptor_result,
                    assessment=recurrent_assessment,
                )
                best_recurrent_candidate = recurrent_assessment.best_candidate
                if recurrent_resolution.outcome == "recurrent_unresolved" and best_recurrent_candidate is not None:
                    target_subject = repo.get_subject(UUID(best_recurrent_candidate.observed_subject_id))
                    if target_subject is not None:
                        target_subject, track = track_service.link_track_to_subject(
                            track=track,
                            source_subject=subject,
                            target_subject=target_subject,
                            resolved_at=message.captured_at,
                            payload=recurrent_resolution.payload,
                            outcome="recurrent_unresolved",
                        )
                        subject = track_service.update_subject_face_profile(
                            subject=target_subject,
                            camera_id=track.camera_id,
                            observed_at=message.captured_at,
                            frame_ref=message.frame_ref,
                            face_detection=face_detection,
                            embedding_result=embedding_result,
                            match_result=match_result,
                            semantic_descriptor_result=semantic_descriptor_result,
                        )
                    track = track_service.record_recurrent_resolution(
                        track=track,
                        recurrent_resolution=recurrent_resolution,
                        resolved_at=message.captured_at,
                    )
            else:
                recurrent_resolution = track_service.load_recurrent_resolution(track=track)
        if recurrent_resolution is not None:
            enriched_resolution = recurrent_subject_service.enrich_resolution_with_descriptor(
                resolution=recurrent_resolution,
                semantic_descriptor_result=semantic_descriptor_result,
            )
            if isinstance(enriched_resolution, RecurrentSubjectResolution):
                recurrent_resolution = enriched_resolution

        decision = presence_service.decide(
            track=track,
            face_detection=face_detection,
            embedding_result=embedding_result,
            match_result=match_result,
            semantic_descriptor_result=semantic_descriptor_result,
        )
        if continuity_resolution and continuity_resolution.outcome != "none":
            decision.payload["continuity_status"] = continuity_resolution.outcome
            decision.payload["subject_continuity"] = continuity_resolution.payload
            if continuity_resolution.requires_human_review:
                decision.payload["requires_human_review"] = True
        if recurrent_resolution and recurrent_resolution.outcome != "none":
            decision.payload["unresolved_recurrence_status"] = recurrent_resolution.outcome
            decision.payload["unresolved_subject_recurrence"] = recurrent_resolution.payload
            if recurrent_resolution.requires_human_review:
                decision.payload["requires_human_review"] = True
            if recurrent_resolution.requires_case_evaluation:
                decision.payload["requires_case_evaluation"] = True

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

        supplemental_resolutions = [resolution for resolution in [continuity_resolution, recurrent_resolution] if resolution]
        for supplemental_resolution in supplemental_resolutions:
            for supplemental in supplemental_resolution.supplemental_decisions:
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


def process_ingestion_jsonl(
    jsonl_path: str,
    *,
    force_replay: bool = False,
    checkpoint_path: str | None = None,
    deduper_path: str | None = None,
    rejected_events_path: str | None = None,
) -> ProcessIngestionOutboxResult:
    from app.ingestion import FileCheckpointStore, FileEventDeduper, RejectedEventStore

    return process_ingestion_outbox(
        jsonl_path,
        processor=process_message,
        frame_search_roots=settings.ingestion_frame_search_root_paths,
        checkpoint_store=FileCheckpointStore(checkpoint_path or settings.ingestion_checkpoint_path),
        event_deduper=FileEventDeduper(deduper_path or settings.ingestion_deduper_path),
        rejected_event_store=RejectedEventStore(rejected_events_path or settings.ingestion_rejected_events_path),
        force_replay=force_replay,
    )


def process_rabbitmq_consumer(
    *,
    max_messages: int | None = None,
    retry_limit: int | None = None,
    deduper_path: str | None = None,
    rejected_events_path: str | None = None,
) -> ProcessRabbitMqFramesResult:
    from app.ingestion import FileEventDeduper, RejectedEventStore

    return process_rabbitmq_frames(
        processor=process_message,
        frame_search_roots=settings.ingestion_frame_search_root_paths,
        event_deduper=FileEventDeduper(deduper_path or settings.ingestion_deduper_path),
        rejected_event_store=RejectedEventStore(rejected_events_path or settings.ingestion_rejected_events_path),
        max_messages=max_messages,
        retry_limit=retry_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap worker for vigilante-recognition slice 7")
    parser.add_argument("--fixture", help="Path to frame.ingested example JSON")
    parser.add_argument(
        "--ingestion-jsonl",
        help="Path to vigilante-ingestion JSONL outbox with frame.ingested events",
    )
    parser.add_argument(
        "--rabbitmq-consumer",
        action="store_true",
        help="Consume frame.ingested from RabbitMQ using the configured broker topology.",
    )
    parser.add_argument(
        "--force-replay",
        action="store_true",
        help="Replay the JSONL from byte offset 0 and ignore persisted event_id dedupe for this run.",
    )
    parser.add_argument(
        "--ingestion-checkpoint-path",
        help="Local checkpoint JSON path. Defaults to INGESTION_CHECKPOINT_PATH.",
    )
    parser.add_argument(
        "--ingestion-deduper-path",
        help="Local processed event_id registry path. Defaults to INGESTION_DEDUPER_PATH.",
    )
    parser.add_argument(
        "--ingestion-rejected-path",
        help="Local rejected events JSONL path. Defaults to INGESTION_REJECTED_EVENTS_PATH.",
    )
    parser.add_argument(
        "--rabbitmq-max-messages",
        type=int,
        help="Stop RabbitMQ consumption after N deliveries. Omit to run until interrupted.",
    )
    parser.add_argument(
        "--rabbitmq-retry-limit",
        type=int,
        help="Retry processing failures this many times before broker DLQ.",
    )
    args = parser.parse_args()
    selected_modes = sum(bool(value) for value in [args.fixture, args.ingestion_jsonl, args.rabbitmq_consumer])
    if selected_modes != 1:
        parser.error("Provide exactly one of --fixture, --ingestion-jsonl or --rabbitmq-consumer")

    configure_logging(settings.log_level)
    init_db()
    try:
        if args.fixture:
            event = process_fixture(args.fixture)
            logger.info("worker_finished event_type=%s track_id=%s", event["event_type"], event["context"]["track_id"])
            return

        if args.ingestion_jsonl:
            result = process_ingestion_jsonl(
                args.ingestion_jsonl,
                force_replay=args.force_replay,
                checkpoint_path=args.ingestion_checkpoint_path,
                deduper_path=args.ingestion_deduper_path,
                rejected_events_path=args.ingestion_rejected_path,
            )
        else:
            result = process_rabbitmq_consumer(
                max_messages=args.rabbitmq_max_messages,
                retry_limit=args.rabbitmq_retry_limit,
                deduper_path=args.ingestion_deduper_path,
                rejected_events_path=args.ingestion_rejected_path,
            )
    except (InvalidCameraIdError, InvalidFrameIngestedEventError, InvalidJsonlLineError, FrameResolutionError) as exc:
        logger.error("worker_failed reason=%s", exc)
        raise SystemExit(2) from exc

    if isinstance(result, ProcessRabbitMqFramesResult):
        logger.info(
            (
                "worker_finished rabbitmq_queue=%s consumed=%s processed=%s acked=%s "
                "retried=%s rejected_to_dlq=%s skipped_duplicate=%s invalid_messages=%s "
                "frame_resolution_errors=%s processing_errors=%s deduper_path=%s rejected_events_path=%s"
            ),
            result.queue_name,
            result.consumed,
            result.processed,
            result.acked,
            result.retried,
            result.rejected_to_dlq,
            result.skipped_duplicate,
            result.invalid_messages,
            result.frame_resolution_errors,
            result.processing_errors,
            result.deduper_path,
            result.rejected_events_path,
        )
        return

    logger.info(
        (
            "worker_finished ingestion_jsonl=%s read=%s processed=%s "
            "skipped_checkpoint=%s skipped_duplicate=%s rejected=%s "
            "frame_resolution_errors=%s processing_errors=%s start_offset=%s final_offset=%s "
            "checkpoint_path=%s deduper_path=%s rejected_events_path=%s"
        ),
        result.source_path,
        result.read,
        result.processed,
        result.skipped_checkpoint,
        result.skipped_duplicate,
        result.rejected,
        result.frame_resolution_errors,
        result.processing_errors,
        result.start_offset,
        result.final_offset,
        result.checkpoint_path,
        result.deduper_path,
        result.rejected_events_path,
    )


if __name__ == "__main__":
    main()
