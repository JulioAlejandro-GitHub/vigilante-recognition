from __future__ import annotations

import argparse
import logging
from uuid import UUID

from app.config import settings
from app.consumer import load_fixture_message
from app.correlation import extract_run_id, source_correlation_payload
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
from app.services.face_backend_selector import FaceBackendSelector
from app.services.face_backend_service import SimpleFaceBackend
from app.services.face_matching_service import FaceMatchingService
from app.services.canonical_frame_ref_service import CanonicalFrameRefService
from app.services.camera_face_metrics_service import log_all_camera_face_metrics
from app.services.camera_runtime_config_service import extract_camera_runtime_config
from app.services.presence_service import PresenceService
from app.services.recurrent_subject_service import RecurrentSubjectService
from app.services.runtime_metrics_http_service import start_runtime_metrics_http_server
from app.services.runtime_metrics_service import RuntimeMetricsService
from app.services.semantic_descriptor_service import SemanticDescriptorService
from app.services.vlm_execution_policy_service import build_vlm_execution_policy_snapshot
from app.services.track_service import TrackService
from app.storage.frame_resolver import FrameResolutionError

logger = logging.getLogger(__name__)
_SEMANTIC_DESCRIPTOR_SERVICE_CACHE: tuple[tuple[object, ...], SemanticDescriptorService] | None = None
_RUNTIME_METRICS_SERVICE_CACHE: tuple[tuple[object, ...], RuntimeMetricsService] | None = None


def _get_semantic_descriptor_service() -> SemanticDescriptorService:
    global _SEMANTIC_DESCRIPTOR_SERVICE_CACHE
    cache_key = (
        SemanticDescriptorService,
        settings.effective_qwen_model_name,
        settings.effective_smolvlm_model_name,
        settings.effective_vlm_device,
        settings.vlm_max_new_tokens,
        settings.vlm_max_image_edge,
        settings.vlm_serialization_guard_enabled,
    )
    if _SEMANTIC_DESCRIPTOR_SERVICE_CACHE is None or _SEMANTIC_DESCRIPTOR_SERVICE_CACHE[0] != cache_key:
        _SEMANTIC_DESCRIPTOR_SERVICE_CACHE = (cache_key, SemanticDescriptorService())
    return _SEMANTIC_DESCRIPTOR_SERVICE_CACHE[1]


def _get_runtime_metrics_service() -> RuntimeMetricsService:
    global _RUNTIME_METRICS_SERVICE_CACHE
    cache_key = (
        settings.runtime_metrics_enabled,
        settings.runtime_metrics_store,
        settings.runtime_metrics_path,
        settings.runtime_metrics_rotate_max_mb,
        settings.runtime_metrics_retention_files,
        settings.runtime_metrics_log_summary_every_n_events,
        settings.runtime_recommendations_enabled,
        settings.runtime_recommendations_path,
        settings.runtime_recommendations_rotate_max_mb,
        settings.runtime_recommendations_retention_files,
        settings.runtime_recommendations_min_events_per_camera,
        settings.runtime_recommendations_window_hours,
        settings.runtime_recommendations_log_every_n_events,
    )
    if _RUNTIME_METRICS_SERVICE_CACHE is None or _RUNTIME_METRICS_SERVICE_CACHE[0] != cache_key:
        _RUNTIME_METRICS_SERVICE_CACHE = (cache_key, RuntimeMetricsService.from_settings())
    return _RUNTIME_METRICS_SERVICE_CACHE[1]


def _safe_generate_semantic_descriptor(
    semantic_descriptor_service: SemanticDescriptorService,
    *,
    frame_ref: str,
    source_frame_ref: str | None = None,
    face_detection,
    event_type_hint: str | None = None,
    camera_id: str | None = None,
    camera_metadata: dict | None = None,
) -> SemanticDescriptorResult:
    published_source_frame_ref = frame_ref if source_frame_ref is None else source_frame_ref
    try:
        return semantic_descriptor_service.generate(
            frame_ref=frame_ref,
            source_frame_ref=published_source_frame_ref,
            face_detection=face_detection,
            event_type_hint=event_type_hint,
            camera_id=camera_id,
            camera_metadata=camera_metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception(
            "semantic_descriptor_generation_failed frame_ref=%s error=%s",
            frame_ref,
            type(exc).__name__,
        )
        generation_trace = {
            "trace_version": "semantic_backend_trace_v1",
            "execution_policy": build_vlm_execution_policy_snapshot(),
            "semantic_backend_requested": settings.semantic_descriptor_backend,
            "semantic_backend_selected": None,
            "semantic_backend_fallback_used": True,
            "semantic_backend_error": f"unexpected_service_error:{type(exc).__name__}",
            "semantic_backend_event_type_hint": event_type_hint,
            "camera_id": camera_id,
            "event_type": event_type_hint,
            "requested_backend": settings.semantic_descriptor_backend,
            "fallback_enabled": settings.semantic_enable_fallback,
            "timeout_seconds": settings.effective_vlm_timeout_seconds,
            "timeout_applied_seconds": settings.effective_vlm_timeout_seconds,
            "max_new_tokens": settings.vlm_max_new_tokens,
            "max_image_edge": settings.vlm_max_image_edge,
            "requested_device": settings.effective_vlm_device,
            "serialization_guard_enabled": settings.vlm_serialization_guard_enabled,
            "descriptor_valid": False,
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
        return SemanticDescriptorResult(
            backend="simple_color_signature_v1",
            source_frame_ref=published_source_frame_ref,
            rejection_reasons=["semantic_descriptor_generation_unexpected_failure"],
            descriptor={
                "source_frame_ref": published_source_frame_ref,
                "semantic_backend_trace": generation_trace,
                "generation_trace": generation_trace,
                "semantic_backend_requested": settings.semantic_descriptor_backend,
                "semantic_backend_selected": None,
                "semantic_backend_fallback_used": True,
                "semantic_backend_error": f"unexpected_service_error:{type(exc).__name__}",
            },
        )


def _build_camera_runtime_config_trace(
    *,
    message: FrameIngestedMessage,
    face_detection,
    semantic_descriptor_result: SemanticDescriptorResult | None,
) -> dict:
    runtime_config = extract_camera_runtime_config(message.payload.metadata)
    trace = runtime_config.trace_payload(camera_id=message.camera_id)

    face_trace = dict(getattr(face_detection, "face_backend_trace", {}) or {})
    face_configuration = face_trace.get("configuration")
    if not isinstance(face_configuration, dict):
        face_configuration = {}
    face_tuning_source = (
        face_configuration.get("face_tuning_source")
        or face_configuration.get("config_source")
        or face_trace.get("config_source")
        or "global"
    )

    vlm_policy_trace = _semantic_vlm_policy_trace(semantic_descriptor_result)
    vlm_policy_source = (
        vlm_policy_trace.get("vlm_policy_source")
        or vlm_policy_trace.get("config_source")
        or "not_evaluated"
    )

    trace.update(
        {
            "face_tuning_source": face_tuning_source,
            "vlm_policy_source": vlm_policy_source,
            "face_effective_config_hash": face_configuration.get("effective_config_hash"),
            "vlm_effective_policy_hash": vlm_policy_trace.get("effective_policy_hash"),
            "face_camera_config_version": face_configuration.get("camera_config_version"),
            "vlm_camera_config_version": vlm_policy_trace.get("camera_config_version"),
            "face_camera_config_hash": face_configuration.get("camera_config_hash"),
            "vlm_camera_config_hash": vlm_policy_trace.get("camera_config_hash"),
        }
    )
    trace["camera_override_applied"] = any(
        _source_applied(source)
        for source in (
            trace.get("face_tuning_source"),
            trace.get("vlm_policy_source"),
        )
    )
    return trace


def _semantic_vlm_policy_trace(semantic_descriptor_result: SemanticDescriptorResult | None) -> dict:
    if semantic_descriptor_result is None:
        return {}
    descriptor = semantic_descriptor_result.descriptor or {}
    policy_trace = descriptor.get("vlm_policy_trace")
    if isinstance(policy_trace, dict):
        return policy_trace
    backend_trace = descriptor.get("semantic_backend_trace") or descriptor.get("generation_trace")
    if isinstance(backend_trace, dict):
        nested_policy_trace = backend_trace.get("vlm_policy_trace")
        if isinstance(nested_policy_trace, dict):
            return nested_policy_trace
    return {}


def _source_applied(source: object) -> bool:
    if source is None:
        return False
    return str(source) not in {"", "global", "global_defaults", "not_provided", "not_evaluated"}


def process_fixture(fixture_path: str) -> dict:
    message = load_fixture_message(fixture_path)
    return process_message(message)


def process_message(message: FrameIngestedMessage) -> dict:
    with get_session() as session:
        source_correlation = source_correlation_payload(message)
        run_id = extract_run_id(message)
        logger.info(
            "frame_ingested_processing_started event_id=%s run_id=%s frame_ref=%s",
            message.event_id,
            run_id or "",
            message.frame_ref,
        )
        frame_ref_service = CanonicalFrameRefService()
        canonical_resolution = frame_ref_service.resolve(message)
        published_frame_ref = canonical_resolution.frame_ref
        published_evidence_refs = [published_frame_ref] if published_frame_ref else []
        semantic_source_frame_ref = published_frame_ref or ""
        processing_frame_ref = message.frame_ref
        if published_frame_ref is None:
            logger.warning(
                "canonical_frame_ref_unavailable event_id=%s cached_path=%s reason=%s",
                message.event_id,
                message.cached_path,
                canonical_resolution.fallback_reason,
            )
        elif canonical_resolution.fallback_reason:
            logger.info(
                "canonical_frame_ref_fallback event_id=%s source=%s reason=%s published_frame_ref=%s cached_path=%s",
                message.event_id,
                canonical_resolution.source,
                canonical_resolution.fallback_reason,
                published_frame_ref,
                message.cached_path,
            )

        repo = RecognitionRepository(session)
        track_service = TrackService(repo)
        presence_service = PresenceService()
        simple_embedding_service = FaceEmbeddingService()
        face_backend_service = FaceBackendSelector(
            simple_backend=SimpleFaceBackend(
                presence_service=presence_service,
                embedding_service=simple_embedding_service,
            )
        )
        matching_service = FaceMatchingService(repo=repo, embedding_service=face_backend_service)
        cross_camera_service = CrossCameraCorrelationService(repo=repo)
        conflict_service = ConflictResolutionService()
        semantic_descriptor_service = _get_semantic_descriptor_service()
        recurrent_subject_service = RecurrentSubjectService(
            repo=repo,
            semantic_descriptor_service=semantic_descriptor_service,
        )
        publisher = EventPublisher()

        subject, track, is_new_appearance = track_service.open_track_from_frame(message)
        track = track_service.confirm_basic_presence(track)
        face_detection = face_backend_service.inspect_face(
            frame_ref=processing_frame_ref,
            quality_metadata=message.payload.quality_metadata,
            camera_id=message.camera_id,
            camera_metadata=message.payload.metadata,
        )
        track_service.register_face_observation(
            track=track,
            face_detection=face_detection,
            frame_ref=published_frame_ref or "",
            detected_at=message.captured_at,
        )
        embedding_result = None
        match_result = None
        semantic_descriptor_result = None
        continuity_resolution = None
        recurrent_resolution = None
        if face_detection.usable:
            embedding_result = face_backend_service.generate(
                frame_ref=processing_frame_ref,
                face_detection=face_detection,
                camera_id=message.camera_id,
                camera_metadata=message.payload.metadata,
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
                    frame_ref=processing_frame_ref,
                    source_frame_ref=semantic_source_frame_ref,
                    face_detection=face_detection,
                    event_type_hint="face_detected_unidentified",
                    camera_id=message.camera_id,
                    camera_metadata=message.payload.metadata,
                )
                semantic_descriptor_result = frame_ref_service.canonicalize_semantic_descriptor(
                    semantic_descriptor_result,
                    canonical_frame_ref=published_frame_ref,
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
                            frame_ref=published_frame_ref or "",
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
                        frame_ref=published_frame_ref or "",
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
                    frame_ref=published_frame_ref or "",
                    face_detection=face_detection,
                    embedding_result=embedding_result,
                    match_result=match_result,
                    semantic_descriptor_result=semantic_descriptor_result,
                )
        else:
            semantic_descriptor_result = _safe_generate_semantic_descriptor(
                semantic_descriptor_service,
                frame_ref=processing_frame_ref,
                source_frame_ref=semantic_source_frame_ref,
                face_detection=face_detection,
                event_type_hint="human_presence_no_face",
                camera_id=message.camera_id,
                camera_metadata=message.payload.metadata,
            )
            semantic_descriptor_result = frame_ref_service.canonicalize_semantic_descriptor(
                semantic_descriptor_result,
                canonical_frame_ref=published_frame_ref,
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
                frame_ref=published_frame_ref or "",
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
                            frame_ref=published_frame_ref or "",
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

        camera_config_trace = _build_camera_runtime_config_trace(
            message=message,
            face_detection=face_detection,
            semantic_descriptor_result=semantic_descriptor_result,
        )
        decision.payload["camera_runtime_config_trace"] = camera_config_trace

        event = build_recognition_event(
            event_type=decision.event_type,
            camera_id=track.camera_id,
            track_id=track.human_track_id,
            subject_id=subject.observed_subject_id,
            severity=decision.severity,
            confidence=decision.confidence,
            decision_reason=decision.decision_reason,
            frame_ref=published_frame_ref,
            evidence_refs=published_evidence_refs,
            payload_details=decision.payload,
            correlation=source_correlation,
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
            evidence_refs=published_evidence_refs,
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
                    frame_ref=published_frame_ref,
                    evidence_refs=published_evidence_refs,
                    payload_details={
                        **supplemental.payload,
                        "camera_runtime_config_trace": camera_config_trace,
                    },
                    correlation=source_correlation,
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
                    evidence_refs=published_evidence_refs,
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
        logger.info(
            "recognition_events_published source_event_id=%s run_id=%s emitted=%s primary_event_id=%s",
            source_correlation.get("source_event_id") or "",
            run_id or "",
            len(events_to_publish),
            event.get("event_id"),
        )
        _get_runtime_metrics_service().record_emitted_events(
            source_message=message,
            emitted_events=events_to_publish,
            face_detection=face_detection,
            semantic_descriptor_result=semantic_descriptor_result,
            camera_runtime_config_trace=camera_config_trace,
        )
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
    if settings.runtime_metrics_enabled and (
        settings.runtime_metrics_enable_http or settings.runtime_recommendations_enable_http
    ):
        runtime_metrics_service = _get_runtime_metrics_service()
        if runtime_metrics_service.store is not None:
            start_runtime_metrics_http_server(
                store=runtime_metrics_service.store,
                host=settings.runtime_metrics_http_host,
                port=settings.runtime_metrics_http_port,
                recommendation_service=(
                    runtime_metrics_service.recommendation_service
                    if settings.runtime_recommendations_enable_http
                    else None
                ),
            )
    init_db()
    try:
        if args.fixture:
            event = process_fixture(args.fixture)
            logger.info("worker_finished event_type=%s track_id=%s", event["event_type"], event["context"]["track_id"])
            log_all_camera_face_metrics()
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
        log_all_camera_face_metrics()
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
    log_all_camera_face_metrics()


if __name__ == "__main__":
    main()
