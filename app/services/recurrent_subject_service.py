from __future__ import annotations

from uuid import UUID

import numpy as np

from app.config import settings
from app.domain.entities import (
    FaceEmbeddingResult,
    RecurrentSubjectAssessment,
    RecurrentSubjectCandidate,
    RecurrentSubjectResolution,
    SemanticDescriptorResult,
    SupplementalRecognitionDecision,
)
from app.infra.repository import RecognitionRepository
from app.services.semantic_descriptor_service import SemanticDescriptorService


class RecurrentSubjectService:
    def __init__(
        self,
        *,
        repo: RecognitionRepository,
        semantic_descriptor_service: SemanticDescriptorService,
    ) -> None:
        self.repo = repo
        self.semantic_descriptor_service = semantic_descriptor_service

    def evaluate(
        self,
        *,
        current_subject,
        current_track,
        observed_at,
        current_camera_id,
        semantic_descriptor_result: SemanticDescriptorResult | None,
        embedding_result: FaceEmbeddingResult | None,
    ) -> RecurrentSubjectAssessment:
        assessment = RecurrentSubjectAssessment(
            current_subject_id=str(current_subject.observed_subject_id),
            current_track_id=str(current_track.human_track_id),
            current_camera_id=str(current_camera_id),
            semantic_similarity_threshold=settings.semantic_similarity_threshold,
            recurrent_subject_threshold=settings.recurrent_subject_threshold,
            case_suggestion_threshold=settings.case_suggestion_threshold,
            manual_review_threshold=settings.manual_review_threshold,
        )
        if semantic_descriptor_result is None or not semantic_descriptor_result.generated:
            assessment.decision_reason = ["semantic_descriptor_not_available"]
            return assessment

        raw_candidates = self.repo.load_recent_unresolved_subject_candidates(
            exclude_subject_id=current_subject.observed_subject_id,
            observed_at=observed_at,
            window_seconds=settings.cross_camera_time_window_seconds,
        )
        if not raw_candidates:
            assessment.decision_reason = ["unresolved_subject_no_recent_candidates"]
            return assessment

        current_vector = None
        if embedding_result and embedding_result.generated:
            current_vector = np.asarray(embedding_result.vector, dtype=np.float32)

        candidates: list[RecurrentSubjectCandidate] = []
        for raw_candidate in raw_candidates:
            subject = raw_candidate["subject"]
            latest_track = raw_candidate["latest_track"]
            candidate_descriptor = SemanticDescriptorResult(
                generated=True,
                backend=raw_candidate["semantic_descriptor"].get("backend", settings.semantic_descriptor_backend),
                descriptor=raw_candidate["semantic_descriptor"].get("descriptor", {}),
                signature=raw_candidate["semantic_descriptor"].get("signature", {}),
                confidence=float(raw_candidate["semantic_descriptor"].get("confidence", 0.0)),
                source_frame_ref=raw_candidate["semantic_descriptor"].get("source_frame_ref"),
            )
            semantic_similarity = self.semantic_descriptor_service.compare(
                semantic_descriptor_result,
                candidate_descriptor,
            )
            if semantic_similarity < settings.semantic_similarity_threshold:
                continue

            visual_similarity = 0.0
            candidate_embedding = raw_candidate.get("embedding")
            if current_vector is not None and candidate_embedding and len(candidate_embedding) == len(current_vector):
                candidate_vector = np.asarray(candidate_embedding, dtype=np.float32)
                visual_similarity = max(0.0, round(float(np.dot(current_vector, candidate_vector)), 4))

            delta_seconds = abs((observed_at - subject.last_seen_at).total_seconds())
            temporal_coherence = max(
                0.0,
                round(1.0 - (delta_seconds / settings.cross_camera_time_window_seconds), 4),
            )
            camera_relation_score = 1.0 if str(subject.last_camera_id) != str(current_camera_id) else 0.8
            continuity_bonus = 0.05 if raw_candidate.get("continuity_resolution") else 0.0

            if visual_similarity > 0.0:
                aggregate_score = min(
                    1.0,
                    round(
                        (0.55 * semantic_similarity)
                        + (0.25 * visual_similarity)
                        + (0.15 * temporal_coherence)
                        + (0.05 * camera_relation_score)
                        + continuity_bonus,
                        4,
                    ),
                )
            else:
                aggregate_score = min(
                    1.0,
                    round(
                        (0.75 * semantic_similarity)
                        + (0.2 * temporal_coherence)
                        + (0.05 * camera_relation_score)
                        + continuity_bonus,
                        4,
                    ),
                )

            candidates.append(
                RecurrentSubjectCandidate(
                    observed_subject_id=str(subject.observed_subject_id),
                    latest_track_id=str(latest_track.human_track_id) if latest_track is not None else None,
                    last_camera_id=str(subject.last_camera_id) if subject.last_camera_id else None,
                    last_seen_at=subject.last_seen_at,
                    recurrence_count=int(subject.recurrence_count or 1),
                    semantic_similarity_score=semantic_similarity,
                    visual_similarity_score=visual_similarity,
                    temporal_coherence_score=temporal_coherence,
                    camera_relation_score=round(camera_relation_score, 4),
                    aggregate_score=aggregate_score,
                    descriptor_summary={
                        "dominant_palette": candidate_descriptor.signature.get("dominant_palette", []),
                        "upper_region_color": candidate_descriptor.signature.get("upper_region_color"),
                        "subject_scale": candidate_descriptor.signature.get("subject_scale"),
                    },
                )
            )

        assessment.evaluated_candidates = len(candidates)
        if not candidates:
            assessment.decision_reason = ["unresolved_subject_no_semantic_match"]
            return assessment

        candidates.sort(key=lambda candidate: candidate.aggregate_score, reverse=True)
        assessment.best_candidate = candidates[0]
        assessment.second_best_candidate = candidates[1] if len(candidates) > 1 else None
        if assessment.second_best_candidate is not None:
            assessment.second_best_margin = round(
                assessment.best_candidate.aggregate_score - assessment.second_best_candidate.aggregate_score,
                4,
            )

        if assessment.best_candidate.aggregate_score >= settings.recurrent_subject_threshold:
            assessment.decision_reason = ["unresolved_recurrence_threshold_passed"]
        else:
            assessment.decision_reason = ["unresolved_recurrence_below_threshold"]
        return assessment

    def resolve(
        self,
        *,
        current_subject,
        current_track,
        semantic_descriptor_result: SemanticDescriptorResult | None,
        assessment: RecurrentSubjectAssessment,
    ) -> RecurrentSubjectResolution:
        resolution = RecurrentSubjectResolution(
            outcome="none",
            subject_id_to_use=str(current_subject.observed_subject_id),
            assessment=assessment,
        )
        best_candidate = assessment.best_candidate
        if best_candidate is None:
            resolution.decision_reason = assessment.decision_reason or ["unresolved_subject_no_candidate"]
            return resolution

        if best_candidate.aggregate_score < settings.recurrent_subject_threshold:
            resolution.decision_reason = assessment.decision_reason or ["unresolved_recurrence_below_threshold"]
            return resolution

        evidence_count = best_candidate.recurrence_count + 1
        payload = {
            "current_subject_id": str(current_subject.observed_subject_id),
            "current_track_id": str(current_track.human_track_id),
            "target_subject_id": best_candidate.observed_subject_id,
            "evidence_count": evidence_count,
            "semantic_descriptor": semantic_descriptor_result.descriptor if semantic_descriptor_result else {},
            "recurrent_subject_assessment": self._serialize_assessment(assessment),
        }
        resolution.outcome = "recurrent_unresolved"
        resolution.subject_id_to_use = best_candidate.observed_subject_id
        resolution.target_track_id = best_candidate.latest_track_id
        resolution.decision_reason = [
            "semantic_subject_recurrence_detected",
            "unresolved_subject_relinked",
        ]
        resolution.payload = payload
        resolution.supplemental_decisions = [
            SupplementalRecognitionDecision(
                event_type="recurrent_unresolved_subject",
                severity="medium",
                confidence=best_candidate.aggregate_score,
                decision_reason=resolution.decision_reason,
                payload={
                    **payload,
                    "semantic_similarity": best_candidate.semantic_similarity_score,
                    "visual_similarity": best_candidate.visual_similarity_score,
                },
                subject_id=best_candidate.observed_subject_id,
            )
        ]

        if evidence_count >= 2 and best_candidate.aggregate_score >= settings.manual_review_threshold:
            resolution.requires_human_review = True
            resolution.supplemental_decisions.append(
                SupplementalRecognitionDecision(
                    event_type="manual_review_required",
                    severity="medium",
                    confidence=best_candidate.aggregate_score,
                    decision_reason=[
                        "recurrent_unresolved_subject_detected",
                        "manual_review_enriched_by_semantics",
                    ],
                    payload={
                        **payload,
                        "review_type": "recurrent_unresolved_subject",
                        "requires_human_review": True,
                        "semantic_similarity": best_candidate.semantic_similarity_score,
                        "visual_similarity": best_candidate.visual_similarity_score,
                    },
                    subject_id=best_candidate.observed_subject_id,
                )
            )

        if evidence_count >= 3 and best_candidate.aggregate_score >= settings.case_suggestion_threshold:
            resolution.requires_case_evaluation = True
            resolution.supplemental_decisions.append(
                SupplementalRecognitionDecision(
                    event_type="case_suggestion_created",
                    severity="medium",
                    confidence=best_candidate.aggregate_score,
                    decision_reason=[
                        "unresolved_subject_evidence_accumulated",
                        "case_suggestion_threshold_passed",
                    ],
                    payload={
                        **payload,
                        "requires_case_evaluation": True,
                        "suggestion_type": "unresolved_subject_case",
                        "suggested_upstream_owner": "vigilante-api",
                        "semantic_similarity": best_candidate.semantic_similarity_score,
                        "visual_similarity": best_candidate.visual_similarity_score,
                    },
                    subject_id=best_candidate.observed_subject_id,
                )
            )

        return resolution

    def enrich_resolution_with_descriptor(
        self,
        *,
        resolution: RecurrentSubjectResolution | None,
        semantic_descriptor_result: SemanticDescriptorResult | None,
    ) -> RecurrentSubjectResolution | None:
        if (
            resolution is None
            or resolution.outcome == "none"
            or semantic_descriptor_result is None
            or not semantic_descriptor_result.generated
        ):
            return resolution

        enriched_descriptor = semantic_descriptor_result.descriptor
        resolution.payload = {
            **dict(resolution.payload or {}),
            "semantic_descriptor": enriched_descriptor,
        }
        for supplemental in resolution.supplemental_decisions:
            supplemental.payload = {
                **dict(supplemental.payload or {}),
                "semantic_descriptor": enriched_descriptor,
            }
        return resolution

    def _serialize_assessment(self, assessment: RecurrentSubjectAssessment) -> dict:
        candidates = []
        for candidate in [assessment.best_candidate, assessment.second_best_candidate]:
            if candidate is None:
                continue
            candidates.append(
                {
                    "observed_subject_id": candidate.observed_subject_id,
                    "latest_track_id": candidate.latest_track_id,
                    "last_camera_id": candidate.last_camera_id,
                    "last_seen_at": candidate.last_seen_at.isoformat(),
                    "recurrence_count": candidate.recurrence_count,
                    "semantic_similarity_score": candidate.semantic_similarity_score,
                    "visual_similarity_score": candidate.visual_similarity_score,
                    "temporal_coherence_score": candidate.temporal_coherence_score,
                    "camera_relation_score": candidate.camera_relation_score,
                    "aggregate_score": candidate.aggregate_score,
                    "descriptor_summary": candidate.descriptor_summary,
                }
            )
        return {
            "semantic_similarity_threshold": assessment.semantic_similarity_threshold,
            "recurrent_subject_threshold": assessment.recurrent_subject_threshold,
            "case_suggestion_threshold": assessment.case_suggestion_threshold,
            "manual_review_threshold": assessment.manual_review_threshold,
            "second_best_margin": assessment.second_best_margin,
            "evaluated_candidates": assessment.evaluated_candidates,
            "decision_reason": assessment.decision_reason,
            "candidates": candidates,
        }
