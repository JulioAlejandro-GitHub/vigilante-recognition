from __future__ import annotations

import numpy as np

from app.config import settings
from app.domain.entities import CrossCameraAssessment, CrossCameraCandidate, FaceEmbeddingResult, FaceMatchResult
from app.infra.repository import RecognitionRepository


class CrossCameraCorrelationService:
    def __init__(self, *, repo: RecognitionRepository) -> None:
        self.repo = repo

    def evaluate(
        self,
        *,
        current_subject,
        current_track,
        observed_at,
        current_camera_id,
        embedding_result: FaceEmbeddingResult | None,
        match_result: FaceMatchResult | None,
    ) -> CrossCameraAssessment:
        assessment = CrossCameraAssessment(
            current_subject_id=str(current_subject.observed_subject_id),
            current_track_id=str(current_track.human_track_id),
            current_camera_id=str(current_camera_id),
            threshold=settings.cross_camera_match_threshold,
            manual_review_threshold=settings.manual_review_threshold,
            second_best_margin_threshold=settings.second_best_margin,
        )
        if embedding_result is None or not embedding_result.generated:
            assessment.decision_reason = ["cross_camera_embedding_not_available"]
            return assessment

        raw_candidates = self.repo.load_recent_subject_candidates(
            exclude_subject_id=current_subject.observed_subject_id,
            observed_at=observed_at,
            window_seconds=settings.cross_camera_time_window_seconds,
        )
        if not raw_candidates:
            assessment.decision_reason = ["cross_camera_no_recent_candidates"]
            return assessment

        current_vector = np.asarray(embedding_result.vector, dtype=np.float32)
        candidates: list[CrossCameraCandidate] = []
        for raw_candidate in raw_candidates:
            candidate_embedding = raw_candidate["embedding"]
            if len(candidate_embedding) != len(embedding_result.vector):
                continue

            subject = raw_candidate["subject"]
            latest_track = raw_candidate["latest_track"]
            resolved_identity = raw_candidate["resolved_identity"] or {}
            candidate_camera_id = str(subject.last_camera_id) if subject.last_camera_id else None
            if candidate_camera_id == str(current_camera_id):
                continue

            candidate_vector = np.asarray(candidate_embedding, dtype=np.float32)
            face_similarity_score = round(float(np.dot(current_vector, candidate_vector)), 4)
            delta_seconds = abs((observed_at - subject.last_seen_at).total_seconds())
            temporal_coherence_score = max(
                0.0,
                round(1.0 - (delta_seconds / settings.cross_camera_time_window_seconds), 4),
            )
            camera_switch_score = 1.0

            identity_bonus = 0.0
            if (
                match_result
                and match_result.identified
                and match_result.best_match is not None
                and resolved_identity.get("person_profile_id") == match_result.best_match.person_profile_id
            ):
                identity_bonus = 0.05

            aggregate_score = min(
                1.0,
                round(
                    (0.75 * face_similarity_score)
                    + (0.15 * temporal_coherence_score)
                    + (0.10 * camera_switch_score)
                    + identity_bonus,
                    4,
                ),
            )
            candidates.append(
                CrossCameraCandidate(
                    observed_subject_id=str(subject.observed_subject_id),
                    latest_track_id=str(latest_track.human_track_id) if latest_track is not None else None,
                    last_camera_id=candidate_camera_id,
                    last_seen_at=subject.last_seen_at,
                    recurrence_count=int(subject.recurrence_count or 1),
                    face_similarity_score=face_similarity_score,
                    temporal_coherence_score=temporal_coherence_score,
                    camera_switch_score=round(camera_switch_score, 4),
                    aggregate_score=aggregate_score,
                    resolved_identity=resolved_identity,
                )
            )

        assessment.evaluated_candidates = len(candidates)
        if not candidates:
            assessment.decision_reason = ["cross_camera_candidates_missing_embeddings"]
            return assessment

        candidates.sort(key=lambda candidate: candidate.aggregate_score, reverse=True)
        assessment.best_candidate = candidates[0]
        assessment.second_best_candidate = candidates[1] if len(candidates) > 1 else None
        if assessment.second_best_candidate is not None:
            assessment.second_best_margin = round(
                assessment.best_candidate.aggregate_score - assessment.second_best_candidate.aggregate_score,
                4,
            )

        if assessment.best_candidate.aggregate_score >= settings.cross_camera_match_threshold:
            if assessment.second_best_candidate and assessment.second_best_margin < settings.second_best_margin:
                assessment.decision_reason = [
                    "cross_camera_candidate_above_threshold",
                    "cross_camera_second_best_margin_not_met",
                ]
            else:
                assessment.decision_reason = ["cross_camera_candidate_above_threshold"]
        elif assessment.best_candidate.aggregate_score >= settings.manual_review_threshold:
            assessment.decision_reason = ["cross_camera_candidate_needs_review"]
        else:
            assessment.decision_reason = ["cross_camera_candidate_below_review_threshold"]
        return assessment
