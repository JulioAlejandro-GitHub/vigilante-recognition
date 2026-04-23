from __future__ import annotations

from app.config import settings
from app.domain.entities import ContinuityResolution, CrossCameraAssessment, FaceMatchResult, SupplementalRecognitionDecision


class ConflictResolutionService:
    def resolve(
        self,
        *,
        current_subject,
        current_track,
        assessment: CrossCameraAssessment,
        match_result: FaceMatchResult | None,
    ) -> ContinuityResolution:
        resolution = ContinuityResolution(
            outcome="none",
            subject_id_to_use=str(current_subject.observed_subject_id),
            assessment=assessment,
        )
        best_candidate = assessment.best_candidate
        if best_candidate is None:
            resolution.decision_reason = assessment.decision_reason or ["cross_camera_no_candidate"]
            return resolution

        current_identity = {}
        if match_result and match_result.identified and match_result.best_match is not None:
            current_identity = {
                "person_profile_id": match_result.best_match.person_profile_id,
                "external_person_key": match_result.best_match.external_person_key,
                "full_name": match_result.best_match.full_name,
                "person_type": match_result.best_match.person_type,
                "risk_level": match_result.best_match.risk_level,
                "match_confidence": match_result.match_confidence,
            }

        payload = {
            "current_subject_id": str(current_subject.observed_subject_id),
            "current_track_id": str(current_track.human_track_id),
            "cross_camera_assessment": self._serialize_assessment(assessment),
        }
        if current_identity:
            payload["current_identity"] = current_identity

        candidate_identity = best_candidate.resolved_identity or {}
        if current_identity and candidate_identity:
            identity_gap = round(abs(current_identity["match_confidence"] - best_candidate.aggregate_score), 4)
            if (
                current_identity["person_profile_id"] != candidate_identity.get("person_profile_id")
                and best_candidate.aggregate_score >= settings.manual_review_threshold
                and identity_gap <= settings.identity_conflict_margin
            ):
                resolution.outcome = "identity_conflict"
                resolution.requires_human_review = True
                resolution.correlation_status = "pending_review"
                resolution.decision_reason = [
                    "identity_signals_incompatible",
                    "cross_camera_candidate_strong",
                    "human_review_required",
                ]
                resolution.payload = {
                    **payload,
                    "conflict_type": "incompatible_identity_signals",
                    "candidate_identity": candidate_identity,
                    "identity_gap": identity_gap,
                    "requires_human_review": True,
                }
                resolution.supplemental_decisions = [
                    SupplementalRecognitionDecision(
                        event_type="identity_conflict",
                        severity="high",
                        confidence=max(current_identity["match_confidence"], best_candidate.aggregate_score),
                        decision_reason=resolution.decision_reason,
                        payload=resolution.payload,
                        subject_id=str(current_subject.observed_subject_id),
                    ),
                    SupplementalRecognitionDecision(
                        event_type="manual_review_required",
                        severity="medium",
                        confidence=best_candidate.aggregate_score,
                        decision_reason=[
                            "identity_conflict_detected",
                            "manual_resolution_required",
                        ],
                        payload={
                            **resolution.payload,
                            "review_type": "identity_conflict",
                            "requires_human_review": True,
                        },
                        subject_id=str(current_subject.observed_subject_id),
                    ),
                ]
                return resolution

        auto_correlation = (
            best_candidate.aggregate_score >= settings.cross_camera_match_threshold
            and (
                assessment.second_best_candidate is None
                or assessment.second_best_margin >= settings.second_best_margin
            )
        )
        if auto_correlation:
            resolution.outcome = "correlated"
            resolution.subject_id_to_use = best_candidate.observed_subject_id
            resolution.target_track_id = best_candidate.latest_track_id
            resolution.correlation_status = "auto"
            resolution.decision_reason = [
                "cross_camera_candidate_above_threshold",
                "cross_camera_auto_resolved",
            ]
            resolution.payload = {
                **payload,
                "source_subject_id": str(current_subject.observed_subject_id),
                "target_subject_id": best_candidate.observed_subject_id,
                "correlation_status": "auto",
            }
            resolution.supplemental_decisions = [
                SupplementalRecognitionDecision(
                    event_type="cross_camera_subject_correlated",
                    severity="medium",
                    confidence=best_candidate.aggregate_score,
                    decision_reason=resolution.decision_reason,
                    payload=resolution.payload,
                    subject_id=best_candidate.observed_subject_id,
                )
            ]
            return resolution

        if best_candidate.aggregate_score >= settings.manual_review_threshold:
            review_reason = ["cross_camera_candidate_needs_review"]
            if assessment.second_best_candidate is not None and assessment.second_best_margin < settings.second_best_margin:
                review_reason.append("cross_camera_second_best_margin_not_met")
            if current_identity and candidate_identity and current_identity["person_profile_id"] != candidate_identity.get("person_profile_id"):
                review_reason.append("identity_signals_divergent_but_not_conflict")

            resolution.outcome = "manual_review_required"
            resolution.requires_human_review = True
            resolution.correlation_status = "pending_review"
            resolution.decision_reason = review_reason
            resolution.payload = {
                **payload,
                "review_type": "cross_camera_correlation",
                "requires_human_review": True,
            }
            resolution.supplemental_decisions = [
                SupplementalRecognitionDecision(
                    event_type="manual_review_required",
                    severity="medium",
                    confidence=best_candidate.aggregate_score,
                    decision_reason=review_reason,
                    payload=resolution.payload,
                    subject_id=str(current_subject.observed_subject_id),
                )
            ]
            return resolution

        resolution.decision_reason = assessment.decision_reason or ["cross_camera_no_resolution"]
        return resolution

    def _serialize_assessment(self, assessment: CrossCameraAssessment) -> dict:
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
                    "face_similarity_score": candidate.face_similarity_score,
                    "temporal_coherence_score": candidate.temporal_coherence_score,
                    "camera_switch_score": candidate.camera_switch_score,
                    "aggregate_score": candidate.aggregate_score,
                    "resolved_identity": candidate.resolved_identity,
                }
            )
        return {
            "threshold": assessment.threshold,
            "manual_review_threshold": assessment.manual_review_threshold,
            "second_best_margin_threshold": assessment.second_best_margin_threshold,
            "second_best_margin": assessment.second_best_margin,
            "evaluated_candidates": assessment.evaluated_candidates,
            "decision_reason": assessment.decision_reason,
            "candidates": candidates,
        }
