from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from app.config import settings
from app.consumer import load_fixture_message
from app.domain.entities import CrossCameraAssessment, CrossCameraCandidate, FaceMatchCandidate, FaceMatchResult
from app.services.conflict_resolution_service import ConflictResolutionService
from app.services.cross_camera_correlation_service import CrossCameraCorrelationService
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.presence_service import PresenceService


class DummyRepo:
    def __init__(self, candidates=None) -> None:
        self.candidates = candidates or []

    def load_recent_subject_candidates(self, **kwargs):
        return self.candidates


def test_cross_camera_assessment_detects_strong_candidate():
    presence_service = PresenceService()
    embedding_service = FaceEmbeddingService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_identified.json")
    face_detection = presence_service.inspect_face(
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )
    embedding_result = embedding_service.generate(frame_ref=fixture.frame_ref, face_detection=face_detection)

    current_subject = SimpleNamespace(observed_subject_id=uuid4())
    current_track = SimpleNamespace(human_track_id=uuid4())
    candidate_subject = SimpleNamespace(
        observed_subject_id=uuid4(),
        last_seen_at=fixture.captured_at - timedelta(seconds=60),
        last_camera_id="11111111-1111-1111-1111-111111111111",
        recurrence_count=2,
    )
    candidate_track = SimpleNamespace(human_track_id=uuid4())
    repo = DummyRepo(
        candidates=[
            {
                "subject": candidate_subject,
                "latest_track": candidate_track,
                "embedding": embedding_result.vector,
                "resolved_identity": {
                    "person_profile_id": "22222222-2222-2222-2222-222222222222",
                    "full_name": "Barack Obama (dev fixture)",
                },
            }
        ]
    )

    assessment = CrossCameraCorrelationService(repo=repo).evaluate(
        current_subject=current_subject,
        current_track=current_track,
        observed_at=fixture.captured_at,
        current_camera_id="22222222-1111-1111-1111-111111111111",
        embedding_result=embedding_result,
        match_result=None,
    )

    assert assessment.best_candidate is not None
    assert assessment.best_candidate.aggregate_score >= settings.cross_camera_match_threshold
    assert assessment.decision_reason == ["cross_camera_candidate_above_threshold"]


def test_conflict_resolution_resolves_auto_correlation():
    current_subject_id = str(uuid4())
    current_track_id = str(uuid4())
    assessment = CrossCameraAssessment(
        current_subject_id=current_subject_id,
        current_track_id=current_track_id,
        current_camera_id="22222222-1111-1111-1111-111111111111",
        threshold=settings.cross_camera_match_threshold,
        manual_review_threshold=settings.manual_review_threshold,
        second_best_margin_threshold=settings.second_best_margin,
        evaluated_candidates=1,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(uuid4()),
            latest_track_id=str(uuid4()),
            last_camera_id="11111111-1111-1111-1111-111111111111",
            last_seen_at=load_fixture_message("tests/fixtures/frame_ingested_identified.json").captured_at,
            recurrence_count=2,
            face_similarity_score=1.0,
            temporal_coherence_score=0.9,
            camera_switch_score=1.0,
            aggregate_score=0.985,
            resolved_identity={
                "person_profile_id": "22222222-2222-2222-2222-222222222222",
            },
        ),
        decision_reason=["cross_camera_candidate_above_threshold"],
    )
    match_result = FaceMatchResult(
        identified=True,
        match_confidence=1.0,
        matching_strategy="cosine_similarity:simple_face_crop_512",
        threshold=settings.face_match_threshold,
        second_best_margin_threshold=settings.second_best_margin,
        best_match=FaceMatchCandidate(
            person_profile_id="22222222-2222-2222-2222-222222222222",
            full_name="Barack Obama (dev fixture)",
            person_type="employee",
            risk_level="low",
            similarity=1.0,
            gallery_source="local_dev_fixture",
        ),
        best_similarity=1.0,
    )

    resolution = ConflictResolutionService().resolve(
        current_subject=SimpleNamespace(observed_subject_id=current_subject_id),
        current_track=SimpleNamespace(human_track_id=current_track_id),
        assessment=assessment,
        match_result=match_result,
    )

    assert resolution.outcome == "correlated"
    assert resolution.correlation_status == "auto"
    assert resolution.subject_id_to_use == assessment.best_candidate.observed_subject_id
    assert resolution.supplemental_decisions[0].event_type == "cross_camera_subject_correlated"


def test_conflict_resolution_detects_identity_conflict():
    current_subject_id = str(uuid4())
    current_track_id = str(uuid4())
    assessment = CrossCameraAssessment(
        current_subject_id=current_subject_id,
        current_track_id=current_track_id,
        current_camera_id="33333333-1111-1111-1111-111111111111",
        threshold=settings.cross_camera_match_threshold,
        manual_review_threshold=settings.manual_review_threshold,
        second_best_margin_threshold=settings.second_best_margin,
        evaluated_candidates=1,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(uuid4()),
            latest_track_id=str(uuid4()),
            last_camera_id="11111111-1111-1111-1111-111111111111",
            last_seen_at=load_fixture_message("tests/fixtures/frame_ingested_identified.json").captured_at,
            recurrence_count=2,
            face_similarity_score=1.0,
            temporal_coherence_score=0.9,
            camera_switch_score=1.0,
            aggregate_score=1.0,
            resolved_identity={
                "person_profile_id": "22222222-2222-2222-2222-222222222222",
                "full_name": "Barack Obama (dev fixture)",
            },
        ),
        decision_reason=["cross_camera_candidate_above_threshold"],
    )
    match_result = FaceMatchResult(
        identified=True,
        match_confidence=1.0,
        matching_strategy="cosine_similarity:simple_face_crop_512",
        threshold=settings.face_match_threshold,
        second_best_margin_threshold=settings.second_best_margin,
        best_match=FaceMatchCandidate(
            person_profile_id="44444444-4444-4444-4444-444444444444",
            full_name="Conflict Identity (dev fixture)",
            person_type="employee",
            risk_level="medium",
            similarity=1.0,
            gallery_source="local_dev_fixture",
        ),
        best_similarity=1.0,
    )

    resolution = ConflictResolutionService().resolve(
        current_subject=SimpleNamespace(observed_subject_id=current_subject_id),
        current_track=SimpleNamespace(human_track_id=current_track_id),
        assessment=assessment,
        match_result=match_result,
    )

    assert resolution.outcome == "identity_conflict"
    assert resolution.requires_human_review is True
    assert [decision.event_type for decision in resolution.supplemental_decisions] == [
        "identity_conflict",
        "manual_review_required",
    ]


def test_conflict_resolution_requests_manual_review_for_uncertain_candidate():
    current_subject_id = str(uuid4())
    current_track_id = str(uuid4())
    assessment = CrossCameraAssessment(
        current_subject_id=current_subject_id,
        current_track_id=current_track_id,
        current_camera_id="44444444-1111-1111-1111-111111111111",
        threshold=settings.cross_camera_match_threshold,
        manual_review_threshold=settings.manual_review_threshold,
        second_best_margin_threshold=settings.second_best_margin,
        evaluated_candidates=2,
        best_candidate=CrossCameraCandidate(
            observed_subject_id=str(uuid4()),
            latest_track_id=str(uuid4()),
            last_camera_id="11111111-1111-1111-1111-111111111111",
            last_seen_at=load_fixture_message("tests/fixtures/frame_ingested_identified.json").captured_at,
            recurrence_count=2,
            face_similarity_score=0.2673,
            temporal_coherence_score=0.75,
            camera_switch_score=1.0,
            aggregate_score=0.4125,
            resolved_identity={},
        ),
        second_best_candidate=CrossCameraCandidate(
            observed_subject_id=str(uuid4()),
            latest_track_id=str(uuid4()),
            last_camera_id="33333333-1111-1111-1111-111111111111",
            last_seen_at=load_fixture_message("tests/fixtures/frame_ingested_identified.json").captured_at,
            recurrence_count=1,
            face_similarity_score=0.261,
            temporal_coherence_score=0.75,
            camera_switch_score=1.0,
            aggregate_score=0.408,
            resolved_identity={},
        ),
        second_best_margin=0.0045,
        decision_reason=["cross_camera_candidate_needs_review"],
    )

    resolution = ConflictResolutionService().resolve(
        current_subject=SimpleNamespace(observed_subject_id=current_subject_id),
        current_track=SimpleNamespace(human_track_id=current_track_id),
        assessment=assessment,
        match_result=None,
    )

    assert resolution.outcome == "manual_review_required"
    assert resolution.requires_human_review is True
    assert resolution.supplemental_decisions[0].event_type == "manual_review_required"
