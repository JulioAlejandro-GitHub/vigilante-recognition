from types import SimpleNamespace
from uuid import uuid4

from app.domain.entities import RecurrentSubjectAssessment, SemanticDescriptorResult
from app.services.recurrent_subject_service import RecurrentSubjectService
from app.services.semantic_descriptor_service import SemanticDescriptorService


def _descriptor(upper: str, middle: str, lower: str) -> SemanticDescriptorResult:
    signature = {
        "upper_region_color": upper,
        "middle_region_color": middle,
        "lower_region_color": lower,
        "dominant_palette": [upper, middle, lower],
        "contrast_level": "medium",
        "saturation_level": "low",
        "frame_aspect_ratio": "portrait",
        "horizontal_position": "center",
        "subject_scale": "upper_body",
        "face_state": "low_quality_face",
    }
    return SemanticDescriptorResult(
        generated=True,
        backend="simple_color_signature_v1",
        descriptor={"signature": signature},
        signature=signature,
        confidence=0.84,
        source_frame_ref="tests/fixtures/images/face_low_quality.jpg",
    )


def _subject(subject_id, camera_id, observed_at, recurrence_count=1):
    return SimpleNamespace(
        observed_subject_id=subject_id,
        last_seen_at=observed_at,
        last_camera_id=camera_id,
        recurrence_count=recurrence_count,
    )


def test_recurrent_subject_resolution_emits_recurrence_and_manual_review():
    semantic_service = SemanticDescriptorService()
    repo = SimpleNamespace(
        load_recent_unresolved_subject_candidates=lambda **_: [
            {
                "subject": _subject(uuid4(), "11111111-1111-1111-1111-111111111111", _NOW, recurrence_count=1),
                "latest_track": SimpleNamespace(human_track_id=uuid4()),
                "semantic_descriptor": {
                    "backend": "simple_color_signature_v1",
                    "descriptor": {},
                    "signature": _descriptor("gray", "gray", "blue").signature,
                    "confidence": 0.84,
                },
                "embedding": None,
                "continuity_resolution": {},
            }
        ]
    )
    service = RecurrentSubjectService(repo=repo, semantic_descriptor_service=semantic_service)
    current_subject = SimpleNamespace(observed_subject_id=uuid4())
    current_track = SimpleNamespace(human_track_id=uuid4())

    assessment = service.evaluate(
        current_subject=current_subject,
        current_track=current_track,
        observed_at=_NOW,
        current_camera_id="55555555-1111-1111-1111-111111111111",
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        embedding_result=None,
    )
    resolution = service.resolve(
        current_subject=current_subject,
        current_track=current_track,
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        assessment=assessment,
    )

    assert assessment.best_candidate is not None
    assert resolution.outcome == "recurrent_unresolved"
    assert resolution.requires_human_review is True
    assert [decision.event_type for decision in resolution.supplemental_decisions] == [
        "recurrent_unresolved_subject",
        "manual_review_required",
    ]


def test_recurrent_subject_resolution_emits_case_suggestion_on_third_occurrence():
    semantic_service = SemanticDescriptorService()
    repo = SimpleNamespace(
        load_recent_unresolved_subject_candidates=lambda **_: [
            {
                "subject": _subject(uuid4(), "11111111-1111-1111-1111-111111111111", _NOW, recurrence_count=2),
                "latest_track": SimpleNamespace(human_track_id=uuid4()),
                "semantic_descriptor": {
                    "backend": "simple_color_signature_v1",
                    "descriptor": {},
                    "signature": _descriptor("gray", "gray", "blue").signature,
                    "confidence": 0.84,
                },
                "embedding": None,
                "continuity_resolution": {},
            }
        ]
    )
    service = RecurrentSubjectService(repo=repo, semantic_descriptor_service=semantic_service)
    current_subject = SimpleNamespace(observed_subject_id=uuid4())
    current_track = SimpleNamespace(human_track_id=uuid4())

    assessment = service.evaluate(
        current_subject=current_subject,
        current_track=current_track,
        observed_at=_NOW,
        current_camera_id="66666666-1111-1111-1111-111111111111",
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        embedding_result=None,
    )
    resolution = service.resolve(
        current_subject=current_subject,
        current_track=current_track,
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        assessment=assessment,
    )

    assert resolution.requires_case_evaluation is True
    assert [decision.event_type for decision in resolution.supplemental_decisions] == [
        "recurrent_unresolved_subject",
        "manual_review_required",
        "case_suggestion_created",
    ]


def test_recurrent_subject_assessment_rejects_low_similarity_candidates():
    semantic_service = SemanticDescriptorService()
    repo = SimpleNamespace(
        load_recent_unresolved_subject_candidates=lambda **_: [
            {
                "subject": _subject(uuid4(), "11111111-1111-1111-1111-111111111111", _NOW, recurrence_count=1),
                "latest_track": SimpleNamespace(human_track_id=uuid4()),
                "semantic_descriptor": {
                    "backend": "simple_color_signature_v1",
                    "descriptor": {},
                    "signature": _descriptor("red", "yellow", "orange").signature,
                    "confidence": 0.84,
                },
                "embedding": None,
                "continuity_resolution": {},
            }
        ]
    )
    service = RecurrentSubjectService(repo=repo, semantic_descriptor_service=semantic_service)
    current_subject = SimpleNamespace(observed_subject_id=uuid4())
    current_track = SimpleNamespace(human_track_id=uuid4())

    assessment = service.evaluate(
        current_subject=current_subject,
        current_track=current_track,
        observed_at=_NOW,
        current_camera_id="55555555-1111-1111-1111-111111111111",
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        embedding_result=None,
    )
    resolution = service.resolve(
        current_subject=current_subject,
        current_track=current_track,
        semantic_descriptor_result=_descriptor("gray", "gray", "blue"),
        assessment=assessment,
    )

    assert assessment.best_candidate is None
    assert resolution.outcome == "none"
    assert resolution.supplemental_decisions == []


_NOW = __import__("datetime").datetime.fromisoformat("2026-05-10T10:05:45.123+00:00")
