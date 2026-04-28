from app.config import settings
from app.consumer import load_fixture_message
from app.domain.entities import FaceEmbeddingResult, KnownFaceGalleryEntry
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.face_matching_service import FaceMatchingService
from app.services.presence_service import PresenceService


class DummyRepo:
    def __init__(self, entries=None) -> None:
        self.entries = entries or []

    def load_known_face_gallery_entries(self, *, embedding_backend: str):
        return [entry for entry in self.entries if entry.embedding_backend == embedding_backend]


def test_match_uses_local_gallery_fallback_for_identified_fixture():
    presence_service = PresenceService()
    embedding_service = FaceEmbeddingService()
    matching_service = FaceMatchingService(repo=DummyRepo(), embedding_service=embedding_service)
    fixture = load_fixture_message("tests/fixtures/frame_ingested_identified.json")

    face_detection = presence_service.inspect_face(
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )
    embedding_result = embedding_service.generate(frame_ref=fixture.frame_ref, face_detection=face_detection)
    match_result = matching_service.match(embedding_result)

    assert match_result.identified is True
    assert match_result.best_match is not None
    assert match_result.best_match.person_profile_id == "22222222-2222-2222-2222-222222222222"
    assert match_result.matching_strategy == f"cosine_similarity:{settings.embedding_backend}"
    assert match_result.match_confidence >= settings.face_match_threshold


def test_match_returns_unidentified_for_face_outside_gallery():
    presence_service = PresenceService()
    embedding_service = FaceEmbeddingService()
    matching_service = FaceMatchingService(repo=DummyRepo(), embedding_service=embedding_service)
    fixture = load_fixture_message("tests/fixtures/frame_ingested_example.json")

    face_detection = presence_service.inspect_face(
        frame_ref=fixture.frame_ref,
        quality_metadata=fixture.payload.quality_metadata,
    )
    embedding_result = embedding_service.generate(frame_ref=fixture.frame_ref, face_detection=face_detection)
    match_result = matching_service.match(embedding_result)

    assert match_result.identified is False
    assert "match_below_threshold" in match_result.rejection_reasons
    assert match_result.best_match is not None
    assert match_result.match_confidence < settings.face_match_threshold


def test_second_best_margin_blocks_identification():
    repo = DummyRepo(
        entries=[
            KnownFaceGalleryEntry(
                person_profile_id="candidate-1",
                full_name="Candidate 1",
                person_type="employee",
                risk_level="low",
                embedding=[1.0, 0.0],
                embedding_backend=settings.embedding_backend,
                gallery_source="test_gallery",
            ),
            KnownFaceGalleryEntry(
                person_profile_id="candidate-2",
                full_name="Candidate 2",
                person_type="employee",
                risk_level="low",
                embedding=[0.98, 0.0],
                embedding_backend=settings.embedding_backend,
                gallery_source="test_gallery",
            ),
        ]
    )
    matching_service = FaceMatchingService(repo=repo, embedding_service=FaceEmbeddingService())
    embedding_result = FaceEmbeddingResult(
        generated=True,
        backend=settings.embedding_backend,
        dimensions=2,
        vector=[1.0, 0.0],
    )

    match_result = matching_service.match(embedding_result)

    assert match_result.identified is False
    assert "second_best_margin_not_met" in match_result.rejection_reasons
    assert match_result.best_similarity == 1.0
    assert match_result.second_best_similarity == 0.98
