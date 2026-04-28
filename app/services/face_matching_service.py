from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.domain.entities import FaceEmbeddingResult, FaceMatchCandidate, FaceMatchResult, KnownFaceGalleryEntry
from app.infra.repository import RecognitionRepository
from app.services.face_embedding_service import FaceEmbeddingService


class FaceMatchingService:
    def __init__(self, *, repo: RecognitionRepository, embedding_service: FaceEmbeddingService) -> None:
        self.repo = repo
        self.embedding_service = embedding_service
        self._local_gallery_cache: dict[str, list[KnownFaceGalleryEntry]] = {}

    def match(
        self,
        embedding_result: FaceEmbeddingResult,
        *,
        gallery_override_path: str | None = None,
    ) -> FaceMatchResult:
        strategy = f"cosine_similarity:{settings.embedding_backend}"
        if not embedding_result.generated:
            return FaceMatchResult(
                matching_strategy=strategy,
                threshold=settings.face_match_threshold,
                second_best_margin_threshold=settings.second_best_margin,
                rejection_reasons=["embedding_not_generated", *embedding_result.rejection_reasons],
            )

        gallery_entries = self._load_gallery(gallery_override_path=gallery_override_path)
        if not gallery_entries:
            return FaceMatchResult(
                matching_strategy=strategy,
                threshold=settings.face_match_threshold,
                second_best_margin_threshold=settings.second_best_margin,
                rejection_reasons=["gallery_empty"],
            )

        subject_vector = np.asarray(embedding_result.vector, dtype=np.float32)
        candidates: list[FaceMatchCandidate] = []
        for entry in gallery_entries:
            if len(entry.embedding) != len(embedding_result.vector):
                continue

            gallery_vector = np.asarray(entry.embedding, dtype=np.float32)
            similarity = round(float(np.dot(subject_vector, gallery_vector)), 4)
            candidates.append(
                FaceMatchCandidate(
                    person_profile_id=entry.person_profile_id,
                    full_name=entry.full_name,
                    person_type=entry.person_type,
                    risk_level=entry.risk_level,
                    external_person_key=entry.external_person_key,
                    similarity=similarity,
                    gallery_source=entry.gallery_source,
                )
            )

        if not candidates:
            return FaceMatchResult(
                matching_strategy=strategy,
                threshold=settings.face_match_threshold,
                second_best_margin_threshold=settings.second_best_margin,
                rejection_reasons=["gallery_dimension_mismatch"],
            )

        candidates.sort(key=lambda candidate: candidate.similarity, reverse=True)
        best_match = candidates[0]
        second_best_match = candidates[1] if len(candidates) > 1 else None
        second_best_similarity = second_best_match.similarity if second_best_match else 0.0
        margin = round(best_match.similarity - second_best_similarity, 4)

        rejection_reasons: list[str] = []
        if best_match.similarity < settings.face_match_threshold:
            rejection_reasons.append("match_below_threshold")
        if second_best_match is not None and margin < settings.second_best_margin:
            rejection_reasons.append("second_best_margin_not_met")

        identified = not rejection_reasons
        return FaceMatchResult(
            identified=identified,
            match_confidence=max(0.0, best_match.similarity),
            matching_strategy=strategy,
            threshold=settings.face_match_threshold,
            second_best_margin_threshold=settings.second_best_margin,
            evaluated_candidates=len(candidates),
            gallery_source=best_match.gallery_source,
            best_match=best_match,
            second_best_match=second_best_match,
            best_similarity=best_match.similarity,
            second_best_similarity=second_best_similarity,
            second_best_margin=margin,
            rejection_reasons=rejection_reasons,
        )

    def _load_gallery(self, gallery_override_path: str | None = None) -> list[KnownFaceGalleryEntry]:
        if gallery_override_path:
            return self._load_local_gallery_entries(gallery_override_path)

        db_gallery = self.repo.load_known_face_gallery_entries(embedding_backend=settings.embedding_backend)
        if db_gallery:
            return db_gallery

        return self._load_local_gallery_entries(settings.known_face_gallery_path)

    def _load_local_gallery_entries(self, gallery_path_value: str) -> list[KnownFaceGalleryEntry]:
        if gallery_path_value in self._local_gallery_cache:
            return self._local_gallery_cache[gallery_path_value]

        gallery_path = Path(gallery_path_value)
        if not gallery_path.is_absolute():
            gallery_path = Path.cwd() / gallery_path
        if not gallery_path.exists():
            return []

        data = json.loads(gallery_path.read_text(encoding="utf-8"))
        entries: list[KnownFaceGalleryEntry] = []
        for raw_entry in data.get("entries", []):
            embedding_result = self.embedding_service.generate(frame_ref=raw_entry["source_image_ref"])
            if not embedding_result.generated:
                continue

            entries.append(
                KnownFaceGalleryEntry(
                    person_profile_id=raw_entry["person_profile_id"],
                    external_person_key=raw_entry.get("external_person_key"),
                    full_name=raw_entry["full_name"],
                    person_type=raw_entry.get("person_type", "unknown"),
                    risk_level=raw_entry.get("risk_level", "low"),
                    embedding=embedding_result.vector,
                    embedding_backend=settings.embedding_backend,
                    source_image_ref=raw_entry["source_image_ref"],
                    gallery_source="local_dev_fixture",
                )
            )
        self._local_gallery_cache[gallery_path_value] = entries
        return entries
