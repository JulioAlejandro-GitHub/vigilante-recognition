from __future__ import annotations

from typing import Protocol

from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult
from app.services.face_embedding_service import FaceEmbeddingService
from app.services.presence_service import PresenceService


class FaceBackendError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        backend_key: str,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.backend_key = backend_key
        self.stage = stage
        self.details = details or {}


class FaceBackend(Protocol):
    backend_key: str
    backend_name: str
    provider_name: str | None

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
    ) -> FaceDetectionResult:
        ...

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
    ) -> FaceEmbeddingResult:
        ...


class SimpleFaceBackend:
    backend_key = "simple"
    backend_name = "simple_opencv_haar"
    provider_name = "opencv_cpu"

    def __init__(
        self,
        *,
        presence_service: PresenceService | None = None,
        embedding_service: FaceEmbeddingService | None = None,
    ) -> None:
        self.presence_service = presence_service or PresenceService()
        self.embedding_service = embedding_service or FaceEmbeddingService()

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
    ) -> FaceDetectionResult:
        return self.presence_service.inspect_face(
            frame_ref=frame_ref,
            quality_metadata=quality_metadata or {},
        )

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
    ) -> FaceEmbeddingResult:
        return self.embedding_service.generate(
            frame_ref=frame_ref,
            face_detection=face_detection,
        )
