from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult


class FaceEmbeddingService:
    def __init__(self) -> None:
        if settings.embedding_backend != "simple_face_crop_512":
            raise RuntimeError(f"Unsupported embedding backend: {settings.embedding_backend}")

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
    ) -> FaceEmbeddingResult:
        frame_path = self._resolve_frame_path(frame_ref)
        if frame_path is None:
            return FaceEmbeddingResult(
                backend=settings.embedding_backend,
                source_frame_ref=frame_ref,
                rejection_reasons=["frame_ref_not_found"],
            )

        image = cv2.imread(str(frame_path))
        if image is None:
            return FaceEmbeddingResult(
                backend=settings.embedding_backend,
                source_frame_ref=frame_ref,
                rejection_reasons=["frame_unreadable"],
            )

        bbox = face_detection.bbox if face_detection and face_detection.bbox else self._detect_best_face_bbox(image)
        if bbox is None:
            return FaceEmbeddingResult(
                backend=settings.embedding_backend,
                source_frame_ref=frame_ref,
                rejection_reasons=["face_not_detected_for_embedding"],
            )

        face_crop = self._extract_face_crop(image=image, bbox=bbox)
        if face_crop.size == 0:
            return FaceEmbeddingResult(
                backend=settings.embedding_backend,
                source_frame_ref=frame_ref,
                bbox=bbox,
                rejection_reasons=["empty_face_crop"],
            )

        grayscale = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(grayscale)
        resized = cv2.resize(equalized, (32, 16), interpolation=cv2.INTER_AREA)
        vector = resized.astype(np.float32).reshape(-1) / 255.0
        vector = vector - float(vector.mean())
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return FaceEmbeddingResult(
                backend=settings.embedding_backend,
                source_frame_ref=frame_ref,
                bbox=bbox,
                rejection_reasons=["embedding_norm_is_zero"],
            )

        normalized_vector = vector / norm
        embedding = [round(float(value), 6) for value in normalized_vector.tolist()]

        return FaceEmbeddingResult(
            generated=True,
            backend=settings.embedding_backend,
            dimensions=len(embedding),
            vector=embedding,
            bbox=bbox,
            source_frame_ref=frame_ref,
        )

    def _detect_best_face_bbox(self, image) -> dict[str, int] | None:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda bbox: int(bbox[2]) * int(bbox[3]))
        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }

    def _extract_face_crop(self, *, image, bbox: dict[str, int]):
        x = max(0, int(bbox["x"]))
        y = max(0, int(bbox["y"]))
        width = max(1, int(bbox["width"]))
        height = max(1, int(bbox["height"]))
        return image[y : y + height, x : x + width]

    def _resolve_frame_path(self, frame_ref: str) -> Path | None:
        frame_path = Path(frame_ref)
        if frame_path.is_absolute() and frame_path.exists():
            return frame_path

        relative_path = Path.cwd() / frame_ref
        if relative_path.exists():
            return relative_path

        return None
