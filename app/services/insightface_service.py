from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult
from app.services.face_backend_service import FaceBackendError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProviderConfig:
    provider_name: str
    providers: list[str]
    ctx_id: int


@dataclass(frozen=True)
class _InsightFaceAnalysis:
    detected: bool
    usable: bool
    image_size: dict[str, int] | None
    bbox: dict[str, int] | None
    quality_score: float
    quality_metrics: dict[str, float]
    rejection_reasons: list[str]
    embedding: list[float]


class InsightFaceService:
    backend_key = "insightface"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        provider: str | None = None,
        model_root: str | None = None,
        det_size: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.insightface_model_name
        self.provider = provider or settings.insightface_provider
        self.model_root = model_root if model_root is not None else settings.insightface_model_root
        self.det_size = det_size or settings.insightface_det_size
        self.backend_name = f"insightface:{self.model_name}"
        self.provider_name = self.provider
        self._app: Any | None = None
        self._provider_config: _ProviderConfig | None = None
        self._analysis_cache: dict[str, _InsightFaceAnalysis] = {}

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
    ) -> FaceDetectionResult:
        analysis = self._analyze(frame_ref=frame_ref)
        return FaceDetectionResult(
            detected=analysis.detected,
            usable=analysis.usable,
            quality_score=analysis.quality_score,
            bbox=analysis.bbox,
            image_size=analysis.image_size,
            rejection_reasons=analysis.rejection_reasons,
            quality_metrics=analysis.quality_metrics,
            frame_quality_metadata=quality_metadata or {},
        )

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
    ) -> FaceEmbeddingResult:
        analysis = self._analyze(frame_ref=frame_ref)
        if face_detection is not None and not face_detection.usable:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                bbox=face_detection.bbox,
                rejection_reasons=["face_not_usable_for_embedding"],
            )
        if not analysis.detected:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                rejection_reasons=["face_not_detected_for_embedding", *analysis.rejection_reasons],
            )
        if not analysis.embedding:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                bbox=analysis.bbox,
                rejection_reasons=["embedding_not_available"],
            )

        return FaceEmbeddingResult(
            generated=True,
            backend=self.backend_name,
            dimensions=len(analysis.embedding),
            vector=analysis.embedding,
            bbox=analysis.bbox,
            source_frame_ref=frame_ref,
        )

    def _analyze(self, *, frame_ref: str) -> _InsightFaceAnalysis:
        cache_key = self._cache_key(frame_ref)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        frame_path = self._resolve_frame_path(frame_ref)
        if frame_path is None:
            analysis = _InsightFaceAnalysis(
                detected=False,
                usable=False,
                image_size=None,
                bbox=None,
                quality_score=0.0,
                quality_metrics={},
                rejection_reasons=["frame_ref_not_found"],
                embedding=[],
            )
            self._analysis_cache[cache_key] = analysis
            return analysis

        image = cv2.imread(str(frame_path))
        if image is None:
            analysis = _InsightFaceAnalysis(
                detected=False,
                usable=False,
                image_size=None,
                bbox=None,
                quality_score=0.0,
                quality_metrics={},
                rejection_reasons=["frame_unreadable"],
                embedding=[],
            )
            self._analysis_cache[cache_key] = analysis
            return analysis

        image_height, image_width = image.shape[:2]
        image_size = {"width": int(image_width), "height": int(image_height)}
        try:
            faces = self._load_app().get(image)
        except FaceBackendError:
            raise
        except Exception as exc:  # pragma: no cover - depends on optional runtime/provider failures
            raise FaceBackendError(
                f"insightface_runtime_error:{type(exc).__name__}",
                backend_key=self.backend_key,
                stage="detect",
                details={"error": str(exc)},
            ) from exc

        if not faces:
            analysis = _InsightFaceAnalysis(
                detected=False,
                usable=False,
                image_size=image_size,
                bbox=None,
                quality_score=0.0,
                quality_metrics={},
                rejection_reasons=["face_not_detected"],
                embedding=[],
            )
            self._analysis_cache[cache_key] = analysis
            return analysis

        face = max(faces, key=lambda candidate: self._face_area(candidate))
        bbox = self._bbox(face, image_width=image_width, image_height=image_height)
        face_crop = self._extract_face_crop(image=image, bbox=bbox)
        quality_metrics = self._compute_quality_metrics(
            image_width=image_width,
            image_height=image_height,
            bbox=bbox,
            face_crop=face_crop,
            detection_score=float(getattr(face, "det_score", 0.0) or 0.0),
        )
        quality_score = round(
            (0.3 * quality_metrics["detection_confidence"])
            + (0.25 * quality_metrics["blur_score"])
            + (0.2 * quality_metrics["brightness_score"])
            + (0.15 * quality_metrics["size_score"])
            + (0.1 * quality_metrics["centered_score"]),
            4,
        )
        usable = quality_score >= settings.face_quality_threshold
        rejection_reasons = [] if usable else ["face_quality_threshold_failed"]
        analysis = _InsightFaceAnalysis(
            detected=True,
            usable=usable,
            image_size=image_size,
            bbox=bbox,
            quality_score=quality_score,
            quality_metrics=quality_metrics,
            rejection_reasons=rejection_reasons,
            embedding=self._embedding(face),
        )
        self._analysis_cache[cache_key] = analysis
        return analysis

    def _load_app(self):
        if self._app is not None:
            return self._app
        if not settings.insightface_enabled:
            raise FaceBackendError(
                "insightface_disabled",
                backend_key=self.backend_key,
                stage="configuration",
            )

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FaceBackendError(
                "insightface_not_installed",
                backend_key=self.backend_key,
                stage="import",
            ) from exc

        provider_config = self._resolve_provider_config()
        root = self._model_root_path()
        kwargs: dict[str, object] = {
            "name": self.model_name,
            "providers": provider_config.providers,
        }
        if root is not None:
            kwargs["root"] = str(root)
        try:
            app = FaceAnalysis(**kwargs)
            app.prepare(ctx_id=provider_config.ctx_id, det_size=self._det_size_tuple())
        except Exception as exc:  # pragma: no cover - depends on optional model/provider state
            raise FaceBackendError(
                f"insightface_load_failed:{type(exc).__name__}",
                backend_key=self.backend_key,
                stage="load",
                details={
                    "error": str(exc),
                    "model_name": self.model_name,
                    "provider": provider_config.provider_name,
                    "providers": provider_config.providers,
                    "model_root": str(root) if root is not None else None,
                },
            ) from exc

        self._app = app
        self._provider_config = provider_config
        self.provider_name = provider_config.provider_name
        logger.info(
            "insightface_backend_loaded model_name=%s provider=%s providers=%s model_root=%s det_size=%s ctx_id=%s",
            self.model_name,
            provider_config.provider_name,
            ",".join(provider_config.providers),
            str(root) if root is not None else "<default>",
            self.det_size,
            provider_config.ctx_id,
        )
        return self._app

    def _resolve_provider_config(self) -> _ProviderConfig:
        raw_provider = (self.provider or "cpu").strip()
        normalized = raw_provider.lower()
        if normalized in {"cpu", "cpu_execution_provider"}:
            return _ProviderConfig(
                provider_name="cpu",
                providers=["CPUExecutionProvider"],
                ctx_id=-1,
            )
        if normalized in {"cuda", "cuda_execution_provider", "gpu"}:
            return _ProviderConfig(
                provider_name="cuda",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                ctx_id=0,
            )
        if "," in raw_provider:
            providers = [part.strip() for part in raw_provider.split(",") if part.strip()]
            if not providers:
                raise FaceBackendError(
                    "insightface_provider_invalid",
                    backend_key=self.backend_key,
                    stage="configuration",
                )
            has_cpu_only = providers == ["CPUExecutionProvider"]
            return _ProviderConfig(
                provider_name=raw_provider,
                providers=providers,
                ctx_id=-1 if has_cpu_only else 0,
            )
        if raw_provider.endswith("ExecutionProvider"):
            return _ProviderConfig(
                provider_name=raw_provider,
                providers=[raw_provider],
                ctx_id=-1 if raw_provider == "CPUExecutionProvider" else 0,
            )
        raise FaceBackendError(
            "insightface_provider_invalid",
            backend_key=self.backend_key,
            stage="configuration",
            details={"provider": raw_provider},
        )

    def _det_size_tuple(self) -> tuple[int, int]:
        try:
            raw_width, raw_height = self.det_size.lower().replace("x", ",").split(",", 1)
            width = int(raw_width.strip())
            height = int(raw_height.strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise FaceBackendError(
                "insightface_det_size_invalid",
                backend_key=self.backend_key,
                stage="configuration",
                details={"det_size": self.det_size},
            ) from exc
        if width <= 0 or height <= 0:
            raise FaceBackendError(
                "insightface_det_size_invalid",
                backend_key=self.backend_key,
                stage="configuration",
                details={"det_size": self.det_size},
            )
        return width, height

    def _model_root_path(self) -> Path | None:
        if not self.model_root:
            return None
        return Path(self.model_root).expanduser()

    def _cache_key(self, frame_ref: str) -> str:
        frame_path = self._resolve_frame_path(frame_ref)
        if frame_path is None:
            return frame_ref
        try:
            stat = frame_path.stat()
        except OSError:
            return str(frame_path)
        return f"{frame_path}:{stat.st_mtime_ns}:{stat.st_size}"

    def _resolve_frame_path(self, frame_ref: str) -> Path | None:
        frame_path = Path(frame_ref)
        if frame_path.is_absolute() and frame_path.exists():
            return frame_path
        relative_path = Path.cwd() / frame_ref
        if relative_path.exists():
            return relative_path
        return None

    def _face_area(self, face) -> float:
        raw_bbox = getattr(face, "bbox", None)
        if raw_bbox is None or len(raw_bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = [float(value) for value in raw_bbox[:4]]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _bbox(self, face, *, image_width: int, image_height: int) -> dict[str, int]:
        raw_bbox = getattr(face, "bbox", None)
        if raw_bbox is None or len(raw_bbox) < 4:
            return {"x": 0, "y": 0, "width": 1, "height": 1}
        x1, y1, x2, y2 = [float(value) for value in raw_bbox[:4]]
        x = max(0, min(int(round(x1)), image_width - 1))
        y = max(0, min(int(round(y1)), image_height - 1))
        right = max(x + 1, min(int(round(x2)), image_width))
        bottom = max(y + 1, min(int(round(y2)), image_height))
        return {
            "x": int(x),
            "y": int(y),
            "width": int(right - x),
            "height": int(bottom - y),
        }

    def _extract_face_crop(self, *, image, bbox: dict[str, int]):
        x = max(0, int(bbox["x"]))
        y = max(0, int(bbox["y"]))
        width = max(1, int(bbox["width"]))
        height = max(1, int(bbox["height"]))
        return image[y : y + height, x : x + width]

    def _compute_quality_metrics(
        self,
        *,
        image_width: int,
        image_height: int,
        bbox: dict[str, int],
        face_crop,
        detection_score: float,
    ) -> dict[str, float]:
        if face_crop.size == 0:
            blur_score = 0.0
            brightness_score = 0.0
        else:
            grayscale = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
            blur_score = min(1.0, blur_variance / 350.0)
            brightness = float(grayscale.mean())
            brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)

        width = max(1, int(bbox["width"]))
        height = max(1, int(bbox["height"]))
        size_score = min(1.0, min(width, height) / 120.0)
        face_center_x = int(bbox["x"]) + (width / 2.0)
        face_center_y = int(bbox["y"]) + (height / 2.0)
        max_distance = ((image_width / 2.0) ** 2 + (image_height / 2.0) ** 2) ** 0.5
        distance_to_center = ((face_center_x - (image_width / 2.0)) ** 2 + (face_center_y - (image_height / 2.0)) ** 2) ** 0.5
        centered_score = max(0.0, 1.0 - (distance_to_center / max_distance))

        return {
            "detection_confidence": round(max(0.0, min(1.0, detection_score)), 4),
            "blur_score": round(blur_score, 4),
            "brightness_score": round(brightness_score, 4),
            "size_score": round(size_score, 4),
            "centered_score": round(centered_score, 4),
        }

    def _embedding(self, face) -> list[float]:
        raw_embedding = getattr(face, "normed_embedding", None)
        if raw_embedding is None:
            raw_embedding = getattr(face, "embedding", None)
        if raw_embedding is None:
            return []

        vector = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return []
        normalized_vector = vector / norm
        return [round(float(value), 6) for value in normalized_vector.tolist()]
