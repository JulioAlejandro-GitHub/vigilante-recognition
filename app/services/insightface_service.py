from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult
from app.services.camera_face_tuning_service import (
    CameraFaceTuningService,
    InsightFaceEffectiveTuning,
)
from app.services.face_backend_service import FaceBackendError
from app.services.insightface_runtime_cache import (
    InsightFaceRuntime,
    build_insightface_runtime_config,
    get_insightface_runtime,
)


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
    trace: dict[str, object] = field(default_factory=dict)


class InsightFaceService:
    backend_key = "insightface"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        provider: str | None = None,
        model_root: str | None = None,
        det_size: str | None = None,
        detection_threshold: float | None = None,
        max_faces: int | None = None,
        tuning_service: CameraFaceTuningService | None = None,
    ) -> None:
        self.model_name = model_name or settings.insightface_model_name
        self.provider = provider or settings.insightface_provider
        self.model_root = model_root if model_root is not None else settings.insightface_model_root
        self.det_size = det_size or settings.insightface_det_size
        self.detection_threshold = (
            settings.insightface_detection_threshold if detection_threshold is None else detection_threshold
        )
        self.max_faces = settings.insightface_max_faces if max_faces is None else max_faces
        self.backend_name = f"insightface:{self.model_name}"
        self.provider_name = self.provider
        self._analysis_cache: dict[str, _InsightFaceAnalysis] = {}
        self._tuning_service = tuning_service or CameraFaceTuningService()

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
        camera_id: str | None = None,
        camera_metadata: dict[str, object] | None = None,
    ) -> FaceDetectionResult:
        analysis = self._analyze(
            frame_ref=frame_ref,
            camera_id=camera_id,
            camera_metadata=camera_metadata,
        )
        return FaceDetectionResult(
            detected=analysis.detected,
            usable=analysis.usable,
            quality_score=analysis.quality_score,
            bbox=analysis.bbox,
            image_size=analysis.image_size,
            rejection_reasons=analysis.rejection_reasons,
            quality_metrics=analysis.quality_metrics,
            frame_quality_metadata=quality_metadata or {},
            face_backend_trace=analysis.trace,
        )

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
        camera_id: str | None = None,
        camera_metadata: dict[str, object] | None = None,
    ) -> FaceEmbeddingResult:
        if face_detection is not None and not face_detection.usable:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                bbox=face_detection.bbox,
                rejection_reasons=["face_not_usable_for_embedding"],
                embedding_backend_trace=face_detection.face_backend_trace,
            )

        analysis = self._analyze(
            frame_ref=frame_ref,
            camera_id=camera_id or self._camera_id_from_detection(face_detection),
            camera_metadata=camera_metadata,
        )
        if not analysis.detected:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                rejection_reasons=["face_not_detected_for_embedding", *analysis.rejection_reasons],
                embedding_backend_trace=analysis.trace,
            )
        if not analysis.embedding:
            return FaceEmbeddingResult(
                backend=self.backend_name,
                source_frame_ref=frame_ref,
                bbox=analysis.bbox,
                rejection_reasons=["embedding_not_available"],
                embedding_backend_trace=analysis.trace,
            )

        return FaceEmbeddingResult(
            generated=True,
            backend=self.backend_name,
            dimensions=len(analysis.embedding),
            vector=analysis.embedding,
            bbox=analysis.bbox,
            source_frame_ref=frame_ref,
            embedding_backend_trace=analysis.trace,
        )

    def _analyze(
        self,
        *,
        frame_ref: str,
        camera_id: str | None = None,
        camera_metadata: dict[str, object] | None = None,
    ) -> _InsightFaceAnalysis:
        tuning = self._resolve_tuning(camera_id=camera_id, camera_metadata=camera_metadata)
        cache_key = self._cache_key(frame_ref, tuning=tuning)
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
                trace=self._unloaded_trace(reason="frame_ref_not_found", tuning=tuning),
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
                trace=self._unloaded_trace(reason="frame_unreadable", tuning=tuning),
            )
            self._analysis_cache[cache_key] = analysis
            return analysis

        image_height, image_width = image.shape[:2]
        image_size = {"width": int(image_width), "height": int(image_height)}
        runtime_reused = False
        runtime: InsightFaceRuntime | None = None
        max_faces = self._max_faces_int(tuning.max_faces)
        try:
            runtime, runtime_reused = self._load_runtime(tuning=tuning)
            detect_started_at = perf_counter()
            faces = runtime.app.get(image, max_num=max_faces)
            detect_elapsed_ms = self._elapsed_ms(detect_started_at)
        except FaceBackendError:
            raise
        except Exception as exc:  # pragma: no cover - depends on optional runtime/provider failures
            raise FaceBackendError(
                f"insightface_runtime_error:{type(exc).__name__}",
                backend_key=self.backend_key,
                stage="detect",
                details={"error": str(exc)},
            ) from exc
        faces = self._filter_faces(faces, detection_threshold=tuning.detection_threshold)
        faces_detected = len(faces)
        trace = self._runtime_trace(
            runtime=runtime,
            runtime_reused=runtime_reused,
            detect_elapsed_ms=detect_elapsed_ms,
            faces_detected=faces_detected,
            selected_face_score=None,
            max_faces=max_faces,
            tuning=tuning,
        )

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
                trace=trace,
            )
            self._analysis_cache[cache_key] = analysis
            return analysis

        face = max(faces, key=lambda candidate: self._face_area(candidate))
        selected_face_score = round(float(getattr(face, "det_score", 0.0) or 0.0), 4)
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
        rejection_reasons = self._quality_rejection_reasons(
            quality_score=quality_score,
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            tuning=tuning,
        )
        usable = not rejection_reasons
        analysis = _InsightFaceAnalysis(
            detected=True,
            usable=usable,
            image_size=image_size,
            bbox=bbox,
            quality_score=quality_score,
            quality_metrics=quality_metrics,
            rejection_reasons=rejection_reasons,
            embedding=self._embedding(face),
            trace=self._runtime_trace(
                runtime=runtime,
                runtime_reused=runtime_reused,
                detect_elapsed_ms=detect_elapsed_ms,
                faces_detected=faces_detected,
                selected_face_score=selected_face_score,
                max_faces=max_faces,
                tuning=tuning,
            ),
        )
        self._analysis_cache[cache_key] = analysis
        return analysis

    def _load_runtime(self, *, tuning: InsightFaceEffectiveTuning) -> tuple[InsightFaceRuntime, bool]:
        if not settings.insightface_enabled:
            raise FaceBackendError(
                "insightface_disabled",
                backend_key=self.backend_key,
                stage="configuration",
            )

        runtime_config = build_insightface_runtime_config(
            model_name=tuning.model_name,
            provider=tuning.provider,
            model_root=tuning.model_root,
            det_size=tuning.det_size,
            detection_threshold=tuning.detection_threshold,
        )
        runtime, reused = get_insightface_runtime(runtime_config)
        self.backend_name = runtime.backend_name
        self.provider_name = runtime.provider_name
        return runtime, reused

    def _filter_faces(self, faces, *, detection_threshold: float) -> list[object]:
        threshold = float(detection_threshold)
        return [
            face
            for face in list(faces or [])
            if float(getattr(face, "det_score", 0.0) or 0.0) >= threshold
        ]

    def _max_faces_int(self, max_faces_value: int) -> int:
        try:
            max_faces = int(max_faces_value)
        except (TypeError, ValueError) as exc:
            raise FaceBackendError(
                "insightface_max_faces_invalid",
                backend_key=self.backend_key,
                stage="configuration",
                details={"max_faces": max_faces_value},
            ) from exc
        if max_faces < 0:
            raise FaceBackendError(
                "insightface_max_faces_invalid",
                backend_key=self.backend_key,
                stage="configuration",
                details={"max_faces": max_faces_value},
            )
        return max_faces

    def _runtime_trace(
        self,
        *,
        runtime: InsightFaceRuntime,
        runtime_reused: bool,
        detect_elapsed_ms: float,
        faces_detected: int,
        selected_face_score: float | None,
        max_faces: int,
        tuning: InsightFaceEffectiveTuning,
    ) -> dict[str, object]:
        configuration = runtime.config.as_trace()
        configuration["max_faces"] = max_faces
        configuration.update(tuning.camera_trace())
        return {
            "backend": self.backend_key,
            "backend_name": runtime.backend_name,
            "provider": runtime.provider_name,
            "camera_id": tuning.camera_id,
            "config_source": tuning.config_source,
            "camera_override_applied": tuning.camera_override_applied,
            "camera_override_key": tuning.camera_override_key,
            "quality_thresholds": tuning.quality_thresholds_trace(),
            "configuration": configuration,
            "runtime_loaded": True,
            "runtime_reused": runtime_reused,
            "backend_load_ms": 0.0 if runtime_reused else runtime.load_elapsed_ms,
            "runtime_load_elapsed_ms": runtime.load_elapsed_ms,
            "detect_elapsed_ms": detect_elapsed_ms,
            "faces_detected": faces_detected,
            "selected_face_score": selected_face_score,
        }

    def _unloaded_trace(self, *, reason: str, tuning: InsightFaceEffectiveTuning) -> dict[str, object]:
        configuration = {
            "model_name": tuning.model_name,
            "provider": tuning.provider,
            "model_root": tuning.model_root or None,
            "det_size": tuning.det_size,
            "detection_threshold": tuning.detection_threshold,
            "max_faces": tuning.max_faces,
            **tuning.camera_trace(),
        }
        return {
            "backend": self.backend_key,
            "backend_name": self.backend_name,
            "provider": self.provider_name,
            "camera_id": tuning.camera_id,
            "config_source": tuning.config_source,
            "camera_override_applied": tuning.camera_override_applied,
            "camera_override_key": tuning.camera_override_key,
            "quality_thresholds": tuning.quality_thresholds_trace(),
            "configuration": configuration,
            "runtime_loaded": False,
            "runtime_reused": False,
            "backend_load_ms": 0.0,
            "runtime_load_elapsed_ms": None,
            "detect_elapsed_ms": 0.0,
            "faces_detected": 0,
            "reason": reason,
        }

    def _elapsed_ms(self, started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000.0, 2)

    def _resolve_tuning(
        self,
        *,
        camera_id: str | None,
        camera_metadata: dict[str, object] | None,
    ) -> InsightFaceEffectiveTuning:
        defaults = self._tuning_service.build_defaults(
            model_name=self.model_name,
            provider=self.provider,
            model_root=self.model_root,
            det_size=self.det_size,
            detection_threshold=self.detection_threshold,
            max_faces=self.max_faces,
        )
        return self._tuning_service.resolve(
            camera_id=camera_id,
            defaults=defaults,
            camera_metadata=camera_metadata,
        )

    def _camera_id_from_detection(self, face_detection: FaceDetectionResult | None) -> str | None:
        if face_detection is None:
            return None
        trace = face_detection.face_backend_trace or {}
        camera_id = trace.get("camera_id")
        if camera_id is not None:
            return str(camera_id)
        configuration = trace.get("configuration", {})
        if isinstance(configuration, dict) and configuration.get("camera_id") is not None:
            return str(configuration["camera_id"])
        return None

    def _cache_key(self, frame_ref: str, *, tuning: InsightFaceEffectiveTuning) -> str:
        frame_path = self._resolve_frame_path(frame_ref)
        tuning_key = repr(tuning.cache_signature())
        if frame_path is None:
            return f"{frame_ref}:{tuning_key}"
        try:
            stat = frame_path.stat()
        except OSError:
            return f"{frame_path}:{tuning_key}"
        return f"{frame_path}:{stat.st_mtime_ns}:{stat.st_size}:{tuning_key}"

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
            "face_area_ratio": round((width * height) / max(1.0, float(image_width * image_height)), 6),
            "face_min_dimension": float(min(width, height)),
        }

    def _quality_rejection_reasons(
        self,
        *,
        quality_score: float,
        bbox: dict[str, int],
        image_width: int,
        image_height: int,
        tuning: InsightFaceEffectiveTuning,
    ) -> list[str]:
        rejection_reasons: list[str] = []
        if quality_score < tuning.face_quality_threshold:
            rejection_reasons.append("face_quality_threshold_failed")

        min_dimension = min(int(bbox["width"]), int(bbox["height"]))
        if tuning.min_face_bbox_size > 0 and min_dimension < tuning.min_face_bbox_size:
            rejection_reasons.append("face_bbox_too_small")

        frame_area = max(1.0, float(image_width * image_height))
        face_area_ratio = (int(bbox["width"]) * int(bbox["height"])) / frame_area
        if tuning.min_face_area_ratio > 0.0 and face_area_ratio < tuning.min_face_area_ratio:
            rejection_reasons.append("face_area_ratio_below_threshold")
        return rejection_reasons

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
