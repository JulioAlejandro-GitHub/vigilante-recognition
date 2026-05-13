from __future__ import annotations

import logging
from time import perf_counter

from app.config import Settings, settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult
from app.logging import compact_value
from app.services.camera_face_metrics_service import record_camera_face_detection
from app.services.face_backend_service import FaceBackend, FaceBackendError, SimpleFaceBackend
from app.services.insightface_service import InsightFaceService

logger = logging.getLogger(__name__)


class FaceBackendSelector:
    VALID_BACKENDS = {"simple", "insightface", "auto"}

    def __init__(
        self,
        *,
        settings_obj: Settings | None = None,
        simple_backend: FaceBackend | None = None,
        insightface_backend: FaceBackend | None = None,
    ) -> None:
        self.settings = settings_obj or settings
        self.simple_backend = simple_backend or SimpleFaceBackend()
        self._insightface_backend = insightface_backend

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
        camera_id: str | None = None,
        camera_metadata: dict[str, object] | None = None,
    ) -> FaceDetectionResult:
        requested_backend = self._requested_backend()
        logger.debug(
            "face_backend_requested stage=detect requested=%s frame_ref=%s",
            requested_backend,
            compact_value(frame_ref),
        )
        if requested_backend == "simple":
            return self._inspect_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                quality_metadata=quality_metadata or {},
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )
        if requested_backend == "insightface":
            if not self.settings.insightface_enabled:
                error = FaceBackendError(
                    "insightface_disabled",
                    backend_key="insightface",
                    stage="configuration",
                )
                self._log_forced_failure(stage="detect", frame_ref=frame_ref, error=error)
                raise error
            return self._inspect_with_backend(
                backend=self._insightface(),
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                quality_metadata=quality_metadata or {},
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )

        attempts: list[dict[str, object]] = []
        if not self.settings.insightface_enabled:
            attempts.append(self._skipped_attempt(stage="detect", reason="insightface_disabled"))
            logger.warning(
                "face_backend_fallback stage=detect requested=auto failed_backend=insightface selected=simple reason=insightface_disabled frame_ref=%s",
                frame_ref,
            )
            return self._inspect_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                quality_metadata=quality_metadata or {},
                camera_id=camera_id,
                camera_metadata=camera_metadata,
                fallback_used=True,
                error_reason="insightface_disabled",
                pre_attempts=attempts,
            )

        try:
            return self._inspect_with_backend(
                backend=self._insightface(),
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                quality_metadata=quality_metadata or {},
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )
        except Exception as exc:
            error = self._coerce_error(exc, backend_key="insightface", stage="detect")
            attempts.append(self._failed_attempt(error))
            logger.warning(
                "face_backend_fallback stage=detect requested=auto failed_backend=insightface selected=simple reason=%s frame_ref=%s",
                error.reason,
                frame_ref,
            )
            return self._inspect_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                quality_metadata=quality_metadata or {},
                camera_id=camera_id,
                camera_metadata=camera_metadata,
                fallback_used=True,
                error_reason=error.reason,
                pre_attempts=attempts,
            )

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
        camera_id: str | None = None,
        camera_metadata: dict[str, object] | None = None,
    ) -> FaceEmbeddingResult:
        requested_backend = self._requested_backend_from_detection(face_detection)
        selected_detection_backend = self._selected_backend_from_detection(face_detection, requested_backend)
        logger.debug(
            "face_backend_requested stage=embedding requested=%s selected_detection_backend=%s frame_ref=%s",
            requested_backend,
            selected_detection_backend,
            compact_value(frame_ref),
        )

        if requested_backend == "simple" or selected_detection_backend == "simple":
            return self._generate_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )

        if requested_backend == "insightface":
            if not self.settings.insightface_enabled:
                error = FaceBackendError(
                    "insightface_disabled",
                    backend_key="insightface",
                    stage="embedding",
                )
                self._log_forced_failure(stage="embedding", frame_ref=frame_ref, error=error)
                raise error
            return self._generate_with_backend(
                backend=self._insightface(),
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )

        attempts: list[dict[str, object]] = []
        if not self.settings.insightface_enabled:
            attempts.append(self._skipped_attempt(stage="embedding", reason="insightface_disabled"))
            logger.warning(
                "face_backend_fallback stage=embedding requested=auto failed_backend=insightface selected=simple reason=insightface_disabled frame_ref=%s",
                frame_ref,
            )
            return self._generate_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
                fallback_used=True,
                error_reason="insightface_disabled",
                pre_attempts=attempts,
            )

        try:
            return self._generate_with_backend(
                backend=self._insightface(),
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )
        except Exception as exc:
            error = self._coerce_error(exc, backend_key="insightface", stage="embedding")
            attempts.append(self._failed_attempt(error))
            logger.warning(
                "face_backend_fallback stage=embedding requested=auto failed_backend=insightface selected=simple reason=%s frame_ref=%s",
                error.reason,
                frame_ref,
            )
            return self._generate_with_backend(
                backend=self.simple_backend,
                requested_backend=requested_backend,
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
                fallback_used=True,
                error_reason=error.reason,
                pre_attempts=attempts,
            )

    def _inspect_with_backend(
        self,
        *,
        backend: FaceBackend,
        requested_backend: str,
        frame_ref: str,
        quality_metadata: dict[str, float],
        camera_id: str | None,
        camera_metadata: dict[str, object] | None,
        fallback_used: bool = False,
        error_reason: str | None = None,
        pre_attempts: list[dict[str, object]] | None = None,
    ) -> FaceDetectionResult:
        started_at = perf_counter()
        try:
            result = backend.inspect_face(
                frame_ref=frame_ref,
                quality_metadata=quality_metadata,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )
        except Exception as exc:
            error = self._coerce_error(exc, backend_key=backend.backend_key, stage="detect")
            self._log_forced_failure(stage="detect", frame_ref=frame_ref, error=error)
            raise error from exc
        elapsed_ms = self._elapsed_ms(started_at)
        attempts = list(pre_attempts or [])
        attempts.append(self._success_attempt(backend=backend, stage="detect", elapsed_ms=elapsed_ms))
        self._annotate_detection(
            result,
            backend=backend,
            requested_backend=requested_backend,
            fallback_used=fallback_used,
            error_reason=error_reason,
            attempts=attempts,
            elapsed_ms=elapsed_ms,
            camera_id=camera_id,
        )
        trace = result.face_backend_trace
        configuration = trace.get("configuration", {}) if isinstance(trace.get("configuration"), dict) else {}
        if camera_id:
            record_camera_face_detection(camera_id=camera_id, face_detection=result)
        logger.info(
            (
                "face_backend_selected stage=detect requested=%s selected=%s fallback_used=%s "
                "camera_id=%s detected=%s usable=%s faces_detected=%s elapsed_ms=%.2f frame_ref=%s"
            ),
            requested_backend,
            backend.backend_key,
            fallback_used,
            camera_id,
            result.detected,
            result.usable,
            trace.get("faces_detected"),
            elapsed_ms,
            compact_value(frame_ref),
        )
        logger.debug(
            (
                "face_backend_trace stage=detect requested=%s selected=%s provider=%s camera_id=%s "
                "detect_elapsed_ms=%s backend_load_ms=%s runtime_reused=%s model_name=%s det_size=%s "
                "detection_threshold=%s max_faces=%s quality_thresholds=%s config_source=%s "
                "camera_override_applied=%s trace=%s"
            ),
            requested_backend,
            backend.backend_key,
            backend.provider_name,
            camera_id,
            trace.get("detect_elapsed_ms"),
            trace.get("backend_load_ms"),
            trace.get("runtime_reused"),
            configuration.get("model_name"),
            configuration.get("det_size"),
            configuration.get("detection_threshold"),
            configuration.get("max_faces"),
            configuration.get("quality_thresholds"),
            configuration.get("config_source"),
            configuration.get("camera_override_applied"),
            trace,
        )
        return result

    def _generate_with_backend(
        self,
        *,
        backend: FaceBackend,
        requested_backend: str,
        frame_ref: str,
        face_detection: FaceDetectionResult | None,
        camera_id: str | None,
        camera_metadata: dict[str, object] | None,
        fallback_used: bool = False,
        error_reason: str | None = None,
        pre_attempts: list[dict[str, object]] | None = None,
    ) -> FaceEmbeddingResult:
        started_at = perf_counter()
        try:
            result = backend.generate(
                frame_ref=frame_ref,
                face_detection=face_detection,
                camera_id=camera_id,
                camera_metadata=camera_metadata,
            )
        except Exception as exc:
            error = self._coerce_error(exc, backend_key=backend.backend_key, stage="embedding")
            self._log_forced_failure(stage="embedding", frame_ref=frame_ref, error=error)
            raise error from exc
        elapsed_ms = self._elapsed_ms(started_at)
        attempts = list(pre_attempts or [])
        attempts.append(self._success_attempt(backend=backend, stage="embedding", elapsed_ms=elapsed_ms))
        backend_trace = dict(result.embedding_backend_trace or {})
        result.embedding_backend_requested = requested_backend
        result.embedding_backend_selected = backend.backend_key
        result.embedding_backend_fallback_used = fallback_used
        result.embedding_backend_error = error_reason
        result.embedding_backend_trace = {
            "requested_backend": requested_backend,
            "selected_backend": backend.backend_key,
            "selected_backend_name": backend.backend_name,
            "fallback_used": fallback_used,
            "error": error_reason,
            "provider": backend.provider_name,
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "generated": result.generated,
            "dimensions": result.dimensions,
            **self._selected_backend_trace_fields(backend_trace),
        }
        logger.info(
            "face_backend_selected stage=embedding requested=%s selected=%s fallback_used=%s generated=%s dimensions=%s elapsed_ms=%.2f frame_ref=%s",
            requested_backend,
            backend.backend_key,
            fallback_used,
            result.generated,
            result.dimensions,
            elapsed_ms,
            compact_value(frame_ref),
        )
        logger.debug(
            "face_backend_trace stage=embedding requested=%s selected=%s provider=%s frame_ref=%s trace=%s",
            requested_backend,
            backend.backend_key,
            backend.provider_name,
            compact_value(frame_ref),
            result.embedding_backend_trace,
        )
        return result

    def _annotate_detection(
        self,
        result: FaceDetectionResult,
        *,
        backend: FaceBackend,
        requested_backend: str,
        fallback_used: bool,
        error_reason: str | None,
        attempts: list[dict[str, object]],
        elapsed_ms: float,
        camera_id: str | None,
    ) -> None:
        backend_trace = dict(result.face_backend_trace or {})
        result.face_backend = backend.backend_key
        result.face_backend_requested = requested_backend
        result.face_backend_selected = backend.backend_key
        result.face_backend_fallback_used = fallback_used
        result.face_backend_error = error_reason
        result.face_backend_trace = {
            "requested_backend": requested_backend,
            "selected_backend": backend.backend_key,
            "selected_backend_name": backend.backend_name,
            "fallback_used": fallback_used,
            "error": error_reason,
            "provider": backend.provider_name,
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "detected": result.detected,
            "usable": result.usable,
            "quality_score": result.quality_score,
            "camera_id": camera_id,
            **self._selected_backend_trace_fields(backend_trace),
        }

    def _selected_backend_trace_fields(self, backend_trace: dict[str, object]) -> dict[str, object]:
        if not backend_trace:
            return {}
        selected_keys = [
            "configuration",
            "runtime_loaded",
            "runtime_reused",
            "backend_load_ms",
            "runtime_load_elapsed_ms",
            "detect_elapsed_ms",
            "faces_detected",
            "selected_face_score",
            "camera_id",
            "config_source",
            "camera_override_applied",
            "camera_override_key",
            "quality_thresholds",
        ]
        return {key: backend_trace[key] for key in selected_keys if key in backend_trace}

    def _insightface(self) -> FaceBackend:
        if self._insightface_backend is None:
            self._insightface_backend = InsightFaceService()
        return self._insightface_backend

    def _requested_backend(self) -> str:
        requested_backend = (self.settings.face_backend or "simple").strip().lower()
        if requested_backend not in self.VALID_BACKENDS:
            raise FaceBackendError(
                "face_backend_invalid",
                backend_key=requested_backend,
                stage="configuration",
                details={"face_backend": self.settings.face_backend},
            )
        return requested_backend

    def _requested_backend_from_detection(self, face_detection: FaceDetectionResult | None) -> str:
        if face_detection and face_detection.face_backend_requested:
            return face_detection.face_backend_requested
        return self._requested_backend()

    def _selected_backend_from_detection(
        self,
        face_detection: FaceDetectionResult | None,
        requested_backend: str,
    ) -> str:
        if face_detection and face_detection.face_backend_selected:
            return face_detection.face_backend_selected
        if requested_backend == "auto":
            return "insightface"
        return requested_backend

    def _success_attempt(self, *, backend: FaceBackend, stage: str, elapsed_ms: float) -> dict[str, object]:
        return {
            "backend_key": backend.backend_key,
            "backend_name": backend.backend_name,
            "provider": backend.provider_name,
            "stage": stage,
            "status": "success",
            "elapsed_ms": elapsed_ms,
        }

    def _failed_attempt(self, error: FaceBackendError) -> dict[str, object]:
        return {
            "backend_key": error.backend_key,
            "backend_name": error.backend_key,
            "stage": error.stage,
            "status": "failed",
            "reason": error.reason,
            **error.details,
        }

    def _skipped_attempt(self, *, stage: str, reason: str) -> dict[str, object]:
        return {
            "backend_key": "insightface",
            "backend_name": "insightface",
            "stage": stage,
            "status": "skipped",
            "reason": reason,
        }

    def _coerce_error(self, exc: Exception, *, backend_key: str, stage: str) -> FaceBackendError:
        if isinstance(exc, FaceBackendError):
            return exc
        return FaceBackendError(
            f"unexpected_backend_error:{type(exc).__name__}",
            backend_key=backend_key,
            stage=stage,
            details={"error": str(exc)},
        )

    def _log_forced_failure(self, *, stage: str, frame_ref: str, error: FaceBackendError) -> None:
        logger.error(
            "face_backend_failed stage=%s backend=%s reason=%s frame_ref=%s",
            stage,
            error.backend_key,
            error.reason,
            frame_ref,
        )

    def _elapsed_ms(self, started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000.0, 2)
