from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import time
from typing import Any

from app.config import settings
from app.domain.entities import FaceDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class CameraFaceMetrics:
    camera_id: str
    frames_processed: int = 0
    faces_detected: int = 0
    face_not_detected: int = 0
    usable_true: int = 0
    usable_false: int = 0
    low_quality_face: int = 0
    detect_latency_ms_total: float = 0.0
    detect_latency_samples: int = 0
    last_backend: str | None = None
    last_provider: str | None = None
    last_config_source: str | None = None
    last_configuration: dict[str, Any] = field(default_factory=dict)
    last_updated_at_epoch: float | None = None

    @property
    def usable_ratio(self) -> float:
        if self.frames_processed <= 0:
            return 0.0
        return round(self.usable_true / self.frames_processed, 4)

    @property
    def unusable_ratio(self) -> float:
        if self.frames_processed <= 0:
            return 0.0
        return round(self.usable_false / self.frames_processed, 4)

    @property
    def average_detect_latency_ms(self) -> float:
        if self.detect_latency_samples <= 0:
            return 0.0
        return round(self.detect_latency_ms_total / self.detect_latency_samples, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "face_not_detected": self.face_not_detected,
            "usable_true": self.usable_true,
            "usable_false": self.usable_false,
            "low_quality_face": self.low_quality_face,
            "usable_ratio": self.usable_ratio,
            "unusable_ratio": self.unusable_ratio,
            "average_detect_latency_ms": self.average_detect_latency_ms,
            "detect_latency_samples": self.detect_latency_samples,
            "last_backend": self.last_backend,
            "last_provider": self.last_provider,
            "last_config_source": self.last_config_source,
            "last_configuration": deepcopy(self.last_configuration),
            "last_updated_at_epoch": self.last_updated_at_epoch,
        }


class CameraFaceMetricsRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._metrics_by_camera: dict[str, CameraFaceMetrics] = {}

    def record_detection(
        self,
        *,
        camera_id: str,
        face_detection: FaceDetectionResult,
    ) -> CameraFaceMetrics:
        trace = face_detection.face_backend_trace or {}
        configuration = trace.get("configuration", {})
        if not isinstance(configuration, dict):
            configuration = {}

        with self._lock:
            metrics = self._metrics_by_camera.setdefault(
                camera_id,
                CameraFaceMetrics(camera_id=camera_id),
            )
            metrics.frames_processed += 1
            metrics.faces_detected += self._int_value(trace.get("faces_detected"), default=1 if face_detection.detected else 0)
            if not face_detection.detected:
                metrics.face_not_detected += 1
            if face_detection.usable:
                metrics.usable_true += 1
            else:
                metrics.usable_false += 1
            if face_detection.detected and not face_detection.usable:
                metrics.low_quality_face += 1

            latency_ms = self._float_value(trace.get("detect_elapsed_ms"))
            if latency_ms is None:
                latency_ms = self._float_value(trace.get("elapsed_ms"))
            if latency_ms is not None:
                metrics.detect_latency_ms_total += latency_ms
                metrics.detect_latency_samples += 1

            metrics.last_backend = str(trace.get("selected_backend") or face_detection.face_backend_selected or "")
            metrics.last_provider = str(trace.get("provider") or "")
            metrics.last_config_source = str(
                trace.get("config_source")
                or configuration.get("config_source")
                or configuration.get("tuning_source")
                or ""
            )
            metrics.last_configuration = deepcopy(configuration)
            metrics.last_updated_at_epoch = round(time(), 3)
            snapshot = deepcopy(metrics)

        self._log_if_due(snapshot)
        return snapshot

    def snapshot(self, camera_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if camera_id is not None:
                metrics = self._metrics_by_camera.get(camera_id)
                return metrics.as_dict() if metrics else {}
            return {
                key: metrics.as_dict()
                for key, metrics in sorted(self._metrics_by_camera.items(), key=lambda item: item[0])
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics_by_camera.clear()

    def log_all(self) -> None:
        snapshots = self.snapshot()
        for snapshot in snapshots.values():
            self._log_summary(snapshot)

    def _log_if_due(self, metrics: CameraFaceMetrics) -> None:
        log_every = max(0, int(settings.insightface_camera_metrics_log_every_n_frames or 0))
        if log_every <= 0:
            return
        if metrics.frames_processed % log_every == 0:
            self._log_summary(metrics.as_dict())

    def _log_summary(self, snapshot: dict[str, Any]) -> None:
        logger.info(
            (
                "camera_face_metrics_summary camera_id=%s frames_processed=%s faces_detected=%s "
                "face_not_detected=%s usable_true=%s usable_false=%s low_quality_face=%s "
                "usable_ratio=%.4f avg_detect_latency_ms=%.2f backend=%s provider=%s config_source=%s"
            ),
            snapshot["camera_id"],
            snapshot["frames_processed"],
            snapshot["faces_detected"],
            snapshot["face_not_detected"],
            snapshot["usable_true"],
            snapshot["usable_false"],
            snapshot["low_quality_face"],
            snapshot["usable_ratio"],
            snapshot["average_detect_latency_ms"],
            snapshot["last_backend"],
            snapshot["last_provider"],
            snapshot["last_config_source"],
        )

    def _int_value(self, value: Any, *, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float_value(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


camera_face_metrics_registry = CameraFaceMetricsRegistry()


def record_camera_face_detection(
    *,
    camera_id: str,
    face_detection: FaceDetectionResult,
) -> CameraFaceMetrics:
    return camera_face_metrics_registry.record_detection(
        camera_id=camera_id,
        face_detection=face_detection,
    )


def get_camera_face_metrics_snapshot(camera_id: str | None = None) -> dict[str, Any]:
    return camera_face_metrics_registry.snapshot(camera_id)


def reset_camera_face_metrics() -> None:
    camera_face_metrics_registry.reset()


def log_all_camera_face_metrics() -> None:
    camera_face_metrics_registry.log_all()
