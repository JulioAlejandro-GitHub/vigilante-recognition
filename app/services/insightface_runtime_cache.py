from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from app.services.face_backend_service import FaceBackendError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsightFaceRuntimeConfig:
    model_name: str
    provider: str
    provider_name: str
    providers: tuple[str, ...]
    ctx_id: int
    model_root: str | None
    det_size: tuple[int, int]
    detection_threshold: float

    def as_trace(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "provider": self.provider_name,
            "provider_requested": self.provider,
            "providers": list(self.providers),
            "ctx_id": self.ctx_id,
            "model_root": self.model_root,
            "det_size": [self.det_size[0], self.det_size[1]],
            "detection_threshold": self.detection_threshold,
        }


@dataclass(frozen=True)
class InsightFaceRuntime:
    app: Any
    config: InsightFaceRuntimeConfig
    backend_name: str
    provider_name: str
    load_elapsed_ms: float


_runtime_lock = RLock()
_runtime_cache: dict[InsightFaceRuntimeConfig, InsightFaceRuntime] = {}


def build_insightface_runtime_config(
    *,
    model_name: str,
    provider: str,
    model_root: str | None,
    det_size: str,
    detection_threshold: float,
) -> InsightFaceRuntimeConfig:
    provider_name, providers, ctx_id = _resolve_provider_config(provider)
    normalized_root = _normalize_model_root(model_root)
    return InsightFaceRuntimeConfig(
        model_name=(model_name or "buffalo_l").strip(),
        provider=(provider or "cpu").strip(),
        provider_name=provider_name,
        providers=providers,
        ctx_id=ctx_id,
        model_root=normalized_root,
        det_size=_parse_det_size(det_size),
        detection_threshold=_parse_detection_threshold(detection_threshold),
    )


def get_insightface_runtime(config: InsightFaceRuntimeConfig) -> tuple[InsightFaceRuntime, bool]:
    with _runtime_lock:
        cached = _runtime_cache.get(config)
        if cached is not None:
            return cached, True

        runtime = _load_runtime(config)
        _runtime_cache[config] = runtime
        return runtime, False


def clear_insightface_runtime_cache() -> None:
    with _runtime_lock:
        _runtime_cache.clear()


def insightface_runtime_cache_size() -> int:
    with _runtime_lock:
        return len(_runtime_cache)


def _load_runtime(config: InsightFaceRuntimeConfig) -> InsightFaceRuntime:
    started_at = perf_counter()
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise FaceBackendError(
            "insightface_not_installed",
            backend_key="insightface",
            stage="import",
        ) from exc

    kwargs: dict[str, object] = {
        "name": config.model_name,
        "providers": list(config.providers),
    }
    if config.model_root is not None:
        kwargs["root"] = config.model_root

    try:
        app = FaceAnalysis(**kwargs)
        app.prepare(
            ctx_id=config.ctx_id,
            det_size=config.det_size,
            det_thresh=config.detection_threshold,
        )
    except Exception as exc:  # pragma: no cover - depends on optional model/provider state
        raise FaceBackendError(
            f"insightface_load_failed:{type(exc).__name__}",
            backend_key="insightface",
            stage="load",
            details={
                "error": str(exc),
                **config.as_trace(),
            },
        ) from exc

    load_elapsed_ms = round((perf_counter() - started_at) * 1000.0, 2)
    runtime = InsightFaceRuntime(
        app=app,
        config=config,
        backend_name=f"insightface:{config.model_name}",
        provider_name=config.provider_name,
        load_elapsed_ms=load_elapsed_ms,
    )
    logger.info(
        "insightface_backend_loaded model_name=%s provider=%s backend_load_ms=%.2f",
        config.model_name,
        config.provider_name,
        load_elapsed_ms,
    )
    logger.debug(
        "insightface_backend_config model_name=%s provider=%s providers=%s model_root=%s det_size=%sx%s detection_threshold=%.3f ctx_id=%s backend_load_ms=%.2f",
        config.model_name,
        config.provider_name,
        ",".join(config.providers),
        config.model_root or "<default>",
        config.det_size[0],
        config.det_size[1],
        config.detection_threshold,
        config.ctx_id,
        load_elapsed_ms,
    )
    return runtime


def _resolve_provider_config(provider: str | None) -> tuple[str, tuple[str, ...], int]:
    raw_provider = (provider or "cpu").strip()
    normalized = raw_provider.lower()
    if normalized in {"cpu", "cpu_execution_provider"}:
        return "cpu", ("CPUExecutionProvider",), -1
    if normalized in {"cuda", "cuda_execution_provider", "gpu"}:
        return "cuda", ("CUDAExecutionProvider", "CPUExecutionProvider"), 0
    if "," in raw_provider:
        providers = tuple(part.strip() for part in raw_provider.split(",") if part.strip())
        if not providers:
            raise FaceBackendError(
                "insightface_provider_invalid",
                backend_key="insightface",
                stage="configuration",
            )
        return raw_provider, providers, -1 if providers == ("CPUExecutionProvider",) else 0
    if raw_provider.endswith("ExecutionProvider"):
        return raw_provider, (raw_provider,), -1 if raw_provider == "CPUExecutionProvider" else 0
    raise FaceBackendError(
        "insightface_provider_invalid",
        backend_key="insightface",
        stage="configuration",
        details={"provider": raw_provider},
    )


def _parse_det_size(det_size: str) -> tuple[int, int]:
    try:
        raw_width, raw_height = det_size.lower().replace("x", ",").split(",", 1)
        width = int(raw_width.strip())
        height = int(raw_height.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise FaceBackendError(
            "insightface_det_size_invalid",
            backend_key="insightface",
            stage="configuration",
            details={"det_size": det_size},
        ) from exc
    if width <= 0 or height <= 0:
        raise FaceBackendError(
            "insightface_det_size_invalid",
            backend_key="insightface",
            stage="configuration",
            details={"det_size": det_size},
        )
    return width, height


def _parse_detection_threshold(detection_threshold: float) -> float:
    try:
        threshold = float(detection_threshold)
    except (TypeError, ValueError) as exc:
        raise FaceBackendError(
            "insightface_detection_threshold_invalid",
            backend_key="insightface",
            stage="configuration",
            details={"detection_threshold": detection_threshold},
        ) from exc
    if threshold < 0.0 or threshold > 1.0:
        raise FaceBackendError(
            "insightface_detection_threshold_invalid",
            backend_key="insightface",
            stage="configuration",
            details={"detection_threshold": detection_threshold},
        )
    return round(threshold, 4)


def _normalize_model_root(model_root: str | None) -> str | None:
    if model_root is None:
        return None
    stripped = model_root.strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())
