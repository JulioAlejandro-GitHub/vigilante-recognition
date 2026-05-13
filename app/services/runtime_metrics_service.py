from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.config import Settings, settings
from app.domain.entities import (
    FaceDetectionResult,
    FrameIngestedMessage,
    SemanticDescriptorResult,
)
from app.services.runtime_metrics_store import JsonlRuntimeMetricsStore
from app.services.runtime_metrics_summary_service import RuntimeMetricsSummaryService
from app.services.runtime_recommendation_service import RuntimeRecommendationService

logger = logging.getLogger(__name__)

RUNTIME_METRICS_EVENT_SCHEMA_VERSION = "runtime_metrics_event_v1"
REAL_SEMANTIC_BACKEND_KEYS = {"qwen", "qwen_vl", "smolvlm"}


class RuntimeMetricsService:
    def __init__(
        self,
        *,
        enabled: bool,
        store: JsonlRuntimeMetricsStore | None,
        log_summary_every_n_events: int = 0,
        summary_service: RuntimeMetricsSummaryService | None = None,
        recommendation_service: RuntimeRecommendationService | None = None,
        recommendation_every_n_events: int = 0,
    ) -> None:
        self.enabled = enabled
        self.store = store
        self.log_summary_every_n_events = max(0, int(log_summary_every_n_events))
        self.summary_service = summary_service or RuntimeMetricsSummaryService()
        self.recommendation_service = recommendation_service
        self.recommendation_every_n_events = max(0, int(recommendation_every_n_events))
        self._recorded_events = 0

    @classmethod
    def from_settings(cls, settings_obj: Settings | None = None) -> "RuntimeMetricsService":
        resolved_settings = settings_obj or settings
        store_kind = (resolved_settings.runtime_metrics_store or "jsonl").strip().lower()
        if store_kind not in {"jsonl", "ndjson"}:
            logger.warning(
                "runtime_metrics_store_unsupported configured=%s fallback=jsonl",
                store_kind,
            )
        store = JsonlRuntimeMetricsStore(
            resolved_settings.runtime_metrics_path,
            rotate_max_mb=resolved_settings.runtime_metrics_rotate_max_mb,
            retention_files=resolved_settings.runtime_metrics_retention_files,
        )
        recommendation_service = None
        if resolved_settings.runtime_recommendations_enabled:
            recommendation_service = RuntimeRecommendationService.from_settings(
                settings_obj=resolved_settings,
                metrics_store=store,
            )
        return cls(
            enabled=resolved_settings.runtime_metrics_enabled,
            store=store,
            log_summary_every_n_events=(
                resolved_settings.runtime_metrics_log_summary_every_n_events
            ),
            recommendation_service=recommendation_service,
            recommendation_every_n_events=(
                resolved_settings.runtime_recommendations_log_every_n_events
            ),
        )

    def record_emitted_events(
        self,
        *,
        source_message: FrameIngestedMessage,
        emitted_events: list[dict[str, Any]],
        face_detection: FaceDetectionResult,
        semantic_descriptor_result: SemanticDescriptorResult | None,
        camera_runtime_config_trace: dict[str, Any],
    ) -> None:
        for emitted_event in emitted_events:
            self.record_event(
                source_message=source_message,
                emitted_event=emitted_event,
                face_detection=face_detection,
                semantic_descriptor_result=semantic_descriptor_result,
                camera_runtime_config_trace=camera_runtime_config_trace,
            )

    def record_event(
        self,
        *,
        source_message: FrameIngestedMessage,
        emitted_event: dict[str, Any],
        face_detection: FaceDetectionResult,
        semantic_descriptor_result: SemanticDescriptorResult | None,
        camera_runtime_config_trace: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.enabled or self.store is None:
            return None

        record = build_runtime_metrics_record(
            source_message=source_message,
            emitted_event=emitted_event,
            face_detection=face_detection,
            semantic_descriptor_result=semantic_descriptor_result,
            camera_runtime_config_trace=camera_runtime_config_trace,
        )
        try:
            self.store.append(record)
        except Exception as exc:  # pragma: no cover - defensive operational guard
            logger.warning(
                "runtime_metrics_persist_failed event_id=%s error=%s",
                record.get("event_id"),
                type(exc).__name__,
            )
            return record

        self._recorded_events += 1
        if (
            self.log_summary_every_n_events > 0
            and self._recorded_events % self.log_summary_every_n_events == 0
        ):
            self._log_summary()
        if (
            self.recommendation_service is not None
            and self.recommendation_every_n_events > 0
            and self._recorded_events % self.recommendation_every_n_events == 0
        ):
            self._persist_recommendations()
        return record

    def _log_summary(self) -> None:
        if self.store is None:
            return
        try:
            summary = self.summary_service.summarize_store(self.store)
        except Exception as exc:  # pragma: no cover - defensive operational guard
            logger.warning("runtime_metrics_summary_failed error=%s", type(exc).__name__)
            return
        logger.info(
            "runtime_metrics_summary total_events=%s camera_count=%s face_backend_count=%s semantic_backend_count=%s event_type_count=%s",
            summary.get("total_events"),
            len(summary.get("by_camera", {})),
            len(summary.get("by_face_backend", {})),
            len(summary.get("by_semantic_backend", {})),
            len(summary.get("by_event_type", {})),
        )
        logger.debug(
            "runtime_metrics_summary_detail cameras=%s face_backends=%s semantic_backends=%s event_types=%s",
            sorted(summary.get("by_camera", {}).keys()),
            sorted(summary.get("by_face_backend", {}).keys()),
            sorted(summary.get("by_semantic_backend", {}).keys()),
            sorted(summary.get("by_event_type", {}).keys()),
        )

    def _persist_recommendations(self) -> None:
        if self.recommendation_service is None:
            return
        try:
            result = self.recommendation_service.generate(persist=True)
        except Exception as exc:  # pragma: no cover - defensive operational guard
            logger.warning(
                "runtime_recommendations_generation_failed error=%s",
                type(exc).__name__,
            )
            return
        logger.info(
            "runtime_recommendations_generated cameras=%s recommendations=%s actionable=%s persisted=%s",
            result.get("camera_count"),
            result.get("recommendation_count"),
            result.get("actionable_recommendation_count"),
            result.get("persisted_count"),
        )


def build_runtime_metrics_record(
    *,
    source_message: FrameIngestedMessage,
    emitted_event: dict[str, Any],
    face_detection: FaceDetectionResult,
    semantic_descriptor_result: SemanticDescriptorResult | None,
    camera_runtime_config_trace: dict[str, Any],
) -> dict[str, Any]:
    context = _as_dict(emitted_event.get("context"))
    camera_id = str(context.get("camera_id") or source_message.camera_id)
    return {
        "schema_version": RUNTIME_METRICS_EVENT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "event_type": emitted_event.get("event_type"),
        "event_id": emitted_event.get("event_id"),
        "source_frame_event_id": source_message.event_id,
        "source_frame_event_type": source_message.event_type,
        "track_id": context.get("track_id"),
        "subject_id": context.get("subject_id"),
        "face": _face_metrics(face_detection),
        "semantic": _semantic_metrics(semantic_descriptor_result),
        "budget": _budget_metrics(semantic_descriptor_result),
        "config": _config_metrics(camera_runtime_config_trace),
    }


def _face_metrics(face_detection: FaceDetectionResult) -> dict[str, Any]:
    trace = _as_dict(face_detection.face_backend_trace)
    configuration = _face_configuration(trace)
    detect_elapsed_ms = (
        _coerce_float(trace.get("detect_elapsed_ms"))
        or _coerce_float(trace.get("elapsed_ms"))
    )
    detected = bool(face_detection.detected)
    usable = bool(face_detection.usable)
    return {
        "requested": face_detection.face_backend_requested
        or trace.get("requested_backend"),
        "selected": face_detection.face_backend_selected
        or trace.get("selected_backend"),
        "fallback_used": bool(
            face_detection.face_backend_fallback_used
            or trace.get("fallback_used")
        ),
        "detect_elapsed_ms": detect_elapsed_ms,
        "detected": detected,
        "usable": usable,
        "low_quality": bool(detected and not usable),
        "quality_score": face_detection.quality_score,
        "quality_metrics": _as_dict(face_detection.quality_metrics),
        "rejection_reasons": _string_list(face_detection.rejection_reasons),
        "bbox": _as_dict(face_detection.bbox),
        "image_size": _as_dict(face_detection.image_size),
        "faces_detected_count": _coerce_int(trace.get("faces_detected")),
        "configuration": configuration,
        "error": face_detection.face_backend_error or trace.get("error"),
        "provider": trace.get("provider"),
    }


def _semantic_metrics(
    semantic_descriptor_result: SemanticDescriptorResult | None,
) -> dict[str, Any]:
    if semantic_descriptor_result is None:
        return {
            "requested": settings.semantic_descriptor_backend,
            "effective_request": None,
            "selected": None,
            "selected_key": None,
            "fallback_used": False,
            "total_duration_ms": None,
            "descriptor_valid": False,
            "success": False,
            "parser_strategy": None,
            "parser_backend_key": None,
            "json_recovered": False,
            "parser_error": None,
            "error": "semantic_descriptor_not_requested",
            "attempts_count": 0,
            "attempted_backend_keys": [],
            "real_backend_attempted": False,
        }

    descriptor = _as_dict(semantic_descriptor_result.descriptor)
    trace = _semantic_trace(semantic_descriptor_result)
    attempts = _attempts(trace)
    selected_key = trace.get("semantic_backend_selected_key") or trace.get(
        "selected_backend_key"
    )
    selected_attempt = _selected_attempt(attempts, selected_key=selected_key)
    parser_attempt = _parser_attempt(attempts, selected_attempt=selected_attempt)
    policy = _semantic_policy(trace=trace, attempts=attempts, selected_attempt=selected_attempt)
    attempted_backend_keys = [
        str(attempt.get("backend_key"))
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("backend_key")
    ]

    descriptor_valid = bool(
        trace.get("descriptor_valid")
        if "descriptor_valid" in trace
        else semantic_descriptor_result.generated
        and descriptor.get("descriptor_schema_version")
    )
    return {
        "requested": descriptor.get("semantic_backend_requested")
        or trace.get("semantic_backend_requested")
        or settings.semantic_descriptor_backend,
        "effective_request": descriptor.get("semantic_backend_effective_request")
        or trace.get("semantic_backend_effective_request"),
        "selected": descriptor.get("semantic_backend_selected")
        or trace.get("semantic_backend_selected")
        or semantic_descriptor_result.backend,
        "selected_key": selected_key or selected_attempt.get("backend_key"),
        "fallback_used": bool(
            descriptor.get("semantic_backend_fallback_used")
            if "semantic_backend_fallback_used" in descriptor
            else trace.get("semantic_backend_fallback_used")
        ),
        "total_duration_ms": _coerce_float(trace.get("total_duration_ms"))
        or _coerce_float(selected_attempt.get("duration_ms")),
        "descriptor_valid": descriptor_valid,
        "success": descriptor_valid,
        "parser_strategy": parser_attempt.get("parse_strategy_used"),
        "parser_backend_key": parser_attempt.get("backend_key"),
        "json_recovered": bool(parser_attempt.get("json_recovered")),
        "parser_error": parser_attempt.get("parser_error"),
        "policy": policy,
        "error": descriptor.get("semantic_backend_error")
        or trace.get("semantic_backend_error"),
        "attempts_count": len(attempts),
        "attempted_backend_keys": attempted_backend_keys,
        "real_backend_attempted": any(
            _normalize_semantic_key(key) in REAL_SEMANTIC_BACKEND_KEYS
            for key in attempted_backend_keys
        ),
    }


def _budget_metrics(
    semantic_descriptor_result: SemanticDescriptorResult | None,
) -> dict[str, Any]:
    if semantic_descriptor_result is None:
        return {
            "status": "not_applicable",
            "observed_rss_mb": None,
            "max_allowed_rss_mb": None,
            "backend_key": None,
            "rejection_reason": None,
            "rejection_reasons": [],
        }

    attempts = _attempts(_semantic_trace(semantic_descriptor_result))
    budget_attempt = _budget_attempt(attempts)
    budget = _as_dict(budget_attempt.get("budget"))
    reasons = budget.get("reasons") if isinstance(budget.get("reasons"), list) else []
    rejection_reason = budget_attempt.get("reason") or (reasons[0] if reasons else None)
    status = budget.get("status") or "not_applicable"
    if budget_attempt.get("status") == "rejected_by_budget":
        status = "rejected"
    return {
        "status": status,
        "backend_key": budget.get("backend_key") or budget_attempt.get("backend_key"),
        "budget_scope": budget.get("budget_scope"),
        "observed_rss_mb": _coerce_float(budget.get("observed_rss_mb")),
        "max_allowed_rss_mb": _coerce_float(budget.get("max_allowed_rss_mb")),
        "observed_latency_seconds": _coerce_float(
            budget.get("observed_latency_seconds")
        ),
        "max_allowed_latency_seconds": _coerce_float(
            budget.get("max_allowed_latency_seconds")
        ),
        "rejection_reason": rejection_reason,
        "rejection_reasons": reasons or ([rejection_reason] if rejection_reason else []),
    }


def _config_metrics(camera_runtime_config_trace: dict[str, Any]) -> dict[str, Any]:
    trace = _as_dict(camera_runtime_config_trace)
    return {
        "source": _normalized_config_source(trace),
        "raw_config_source": trace.get("config_source"),
        "face_tuning_source": trace.get("face_tuning_source"),
        "vlm_policy_source": trace.get("vlm_policy_source"),
        "camera_config_version": trace.get("camera_config_version"),
        "face_camera_config_version": trace.get("face_camera_config_version"),
        "vlm_camera_config_version": trace.get("vlm_camera_config_version"),
        "camera_config_hash": trace.get("camera_config_hash"),
        "effective_config_hash": trace.get("effective_config_hash"),
        "face_effective_config_hash": trace.get("face_effective_config_hash"),
        "vlm_effective_policy_hash": trace.get("vlm_effective_policy_hash"),
        "camera_override_applied": bool(trace.get("camera_override_applied")),
    }


def _face_configuration(trace: dict[str, Any]) -> dict[str, Any]:
    configuration = _as_dict(trace.get("configuration"))
    quality_thresholds = _as_dict(
        configuration.get("quality_thresholds") or trace.get("quality_thresholds")
    )
    return {
        "model_name": configuration.get("model_name"),
        "provider": configuration.get("provider") or trace.get("provider"),
        "det_size": configuration.get("det_size"),
        "detection_threshold": _coerce_float(configuration.get("detection_threshold")),
        "max_faces": _coerce_int(configuration.get("max_faces")),
        "face_quality_threshold": _coerce_float(
            quality_thresholds.get("face_quality_threshold")
            or configuration.get("face_quality_threshold")
        ),
        "min_face_bbox_size": _coerce_int(
            quality_thresholds.get("min_face_bbox_size")
            or configuration.get("min_face_bbox_size")
        ),
        "min_face_area_ratio": _coerce_float(
            quality_thresholds.get("min_face_area_ratio")
            or configuration.get("min_face_area_ratio")
        ),
        "config_source": configuration.get("config_source") or trace.get("config_source"),
        "camera_config_version": configuration.get("camera_config_version"),
        "camera_config_hash": configuration.get("camera_config_hash"),
        "effective_config_hash": configuration.get("effective_config_hash"),
    }


def _semantic_policy(
    *,
    trace: dict[str, Any],
    attempts: list[dict[str, Any]],
    selected_attempt: dict[str, Any],
) -> dict[str, Any]:
    policy_trace = _as_dict(trace.get("vlm_policy_trace"))
    if not policy_trace:
        policy_trace = _as_dict(selected_attempt.get("policy"))
    if not policy_trace:
        for attempt in attempts:
            policy_trace = _as_dict(attempt.get("policy"))
            if policy_trace:
                break

    camera_policy = _as_dict(policy_trace.get("camera_policy"))
    event_policy = _as_dict(policy_trace.get("event_policy"))
    budget = _as_dict(policy_trace.get("budget") or trace.get("policy_budget"))
    backend_budgets = _as_dict(budget.get("backend_budgets"))
    qwen_budget = _as_dict(backend_budgets.get("qwen"))
    smolvlm_budget = _as_dict(backend_budgets.get("smolvlm"))
    return {
        "requested_backend": policy_trace.get("requested_backend")
        or trace.get("semantic_backend_requested"),
        "effective_backend_request": policy_trace.get("effective_backend_request")
        or trace.get("semantic_backend_effective_request"),
        "effective_backend_key": policy_trace.get("effective_backend_key")
        or trace.get("semantic_backend_effective_key"),
        "allowed_backend_key": policy_trace.get("allowed_backend_key")
        or trace.get("semantic_backend_allowed_key"),
        "backend_chain": _string_list(
            policy_trace.get("backend_chain") or trace.get("semantic_backend_candidate_chain")
        ),
        "enabled": camera_policy.get("enabled"),
        "force_simple": camera_policy.get("force_simple"),
        "backend": camera_policy.get("backend"),
        "preferred_backend": camera_policy.get("preferred_backend"),
        "secondary_backend": camera_policy.get("secondary_backend"),
        "enabled_event_types": _string_list(event_policy.get("enabled_event_types")),
        "disabled_event_types": _string_list(event_policy.get("disabled_event_types")),
        "degradation_policy": policy_trace.get("degradation_policy"),
        "max_allowed_latency_seconds": _coerce_float(
            budget.get("max_allowed_latency_seconds")
        ),
        "max_allowed_rss_mb": _coerce_float(budget.get("max_allowed_rss_mb")),
        "qwen_max_allowed_rss_mb": _coerce_float(
            qwen_budget.get("max_allowed_rss_mb")
        ),
        "smolvlm_max_allowed_rss_mb": _coerce_float(
            smolvlm_budget.get("max_allowed_rss_mb")
        ),
        "max_concurrent_inferences": _coerce_int(
            budget.get("max_concurrent_inferences")
        ),
        "config_source": policy_trace.get("config_source")
        or trace.get("semantic_backend_policy_sources"),
        "camera_config_version": policy_trace.get("camera_config_version"),
        "camera_config_hash": policy_trace.get("camera_config_hash"),
        "effective_policy_hash": policy_trace.get("effective_policy_hash"),
    }


def _semantic_trace(result: SemanticDescriptorResult) -> dict[str, Any]:
    descriptor = _as_dict(result.descriptor)
    trace = descriptor.get("semantic_backend_trace") or descriptor.get("generation_trace")
    return _as_dict(trace)


def _attempts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = trace.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [attempt for attempt in attempts if isinstance(attempt, dict)]


def _selected_attempt(
    attempts: list[dict[str, Any]],
    *,
    selected_key: Any,
) -> dict[str, Any]:
    normalized_selected = _normalize_semantic_key(selected_key)
    for attempt in attempts:
        if (
            attempt.get("status") == "success"
            and _normalize_semantic_key(attempt.get("backend_key")) == normalized_selected
        ):
            return attempt
    for attempt in attempts:
        if attempt.get("status") == "success":
            return attempt
    return {}


def _parser_attempt(
    attempts: list[dict[str, Any]],
    *,
    selected_attempt: dict[str, Any],
) -> dict[str, Any]:
    if selected_attempt.get("parse_strategy_used") or selected_attempt.get("parser_error"):
        return selected_attempt
    for attempt in reversed(attempts):
        if attempt.get("parse_strategy_used") or attempt.get("parser_error"):
            return attempt
    return {}


def _budget_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    for attempt in attempts:
        if attempt.get("status") == "rejected_by_budget":
            return attempt
    for attempt in reversed(attempts):
        budget = _as_dict(attempt.get("budget"))
        if budget and budget.get("status") not in {None, "not_applicable"}:
            return attempt
    return {}


def _normalized_config_source(trace: dict[str, Any]) -> str:
    sources = [
        trace.get("face_tuning_source"),
        trace.get("vlm_policy_source"),
        trace.get("config_source"),
    ]
    for source in sources:
        normalized = str(source or "").strip()
        if normalized in {"api.camera.metadata", "api_camera_metadata"}:
            return "api_camera_metadata"
        if normalized in {"env", "environment"}:
            return "env"
        if normalized in {"global", "global_defaults"}:
            return "global"
        if normalized in {"default", "defaults"}:
            return "default"
        if normalized and normalized not in {"not_provided", "not_evaluated"}:
            return normalized
    return "global"


def _normalize_semantic_key(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"qwen_vl", "qwen-vl"}:
        return "qwen"
    if normalized in {"smol_vlm", "smol-vlm"}:
        return "smolvlm"
    if normalized == "simple_color_signature_v1":
        return "simple"
    return normalized


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return [str(item) for item in value if item is not None]
