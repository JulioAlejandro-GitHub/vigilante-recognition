from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Iterable

from app.config import Settings, settings
from app.services.runtime_metrics_store import JsonlRuntimeMetricsStore
from app.services.runtime_recommendation_rules import (
    RECOMMENDATION_RULESET_VERSION,
    evaluate_camera_recommendations,
)
from app.services.runtime_recommendation_store import JsonlRuntimeRecommendationStore

RUNTIME_RECOMMENDATIONS_RESULT_SCHEMA_VERSION = "runtime_recommendations_result_v1"
REAL_SEMANTIC_BACKENDS = {"qwen", "smolvlm"}


class RuntimeRecommendationService:
    def __init__(
        self,
        *,
        enabled: bool,
        metrics_store: JsonlRuntimeMetricsStore,
        recommendation_store: JsonlRuntimeRecommendationStore | None = None,
        min_events_per_camera: int = 20,
        window_hours: float = 24.0,
        default_face_tuning: dict[str, Any] | None = None,
        default_vlm_policy: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.metrics_store = metrics_store
        self.recommendation_store = recommendation_store
        self.min_events_per_camera = max(1, int(min_events_per_camera))
        self.window_hours = max(0.0, float(window_hours))
        self.default_face_tuning = deepcopy(default_face_tuning or {})
        self.default_vlm_policy = deepcopy(default_vlm_policy or {})

    @classmethod
    def from_settings(
        cls,
        *,
        settings_obj: Settings | None = None,
        metrics_store: JsonlRuntimeMetricsStore | None = None,
    ) -> "RuntimeRecommendationService":
        resolved_settings = settings_obj or settings
        resolved_metrics_store = metrics_store or JsonlRuntimeMetricsStore(
            resolved_settings.runtime_metrics_path,
            rotate_max_mb=resolved_settings.runtime_metrics_rotate_max_mb,
            retention_files=resolved_settings.runtime_metrics_retention_files,
        )
        recommendation_store = JsonlRuntimeRecommendationStore(
            resolved_settings.runtime_recommendations_path,
            rotate_max_mb=resolved_settings.runtime_recommendations_rotate_max_mb,
            retention_files=resolved_settings.runtime_recommendations_retention_files,
        )
        return cls(
            enabled=resolved_settings.runtime_recommendations_enabled,
            metrics_store=resolved_metrics_store,
            recommendation_store=recommendation_store,
            min_events_per_camera=(
                resolved_settings.runtime_recommendations_min_events_per_camera
            ),
            window_hours=resolved_settings.runtime_recommendations_window_hours,
            default_face_tuning=_default_face_tuning(resolved_settings),
            default_vlm_policy=_default_vlm_policy(resolved_settings),
        )

    def generate(
        self,
        *,
        persist: bool = False,
        include_status: bool = True,
        camera_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        generated_at = now.isoformat()
        if not self.enabled and not force:
            return {
                "schema_version": RUNTIME_RECOMMENDATIONS_RESULT_SCHEMA_VERSION,
                "rule_set_version": RECOMMENDATION_RULESET_VERSION,
                "generated_at": generated_at,
                "enabled": False,
                "status": "disabled",
                "window_summary": self._empty_window_summary(now=now),
                "recommendations": [],
                "by_camera": {},
                "persisted_count": 0,
            }

        records, window_summary = self._load_window_records(now=now)
        aggregate = aggregate_recommendation_metrics(
            records,
            default_face_tuning=self.default_face_tuning,
            default_vlm_policy=self.default_vlm_policy,
        )

        recommendations: list[dict[str, Any]] = []
        by_camera = aggregate["by_camera"]
        for candidate_camera_id, camera_metrics in by_camera.items():
            if camera_id and str(candidate_camera_id) != str(camera_id):
                continue
            recommendations.extend(
                evaluate_camera_recommendations(
                    camera_metrics,
                    generated_at=generated_at,
                    window_summary=window_summary,
                    min_events_per_camera=self.min_events_per_camera,
                    include_status=include_status,
                )
            )

        persisted_count = 0
        if persist and self.recommendation_store is not None and recommendations:
            persisted_count = self.recommendation_store.append_many(recommendations)

        actionable_count = sum(1 for item in recommendations if item.get("actionable"))
        return {
            "schema_version": RUNTIME_RECOMMENDATIONS_RESULT_SCHEMA_VERSION,
            "rule_set_version": RECOMMENDATION_RULESET_VERSION,
            "generated_at": generated_at,
            "enabled": self.enabled,
            "status": "ok",
            "window_summary": window_summary,
            "camera_count": len(by_camera),
            "recommendation_count": len(recommendations),
            "actionable_recommendation_count": actionable_count,
            "recommendations": recommendations,
            "by_camera": by_camera,
            "persisted_count": persisted_count,
        }

    def read_persisted(
        self,
        *,
        camera_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.recommendation_store is None:
            return []
        records = [
            record
            for record in self.recommendation_store.iter_records()
            if not camera_id or str(record.get("camera_id")) == str(camera_id)
        ]
        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def _load_window_records(self, *, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cutoff = now - timedelta(hours=self.window_hours) if self.window_hours > 0 else None
        records: list[dict[str, Any]] = []
        records_seen = 0
        outside_window = 0
        without_timestamp = 0
        first_event_at: datetime | None = None
        last_event_at: datetime | None = None

        for record in self.metrics_store.iter_records():
            records_seen += 1
            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is None:
                without_timestamp += 1
            elif cutoff is not None and timestamp < cutoff:
                outside_window += 1
                continue

            records.append(record)
            if timestamp is not None:
                if first_event_at is None or timestamp < first_event_at:
                    first_event_at = timestamp
                if last_event_at is None or timestamp > last_event_at:
                    last_event_at = timestamp

        return records, {
            "window_hours": self.window_hours,
            "window_start": cutoff.isoformat() if cutoff is not None else None,
            "window_end": now.isoformat(),
            "first_event_at": first_event_at.isoformat() if first_event_at else None,
            "last_event_at": last_event_at.isoformat() if last_event_at else None,
            "metrics_records_seen": records_seen,
            "metrics_records_in_window": len(records),
            "metrics_records_outside_window": outside_window,
            "metrics_records_without_timestamp": without_timestamp,
            "rule_set_version": RECOMMENDATION_RULESET_VERSION,
        }

    def _empty_window_summary(self, *, now: datetime) -> dict[str, Any]:
        cutoff = now - timedelta(hours=self.window_hours) if self.window_hours > 0 else None
        return {
            "window_hours": self.window_hours,
            "window_start": cutoff.isoformat() if cutoff is not None else None,
            "window_end": now.isoformat(),
            "first_event_at": None,
            "last_event_at": None,
            "metrics_records_seen": 0,
            "metrics_records_in_window": 0,
            "metrics_records_outside_window": 0,
            "metrics_records_without_timestamp": 0,
            "rule_set_version": RECOMMENDATION_RULESET_VERSION,
        }


def aggregate_recommendation_metrics(
    records: Iterable[dict[str, Any]],
    *,
    default_face_tuning: dict[str, Any] | None = None,
    default_vlm_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_camera: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        camera_id = str(record.get("camera_id") or "unknown")
        camera = by_camera.setdefault(
            camera_id,
            _new_camera_metrics(
                camera_id,
                default_face_tuning=default_face_tuning or {},
                default_vlm_policy=default_vlm_policy or {},
            ),
        )
        _accumulate_record(camera, record)

    finalized = {
        camera_id: _finalize_camera_metrics(camera)
        for camera_id, camera in sorted(by_camera.items())
    }
    return {"by_camera": finalized}


def _new_camera_metrics(
    camera_id: str,
    *,
    default_face_tuning: dict[str, Any],
    default_vlm_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "processed_events": 0,
        "face_detected_count": 0,
        "face_usable_count": 0,
        "low_quality_face_count": 0,
        "small_face_signal_count": 0,
        "semantic_selected_count": 0,
        "semantic_success_count": 0,
        "semantic_fallback_count": 0,
        "fallback_to_simple_count": 0,
        "vlm_attempted_count": 0,
        "vlm_success_count": 0,
        "parser_seen_count": 0,
        "parser_recovered_count": 0,
        "invalid_json_count": 0,
        "budget_rejected_count": 0,
        "current_face_tuning": deepcopy(default_face_tuning),
        "current_vlm_policy": deepcopy(default_vlm_policy),
        "_face_latencies_ms": [],
        "_vlm_durations_ms": [],
        "_semantic_backend_counts": Counter(),
        "_face_rejection_reasons": Counter(),
        "_face_size_scores": [],
        "_face_backends": {},
        "_semantic_backends": {},
        "_event_types": {},
    }


def _new_face_backend_metrics() -> dict[str, Any]:
    return {
        "selected_count": 0,
        "requested_count": 0,
        "detected_count": 0,
        "usable_count": 0,
        "low_quality_count": 0,
        "fallback_count": 0,
        "fallback_away_count": 0,
        "error_count": 0,
        "_latencies_ms": [],
    }


def _new_semantic_backend_metrics() -> dict[str, Any]:
    return {
        "attempted_count": 0,
        "selected_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "fallback_away_count": 0,
        "parser_seen_count": 0,
        "parser_recovered_count": 0,
        "invalid_json_count": 0,
        "budget_rejected_count": 0,
        "_durations_ms": [],
        "_observed_rss_mb": [],
    }


def _new_event_type_metrics() -> dict[str, Any]:
    return {
        "processed_events": 0,
        "semantic_selected_count": 0,
        "semantic_success_count": 0,
        "semantic_fallback_count": 0,
        "fallback_to_simple_count": 0,
        "vlm_attempted_count": 0,
        "vlm_success_count": 0,
        "budget_rejected_count": 0,
        "_semantic_backend_counts": Counter(),
        "_vlm_durations_ms": [],
    }


def _accumulate_record(camera: dict[str, Any], record: dict[str, Any]) -> None:
    camera["processed_events"] += 1
    event_type = str(record.get("event_type") or "unknown")
    event_metrics = camera["_event_types"].setdefault(
        event_type,
        _new_event_type_metrics(),
    )
    event_metrics["processed_events"] += 1

    _accumulate_face(camera, record)
    _accumulate_semantic(camera, event_metrics, record)


def _accumulate_face(camera: dict[str, Any], record: dict[str, Any]) -> None:
    face = _dict(record.get("face"))
    selected = _clean_face_key(face.get("selected")) or "unknown"
    requested = _clean_face_key(face.get("requested"))
    selected_bucket = camera["_face_backends"].setdefault(
        selected,
        _new_face_backend_metrics(),
    )
    selected_bucket["selected_count"] += 1

    if requested:
        requested_bucket = camera["_face_backends"].setdefault(
            requested,
            _new_face_backend_metrics(),
        )
        requested_bucket["requested_count"] += 1
        if face.get("fallback_used") is True and requested != selected:
            requested_bucket["fallback_away_count"] += 1

    detected = face.get("detected") is True
    usable = face.get("usable") is True
    low_quality = face.get("low_quality") is True
    if detected:
        camera["face_detected_count"] += 1
        selected_bucket["detected_count"] += 1
    if usable:
        camera["face_usable_count"] += 1
        selected_bucket["usable_count"] += 1
    if low_quality:
        camera["low_quality_face_count"] += 1
        selected_bucket["low_quality_count"] += 1
    if face.get("fallback_used") is True:
        selected_bucket["fallback_count"] += 1
    if face.get("error"):
        selected_bucket["error_count"] += 1

    _append_number(camera["_face_latencies_ms"], face.get("detect_elapsed_ms"))
    _append_number(selected_bucket["_latencies_ms"], face.get("detect_elapsed_ms"))

    quality_metrics = _dict(face.get("quality_metrics"))
    size_score = _coerce_float(quality_metrics.get("size_score"))
    if size_score is not None:
        camera["_face_size_scores"].append(size_score)

    reasons = _string_list(face.get("rejection_reasons"))
    for reason in reasons:
        camera["_face_rejection_reasons"][reason] += 1
    if any(_small_face_reason(reason) for reason in reasons) or (
        low_quality and size_score is not None and size_score < 0.35
    ):
        camera["small_face_signal_count"] += 1

    face_configuration = _extract_face_configuration(face)
    if face_configuration:
        camera["current_face_tuning"].update(face_configuration)
    if requested:
        camera["current_face_tuning"]["face_backend"] = requested


def _accumulate_semantic(
    camera: dict[str, Any],
    event_metrics: dict[str, Any],
    record: dict[str, Any],
) -> None:
    semantic = _dict(record.get("semantic"))
    budget = _dict(record.get("budget"))
    selected = _semantic_backend_key(semantic)
    attempted = _attempted_semantic_backend_keys(semantic, selected=selected)
    real_attempted = any(key in REAL_SEMANTIC_BACKENDS for key in attempted)
    descriptor_valid = semantic.get("descriptor_valid") is True or semantic.get("success") is True
    fallback_used = semantic.get("fallback_used") is True
    duration_ms = semantic.get("total_duration_ms")

    if selected:
        camera["_semantic_backend_counts"][selected] += 1
        event_metrics["_semantic_backend_counts"][selected] += 1
        camera["semantic_selected_count"] += 1
        event_metrics["semantic_selected_count"] += 1
        selected_bucket = camera["_semantic_backends"].setdefault(
            selected,
            _new_semantic_backend_metrics(),
        )
        selected_bucket["selected_count"] += 1
        if descriptor_valid:
            selected_bucket["success_count"] += 1
        if fallback_used:
            selected_bucket["fallback_count"] += 1
        _append_number(selected_bucket["_durations_ms"], duration_ms)

    for backend_key in attempted:
        backend_bucket = camera["_semantic_backends"].setdefault(
            backend_key,
            _new_semantic_backend_metrics(),
        )
        backend_bucket["attempted_count"] += 1
        if fallback_used and selected and backend_key != selected:
            backend_bucket["fallback_away_count"] += 1

    if descriptor_valid:
        camera["semantic_success_count"] += 1
        event_metrics["semantic_success_count"] += 1
        if selected in REAL_SEMANTIC_BACKENDS:
            camera["vlm_success_count"] += 1
            event_metrics["vlm_success_count"] += 1
    if fallback_used:
        camera["semantic_fallback_count"] += 1
        event_metrics["semantic_fallback_count"] += 1
        if selected == "simple":
            camera["fallback_to_simple_count"] += 1
            event_metrics["fallback_to_simple_count"] += 1
    if real_attempted:
        camera["vlm_attempted_count"] += 1
        event_metrics["vlm_attempted_count"] += 1

    parser_backend = _clean_semantic_key(semantic.get("parser_backend_key")) or selected
    parser_seen = bool(semantic.get("parser_strategy") or semantic.get("parser_error"))
    if parser_seen:
        camera["parser_seen_count"] += 1
        if parser_backend:
            parser_bucket = camera["_semantic_backends"].setdefault(
                parser_backend,
                _new_semantic_backend_metrics(),
            )
            parser_bucket["parser_seen_count"] += 1
            if semantic.get("json_recovered") is True:
                parser_bucket["parser_recovered_count"] += 1
            if _invalid_json_error(semantic.get("parser_error")):
                parser_bucket["invalid_json_count"] += 1
        if semantic.get("json_recovered") is True:
            camera["parser_recovered_count"] += 1
        if _invalid_json_error(semantic.get("parser_error")):
            camera["invalid_json_count"] += 1

    if _budget_rejected(budget):
        camera["budget_rejected_count"] += 1
        event_metrics["budget_rejected_count"] += 1
        budget_backend = _clean_semantic_key(budget.get("backend_key")) or selected
        if budget_backend:
            budget_bucket = camera["_semantic_backends"].setdefault(
                budget_backend,
                _new_semantic_backend_metrics(),
            )
            budget_bucket["budget_rejected_count"] += 1
            if budget_backend not in attempted:
                budget_bucket["attempted_count"] += 1

    budget_backend = _clean_semantic_key(budget.get("backend_key")) or selected
    if budget_backend:
        budget_bucket = camera["_semantic_backends"].setdefault(
            budget_backend,
            _new_semantic_backend_metrics(),
        )
        _append_number(budget_bucket["_observed_rss_mb"], budget.get("observed_rss_mb"))

    _append_number(camera["_vlm_durations_ms"], duration_ms)
    _append_number(event_metrics["_vlm_durations_ms"], duration_ms)

    policy = _extract_vlm_policy(semantic)
    if policy:
        camera["current_vlm_policy"].update(policy)


def _finalize_camera_metrics(camera: dict[str, Any]) -> dict[str, Any]:
    processed = int(camera["processed_events"])
    face_latencies = camera.pop("_face_latencies_ms")
    vlm_durations = camera.pop("_vlm_durations_ms")
    semantic_counts = camera.pop("_semantic_backend_counts")
    face_rejection_reasons = camera.pop("_face_rejection_reasons")
    face_size_scores = camera.pop("_face_size_scores")
    face_backends = camera.pop("_face_backends")
    semantic_backends = camera.pop("_semantic_backends")
    event_types = camera.pop("_event_types")

    camera["face_detected_rate"] = _rate(camera["face_detected_count"], processed)
    camera["face_usable_rate"] = _rate(camera["face_usable_count"], processed)
    camera["low_quality_face_rate"] = _rate(
        camera["low_quality_face_count"],
        processed,
    )
    camera["small_face_rejection_rate"] = _rate(
        camera["small_face_signal_count"],
        processed,
    )
    camera["face_rejection_reasons"] = dict(sorted(face_rejection_reasons.items()))
    camera["face_size_score"] = _value_summary(face_size_scores)
    camera["semantic_backend_most_used"] = _most_common(semantic_counts)
    camera["semantic_success_rate"] = _rate(
        camera["semantic_success_count"],
        camera["semantic_selected_count"],
    )
    camera["semantic_fallback_rate"] = _rate(
        camera["semantic_fallback_count"],
        camera["semantic_selected_count"],
    )
    camera["vlm_success_rate"] = _rate(
        camera["vlm_success_count"],
        camera["vlm_attempted_count"],
    )
    camera["fallback_to_simple_rate"] = _rate(
        camera["fallback_to_simple_count"],
        processed,
    )
    camera["fallback_to_simple_per_vlm_attempt_rate"] = _rate(
        camera["fallback_to_simple_count"],
        camera["vlm_attempted_count"],
    )
    camera["parser_recovery_rate"] = _rate(
        camera["parser_recovered_count"],
        camera["parser_seen_count"],
    )
    camera["invalid_json_rate"] = _rate(
        camera["invalid_json_count"],
        camera["parser_seen_count"],
    )
    camera["budget_rejection_rate"] = _rate(
        camera["budget_rejected_count"],
        camera["vlm_attempted_count"],
    )
    camera["face_detect_latency_ms"] = _latency_summary(face_latencies)
    camera["vlm_duration_ms"] = _latency_summary(vlm_durations)
    camera["face_backends"] = _finalize_face_backends(face_backends, total=processed)
    camera["semantic_backends"] = _finalize_semantic_backends(semantic_backends)
    camera["event_types"] = _finalize_event_types(event_types)
    return camera


def _finalize_face_backends(
    buckets: dict[str, dict[str, Any]],
    *,
    total: int,
) -> dict[str, dict[str, Any]]:
    finalized = {}
    for backend_key, bucket in sorted(buckets.items()):
        latencies = bucket.pop("_latencies_ms")
        selected_count = int(bucket["selected_count"])
        requested_count = int(bucket["requested_count"])
        bucket["usage_rate"] = _rate(selected_count, total)
        bucket["detected_rate"] = _rate(bucket["detected_count"], selected_count)
        bucket["usable_rate"] = _rate(bucket["usable_count"], selected_count)
        bucket["low_quality_rate"] = _rate(bucket["low_quality_count"], selected_count)
        bucket["fallback_rate"] = _rate(bucket["fallback_count"], selected_count)
        bucket["fallback_away_rate"] = _rate(
            bucket["fallback_away_count"],
            requested_count,
        )
        bucket["detect_latency_ms"] = _latency_summary(latencies)
        finalized[backend_key] = bucket
    return finalized


def _finalize_semantic_backends(
    buckets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized = {}
    for backend_key, bucket in sorted(buckets.items()):
        durations = bucket.pop("_durations_ms")
        rss_values = bucket.pop("_observed_rss_mb")
        attempted_count = int(bucket["attempted_count"])
        selected_count = int(bucket["selected_count"])
        denominator = attempted_count or selected_count
        bucket["success_rate"] = _rate(bucket["success_count"], denominator)
        bucket["selected_success_rate"] = _rate(bucket["success_count"], selected_count)
        bucket["fallback_rate"] = _rate(bucket["fallback_count"], selected_count)
        bucket["fallback_away_rate"] = _rate(bucket["fallback_away_count"], denominator)
        bucket["parser_recovery_rate"] = _rate(
            bucket["parser_recovered_count"],
            bucket["parser_seen_count"],
        )
        bucket["invalid_json_rate"] = _rate(
            bucket["invalid_json_count"],
            bucket["parser_seen_count"],
        )
        bucket["budget_rejection_rate"] = _rate(
            bucket["budget_rejected_count"],
            denominator,
        )
        bucket["duration_ms"] = _latency_summary(durations)
        bucket["observed_rss_mb"] = _value_summary(rss_values)
        finalized[backend_key] = bucket
    return finalized


def _finalize_event_types(
    buckets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized = {}
    for event_type, bucket in sorted(buckets.items()):
        durations = bucket.pop("_vlm_durations_ms")
        semantic_counts = bucket.pop("_semantic_backend_counts")
        processed = int(bucket["processed_events"])
        bucket["semantic_backend_most_used"] = _most_common(semantic_counts)
        bucket["semantic_success_rate"] = _rate(
            bucket["semantic_success_count"],
            bucket["semantic_selected_count"],
        )
        bucket["vlm_success_rate"] = _rate(
            bucket["vlm_success_count"],
            bucket["vlm_attempted_count"],
        )
        bucket["fallback_to_simple_rate"] = _rate(
            bucket["fallback_to_simple_count"],
            processed,
        )
        bucket["fallback_to_simple_per_vlm_attempt_rate"] = _rate(
            bucket["fallback_to_simple_count"],
            bucket["vlm_attempted_count"],
        )
        bucket["budget_rejection_rate"] = _rate(
            bucket["budget_rejected_count"],
            bucket["vlm_attempted_count"],
        )
        bucket["vlm_duration_ms"] = _latency_summary(durations)
        finalized[event_type] = bucket
    return finalized


def _default_face_tuning(settings_obj: Settings) -> dict[str, Any]:
    return {
        "face_backend": settings_obj.face_backend,
        "det_size": settings_obj.insightface_det_size,
        "detection_threshold": settings_obj.insightface_detection_threshold,
        "max_faces": settings_obj.insightface_max_faces,
        "face_quality_threshold": settings_obj.face_quality_threshold,
        "min_face_bbox_size": settings_obj.insightface_min_face_bbox_size,
        "min_face_area_ratio": settings_obj.insightface_min_face_area_ratio,
        "config_source": "settings_defaults",
    }


def _default_vlm_policy(settings_obj: Settings) -> dict[str, Any]:
    return {
        "backend": settings_obj.semantic_descriptor_backend,
        "preferred_backend": settings_obj.vlm_auto_preferred_backend,
        "secondary_backend": settings_obj.vlm_secondary_backend,
        "enabled_event_types": settings_obj.vlm_event_type_policy,
        "disabled_event_types": [],
        "degradation_policy": settings_obj.vlm_degradation_policy,
        "max_allowed_latency_seconds": settings_obj.vlm_max_allowed_latency_seconds,
        "max_allowed_rss_mb": settings_obj.vlm_max_allowed_rss_mb,
        "qwen_max_allowed_rss_mb": settings_obj.qwen_max_allowed_rss_mb,
        "smolvlm_max_allowed_rss_mb": settings_obj.smolvlm_max_allowed_rss_mb,
        "max_concurrent_inferences": settings_obj.vlm_max_concurrent_inferences,
        "config_source": "settings_defaults",
    }


def _extract_face_configuration(face: dict[str, Any]) -> dict[str, Any]:
    configuration = _dict(face.get("configuration"))
    result = {}
    for key in [
        "det_size",
        "detection_threshold",
        "max_faces",
        "face_quality_threshold",
        "min_face_bbox_size",
        "min_face_area_ratio",
        "config_source",
        "camera_config_version",
        "camera_config_hash",
        "effective_config_hash",
    ]:
        value = configuration.get(key)
        if value is not None:
            result[key] = value
    return result


def _extract_vlm_policy(semantic: dict[str, Any]) -> dict[str, Any]:
    policy = _dict(semantic.get("policy"))
    result = {}
    for key in [
        "requested_backend",
        "effective_backend_request",
        "effective_backend_key",
        "allowed_backend_key",
        "backend_chain",
        "enabled",
        "force_simple",
        "backend",
        "preferred_backend",
        "secondary_backend",
        "enabled_event_types",
        "disabled_event_types",
        "degradation_policy",
        "max_allowed_latency_seconds",
        "max_allowed_rss_mb",
        "qwen_max_allowed_rss_mb",
        "smolvlm_max_allowed_rss_mb",
        "max_concurrent_inferences",
        "config_source",
        "camera_config_version",
        "camera_config_hash",
        "effective_policy_hash",
    ]:
        value = policy.get(key)
        if value is not None and value != []:
            result[key] = value
    return result


def _attempted_semantic_backend_keys(
    semantic: dict[str, Any],
    *,
    selected: str | None,
) -> list[str]:
    keys = [
        key
        for key in (_clean_semantic_key(value) for value in _string_list(semantic.get("attempted_backend_keys")))
        if key
    ]
    if not keys:
        effective = _clean_semantic_key(semantic.get("effective_request"))
        if effective:
            keys.append(effective)
    if selected and selected not in keys:
        keys.append(selected)
    result: list[str] = []
    for key in keys:
        if key not in result:
            result.append(key)
    return result


def _semantic_backend_key(semantic: dict[str, Any]) -> str | None:
    return _clean_semantic_key(semantic.get("selected_key")) or _clean_semantic_key(
        semantic.get("selected")
    )


def _clean_face_key(value: Any) -> str | None:
    cleaned = _clean_key(value)
    if cleaned in {"simple_opencv_haar"}:
        return "simple"
    return cleaned


def _clean_semantic_key(value: Any) -> str | None:
    cleaned = _clean_key(value)
    if cleaned in {"qwen_vl", "qwen-vl"}:
        return "qwen"
    if cleaned in {"smol_vlm", "smol-vlm"}:
        return "smolvlm"
    if cleaned in {"simple_color_signature_v1"}:
        return "simple"
    return cleaned


def _clean_key(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned:
        return None
    aliases = {
        "qwen/qwen2.5-vl-3b-instruct": "qwen",
        "qwen2.5-vl-3b-instruct": "qwen",
        "huggingfacetb/smolvlm2-2.2b-instruct": "smolvlm",
        "smolvlm2-2.2b-instruct": "smolvlm",
    }
    return aliases.get(cleaned, cleaned)


def _budget_rejected(budget: dict[str, Any]) -> bool:
    status = _clean_key(budget.get("status"))
    if status in {"rejected", "exceeded"}:
        return True
    reasons = budget.get("rejection_reasons")
    return bool(budget.get("rejection_reason") or (isinstance(reasons, list) and reasons))


def _invalid_json_error(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).lower()
    return "invalid_json" in normalized or "missing_json" in normalized


def _small_face_reason(reason: str) -> bool:
    normalized = str(reason).lower()
    return any(token in normalized for token in ["bbox", "area", "small", "size"])


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _append_number(values: list[float], value: Any) -> None:
    coerced = _coerce_float(value)
    if coerced is not None:
        values.append(coerced)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if item is not None]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 2),
        "p50": round(_percentile(ordered, 50), 2),
        "p95": round(_percentile(ordered, 95), 2),
    }


def _value_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 2),
        "max": round(max(values), 2),
    }


def _percentile(ordered_values: list[float], percentile: int) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (percentile / 100) * (len(ordered_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = rank - lower
    return ordered_values[lower] + (
        (ordered_values[upper] - ordered_values[lower]) * weight
    )


def _most_common(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]
