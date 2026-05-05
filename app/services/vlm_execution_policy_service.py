from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.services.vlm_policy_service import POLICY_VERSION as OPERATIONAL_POLICY_VERSION


POLICY_VERSION = "vlm_execution_policy_v1"


def build_vlm_execution_policy_snapshot() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "global_backend_default": settings.semantic_descriptor_backend,
        "fallback_enabled": settings.semantic_enable_fallback,
        "auto_preferred_backend": settings.vlm_auto_preferred_backend,
        "qwen_enabled": settings.effective_qwen_vl_enabled,
        "smolvlm_enabled": settings.effective_smolvlm_enabled,
        "qwen_model_name": settings.effective_qwen_model_name,
        "smolvlm_model_name": settings.effective_smolvlm_model_name,
        "timeout_seconds": settings.effective_vlm_timeout_seconds,
        "max_new_tokens": settings.vlm_max_new_tokens,
        "max_image_edge": settings.vlm_max_image_edge,
        "requested_device": settings.effective_vlm_device,
        "enabled_event_types": settings.vlm_event_type_policy,
        "disabled_camera_ids": settings.vlm_disabled_camera_id_policy,
        "camera_policy_overrides_configured": bool(settings.vlm_camera_policy_overrides),
        "max_allowed_latency_seconds": settings.vlm_max_allowed_latency_seconds,
        "max_allowed_rss_mb": settings.vlm_max_allowed_rss_mb,
        "max_concurrent_inferences": settings.vlm_max_concurrent_inferences,
        "concurrency_acquire_timeout_seconds": settings.vlm_concurrency_acquire_timeout_seconds,
        "degradation_policy": settings.vlm_degradation_policy,
        "secondary_backend": settings.vlm_secondary_backend,
        "recent_failure_threshold": settings.vlm_recent_failure_threshold,
        "circuit_breaker_window_seconds": settings.vlm_circuit_breaker_window_seconds,
        "circuit_breaker_cooldown_seconds": settings.vlm_circuit_breaker_cooldown_seconds,
        "operational_policy_version": OPERATIONAL_POLICY_VERSION,
        "subprocess_isolation": True,
        "serialization_guard_enabled": settings.vlm_serialization_guard_enabled,
        "serialization_guard": "single_inflight_request_per_backend_subprocess",
    }


def validate_vlm_runtime_config() -> dict[str, Any]:
    snapshot = build_vlm_execution_policy_snapshot()
    warnings: list[str] = []
    errors: list[str] = []

    if snapshot["timeout_seconds"] <= 0:
        errors.append("vlm_timeout_seconds_must_be_positive")
    if snapshot["max_new_tokens"] <= 0:
        errors.append("vlm_max_new_tokens_must_be_positive")
    if snapshot["max_image_edge"] < 0:
        errors.append("vlm_max_image_edge_must_be_zero_or_positive")
    if snapshot["auto_preferred_backend"] not in {"qwen", "qwen_vl", "smolvlm"}:
        errors.append("vlm_auto_preferred_backend_must_be_qwen_or_smolvlm")
    if snapshot["secondary_backend"] not in {"qwen", "qwen_vl", "smolvlm"}:
        errors.append("vlm_secondary_backend_must_be_qwen_or_smolvlm")
    if snapshot["max_allowed_latency_seconds"] < 0:
        errors.append("vlm_max_allowed_latency_seconds_must_be_zero_or_positive")
    if snapshot["max_allowed_rss_mb"] < 0:
        errors.append("vlm_max_allowed_rss_mb_must_be_zero_or_positive")
    if snapshot["max_concurrent_inferences"] < 1:
        errors.append("vlm_max_concurrent_inferences_must_be_positive")
    if snapshot["degradation_policy"] not in {
        "auto_then_secondary_then_simple",
        "preferred_then_secondary_then_simple",
        "preferred_then_simple",
        "simple_only",
    }:
        errors.append("vlm_degradation_policy_unknown")
    raw_camera_overrides = settings.vlm_camera_policy_overrides_json.strip()
    if raw_camera_overrides:
        try:
            parsed_camera_overrides = json.loads(raw_camera_overrides)
        except json.JSONDecodeError:
            errors.append("vlm_camera_policy_overrides_json_invalid")
        else:
            if not isinstance(parsed_camera_overrides, dict):
                errors.append("vlm_camera_policy_overrides_json_invalid")
    if not snapshot["enabled_event_types"]:
        warnings.append("vlm_event_type_policy_empty")
    if snapshot["global_backend_default"] != "simple" and not snapshot["fallback_enabled"]:
        warnings.append("vlm_enabled_without_fallback")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "policy": snapshot,
    }
