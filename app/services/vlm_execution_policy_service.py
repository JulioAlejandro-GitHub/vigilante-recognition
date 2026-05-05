from __future__ import annotations

from typing import Any

from app.config import settings


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
