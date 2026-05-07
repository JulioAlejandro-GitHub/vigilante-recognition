from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from typing import Any

from app.config import settings
from app.domain.entities import FaceDetectionResult
from app.services.camera_runtime_config_service import extract_camera_runtime_config, stable_config_hash
from app.services.vlm_degradation_service import vlm_degradation_state


logger = logging.getLogger(__name__)

POLICY_VERSION = "vlm_operational_policy_v1"
REAL_BACKEND_KEYS = {"qwen", "qwen_vl", "smolvlm"}
NORMALIZED_REAL_BACKEND_KEYS = {"qwen", "smolvlm"}
SIMPLE_BACKEND_KEY = "simple"
AUTO_BACKEND_KEY = "auto"


@dataclass
class VlmPolicyDecision:
    requested_backend: str
    requested_backend_key: str
    effective_backend_request: str
    effective_backend_key: str
    allowed_backend_key: str
    backend_chain: list[str]
    vlm_allowed: bool
    reason: str
    gate_reasons: list[str]
    event_type: str | None
    normalized_event_type: str
    camera_id: str | None
    policy_sources: list[str]
    budget: dict[str, Any]
    degradation_policy: str
    skipped_backends: list[dict[str, Any]]
    event_policy: dict[str, Any]
    camera_policy: dict[str, Any]
    config_source: str = "global_defaults"
    camera_config_version: str | None = None
    camera_config_hash: str | None = None
    effective_policy_hash: str | None = None
    policy_errors: list[str] = field(default_factory=list)

    def trace_payload(self) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "requested_backend": self.requested_backend,
            "requested_backend_key": self.requested_backend_key,
            "effective_backend_request": self.effective_backend_request,
            "effective_backend_key": self.effective_backend_key,
            "allowed_backend_key": self.allowed_backend_key,
            "backend_chain": list(self.backend_chain),
            "vlm_allowed": self.vlm_allowed,
            "reason": self.reason,
            "gate_reasons": list(self.gate_reasons),
            "event_type": self.event_type,
            "normalized_event_type": self.normalized_event_type,
            "camera_id": self.camera_id,
            "policy_sources": list(self.policy_sources),
            "budget": deepcopy(self.budget),
            "degradation_policy": self.degradation_policy,
            "skipped_backends": deepcopy(self.skipped_backends),
            "event_policy": deepcopy(self.event_policy),
            "camera_policy": deepcopy(self.camera_policy),
            "config_source": self.config_source,
            "vlm_policy_source": self.config_source,
            "camera_config_version": self.camera_config_version,
            "camera_config_hash": self.camera_config_hash,
            "effective_policy_hash": self.effective_policy_hash,
            "camera_override_applied": self.config_source != "global_defaults",
            "policy_errors": list(self.policy_errors),
        }


def resolve_vlm_policy(
    *,
    requested_backend: str | None,
    event_type_hint: str | None,
    camera_id: str | None = None,
    camera_metadata: dict[str, Any] | None = None,
    face_detection: FaceDetectionResult | None = None,
) -> VlmPolicyDecision:
    requested_backend_value = (requested_backend or settings.semantic_descriptor_backend).strip() or "simple"
    requested_key = normalize_backend_key(requested_backend_value)
    normalized_event_type = _normalize_label(event_type_hint)
    effective_policy = _build_effective_policy(
        camera_id=camera_id,
        camera_metadata=camera_metadata or {},
    )
    policy_sources = list(effective_policy.pop("_policy_sources", ["global_defaults"]))
    policy_errors = list(effective_policy.pop("_policy_errors", []))
    camera_config_version = effective_policy.pop("_camera_config_version", None)
    camera_config_hash = effective_policy.pop("_camera_config_hash", None)
    explicit_camera_enable = bool(effective_policy.pop("_explicit_camera_enable", False))
    explicit_backend_override = bool(effective_policy.pop("_explicit_backend_override", False))
    config_source = policy_sources[-1] if policy_sources else "global_defaults"

    backend_override = _first_text(
        effective_policy.get("backend"),
        effective_policy.get("preferred_backend_override"),
        effective_policy.get("force_backend"),
    )
    if backend_override:
        effective_backend_request = backend_override
    elif requested_key == SIMPLE_BACKEND_KEY and explicit_camera_enable:
        effective_backend_request = AUTO_BACKEND_KEY
    else:
        effective_backend_request = requested_backend_value

    effective_key = normalize_backend_key(effective_backend_request)
    preferred_key = normalize_real_backend_key(
        effective_policy.get("preferred_backend") or settings.vlm_auto_preferred_backend
    )
    secondary_key = normalize_real_backend_key(
        effective_policy.get("secondary_backend") or settings.vlm_secondary_backend
    )
    if secondary_key == preferred_key:
        secondary_key = "smolvlm" if preferred_key == "qwen" else "qwen"

    degradation_policy = _normalize_degradation_policy(
        str(effective_policy.get("degradation_policy") or settings.vlm_degradation_policy)
    )
    budget = _build_budget(effective_policy)
    enabled_event_types = _normalize_list(
        effective_policy.get("enabled_event_types"),
        fallback=settings.vlm_event_type_policy,
    )
    disabled_event_types = _normalize_list(effective_policy.get("disabled_event_types"))
    event_policy = {
        "enabled_event_types": enabled_event_types,
        "disabled_event_types": disabled_event_types,
        "default_high_value_event_types": _default_high_value_event_types(),
    }

    camera_policy = {
        "enabled": bool(effective_policy.get("enabled", True)),
        "force_simple": bool(effective_policy.get("force_simple", False)),
        "backend": backend_override,
        "preferred_backend": preferred_key,
        "secondary_backend": secondary_key,
        "explicit_camera_enable": explicit_camera_enable,
        "explicit_backend_override": explicit_backend_override,
    }
    effective_policy_hash = stable_config_hash(
        {
            "camera_policy": camera_policy,
            "event_policy": event_policy,
            "budget": budget,
            "degradation_policy": degradation_policy,
            "policy_sources": policy_sources,
        }
    )

    gate_reasons: list[str] = []
    if not camera_policy["enabled"]:
        gate_reasons.append("vlm_camera_policy_disabled")
    if camera_policy["force_simple"] or effective_key == SIMPLE_BACKEND_KEY:
        gate_reasons.append("vlm_policy_simple_backend")
    event_allowed, event_reason = _event_allowed(
        normalized_event_type=normalized_event_type,
        enabled_event_types=enabled_event_types,
        disabled_event_types=disabled_event_types,
        face_detection=face_detection,
    )
    if not event_allowed:
        gate_reasons.append(event_reason)
    if budget["max_concurrent_inferences"] <= 0:
        gate_reasons.append("vlm_concurrency_disabled")

    if gate_reasons:
        return VlmPolicyDecision(
            requested_backend=requested_backend_value,
            requested_backend_key=requested_key,
            effective_backend_request=effective_backend_request,
            effective_backend_key=effective_key,
            allowed_backend_key=SIMPLE_BACKEND_KEY,
            backend_chain=[SIMPLE_BACKEND_KEY],
            vlm_allowed=False,
            reason=gate_reasons[0],
            gate_reasons=gate_reasons,
            event_type=event_type_hint,
            normalized_event_type=normalized_event_type,
            camera_id=camera_id,
            policy_sources=policy_sources,
            budget=budget,
            degradation_policy=degradation_policy,
            skipped_backends=[],
            event_policy=event_policy,
            camera_policy=camera_policy,
            config_source=config_source,
            camera_config_version=camera_config_version,
            camera_config_hash=camera_config_hash,
            effective_policy_hash=effective_policy_hash,
            policy_errors=policy_errors,
        )

    raw_chain = _build_backend_chain(
        requested_key=effective_key,
        preferred_key=preferred_key,
        secondary_key=secondary_key,
        degradation_policy=degradation_policy,
    )
    backend_chain, skipped_backends = _filter_degraded_backends(raw_chain)
    if not backend_chain:
        backend_chain = [SIMPLE_BACKEND_KEY]
    if settings.semantic_enable_fallback and SIMPLE_BACKEND_KEY not in backend_chain:
        backend_chain.append(SIMPLE_BACKEND_KEY)

    allowed_backend_key = next(
        (backend_key for backend_key in backend_chain if backend_key in NORMALIZED_REAL_BACKEND_KEYS),
        SIMPLE_BACKEND_KEY,
    )
    vlm_allowed = allowed_backend_key != SIMPLE_BACKEND_KEY
    reason = event_reason if vlm_allowed else "vlm_no_real_backend_available"

    return VlmPolicyDecision(
        requested_backend=requested_backend_value,
        requested_backend_key=requested_key,
        effective_backend_request=effective_backend_request,
        effective_backend_key=effective_key,
        allowed_backend_key=allowed_backend_key,
        backend_chain=backend_chain,
        vlm_allowed=vlm_allowed,
        reason=reason,
        gate_reasons=[] if vlm_allowed else [reason],
        event_type=event_type_hint,
        normalized_event_type=normalized_event_type,
        camera_id=camera_id,
        policy_sources=policy_sources,
        budget=budget,
        degradation_policy=degradation_policy,
        skipped_backends=skipped_backends,
        event_policy=event_policy,
        camera_policy=camera_policy,
        config_source=config_source,
        camera_config_version=camera_config_version,
        camera_config_hash=camera_config_hash,
        effective_policy_hash=effective_policy_hash,
        policy_errors=policy_errors,
    )


def evaluate_attempt_budget(
    *,
    backend_key: str,
    attempt: dict[str, Any],
    policy_decision: VlmPolicyDecision,
) -> tuple[bool, list[str], dict[str, Any]]:
    if backend_key not in NORMALIZED_REAL_BACKEND_KEYS:
        return True, [], {"status": "not_applicable", "reasons": []}

    budget = policy_decision.budget
    reasons: list[str] = []
    backend_budget = _resolve_backend_budget(budget, backend_key)
    observed_latency_seconds = _coerce_float(attempt.get("duration_ms")) / 1000.0
    max_latency_seconds = _coerce_float(backend_budget.get("max_allowed_latency_seconds"))
    if max_latency_seconds > 0 and observed_latency_seconds > max_latency_seconds:
        reasons.append("vlm_latency_budget_exceeded")

    observed_rss_mb = _max_observed_rss_mb(attempt)
    max_rss_mb = _coerce_float(backend_budget.get("max_allowed_rss_mb"))
    if max_rss_mb > 0 and observed_rss_mb is not None and observed_rss_mb > max_rss_mb:
        reasons.append("vlm_memory_budget_exceeded")

    trace = {
        "status": "ok" if not reasons else "exceeded",
        "reasons": reasons,
        "backend_key": backend_key,
        "budget_scope": backend_budget["budget_scope"],
        "rss_budget_source": backend_budget["rss_budget_source"],
        "observed_latency_seconds": round(observed_latency_seconds, 4),
        "max_allowed_latency_seconds": max_latency_seconds,
        "observed_rss_mb": observed_rss_mb,
        "max_allowed_rss_mb": max_rss_mb,
        "configured_global_max_allowed_rss_mb": backend_budget[
            "configured_global_max_allowed_rss_mb"
        ],
        "configured_backend_max_allowed_rss_mb": backend_budget[
            "configured_backend_max_allowed_rss_mb"
        ],
    }
    return not reasons, reasons, trace


def normalize_backend_key(requested_backend: str | None) -> str:
    normalized = (requested_backend or "").strip().lower()
    alias_map = {
        "auto": AUTO_BACKEND_KEY,
        "simple": SIMPLE_BACKEND_KEY,
        "simple_color_signature_v1": SIMPLE_BACKEND_KEY,
        "qwen": "qwen",
        "qwen_vl": "qwen",
        "qwen-vl": "qwen",
        "qwen/qwen2.5-vl-3b-instruct": "qwen",
        "qwen2.5-vl-3b-instruct": "qwen",
        "smolvlm": "smolvlm",
        "smol_vlm": "smolvlm",
        "smol-vlm": "smolvlm",
        "huggingfacetb/smolvlm2-2.2b-instruct": "smolvlm",
        "smolvlm2-2.2b-instruct": "smolvlm",
    }
    return alias_map.get(normalized, SIMPLE_BACKEND_KEY)


def normalize_real_backend_key(requested_backend: str | None) -> str:
    normalized = normalize_backend_key(requested_backend)
    if normalized in NORMALIZED_REAL_BACKEND_KEYS:
        return normalized
    return "qwen"


def _build_effective_policy(
    *,
    camera_id: str | None,
    camera_metadata: dict[str, Any],
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "enabled": True,
        "force_simple": False,
        "preferred_backend": settings.vlm_auto_preferred_backend,
        "secondary_backend": settings.vlm_secondary_backend,
        "enabled_event_types": settings.vlm_event_type_policy,
        "disabled_event_types": [],
        "degradation_policy": settings.vlm_degradation_policy,
        "max_allowed_latency_seconds": settings.vlm_max_allowed_latency_seconds,
        "max_allowed_rss_mb": settings.vlm_max_allowed_rss_mb,
        "backend_budgets": _default_backend_budgets(),
        "max_concurrent_inferences": settings.vlm_max_concurrent_inferences,
        "_policy_sources": ["global_defaults"],
        "_policy_errors": [],
        "_explicit_camera_enable": False,
        "_explicit_backend_override": False,
    }

    normalized_camera_id = (camera_id or "").strip().lower()
    if normalized_camera_id and normalized_camera_id in settings.vlm_disabled_camera_id_policy:
        policy["enabled"] = False
        policy["_policy_sources"].append("env_disabled_camera_ids")

    env_overrides = settings.vlm_camera_policy_overrides
    camera_override = _camera_override_for(env_overrides, normalized_camera_id)
    if camera_override:
        _apply_override(policy, camera_override, source="env_camera_policy_json")

    metadata_override = _metadata_policy_override(camera_metadata)
    if metadata_override:
        _apply_override(policy, metadata_override, source="camera_metadata")

    runtime_config = extract_camera_runtime_config(camera_metadata)
    if runtime_config.errors:
        policy["_policy_errors"].extend(f"{runtime_config.source_label}:{error}" for error in runtime_config.errors)
        logger.warning(
            "camera_runtime_config_invalid camera_id=%s source=%s errors=%s",
            camera_id or "<unknown>",
            runtime_config.config_source,
            ";".join(runtime_config.errors),
        )
    runtime_override = dict(runtime_config.vlm_policy)
    if runtime_config.recognition_enabled is not None and "enabled" not in runtime_override:
        runtime_override["enabled"] = runtime_config.recognition_enabled
    if runtime_override:
        _apply_override(
            policy,
            runtime_override,
            source=runtime_config.source_label,
            camera_config_version=runtime_config.camera_config_version,
            camera_config_hash=runtime_config.config_hash or runtime_config.effective_config_hash,
        )

    return policy


def _apply_override(
    policy: dict[str, Any],
    override: dict[str, Any],
    *,
    source: str,
    camera_config_version: str | None = None,
    camera_config_hash: str | None = None,
) -> None:
    normalized_override = _normalize_override_keys(override)
    if "enabled" in normalized_override:
        enabled = _coerce_bool(normalized_override["enabled"])
        if enabled is None:
            _record_policy_error(policy, source, "enabled_invalid")
        else:
            policy["enabled"] = enabled
            policy["_explicit_camera_enable"] = enabled
    if "vlm_enabled" in normalized_override:
        enabled = _coerce_bool(normalized_override["vlm_enabled"])
        if enabled is None:
            _record_policy_error(policy, source, "vlm_enabled_invalid")
        else:
            policy["enabled"] = enabled
            policy["_explicit_camera_enable"] = enabled
    if "force_simple" in normalized_override:
        force_simple = _coerce_bool(normalized_override["force_simple"])
        if force_simple is None:
            _record_policy_error(policy, source, "force_simple_invalid")
        else:
            policy["force_simple"] = force_simple
    if "backend" in normalized_override:
        policy["backend"] = str(normalized_override["backend"])
        policy["_explicit_backend_override"] = True
    if "force_backend" in normalized_override:
        policy["backend"] = str(normalized_override["force_backend"])
        policy["_explicit_backend_override"] = True
    if "preferred_backend" in normalized_override:
        policy["preferred_backend"] = str(normalized_override["preferred_backend"])
    if "secondary_backend" in normalized_override:
        policy["secondary_backend"] = str(normalized_override["secondary_backend"])
    if "enable_for_event_types" in normalized_override:
        policy["enabled_event_types"] = _normalize_list(normalized_override["enable_for_event_types"])
    if "enabled_event_types" in normalized_override:
        policy["enabled_event_types"] = _normalize_list(normalized_override["enabled_event_types"])
    if "disable_for_event_types" in normalized_override:
        policy["disabled_event_types"] = _normalize_list(normalized_override["disable_for_event_types"])
    if "disabled_event_types" in normalized_override:
        policy["disabled_event_types"] = _normalize_list(normalized_override["disabled_event_types"])
    if "max_latency_seconds" in normalized_override:
        _apply_float_override(policy, normalized_override, "max_latency_seconds", "max_allowed_latency_seconds", source)
    if "max_allowed_latency_seconds" in normalized_override:
        _apply_float_override(policy, normalized_override, "max_allowed_latency_seconds", "max_allowed_latency_seconds", source)
    if "max_rss_mb" in normalized_override:
        _apply_global_rss_override(policy, normalized_override, "max_rss_mb", source)
    if "max_allowed_rss_mb" in normalized_override:
        _apply_global_rss_override(policy, normalized_override, "max_allowed_rss_mb", source)
    if "qwen_max_allowed_rss_mb" in normalized_override:
        _apply_backend_rss_override(
            policy,
            backend_key="qwen",
            value=normalized_override["qwen_max_allowed_rss_mb"],
            source=source,
        )
    if "smolvlm_max_allowed_rss_mb" in normalized_override:
        _apply_backend_rss_override(
            policy,
            backend_key="smolvlm",
            value=normalized_override["smolvlm_max_allowed_rss_mb"],
            source=source,
        )
    if "backend_budgets" in normalized_override:
        _apply_backend_budget_overrides(
            policy,
            normalized_override["backend_budgets"],
            source=source,
        )
    if "max_concurrent_inferences" in normalized_override:
        value = _coerce_optional_float(normalized_override["max_concurrent_inferences"])
        if value is None or value < 0:
            _record_policy_error(policy, source, "max_concurrent_inferences_invalid")
        else:
            policy["max_concurrent_inferences"] = int(value)
    if "degradation_policy" in normalized_override:
        policy["degradation_policy"] = str(normalized_override["degradation_policy"])
    if camera_config_version:
        policy["_camera_config_version"] = camera_config_version
    if camera_config_hash:
        policy["_camera_config_hash"] = camera_config_hash
    policy["_policy_sources"].append(source)


def _camera_override_for(overrides: dict[str, Any], normalized_camera_id: str) -> dict[str, Any]:
    if not normalized_camera_id:
        return {}
    raw_override = overrides.get(normalized_camera_id)
    if raw_override is None:
        raw_override = overrides.get(normalized_camera_id.upper())
    if raw_override is None:
        for raw_camera_id, candidate in overrides.items():
            if str(raw_camera_id).strip().lower() == normalized_camera_id:
                raw_override = candidate
                break
    return raw_override if isinstance(raw_override, dict) else {}


def _metadata_policy_override(camera_metadata: dict[str, Any]) -> dict[str, Any]:
    recognition = camera_metadata.get("recognition")
    if isinstance(recognition, dict):
        for key in ["vlm_policy", "semantic_vlm_policy", "semantic_descriptor_policy"]:
            value = recognition.get(key)
            if isinstance(value, dict):
                return value

    for key in ["vlm_policy", "semantic_vlm_policy", "semantic_descriptor_policy"]:
        value = camera_metadata.get(key)
        if isinstance(value, dict):
            return value

    flat_keys = {
        "vlm_enabled",
        "semantic_vlm_enabled",
        "vlm_force_simple",
        "vlm_backend",
        "semantic_backend",
        "semantic_descriptor_backend",
        "vlm_preferred_backend",
        "vlm_secondary_backend",
        "vlm_enable_for_event_types",
        "vlm_disable_for_event_types",
        "vlm_max_allowed_latency_seconds",
        "vlm_max_allowed_rss_mb",
        "qwen_max_allowed_rss_mb",
        "vlm_qwen_max_allowed_rss_mb",
        "qwen_max_rss_mb",
        "smolvlm_max_allowed_rss_mb",
        "vlm_smolvlm_max_allowed_rss_mb",
        "smolvlm_max_rss_mb",
        "vlm_backend_budgets",
        "backend_budgets",
        "vlm_max_concurrent_inferences",
        "vlm_degradation_policy",
    }
    flat_override = {key: camera_metadata[key] for key in flat_keys if key in camera_metadata}
    return flat_override


def _normalize_override_keys(override: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    aliases = {
        "vlm_backend": "backend",
        "vlm_force_simple": "force_simple",
        "semantic_vlm_enabled": "enabled",
        "semantic_backend": "backend",
        "semantic_descriptor_backend": "backend",
        "vlm_preferred_backend": "preferred_backend",
        "vlm_secondary_backend": "secondary_backend",
        "vlm_enable_for_event_types": "enable_for_event_types",
        "eligible_events": "enable_for_event_types",
        "enabled_event_types": "enable_for_event_types",
        "vlm_disable_for_event_types": "disable_for_event_types",
        "disabled_event_types": "disable_for_event_types",
        "vlm_max_allowed_latency_seconds": "max_allowed_latency_seconds",
        "latency_budget_seconds": "max_latency_seconds",
        "vlm_max_allowed_rss_mb": "max_allowed_rss_mb",
        "memory_budget_mb": "max_rss_mb",
        "max_memory_mb": "max_rss_mb",
        "qwen_max_rss_mb": "qwen_max_allowed_rss_mb",
        "vlm_qwen_max_rss_mb": "qwen_max_allowed_rss_mb",
        "vlm_qwen_max_allowed_rss_mb": "qwen_max_allowed_rss_mb",
        "smolvlm_max_rss_mb": "smolvlm_max_allowed_rss_mb",
        "vlm_smolvlm_max_rss_mb": "smolvlm_max_allowed_rss_mb",
        "vlm_smolvlm_max_allowed_rss_mb": "smolvlm_max_allowed_rss_mb",
        "vlm_backend_budgets": "backend_budgets",
        "per_backend_budgets": "backend_budgets",
        "per_backend_budget": "backend_budgets",
        "vlm_max_concurrent_inferences": "max_concurrent_inferences",
        "max_concurrency": "max_concurrent_inferences",
        "max_camera_concurrency": "max_concurrent_inferences",
        "vlm_degradation_policy": "degradation_policy",
    }
    for key, value in override.items():
        normalized_key = aliases.get(str(key).strip().lower(), str(key).strip().lower())
        normalized[normalized_key] = value
    return normalized


def _apply_float_override(
    policy: dict[str, Any],
    normalized_override: dict[str, Any],
    source_key: str,
    target_key: str,
    source: str,
) -> None:
    value = _coerce_optional_float(normalized_override[source_key])
    if value is None or value < 0:
        _record_policy_error(policy, source, f"{target_key}_invalid")
        return
    policy[target_key] = value


def _apply_global_rss_override(
    policy: dict[str, Any],
    normalized_override: dict[str, Any],
    source_key: str,
    source: str,
) -> None:
    value = _coerce_optional_float(normalized_override[source_key])
    if value is None or value < 0:
        _record_policy_error(policy, source, "max_allowed_rss_mb_invalid")
        return
    policy["max_allowed_rss_mb"] = value


def _apply_backend_rss_override(
    policy: dict[str, Any],
    *,
    backend_key: str,
    value: Any,
    source: str,
) -> None:
    normalized_backend_key = normalize_backend_key(backend_key)
    if normalized_backend_key not in NORMALIZED_REAL_BACKEND_KEYS:
        _record_policy_error(policy, source, f"backend_budget_unknown_backend:{backend_key}")
        return
    parsed_value = _coerce_optional_float(value)
    if parsed_value is None or parsed_value < 0:
        _record_policy_error(policy, source, f"{normalized_backend_key}_max_allowed_rss_mb_invalid")
        return
    policy.setdefault("backend_budgets", _default_backend_budgets()).setdefault(
        normalized_backend_key,
        {},
    )["max_allowed_rss_mb"] = parsed_value


def _apply_backend_budget_overrides(
    policy: dict[str, Any],
    value: Any,
    *,
    source: str,
) -> None:
    if not isinstance(value, dict):
        _record_policy_error(policy, source, "backend_budgets_invalid")
        return
    for raw_backend_key, raw_budget in value.items():
        backend_key = normalize_backend_key(str(raw_backend_key))
        if backend_key not in NORMALIZED_REAL_BACKEND_KEYS:
            _record_policy_error(policy, source, f"backend_budget_unknown_backend:{raw_backend_key}")
            continue
        if isinstance(raw_budget, dict):
            normalized_budget = _normalize_override_keys(raw_budget)
            budget_value = (
                normalized_budget.get("max_allowed_rss_mb")
                if "max_allowed_rss_mb" in normalized_budget
                else normalized_budget.get("max_rss_mb")
            )
        else:
            budget_value = raw_budget
        _apply_backend_rss_override(
            policy,
            backend_key=backend_key,
            value=budget_value,
            source=source,
        )


def _record_policy_error(policy: dict[str, Any], source: str, error: str) -> None:
    policy.setdefault("_policy_errors", []).append(f"{source}:{error}")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _event_allowed(
    *,
    normalized_event_type: str,
    enabled_event_types: list[str],
    disabled_event_types: list[str],
    face_detection: FaceDetectionResult | None,
) -> tuple[bool, str]:
    if normalized_event_type and normalized_event_type in disabled_event_types:
        return False, f"vlm_event_type_disabled:{normalized_event_type}"
    if "*" in enabled_event_types or "all" in enabled_event_types:
        return True, "vlm_event_policy_all"
    if not enabled_event_types:
        return False, "vlm_event_policy_empty"
    if not normalized_event_type:
        return True, "vlm_event_policy_no_event_type_hint"
    if normalized_event_type in enabled_event_types:
        return True, f"vlm_event_type_enabled:{normalized_event_type}"
    if "person_detected" in enabled_event_types and face_detection is not None:
        return True, "vlm_event_policy_person_detected"
    if "face_detected" in enabled_event_types and face_detection and face_detection.detected:
        return True, "vlm_event_policy_face_detected"
    if "face_usable" in enabled_event_types and face_detection and face_detection.usable:
        return True, "vlm_event_policy_face_usable"
    if "no_face" in enabled_event_types and face_detection and not face_detection.usable:
        return True, "vlm_event_policy_no_face"
    return False, f"vlm_event_type_not_enabled:{normalized_event_type}"


def _build_backend_chain(
    *,
    requested_key: str,
    preferred_key: str,
    secondary_key: str,
    degradation_policy: str,
) -> list[str]:
    if requested_key == SIMPLE_BACKEND_KEY:
        return [SIMPLE_BACKEND_KEY]
    if degradation_policy == "simple_only":
        return [SIMPLE_BACKEND_KEY]

    if requested_key == AUTO_BACKEND_KEY:
        raw_chain = [preferred_key]
        if degradation_policy in {
            "auto_then_secondary_then_simple",
            "preferred_then_secondary_then_simple",
        }:
            raw_chain.append(secondary_key)
        if settings.semantic_enable_fallback:
            raw_chain.append(SIMPLE_BACKEND_KEY)
    elif requested_key in NORMALIZED_REAL_BACKEND_KEYS:
        raw_chain = [requested_key]
        if degradation_policy == "preferred_then_secondary_then_simple":
            raw_chain.append(secondary_key if secondary_key != requested_key else preferred_key)
        if settings.semantic_enable_fallback:
            raw_chain.append(SIMPLE_BACKEND_KEY)
    else:
        raw_chain = [SIMPLE_BACKEND_KEY]

    chain: list[str] = []
    for backend_key in raw_chain:
        if backend_key and backend_key not in chain:
            chain.append(backend_key)
    return chain


def _filter_degraded_backends(raw_chain: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    chain: list[str] = []
    skipped: list[dict[str, Any]] = []
    for backend_key in raw_chain:
        if backend_key not in NORMALIZED_REAL_BACKEND_KEYS:
            chain.append(backend_key)
            continue
        available, reason, state = vlm_degradation_state.is_backend_available(backend_key)
        if available:
            chain.append(backend_key)
            continue
        skipped.append(
            {
                "backend_key": backend_key,
                "status": "skipped",
                "reason": reason,
                "health": state,
            }
        )
    return chain, skipped


def _build_budget(policy: dict[str, Any]) -> dict[str, Any]:
    backend_budgets = _coerce_backend_budgets(policy.get("backend_budgets"))
    return {
        "max_allowed_latency_seconds": _coerce_float(
            policy.get("max_allowed_latency_seconds", settings.vlm_max_allowed_latency_seconds)
        ),
        "max_allowed_rss_mb": _coerce_float(
            policy.get("max_allowed_rss_mb", settings.vlm_max_allowed_rss_mb)
        ),
        "backend_budgets": backend_budgets,
        "max_concurrent_inferences": int(
            _coerce_float(
                policy.get("max_concurrent_inferences", settings.vlm_max_concurrent_inferences)
            )
        ),
        "concurrency_acquire_timeout_seconds": settings.vlm_concurrency_acquire_timeout_seconds,
        "timeout_seconds": settings.effective_vlm_timeout_seconds,
    }


def _default_backend_budgets() -> dict[str, dict[str, float]]:
    return {
        "qwen": {
            "max_allowed_rss_mb": _coerce_float(settings.qwen_max_allowed_rss_mb),
        },
        "smolvlm": {
            "max_allowed_rss_mb": _coerce_float(settings.smolvlm_max_allowed_rss_mb),
        },
    }


def _coerce_backend_budgets(value: Any) -> dict[str, dict[str, float]]:
    defaults = _default_backend_budgets()
    if not isinstance(value, dict):
        return defaults
    result = defaults
    for raw_backend_key, raw_budget in value.items():
        backend_key = normalize_backend_key(str(raw_backend_key))
        if backend_key not in NORMALIZED_REAL_BACKEND_KEYS:
            continue
        if isinstance(raw_budget, dict):
            max_rss = _coerce_float(raw_budget.get("max_allowed_rss_mb"))
        else:
            max_rss = _coerce_float(raw_budget)
        result.setdefault(backend_key, {})["max_allowed_rss_mb"] = max_rss
    return result


def _resolve_backend_budget(budget: dict[str, Any], backend_key: str) -> dict[str, Any]:
    global_max_rss = _coerce_float(budget.get("max_allowed_rss_mb"))
    backend_budgets = budget.get("backend_budgets")
    backend_budget = {}
    if isinstance(backend_budgets, dict):
        backend_budget = backend_budgets.get(backend_key) or {}
    backend_max_rss = _coerce_float(backend_budget.get("max_allowed_rss_mb"))
    if backend_max_rss > 0:
        max_rss = backend_max_rss
        rss_budget_source = f"{backend_key}_max_allowed_rss_mb"
        budget_scope = "backend"
    else:
        max_rss = global_max_rss
        rss_budget_source = "vlm_max_allowed_rss_mb"
        budget_scope = "global"

    return {
        "max_allowed_latency_seconds": _coerce_float(
            budget.get("max_allowed_latency_seconds")
        ),
        "max_allowed_rss_mb": max_rss,
        "configured_global_max_allowed_rss_mb": global_max_rss,
        "configured_backend_max_allowed_rss_mb": backend_max_rss,
        "rss_budget_source": rss_budget_source,
        "budget_scope": budget_scope,
    }


def _normalize_degradation_policy(value: str) -> str:
    normalized = _normalize_label(value, default="auto_then_secondary_then_simple")
    allowed = {
        "auto_then_secondary_then_simple",
        "preferred_then_secondary_then_simple",
        "preferred_then_simple",
        "simple_only",
    }
    if normalized in allowed:
        return normalized
    return "auto_then_secondary_then_simple"


def _max_observed_rss_mb(attempt: dict[str, Any]) -> float | None:
    rss_values = [
        _coerce_optional_float(value)
        for key, value in attempt.items()
        if key.endswith("_rss_mb") or key.endswith("_max_rss_mb")
    ]
    rss_values = [value for value in rss_values if value is not None]
    if not rss_values:
        return None
    return round(max(rss_values), 2)


def _normalize_list(value: Any, *, fallback: list[str] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        raw_values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    return [
        _normalize_label(raw_value)
        for raw_value in raw_values
        if _normalize_label(raw_value)
    ]


def _default_high_value_event_types() -> list[str]:
    return [
        "manual_review_required",
        "identity_conflict",
        "recurrent_unresolved_subject",
        "case_suggestion_created",
    ]


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_label(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_").replace(".", "_")
    return normalized or default


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
