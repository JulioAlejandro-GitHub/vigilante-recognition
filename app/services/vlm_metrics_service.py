from __future__ import annotations

import json
import platform
from typing import Any

from app.domain.entities import SemanticDescriptorResult

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None


METRICS_VERSION = "vlm_attempt_metrics_v1"


def current_memory_snapshot(*, prefix: str = "process") -> dict[str, float]:
    snapshot: dict[str, float] = {}

    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        snapshot[f"{prefix}_rss_mb"] = _mb(memory.rss)
        snapshot[f"{prefix}_vms_mb"] = _mb(memory.vms)
    except Exception:
        pass

    try:
        if resource is None:
            raise RuntimeError("resource_unavailable")
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system().lower() == "darwin":
            max_rss_mb = max_rss / (1024 * 1024)
        else:
            max_rss_mb = max_rss / 1024
        snapshot[f"{prefix}_max_rss_mb"] = round(max_rss_mb, 2)
    except Exception:
        pass

    return snapshot


def payload_size_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def build_success_attempt_metrics(
    *,
    descriptor: dict[str, Any],
    signature: dict[str, Any],
    raw_summary: str | None,
    duration_ms: int,
) -> dict[str, Any]:
    return {
        "metrics_version": METRICS_VERSION,
        "duration_ms": duration_ms,
        "inference_duration_ms": duration_ms,
        "descriptor_valid": bool(descriptor.get("descriptor_schema_version") and signature),
        "descriptor_output_chars": payload_size_chars(descriptor),
        "signature_output_chars": payload_size_chars(signature),
        "raw_summary_chars": len(raw_summary or ""),
        **current_memory_snapshot(prefix="worker_after_attempt"),
    }


def extract_vlm_trace_summary(result: SemanticDescriptorResult) -> dict[str, Any]:
    descriptor = result.descriptor or {}
    trace = descriptor.get("semantic_backend_trace") or descriptor.get("generation_trace") or {}
    attempts = trace.get("attempts") if isinstance(trace, dict) else []
    attempts = attempts if isinstance(attempts, list) else []
    success_attempt = next(
        (attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("status") == "success"),
        {},
    )
    return {
        "backend": result.backend,
        "generated": result.generated,
        "source_frame_ref": result.source_frame_ref,
        "requested_backend": descriptor.get("semantic_backend_requested")
        or trace.get("semantic_backend_requested"),
        "selected_backend": descriptor.get("semantic_backend_selected")
        or trace.get("semantic_backend_selected")
        or result.backend,
        "selected_backend_key": trace.get("semantic_backend_selected_key")
        or trace.get("selected_backend_key")
        or success_attempt.get("backend_key"),
        "fallback_used": bool(
            descriptor.get("semantic_backend_fallback_used")
            if "semantic_backend_fallback_used" in descriptor
            else trace.get("semantic_backend_fallback_used")
        ),
        "error": descriptor.get("semantic_backend_error") or trace.get("semantic_backend_error"),
        "duration_ms": trace.get("total_duration_ms") or success_attempt.get("duration_ms"),
        "timeout_seconds": trace.get("timeout_seconds")
        or success_attempt.get("timeout_seconds")
        or success_attempt.get("timeout_applied_seconds"),
        "device": success_attempt.get("device"),
        "requested_device": success_attempt.get("requested_device") or trace.get("requested_device"),
        "image_original_size": success_attempt.get("image_original_size"),
        "image_inference_size": success_attempt.get("image_inference_size"),
        "image_resized": success_attempt.get("image_resized"),
        "descriptor_output_chars": success_attempt.get("descriptor_output_chars")
        or payload_size_chars(descriptor),
        "raw_output_chars": success_attempt.get("raw_output_chars"),
        "descriptor_valid": bool(result.generated and descriptor.get("descriptor_schema_version")),
        "raw_summary": descriptor.get("raw_summary"),
    }


def compare_vlm_trace_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    real_candidates = [
        summary
        for summary in summaries
        if summary.get("generated")
        and not summary.get("fallback_used")
        and summary.get("selected_backend_key") in {"qwen", "smolvlm"}
    ]
    if not real_candidates:
        return {
            "recommended_backend": "simple",
            "reason": "no_real_vlm_backend_completed_without_fallback",
            "summaries": summaries,
        }

    def sort_key(summary: dict[str, Any]) -> tuple[float, int]:
        duration = summary.get("duration_ms")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = float("inf")
        backend_bias = 0 if summary.get("selected_backend_key") == "smolvlm" else 1
        return duration_value, backend_bias

    selected = sorted(real_candidates, key=sort_key)[0]
    return {
        "recommended_backend": selected.get("selected_backend_key") or selected.get("backend"),
        "reason": "lowest_successful_latency_without_fallback",
        "summaries": summaries,
    }


def _mb(value: float) -> float:
    return round(float(value) / (1024 * 1024), 2)
