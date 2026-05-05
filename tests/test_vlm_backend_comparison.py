from __future__ import annotations

from app.domain.entities import SemanticDescriptorResult
from app.services.vlm_metrics_service import (
    compare_vlm_trace_summaries,
    extract_vlm_trace_summary,
)


def test_vlm_backend_comparison_prefers_successful_lower_latency_backend() -> None:
    qwen = _result(
        backend="Qwen/Qwen2.5-VL-3B-Instruct",
        backend_key="qwen",
        duration_ms=8400,
        fallback_used=False,
    )
    smolvlm = _result(
        backend="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        backend_key="smolvlm",
        duration_ms=3100,
        fallback_used=False,
    )

    comparison = compare_vlm_trace_summaries(
        [extract_vlm_trace_summary(qwen), extract_vlm_trace_summary(smolvlm)]
    )

    assert comparison["recommended_backend"] == "smolvlm"
    assert comparison["reason"] == "lowest_successful_latency_without_fallback"


def test_vlm_backend_comparison_recommends_simple_when_real_backends_fallback() -> None:
    qwen = _result(
        backend="simple_color_signature_v1",
        backend_key="simple",
        duration_ms=200,
        fallback_used=True,
        requested_backend="qwen",
        error="backend_timeout",
    )
    smolvlm = _result(
        backend="simple_color_signature_v1",
        backend_key="simple",
        duration_ms=180,
        fallback_used=True,
        requested_backend="smolvlm",
        error="model_load_failed:RuntimeError",
    )

    comparison = compare_vlm_trace_summaries(
        [extract_vlm_trace_summary(qwen), extract_vlm_trace_summary(smolvlm)]
    )

    assert comparison["recommended_backend"] == "simple"
    assert comparison["reason"] == "no_real_vlm_backend_completed_without_fallback"


def _result(
    *,
    backend: str,
    backend_key: str,
    duration_ms: int,
    fallback_used: bool,
    requested_backend: str | None = None,
    error: str | None = None,
) -> SemanticDescriptorResult:
    requested = requested_backend or backend_key
    return SemanticDescriptorResult(
        generated=True,
        backend=backend,
        confidence=0.7,
        source_frame_ref="s3://vigilante-frames/vlm-validation/frame.jpg",
        descriptor={
            "descriptor_schema_version": "semantic_descriptor_v2",
            "descriptor_backend": backend,
            "semantic_backend_requested": requested,
            "semantic_backend_selected": backend,
            "semantic_backend_fallback_used": fallback_used,
            "semantic_backend_error": error,
            "raw_summary": "person with neutral clothing",
            "semantic_backend_trace": {
                "semantic_backend_requested": requested,
                "semantic_backend_selected": backend,
                "semantic_backend_selected_key": backend_key,
                "semantic_backend_fallback_used": fallback_used,
                "semantic_backend_error": error,
                "total_duration_ms": duration_ms,
                "attempts": [
                    {
                        "backend_key": backend_key,
                        "backend_name": backend,
                        "status": "success",
                        "duration_ms": duration_ms,
                        "timeout_seconds": 60,
                        "device": "mps",
                        "descriptor_output_chars": 420,
                    }
                ],
            },
        },
        signature={"dominant_palette": ["gray"]},
    )
