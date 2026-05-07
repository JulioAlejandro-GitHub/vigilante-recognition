from __future__ import annotations

from app.services.runtime_metrics_summary_service import RuntimeMetricsSummaryService


def test_runtime_metrics_summary_by_camera_rates_and_latencies():
    summary = RuntimeMetricsSummaryService().summarize(
        [
            _record(
                camera_id="cam-a",
                face_detected=True,
                face_usable=True,
                face_latency=10,
                semantic_selected_key="qwen",
                semantic_valid=True,
                vlm_attempted=True,
                semantic_duration=100,
            ),
            _record(
                camera_id="cam-a",
                event_type="human_presence_no_face",
                face_detected=True,
                face_usable=False,
                face_latency=30,
                semantic_selected_key="simple",
                semantic_valid=True,
                semantic_fallback=True,
                semantic_duration=20,
            ),
            _record(
                camera_id="cam-b",
                event_type="human_presence_no_face",
                face_detected=False,
                face_usable=False,
                semantic_selected_key=None,
                semantic_valid=False,
            ),
        ]
    )

    cam_a = summary["by_camera"]["cam-a"]
    assert cam_a["processed_events"] == 2
    assert cam_a["face_detected_rate"] == 1.0
    assert cam_a["face_usable_rate"] == 0.5
    assert cam_a["low_quality_face_rate"] == 0.5
    assert cam_a["semantic_backend_most_used"] == "qwen"
    assert cam_a["fallback_to_simple_rate"] == 0.5
    assert cam_a["face_detect_latency_ms"]["p50"] == 20.0
    assert cam_a["vlm_duration_ms"]["p95"] == 96.0


def test_runtime_metrics_summary_by_face_backend():
    summary = RuntimeMetricsSummaryService().summarize(
        [
            _record(face_selected="insightface", face_requested="insightface"),
            _record(face_selected="simple", face_requested="auto", face_fallback=True),
            _record(face_selected="simple", face_requested="simple", face_usable=False),
        ]
    )

    insightface = summary["by_face_backend"]["insightface"]
    simple = summary["by_face_backend"]["simple"]
    auto = summary["by_face_backend"]["auto"]
    assert insightface["selected_count"] == 1
    assert insightface["usable_rate"] == 1.0
    assert simple["selected_count"] == 2
    assert simple["fallback_count"] == 1
    assert simple["fallback_rate"] == 0.5
    assert auto["requested_count"] == 1
    assert auto["fallback_away_count"] == 1


def test_runtime_metrics_summary_by_semantic_backend_counts_parser_and_budget():
    summary = RuntimeMetricsSummaryService().summarize(
        [
            _record(
                semantic_selected_key="qwen",
                semantic_valid=True,
                vlm_attempted=True,
                json_recovered=True,
                parser_strategy="fenced_json",
                budget_backend="qwen",
                observed_rss_mb=1200,
            ),
            _record(
                semantic_selected_key="simple",
                semantic_valid=True,
                semantic_fallback=True,
                budget_status="rejected",
                budget_reason="vlm_memory_budget_exceeded",
                budget_backend="smolvlm",
                observed_rss_mb=9999,
                max_allowed_rss_mb=1024,
            ),
            _record(
                semantic_selected_key="qwen",
                semantic_valid=False,
                parser_error="vlm_output_invalid_json",
                parser_backend_key="qwen",
                budget_backend="qwen",
            ),
        ]
    )

    qwen = summary["by_semantic_backend"]["qwen"]
    smolvlm = summary["by_semantic_backend"]["smolvlm"]
    simple = summary["by_semantic_backend"]["simple"]
    assert qwen["selected_count"] == 2
    assert qwen["success_count"] == 1
    assert qwen["parser_recovered_count"] == 1
    assert qwen["invalid_json_count"] == 1
    assert qwen["observed_rss_mb"]["max"] == 1200.0
    assert smolvlm["budget_rejected_count"] == 1
    assert smolvlm["observed_rss_mb"]["max"] == 9999.0
    assert simple["fallback_count"] == 1


def test_runtime_metrics_summary_by_event_type_shows_operational_differences():
    summary = RuntimeMetricsSummaryService().summarize(
        [
            _record(
                event_type="face_detected_unidentified",
                semantic_selected_key="qwen",
                semantic_valid=True,
                vlm_attempted=True,
            ),
            _record(
                event_type="human_presence_no_face",
                semantic_selected_key="simple",
                semantic_valid=True,
                semantic_fallback=True,
                budget_status="rejected",
                budget_reason="vlm_memory_budget_exceeded",
                budget_backend="smolvlm",
            ),
        ]
    )

    unidentified = summary["by_event_type"]["face_detected_unidentified"]
    no_face = summary["by_event_type"]["human_presence_no_face"]
    assert unidentified["vlm_attempted_count"] == 1
    assert unidentified["vlm_success_rate"] == 1.0
    assert no_face["fallback_to_simple_count"] == 1
    assert no_face["budget_rejected_count"] == 1


def _record(
    *,
    camera_id: str = "cam-a",
    event_type: str = "face_detected_unidentified",
    face_selected: str = "insightface",
    face_requested: str = "auto",
    face_fallback: bool = False,
    face_detected: bool = True,
    face_usable: bool = True,
    face_latency: float = 12,
    semantic_selected_key: str | None = "qwen",
    semantic_valid: bool = True,
    semantic_fallback: bool = False,
    semantic_duration: float | None = 100,
    vlm_attempted: bool = False,
    parser_strategy: str | None = "direct_json",
    json_recovered: bool = False,
    parser_error: str | None = None,
    parser_backend_key: str | None = None,
    budget_status: str = "ok",
    budget_reason: str | None = None,
    budget_backend: str | None = None,
    observed_rss_mb: float | None = None,
    max_allowed_rss_mb: float | None = 4096,
) -> dict:
    return {
        "schema_version": "runtime_metrics_event_v1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "camera_id": camera_id,
        "event_type": event_type,
        "event_id": f"evt-{camera_id}-{event_type}",
        "face": {
            "requested": face_requested,
            "selected": face_selected,
            "fallback_used": face_fallback,
            "detect_elapsed_ms": face_latency,
            "detected": face_detected,
            "usable": face_usable,
            "low_quality": bool(face_detected and not face_usable),
            "quality_score": 0.9 if face_usable else 0.2,
        },
        "semantic": {
            "requested": "auto",
            "effective_request": semantic_selected_key,
            "selected": semantic_selected_key,
            "selected_key": semantic_selected_key,
            "fallback_used": semantic_fallback,
            "total_duration_ms": semantic_duration,
            "descriptor_valid": semantic_valid,
            "success": semantic_valid,
            "parser_strategy": parser_strategy,
            "parser_backend_key": parser_backend_key or semantic_selected_key,
            "json_recovered": json_recovered,
            "parser_error": parser_error,
            "attempts_count": 1 if semantic_selected_key else 0,
            "attempted_backend_keys": [semantic_selected_key] if semantic_selected_key else [],
            "real_backend_attempted": vlm_attempted,
        },
        "budget": {
            "status": budget_status,
            "backend_key": budget_backend or semantic_selected_key,
            "observed_rss_mb": observed_rss_mb,
            "max_allowed_rss_mb": max_allowed_rss_mb,
            "rejection_reason": budget_reason,
            "rejection_reasons": [budget_reason] if budget_reason else [],
        },
        "config": {"source": "global"},
    }
