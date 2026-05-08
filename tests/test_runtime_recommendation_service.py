from __future__ import annotations

import os
import subprocess
import sys

from app.services.runtime_metrics_store import JsonlRuntimeMetricsStore
from app.services.runtime_recommendation_service import RuntimeRecommendationService
from app.services.runtime_recommendation_store import JsonlRuntimeRecommendationStore


def test_runtime_recommendation_service_generates_per_camera_recommendations(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    for index in range(6):
        metrics_store.append(
            _record(
                camera_id="cam-face",
                event_index=index,
                face_usable=index < 2,
            )
        )
    for index in range(6):
        metrics_store.append(
            _record(
                camera_id="cam-healthy",
                event_index=index,
                face_usable=True,
                semantic_selected_key=None,
            )
        )

    service = _service(metrics_store=metrics_store, tmp_path=tmp_path)
    result = service.generate()

    assert result["camera_count"] == 2
    assert _has_recommendation(result, "cam-face", "face_tuning")
    healthy = _recommendations_for(result, "cam-healthy")[0]
    assert healthy["suggested_value"] == "keep_current_configuration"
    assert healthy["auto_apply"] is False


def test_runtime_recommendation_service_marks_insufficient_evidence(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    metrics_store.append(_record(camera_id="cam-low", event_index=0))

    result = _service(metrics_store=metrics_store, tmp_path=tmp_path).generate()

    [recommendation] = result["recommendations"]
    assert recommendation["camera_id"] == "cam-low"
    assert recommendation["suggested_value"] == "insufficient_evidence"


def test_runtime_recommendation_service_generates_vlm_fallback_policy(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    for index in range(6):
        metrics_store.append(
            _record(
                camera_id="cam-vlm",
                event_index=index,
                event_type="human_presence_no_face",
                semantic_selected_key="simple",
                semantic_fallback=True,
                attempted_backend_keys=["qwen", "smolvlm", "simple"],
                real_backend_attempted=True,
            )
        )

    result = _service(metrics_store=metrics_store, tmp_path=tmp_path).generate()

    assert _has_recommendation(result, "cam-vlm", "event_policy")


def test_runtime_recommendation_service_generates_smolvlm_budget_recommendation(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    for index in range(6):
        metrics_store.append(
            _record(
                camera_id="cam-budget",
                event_index=index,
                semantic_selected_key="simple",
                semantic_fallback=True,
                attempted_backend_keys=["smolvlm", "simple"],
                real_backend_attempted=True,
                budget_status="rejected",
                budget_backend="smolvlm",
                observed_rss_mb=20000,
                max_allowed_rss_mb=10240,
            )
        )

    result = _service(metrics_store=metrics_store, tmp_path=tmp_path).generate()

    budget = next(
        recommendation
        for recommendation in result["recommendations"]
        if recommendation["recommendation_type"] == "budget"
    )
    assert budget["camera_id"] == "cam-budget"
    assert budget["suggested_value"]["disabled_backend_candidate"] == "smolvlm"


def test_runtime_recommendation_service_persists_recommendations(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    for index in range(6):
        metrics_store.append(
            _record(camera_id="cam-face", event_index=index, face_usable=False)
        )
    recommendation_store = JsonlRuntimeRecommendationStore(
        tmp_path / "recommendations.jsonl"
    )
    service = _service(
        metrics_store=metrics_store,
        recommendation_store=recommendation_store,
        tmp_path=tmp_path,
    )

    result = service.generate(persist=True)
    persisted = list(recommendation_store.iter_records())

    assert result["persisted_count"] == len(result["recommendations"])
    assert persisted
    assert persisted[0]["camera_id"] == "cam-face"
    assert persisted[0]["status"] == "pending"
    assert persisted[0]["suggested_value"]


def test_runtime_recommendation_cli_outputs_readable_summary(tmp_path) -> None:
    metrics_store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    for index in range(6):
        metrics_store.append(
            _record(camera_id="cam-cli", event_index=index, face_usable=False)
        )

    repo_root = os.getcwd()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/show_runtime_recommendations.py",
            "--metrics-path",
            str(tmp_path / "events.jsonl"),
            "--recommendations-path",
            str(tmp_path / "recommendations.jsonl"),
            "--min-events-per-camera",
            "5",
            "--window-hours",
            "0",
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": "."},
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Runtime recommendations:" in completed.stdout
    assert "Camera cam-cli" in completed.stdout
    assert "[pending|" in completed.stdout
    assert "Ajustar face_quality_threshold" in completed.stdout
    assert "suggested:" in completed.stdout


def _service(
    *,
    metrics_store,
    tmp_path,
    recommendation_store=None,
) -> RuntimeRecommendationService:
    return RuntimeRecommendationService(
        enabled=True,
        metrics_store=metrics_store,
        recommendation_store=recommendation_store
        or JsonlRuntimeRecommendationStore(tmp_path / "recommendations.jsonl"),
        min_events_per_camera=5,
        window_hours=0,
        default_face_tuning={
            "face_backend": "auto",
            "det_size": "640,640",
            "detection_threshold": 0.5,
            "max_faces": 1,
            "face_quality_threshold": 0.75,
            "min_face_bbox_size": 0,
            "min_face_area_ratio": 0.0,
            "config_source": "settings_defaults",
        },
        default_vlm_policy={
            "backend": "auto",
            "preferred_backend": "qwen",
            "secondary_backend": "smolvlm",
            "enabled_event_types": ["manual_review_required"],
            "disabled_event_types": [],
            "degradation_policy": "auto_then_secondary_then_simple",
            "max_allowed_latency_seconds": 60.0,
            "max_allowed_rss_mb": 8192.0,
            "qwen_max_allowed_rss_mb": 12288.0,
            "smolvlm_max_allowed_rss_mb": 10240.0,
            "max_concurrent_inferences": 1,
            "config_source": "settings_defaults",
        },
    )


def _record(
    *,
    camera_id: str,
    event_index: int,
    event_type: str = "face_detected_unidentified",
    face_detected: bool = True,
    face_usable: bool = True,
    semantic_selected_key: str | None = "qwen",
    semantic_fallback: bool = False,
    attempted_backend_keys: list[str] | None = None,
    real_backend_attempted: bool = True,
    budget_status: str = "ok",
    budget_backend: str | None = None,
    observed_rss_mb: float | None = 1024,
    max_allowed_rss_mb: float | None = 4096,
) -> dict:
    attempted_backend_keys = (
        attempted_backend_keys
        if attempted_backend_keys is not None
        else ([semantic_selected_key] if semantic_selected_key else [])
    )
    low_quality = bool(face_detected and not face_usable)
    return {
        "schema_version": "runtime_metrics_event_v1",
        "timestamp": f"2026-05-07T12:00:{event_index:02d}+00:00",
        "camera_id": camera_id,
        "event_type": event_type,
        "event_id": f"evt-{camera_id}-{event_index}",
        "face": {
            "requested": "auto",
            "selected": "insightface",
            "fallback_used": False,
            "detect_elapsed_ms": 20,
            "detected": face_detected,
            "usable": face_usable,
            "low_quality": low_quality,
            "quality_score": 0.9 if face_usable else 0.2,
            "quality_metrics": {"size_score": 0.6},
            "rejection_reasons": [] if face_usable else ["face_quality_threshold_failed"],
            "configuration": {
                "det_size": "640,640",
                "detection_threshold": 0.5,
                "max_faces": 1,
                "face_quality_threshold": 0.75,
                "min_face_bbox_size": 0,
                "min_face_area_ratio": 0.0,
                "config_source": "api_camera_metadata",
            },
        },
        "semantic": {
            "requested": "auto",
            "effective_request": semantic_selected_key or "simple",
            "selected": semantic_selected_key,
            "selected_key": semantic_selected_key,
            "fallback_used": semantic_fallback,
            "total_duration_ms": 100,
            "descriptor_valid": semantic_selected_key is not None,
            "success": semantic_selected_key is not None,
            "parser_strategy": "direct_json" if semantic_selected_key in {"qwen", "smolvlm"} else None,
            "parser_backend_key": semantic_selected_key,
            "json_recovered": False,
            "parser_error": None,
            "attempts_count": len(attempted_backend_keys),
            "attempted_backend_keys": attempted_backend_keys,
            "real_backend_attempted": real_backend_attempted,
            "policy": {
                "backend": "auto",
                "preferred_backend": "qwen",
                "secondary_backend": "smolvlm",
                "enabled_event_types": ["manual_review_required"],
                "degradation_policy": "auto_then_secondary_then_simple",
                "max_allowed_latency_seconds": 60.0,
                "max_allowed_rss_mb": 8192.0,
                "qwen_max_allowed_rss_mb": 12288.0,
                "smolvlm_max_allowed_rss_mb": 10240.0,
            },
        },
        "budget": {
            "status": budget_status,
            "backend_key": budget_backend or semantic_selected_key,
            "observed_rss_mb": observed_rss_mb,
            "max_allowed_rss_mb": max_allowed_rss_mb,
            "rejection_reason": (
                "vlm_memory_budget_exceeded" if budget_status == "rejected" else None
            ),
            "rejection_reasons": (
                ["vlm_memory_budget_exceeded"] if budget_status == "rejected" else []
            ),
        },
        "config": {"source": "api_camera_metadata"},
    }


def _recommendations_for(result, camera_id):
    return [
        recommendation
        for recommendation in result["recommendations"]
        if recommendation["camera_id"] == camera_id
    ]


def _has_recommendation(result, camera_id, recommendation_type):
    return any(
        recommendation["recommendation_type"] == recommendation_type
        for recommendation in _recommendations_for(result, camera_id)
    )
