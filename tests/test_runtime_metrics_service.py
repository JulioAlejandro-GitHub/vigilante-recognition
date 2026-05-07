from __future__ import annotations

import json
from datetime import datetime, timezone

from app.domain.entities import FaceDetectionResult, FrameIngestedMessage, SemanticDescriptorResult
from app.services.runtime_metrics_service import RuntimeMetricsService
from app.services.runtime_metrics_store import JsonlRuntimeMetricsStore


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_runtime_metrics_service_persists_event_record(tmp_path):
    store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    service = RuntimeMetricsService(
        enabled=True,
        store=store,
        log_summary_every_n_events=0,
    )

    record = service.record_event(
        source_message=_message(),
        emitted_event=_event("face_detected_unidentified"),
        face_detection=_face_detection(),
        semantic_descriptor_result=_semantic_descriptor(
            selected_key="qwen",
            selected="Qwen/Qwen2.5-VL-3B-Instruct",
            parse_strategy="fenced_json",
            json_recovered=True,
        ),
        camera_runtime_config_trace=_camera_trace(),
    )

    persisted = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert record is not None
    assert len(persisted) == 1
    assert persisted[0]["camera_id"] == CAMERA_ID
    assert persisted[0]["event_type"] == "face_detected_unidentified"
    assert persisted[0]["source_frame_event_id"] == "frame-event-1"
    assert persisted[0]["face"]["selected"] == "insightface"
    assert persisted[0]["face"]["detect_elapsed_ms"] == 12.5
    assert persisted[0]["semantic"]["selected_key"] == "qwen"
    assert persisted[0]["semantic"]["parser_strategy"] == "fenced_json"
    assert persisted[0]["semantic"]["json_recovered"] is True
    assert persisted[0]["budget"]["status"] == "ok"
    assert persisted[0]["budget"]["observed_rss_mb"] == 1024.0
    assert persisted[0]["config"]["source"] == "api_camera_metadata"
    assert persisted[0]["config"]["camera_config_version"] == "cam-v7"


def test_runtime_metrics_store_rotates_and_retains_files(tmp_path):
    store = JsonlRuntimeMetricsStore(
        tmp_path / "events.jsonl",
        rotate_max_mb=0.00025,
        retention_files=1,
    )

    for index in range(5):
        store.append({"index": index, "payload": "x" * 500})

    files = sorted(path.name for path in tmp_path.glob("events*.jsonl"))
    rotated = [name for name in files if name != "events.jsonl"]
    assert "events.jsonl" in files
    assert len(rotated) <= 1


def test_runtime_metrics_service_counts_budget_rejection(tmp_path):
    store = JsonlRuntimeMetricsStore(tmp_path / "events.jsonl")
    service = RuntimeMetricsService(enabled=True, store=store)

    service.record_event(
        source_message=_message(),
        emitted_event=_event("human_presence_no_face"),
        face_detection=_face_detection(detected=False, usable=False),
        semantic_descriptor_result=_semantic_descriptor(
            selected_key="simple",
            selected="simple_color_signature_v1",
            fallback_used=True,
            budget_status="rejected",
            budget_backend="smolvlm",
            budget_reason="vlm_memory_budget_exceeded",
            observed_rss_mb=9999,
            max_allowed_rss_mb=1024,
        ),
        camera_runtime_config_trace=_camera_trace(source="global"),
    )

    [persisted] = list(store.iter_records())
    assert persisted["semantic"]["fallback_used"] is True
    assert persisted["budget"]["status"] == "rejected"
    assert persisted["budget"]["backend_key"] == "smolvlm"
    assert persisted["budget"]["rejection_reason"] == "vlm_memory_budget_exceeded"


def _message() -> FrameIngestedMessage:
    now = datetime.now(timezone.utc)
    return FrameIngestedMessage(
        event_id="frame-event-1",
        event_type="frame.ingested",
        event_version="1.0",
        occurred_at=now,
        payload={
            "camera_id": CAMERA_ID,
            "captured_at": now,
            "frame_ref": "tests/fixtures/images/face_detectable.jpg",
            "metadata": {},
        },
        context={},
    )


def _event(event_type: str) -> dict:
    return {
        "event_id": f"evt-test-{event_type}",
        "event_type": event_type,
        "context": {
            "camera_id": CAMERA_ID,
            "track_id": "track-1",
            "subject_id": "subject-1",
        },
        "payload": {},
    }


def _face_detection(*, detected: bool = True, usable: bool = True) -> FaceDetectionResult:
    return FaceDetectionResult(
        detected=detected,
        usable=usable,
        quality_score=0.88 if usable else 0.22,
        face_backend="insightface",
        face_backend_requested="auto",
        face_backend_selected="insightface",
        face_backend_fallback_used=False,
        face_backend_trace={
            "requested_backend": "auto",
            "selected_backend": "insightface",
            "fallback_used": False,
            "elapsed_ms": 12.5,
            "detect_elapsed_ms": 12.5,
            "provider": "insightface_provider",
        },
    )


def _semantic_descriptor(
    *,
    selected_key: str,
    selected: str,
    parse_strategy: str | None = "direct_json",
    json_recovered: bool = False,
    parser_error: str | None = None,
    fallback_used: bool = False,
    budget_status: str = "ok",
    budget_backend: str | None = None,
    budget_reason: str | None = None,
    observed_rss_mb: float = 1024.0,
    max_allowed_rss_mb: float = 4096.0,
) -> SemanticDescriptorResult:
    budget_backend = budget_backend or selected_key
    attempt_status = "rejected_by_budget" if budget_status == "rejected" else "success"
    attempt = {
        "backend_key": budget_backend,
        "backend_name": selected,
        "status": attempt_status,
        "reason": budget_reason,
        "duration_ms": 45,
        "descriptor_valid": attempt_status == "success",
        "parse_strategy_used": parse_strategy,
        "json_recovered": json_recovered,
        "parser_error": parser_error,
        "budget": {
            "status": "exceeded" if budget_status == "rejected" else budget_status,
            "backend_key": budget_backend,
            "observed_rss_mb": observed_rss_mb,
            "max_allowed_rss_mb": max_allowed_rss_mb,
            "reasons": [budget_reason] if budget_reason else [],
        },
    }
    if fallback_used and selected_key == "simple":
        attempt = {
            **attempt,
            "backend_key": "smolvlm",
            "backend_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        }
        attempts = [
            attempt,
            {
                "backend_key": "simple",
                "backend_name": "simple_color_signature_v1",
                "status": "success",
                "duration_ms": 3,
                "descriptor_valid": True,
                "budget": {"status": "not_applicable", "reasons": []},
            },
        ]
    else:
        attempts = [attempt]

    trace = {
        "semantic_backend_requested": "auto",
        "semantic_backend_effective_request": "qwen",
        "semantic_backend_selected": selected,
        "semantic_backend_selected_key": selected_key,
        "semantic_backend_fallback_used": fallback_used,
        "descriptor_valid": True,
        "total_duration_ms": 45,
        "attempts": attempts,
    }
    descriptor = {
        "descriptor_schema_version": "semantic_descriptor_v2",
        "semantic_backend_requested": "auto",
        "semantic_backend_effective_request": "qwen",
        "semantic_backend_selected": selected,
        "semantic_backend_fallback_used": fallback_used,
        "semantic_backend_trace": trace,
    }
    return SemanticDescriptorResult(
        generated=True,
        backend=selected,
        descriptor=descriptor,
        signature={"dominant_palette": ["gray"]},
        confidence=0.7,
    )


def _camera_trace(source: str = "api.camera.metadata") -> dict:
    return {
        "config_source": source,
        "camera_config_version": "cam-v7",
        "camera_config_hash": "hash-camera",
        "effective_config_hash": "hash-effective",
        "face_tuning_source": "api_camera_metadata" if source != "global" else "global",
        "vlm_policy_source": "api_camera_metadata" if source != "global" else "global",
        "face_effective_config_hash": "hash-face",
        "vlm_effective_policy_hash": "hash-vlm",
        "camera_override_applied": source != "global",
    }
