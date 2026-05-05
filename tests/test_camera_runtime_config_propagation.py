from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

from app.config import settings
from app.domain.entities import FaceDetectionResult, FrameIngestedMessage, SemanticDescriptorResult
from app.services.camera_face_tuning_service import CameraFaceTuningService
from app.services.camera_runtime_config_service import extract_camera_runtime_config
from app.services.vlm_policy_service import resolve_vlm_policy
from app.worker import _build_camera_runtime_config_trace


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_runtime_config_extractor_reads_api_camera_block() -> None:
    runtime_config = extract_camera_runtime_config(_metadata_from_api_camera())

    assert runtime_config.present is True
    assert runtime_config.source_label == "api_camera_metadata"
    assert runtime_config.camera_config_version == "ops-v1"
    assert runtime_config.face_tuning["det_size"] == "320,320"
    assert runtime_config.vlm_policy["backend"] == "auto"


def test_face_tuning_prefers_api_camera_runtime_config_over_payload_and_env() -> None:
    service = CameraFaceTuningService()
    defaults = service.build_defaults(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="640,640",
        detection_threshold=0.5,
        max_faces=1,
    )
    env_overrides = {CAMERA_ID: {"det_size": "960,960", "detection_threshold": 0.4}}
    metadata = {
        **_metadata_from_api_camera(),
        "face_tuning": {"insightface": {"det_size": "800,800", "detection_threshold": 0.45}},
    }

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(env_overrides)):
        tuning = service.resolve(camera_id=CAMERA_ID, defaults=defaults, camera_metadata=metadata)

    assert tuning.config_source == "api_camera_metadata"
    assert tuning.camera_config_version == "ops-v1"
    assert tuning.camera_config_hash == "api-hash"
    assert tuning.det_size == "320,320"
    assert tuning.detection_threshold == 0.7
    assert tuning.max_faces == 2
    assert tuning.camera_override_applied is True


def test_vlm_policy_prefers_api_camera_runtime_config_over_payload_and_env() -> None:
    metadata = {
        **_metadata_from_api_camera(),
        "vlm_policy": {
            "enabled": True,
            "backend": "qwen",
            "enable_for_event_types": ["human_presence_no_face"],
        },
    }

    with patch.object(settings, "vlm_disable_for_camera_ids", CAMERA_ID), patch.object(
        settings,
        "semantic_descriptor_backend",
        "simple",
    ):
        decision = resolve_vlm_policy(
            requested_backend=settings.semantic_descriptor_backend,
            event_type_hint="human_presence_no_face",
            camera_id=CAMERA_ID,
            camera_metadata=metadata,
        )

    assert decision.config_source == "api_camera_metadata"
    assert decision.camera_config_version == "ops-v1"
    assert decision.camera_config_hash == "api-hash"
    assert decision.vlm_allowed is True
    assert decision.effective_backend_key == "auto"
    assert decision.allowed_backend_key == "smolvlm"
    assert decision.backend_chain[:2] == ["smolvlm", "qwen"]
    assert decision.budget["max_allowed_latency_seconds"] == 11
    assert decision.budget["max_allowed_rss_mb"] == 2048
    assert decision.policy_sources[-1] == "api_camera_metadata"


def test_invalid_runtime_vlm_policy_does_not_break_resolution() -> None:
    metadata = _metadata_from_api_camera(
        vlm_policy={
            "enabled": "not-a-bool",
            "backend": "auto",
            "enable_for_event_types": ["manual_review_required"],
            "max_concurrent_inferences": "not-a-number",
        }
    )

    decision = resolve_vlm_policy(
        requested_backend="auto",
        event_type_hint="manual_review_required",
        camera_id=CAMERA_ID,
        camera_metadata=metadata,
    )

    assert decision.vlm_allowed is True
    assert decision.allowed_backend_key == "qwen"
    assert "api_camera_metadata:enabled_invalid" in decision.policy_errors
    assert "api_camera_metadata:max_concurrent_inferences_invalid" in decision.policy_errors


def test_worker_trace_summarizes_runtime_face_and_vlm_sources() -> None:
    message = FrameIngestedMessage.model_validate(
        {
            "event_id": "evt_1",
            "event_type": "frame.ingested",
            "event_version": "1.0",
            "occurred_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "payload": {
                "camera_id": CAMERA_ID,
                "captured_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
                "frame_ref": "tests/fixtures/images/face_detectable.jpg",
                "metadata": _metadata_from_api_camera(),
                "quality_metadata": {},
            },
            "context": {},
        }
    )
    face_detection = FaceDetectionResult(
        face_backend_trace={
            "configuration": {
                "config_source": "api_camera_metadata",
                "camera_config_version": "ops-v1",
                "camera_config_hash": "api-hash",
                "effective_config_hash": "face-effective-hash",
            }
        }
    )
    semantic_descriptor = SemanticDescriptorResult(
        generated=True,
        backend="simple_color_signature_v1",
        descriptor={
            "vlm_policy_trace": {
                "config_source": "api_camera_metadata",
                "vlm_policy_source": "api_camera_metadata",
                "camera_config_version": "ops-v1",
                "camera_config_hash": "api-hash",
                "effective_policy_hash": "vlm-effective-hash",
            }
        },
    )

    trace = _build_camera_runtime_config_trace(
        message=message,
        face_detection=face_detection,
        semantic_descriptor_result=semantic_descriptor,
    )

    assert trace["config_source"] == "api.camera.metadata"
    assert trace["camera_config_version"] == "ops-v1"
    assert trace["camera_override_applied"] is True
    assert trace["face_tuning_source"] == "api_camera_metadata"
    assert trace["vlm_policy_source"] == "api_camera_metadata"
    assert trace["face_effective_config_hash"] == "face-effective-hash"
    assert trace["vlm_effective_policy_hash"] == "vlm-effective-hash"


def _metadata_from_api_camera(*, vlm_policy: dict | None = None) -> dict:
    return {
        "camera_runtime_config": {
            "schema_version": "camera_runtime_config_v1",
            "config_source": "api.camera.metadata",
            "camera_id": CAMERA_ID,
            "camera_config_version": "ops-v1",
            "config_hash": "api-hash",
            "effective_config_hash": "api-effective-hash",
            "recognition": {
                "enabled": True,
                "face_tuning": {
                    "det_size": "320,320",
                    "detection_threshold": 0.7,
                    "max_faces": 2,
                    "face_quality_threshold": 0.55,
                    "min_face_bbox_size": 40,
                    "min_face_area_ratio": 0.02,
                },
                "vlm_policy": vlm_policy
                or {
                    "enabled": True,
                    "backend": "auto",
                    "preferred_backend": "smolvlm",
                    "secondary_backend": "qwen",
                    "enable_for_event_types": ["human_presence_no_face"],
                    "max_latency_seconds": 11,
                    "max_rss_mb": 2048,
                    "max_concurrent_inferences": 1,
                    "degradation_policy": "preferred_then_secondary_then_simple",
                },
            },
        }
    }
