from __future__ import annotations

import json
import logging
from unittest.mock import patch

from app.config import settings
from app.domain.entities import FaceDetectionResult
from app.services.camera_face_metrics_service import (
    get_camera_face_metrics_snapshot,
    record_camera_face_detection,
    reset_camera_face_metrics,
)
from app.services.camera_face_tuning_service import CameraFaceTuningService

CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_camera_face_tuning_uses_global_defaults_without_override():
    service = CameraFaceTuningService()
    defaults = service.build_defaults(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="640,640",
        detection_threshold=0.5,
        max_faces=1,
    )

    with patch.object(settings, "insightface_camera_overrides_json", ""), patch.object(
        settings,
        "face_quality_threshold",
        0.75,
    ), patch.object(settings, "insightface_min_face_bbox_size", 0), patch.object(
        settings,
        "insightface_min_face_area_ratio",
        0.0,
    ):
        tuning = service.resolve(camera_id=CAMERA_ID, defaults=defaults, camera_metadata={})

    assert tuning.config_source == "global"
    assert tuning.camera_override_applied is False
    assert tuning.det_size == "640,640"
    assert tuning.detection_threshold == 0.5
    assert tuning.max_faces == 1
    assert tuning.quality_thresholds_trace() == {
        "face_quality_threshold": 0.75,
        "min_face_bbox_size": 0,
        "min_face_area_ratio": 0.0,
    }


def test_camera_face_tuning_prefers_camera_metadata_over_env_json():
    service = CameraFaceTuningService()
    defaults = service.build_defaults(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="640,640",
        detection_threshold=0.5,
        max_faces=1,
    )
    env_overrides = {
        CAMERA_ID: {
            "det_size": "960,960",
            "detection_threshold": 0.4,
            "max_faces": 4,
        }
    }
    camera_metadata = {
        "face_tuning": {
            "insightface": {
                "det_size": [320, 320],
                "detection_threshold": 0.65,
                "max_faces": 2,
                "face_quality_threshold": 0.6,
                "min_face_bbox_size": 48,
                "min_face_area_ratio": 0.01,
            }
        }
    }

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(env_overrides)):
        tuning = service.resolve(
            camera_id=CAMERA_ID,
            defaults=defaults,
            camera_metadata=camera_metadata,
        )

    assert tuning.config_source == "camera_metadata"
    assert tuning.camera_override_key == "face_tuning.insightface"
    assert tuning.camera_override_applied is True
    assert tuning.det_size == "320,320"
    assert tuning.detection_threshold == 0.65
    assert tuning.max_faces == 2
    assert tuning.face_quality_threshold == 0.6
    assert tuning.min_face_bbox_size == 48
    assert tuning.min_face_area_ratio == 0.01


def test_camera_face_tuning_falls_back_to_global_when_override_is_invalid(caplog):
    service = CameraFaceTuningService()
    defaults = service.build_defaults(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="640,640",
        detection_threshold=0.5,
        max_faces=1,
    )
    env_overrides = {
        CAMERA_ID: {
            "det_size": "not-a-size",
            "detection_threshold": 1.4,
            "max_faces": -1,
        }
    }

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(env_overrides)):
        with caplog.at_level(logging.WARNING):
            tuning = service.resolve(camera_id=CAMERA_ID, defaults=defaults, camera_metadata={})

    assert tuning.config_source == "global"
    assert tuning.camera_override_applied is False
    assert tuning.det_size == "640,640"
    assert tuning.detection_threshold == 0.5
    assert tuning.max_faces == 1
    assert tuning.camera_override_errors == (
        "camera_overrides_json:det_size_invalid",
        "camera_overrides_json:detection_threshold_invalid",
        "camera_overrides_json:max_faces_invalid",
    )
    assert "insightface_camera_override_invalid" in caplog.text


def test_camera_face_metrics_accumulates_comparable_camera_summary():
    reset_camera_face_metrics()
    detection = FaceDetectionResult(
        detected=True,
        usable=False,
        rejection_reasons=["face_quality_threshold_failed"],
        face_backend_selected="insightface",
        face_backend_trace={
            "selected_backend": "insightface",
            "provider": "cpu",
            "faces_detected": 1,
            "detect_elapsed_ms": 12.5,
            "configuration": {
                "det_size": [640, 640],
                "detection_threshold": 0.5,
                "max_faces": 1,
                "config_source": "global",
            },
        },
    )
    no_face = FaceDetectionResult(
        detected=False,
        usable=False,
        rejection_reasons=["face_not_detected"],
        face_backend_selected="insightface",
        face_backend_trace={
            "selected_backend": "insightface",
            "provider": "cpu",
            "faces_detected": 0,
            "detect_elapsed_ms": 10.0,
            "configuration": {"config_source": "global"},
        },
    )

    record_camera_face_detection(camera_id=CAMERA_ID, face_detection=detection)
    record_camera_face_detection(camera_id=CAMERA_ID, face_detection=no_face)

    snapshot = get_camera_face_metrics_snapshot(CAMERA_ID)
    assert snapshot["frames_processed"] == 2
    assert snapshot["faces_detected"] == 1
    assert snapshot["face_not_detected"] == 1
    assert snapshot["usable_true"] == 0
    assert snapshot["usable_false"] == 2
    assert snapshot["low_quality_face"] == 1
    assert snapshot["usable_ratio"] == 0.0
    assert snapshot["average_detect_latency_ms"] == 11.25
