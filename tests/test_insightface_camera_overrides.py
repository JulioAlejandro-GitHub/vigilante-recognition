from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.config import settings
from app.services.insightface_runtime_cache import clear_insightface_runtime_cache
from app.services.insightface_service import InsightFaceService

CAMERA_A_ID = "11111111-1111-1111-1111-111111111111"
CAMERA_B_ID = "22222222-1111-1111-1111-111111111111"


def test_insightface_camera_override_is_applied_and_traced(tmp_path, monkeypatch):
    clear_insightface_runtime_cache()
    calls: dict[str, object] = {}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def prepare(self, *, ctx_id, det_size, det_thresh):
            calls["prepare"] = {"ctx_id": ctx_id, "det_size": det_size, "det_thresh": det_thresh}

        def get(self, image, max_num=0):
            calls["get"] = {"max_num": max_num}
            return [
                SimpleNamespace(
                    bbox=np.asarray([20.0, 20.0, 100.0, 100.0], dtype=np.float32),
                    det_score=0.82,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")
    overrides = {
        CAMERA_A_ID: {
            "det_size": "320,320",
            "detection_threshold": 0.8,
            "max_faces": 3,
            "face_quality_threshold": 0.1,
            "min_face_bbox_size": 20,
            "min_face_area_ratio": 0.005,
        }
    }

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(overrides)), patch.object(
        settings,
        "face_quality_threshold",
        0.75,
    ):
        service = InsightFaceService(model_name="buffalo_l", provider="cpu")
        detection = service.inspect_face(frame_ref=str(frame_path), camera_id=CAMERA_A_ID)

    trace = detection.face_backend_trace
    config = trace["configuration"]
    assert detection.detected is True
    assert detection.usable is True
    assert calls["prepare"] == {"ctx_id": -1, "det_size": (320, 320), "det_thresh": 0.8}
    assert calls["get"] == {"max_num": 3}
    assert config["camera_id"] == CAMERA_A_ID
    assert config["det_size"] == [320, 320]
    assert config["detection_threshold"] == 0.8
    assert config["max_faces"] == 3
    assert config["config_source"] == "camera_overrides_json"
    assert config["camera_override_applied"] is True
    assert config["camera_override_key"] == CAMERA_A_ID
    assert config["quality_thresholds"] == {
        "face_quality_threshold": 0.1,
        "min_face_bbox_size": 20,
        "min_face_area_ratio": 0.005,
    }
    assert trace["camera_id"] == CAMERA_A_ID
    assert trace["config_source"] == "camera_overrides_json"
    assert trace["quality_thresholds"] == config["quality_thresholds"]


def test_insightface_invalid_camera_override_falls_back_to_global_and_traces_error(tmp_path, monkeypatch):
    clear_insightface_runtime_cache()
    calls: dict[str, object] = {}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def prepare(self, *, ctx_id, det_size, det_thresh):
            calls["prepare"] = {"ctx_id": ctx_id, "det_size": det_size, "det_thresh": det_thresh}

        def get(self, image, max_num=0):
            calls["get"] = {"max_num": max_num}
            return [
                SimpleNamespace(
                    bbox=np.asarray([20.0, 20.0, 100.0, 100.0], dtype=np.float32),
                    det_score=0.9,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")
    overrides = {CAMERA_A_ID: {"det_size": "bad", "detection_threshold": 2.0, "max_faces": -1}}

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(overrides)), patch.object(
        settings,
        "face_quality_threshold",
        0.1,
    ):
        service = InsightFaceService(det_size="640,640", detection_threshold=0.5, max_faces=1)
        detection = service.inspect_face(frame_ref=str(frame_path), camera_id=CAMERA_A_ID)

    config = detection.face_backend_trace["configuration"]
    assert detection.detected is True
    assert calls["prepare"] == {"ctx_id": -1, "det_size": (640, 640), "det_thresh": 0.5}
    assert calls["get"] == {"max_num": 1}
    assert config["config_source"] == "global"
    assert config["camera_override_applied"] is False
    assert config["camera_override_errors"] == [
        "camera_overrides_json:det_size_invalid",
        "camera_overrides_json:detection_threshold_invalid",
        "camera_overrides_json:max_faces_invalid",
    ]


def test_insightface_runtime_cache_reuses_runtime_when_only_max_faces_changes(tmp_path, monkeypatch):
    clear_insightface_runtime_cache()
    calls: dict[str, list[object]] = {"init": [], "prepare": [], "get": []}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)

        def prepare(self, *, ctx_id, det_size, det_thresh):
            calls["prepare"].append({"ctx_id": ctx_id, "det_size": det_size, "det_thresh": det_thresh})

        def get(self, image, max_num=0):
            calls["get"].append(max_num)
            return [
                SimpleNamespace(
                    bbox=np.asarray([20.0, 20.0, 100.0, 100.0], dtype=np.float32),
                    det_score=0.95,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")
    overrides = {
        CAMERA_A_ID: {"max_faces": 1, "face_quality_threshold": 0.1},
        CAMERA_B_ID: {"max_faces": 3, "face_quality_threshold": 0.1},
    }

    with patch.object(settings, "insightface_camera_overrides_json", json.dumps(overrides)):
        service = InsightFaceService(det_size="640,640", detection_threshold=0.5)
        first_detection = service.inspect_face(frame_ref=str(frame_path), camera_id=CAMERA_A_ID)
        second_detection = service.inspect_face(frame_ref=str(frame_path), camera_id=CAMERA_B_ID)

    assert len(calls["init"]) == 1
    assert calls["prepare"] == [{"ctx_id": -1, "det_size": (640, 640), "det_thresh": 0.5}]
    assert calls["get"] == [1, 3]
    assert first_detection.face_backend_trace["runtime_reused"] is False
    assert second_detection.face_backend_trace["runtime_reused"] is True
    assert first_detection.face_backend_trace["configuration"]["max_faces"] == 1
    assert second_detection.face_backend_trace["configuration"]["max_faces"] == 3


def _install_fake_insightface(monkeypatch, face_analysis_class) -> None:
    package = types.ModuleType("insightface")
    app_module = types.ModuleType("insightface.app")
    app_module.FaceAnalysis = face_analysis_class
    package.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)


def _write_frame(path: Path) -> Path:
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (100, 100), (180, 180, 180), thickness=-1)
    assert cv2.imwrite(str(path), image)
    return path
