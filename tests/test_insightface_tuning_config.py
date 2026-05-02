from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.face_backend_service import FaceBackendError
from app.services.insightface_service import InsightFaceService
from app.services.insightface_runtime_cache import clear_insightface_runtime_cache


def test_insightface_tuning_config_is_applied_and_traced(tmp_path, monkeypatch):
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
                    bbox=np.asarray([20.0, 20.0, 80.0, 80.0], dtype=np.float32),
                    det_score=0.72,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")

    with patch.object(settings, "face_quality_threshold", 0.1):
        service = InsightFaceService(
            model_name="buffalo_s",
            provider="CPUExecutionProvider",
            model_root=str(tmp_path / "models"),
            det_size="320x320",
            detection_threshold=0.7,
            max_faces=2,
        )
        detection = service.inspect_face(frame_ref=str(frame_path))

    trace = detection.face_backend_trace
    config = trace["configuration"]
    assert detection.detected is True
    assert calls["init_kwargs"] == {
        "name": "buffalo_s",
        "providers": ["CPUExecutionProvider"],
        "root": str(tmp_path / "models"),
    }
    assert calls["prepare"] == {"ctx_id": -1, "det_size": (320, 320), "det_thresh": 0.7}
    assert calls["get"] == {"max_num": 2}
    assert config["model_name"] == "buffalo_s"
    assert config["provider"] == "CPUExecutionProvider"
    assert config["det_size"] == [320, 320]
    assert config["detection_threshold"] == 0.7
    assert config["max_faces"] == 2
    assert trace["faces_detected"] == 1
    assert trace["detect_elapsed_ms"] >= 0.0
    assert trace["backend_load_ms"] >= 0.0


def test_insightface_detection_threshold_filters_low_confidence_faces(tmp_path, monkeypatch):
    clear_insightface_runtime_cache()

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            pass

        def prepare(self, *, ctx_id, det_size, det_thresh):
            pass

        def get(self, image, max_num=0):
            return [
                SimpleNamespace(
                    bbox=np.asarray([20.0, 20.0, 80.0, 80.0], dtype=np.float32),
                    det_score=0.49,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")

    service = InsightFaceService(detection_threshold=0.5)
    detection = service.inspect_face(frame_ref=str(frame_path))

    assert detection.detected is False
    assert detection.rejection_reasons == ["face_not_detected"]
    assert detection.face_backend_trace["faces_detected"] == 0
    assert detection.face_backend_trace["configuration"]["detection_threshold"] == 0.5


def test_insightface_invalid_tuning_config_fails_clearly():
    clear_insightface_runtime_cache()

    service = InsightFaceService(detection_threshold=1.5)

    with pytest.raises(FaceBackendError) as exc_info:
        service.inspect_face(frame_ref="tests/fixtures/images/face_detectable.jpg")

    assert exc_info.value.reason == "insightface_detection_threshold_invalid"
    assert exc_info.value.stage == "configuration"


def _install_fake_insightface(monkeypatch, face_analysis_class) -> None:
    package = types.ModuleType("insightface")
    app_module = types.ModuleType("insightface.app")
    app_module.FaceAnalysis = face_analysis_class
    package.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)


def _write_frame(path: Path) -> Path:
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (180, 180, 180), thickness=-1)
    assert cv2.imwrite(str(path), image)
    return path
