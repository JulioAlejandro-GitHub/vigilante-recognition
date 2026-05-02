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


def test_insightface_service_detects_face_and_generates_embedding_with_fake_runtime(tmp_path, monkeypatch):
    calls: dict[str, object] = {}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def prepare(self, *, ctx_id, det_size):
            calls["prepare"] = {"ctx_id": ctx_id, "det_size": det_size}

        def get(self, image):
            assert image.shape[:2] == (120, 160)
            return [
                SimpleNamespace(
                    bbox=np.asarray([30.0, 20.0, 110.0, 100.0], dtype=np.float32),
                    det_score=0.99,
                    normed_embedding=np.ones(512, dtype=np.float32),
                )
            ]

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")

    with patch.object(settings, "face_quality_threshold", 0.1), patch.object(
        settings,
        "insightface_enabled",
        True,
    ):
        service = InsightFaceService(model_name="buffalo_l", provider="cpu", model_root=str(tmp_path / "models"))
        detection = service.inspect_face(frame_ref=str(frame_path), quality_metadata={"capture_fps": 1.0})
        embedding = service.generate(frame_ref=str(frame_path), face_detection=detection)

    assert detection.detected is True
    assert detection.usable is True
    assert detection.bbox == {"x": 30, "y": 20, "width": 80, "height": 80}
    assert detection.image_size == {"width": 160, "height": 120}
    assert detection.quality_metrics["detection_confidence"] == 0.99
    assert detection.frame_quality_metadata == {"capture_fps": 1.0}
    assert embedding.generated is True
    assert embedding.backend == "insightface:buffalo_l"
    assert embedding.dimensions == 512
    assert len(embedding.vector) == 512
    assert calls["init_kwargs"]["providers"] == ["CPUExecutionProvider"]
    assert calls["prepare"] == {"ctx_id": -1, "det_size": (640, 640)}


def test_insightface_service_fails_clearly_when_disabled(tmp_path, monkeypatch):
    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            raise AssertionError("InsightFace should not load while disabled")

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)
    frame_path = _write_frame(tmp_path / "frame.jpg")

    with patch.object(settings, "insightface_enabled", False):
        service = InsightFaceService()
        with pytest.raises(FaceBackendError) as exc_info:
            service.inspect_face(frame_ref=str(frame_path))

    assert exc_info.value.reason == "insightface_disabled"
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
    cv2.rectangle(image, (30, 20), (110, 100), (180, 180, 180), thickness=-1)
    assert cv2.imwrite(str(path), image)
    return path
