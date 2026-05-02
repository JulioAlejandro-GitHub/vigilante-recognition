from __future__ import annotations

import sys
import types

from app.services.insightface_runtime_cache import (
    build_insightface_runtime_config,
    clear_insightface_runtime_cache,
    get_insightface_runtime,
    insightface_runtime_cache_size,
)


def test_insightface_runtime_cache_reuses_face_analysis_for_same_config(monkeypatch):
    clear_insightface_runtime_cache()
    calls: dict[str, list[dict[str, object]]] = {"init": [], "prepare": []}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)

        def prepare(self, *, ctx_id, det_size, det_thresh):
            calls["prepare"].append(
                {
                    "ctx_id": ctx_id,
                    "det_size": det_size,
                    "det_thresh": det_thresh,
                }
            )

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)

    config = build_insightface_runtime_config(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="320x320",
        detection_threshold=0.42,
    )
    first_runtime, first_reused = get_insightface_runtime(config)
    second_runtime, second_reused = get_insightface_runtime(config)

    assert first_runtime is second_runtime
    assert first_reused is False
    assert second_reused is True
    assert insightface_runtime_cache_size() == 1
    assert calls["init"] == [{"name": "buffalo_l", "providers": ["CPUExecutionProvider"]}]
    assert calls["prepare"] == [{"ctx_id": -1, "det_size": (320, 320), "det_thresh": 0.42}]


def test_insightface_runtime_cache_loads_new_runtime_when_load_config_changes(monkeypatch):
    clear_insightface_runtime_cache()
    calls: dict[str, list[dict[str, object]]] = {"init": [], "prepare": []}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)

        def prepare(self, *, ctx_id, det_size, det_thresh):
            calls["prepare"].append(
                {
                    "ctx_id": ctx_id,
                    "det_size": det_size,
                    "det_thresh": det_thresh,
                }
            )

    _install_fake_insightface(monkeypatch, FakeFaceAnalysis)

    base_config = build_insightface_runtime_config(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="320,320",
        detection_threshold=0.42,
    )
    changed_det_size = build_insightface_runtime_config(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="640,640",
        detection_threshold=0.42,
    )
    changed_threshold = build_insightface_runtime_config(
        model_name="buffalo_l",
        provider="cpu",
        model_root="",
        det_size="320,320",
        detection_threshold=0.35,
    )

    get_insightface_runtime(base_config)
    get_insightface_runtime(base_config)
    get_insightface_runtime(changed_det_size)
    get_insightface_runtime(changed_threshold)

    assert insightface_runtime_cache_size() == 3
    assert len(calls["init"]) == 3
    assert calls["prepare"] == [
        {"ctx_id": -1, "det_size": (320, 320), "det_thresh": 0.42},
        {"ctx_id": -1, "det_size": (640, 640), "det_thresh": 0.42},
        {"ctx_id": -1, "det_size": (320, 320), "det_thresh": 0.35},
    ]


def _install_fake_insightface(monkeypatch, face_analysis_class) -> None:
    package = types.ModuleType("insightface")
    app_module = types.ModuleType("insightface.app")
    app_module.FaceAnalysis = face_analysis_class
    package.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)
