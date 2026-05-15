import pytest

from app.config import settings
from app.services.vlm_degradation_service import vlm_degradation_state


@pytest.fixture(autouse=True)
def stable_unit_test_runtime_settings():
    overrides = {
        "semantic_descriptor_backend": "simple",
        "semantic_enable_fallback": True,
        "face_backend": "simple",
        "embedding_backend": "simple_face_crop_512",
        "known_face_gallery_path": "tests/fixtures/gallery/dev_known_face_gallery.json",
        "semantic_use_real_vlm": None,
        "qwen_vl_enabled": False,
        "smolvlm_enabled": False,
        "vlm_auto_preferred_backend": "qwen",
        "vlm_secondary_backend": "smolvlm",
        "vlm_device": "auto",
        "semantic_device": "",
        "vlm_timeout_seconds": 60,
        "semantic_timeout_seconds": None,
        "vlm_max_new_tokens": 192,
        "vlm_max_image_edge": 384,
        "vlm_serialization_guard_enabled": True,
        "vlm_enable_for_event_types": (
            "manual_review_required,identity_conflict,"
            "recurrent_unresolved_subject,case_suggestion_created"
        ),
        "vlm_disable_for_camera_ids": "",
        "vlm_camera_policy_overrides_json": "",
        "vlm_max_allowed_latency_seconds": 60.0,
        "vlm_max_allowed_rss_mb": 8192.0,
        "qwen_max_allowed_rss_mb": 12288.0,
        "smolvlm_max_allowed_rss_mb": 10240.0,
        "vlm_max_concurrent_inferences": 1,
        "vlm_concurrency_acquire_timeout_seconds": 0.0,
        "vlm_degradation_policy": "auto_then_secondary_then_simple",
        "vlm_recent_failure_threshold": 3,
        "vlm_circuit_breaker_window_seconds": 300,
        "vlm_circuit_breaker_cooldown_seconds": 300,
    }
    previous = {key: getattr(settings, key) for key in overrides}
    for key, value in overrides.items():
        setattr(settings, key, value)
    vlm_degradation_state.reset()
    yield
    for key, value in previous.items():
        setattr(settings, key, value)
    vlm_degradation_state.reset()
