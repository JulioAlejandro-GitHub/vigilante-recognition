from __future__ import annotations

from unittest.mock import patch

from app.config import settings
from app.services.vlm_policy_service import resolve_vlm_policy


def test_policy_sends_low_priority_event_directly_to_simple_by_default() -> None:
    decision = resolve_vlm_policy(
        requested_backend="auto",
        event_type_hint="human_presence_no_face",
        camera_id="cam-low",
    )

    assert decision.backend_chain == ["simple"]
    assert decision.vlm_allowed is False
    assert decision.allowed_backend_key == "simple"
    assert decision.reason == "vlm_event_type_not_enabled:human_presence_no_face"
    assert "manual_review_required" in decision.event_policy["enabled_event_types"]


def test_policy_allows_high_value_event_with_preferred_backend() -> None:
    decision = resolve_vlm_policy(
        requested_backend="auto",
        event_type_hint="manual_review_required",
        camera_id="cam-review",
    )

    assert decision.vlm_allowed is True
    assert decision.allowed_backend_key == "qwen"
    assert decision.backend_chain[:2] == ["qwen", "smolvlm"]
    assert decision.reason == "vlm_event_type_enabled:manual_review_required"


def test_camera_json_override_can_enable_vlm_and_change_budget() -> None:
    camera_id = "11111111-1111-1111-1111-111111111111"
    override_json = (
        '{"11111111-1111-1111-1111-111111111111": {'
        '"enabled": true,'
        '"backend": "auto",'
        '"preferred_backend": "smolvlm",'
        '"secondary_backend": "qwen",'
        '"enable_for_event_types": ["human_presence_no_face"],'
        '"max_latency_seconds": 12,'
        '"max_rss_mb": 4096'
        "}}"
    )

    with patch.object(settings, "semantic_descriptor_backend", "simple"), patch.object(
        settings,
        "vlm_camera_policy_overrides_json",
        override_json,
    ):
        decision = resolve_vlm_policy(
            requested_backend=settings.semantic_descriptor_backend,
            event_type_hint="human_presence_no_face",
            camera_id=camera_id,
        )

    assert decision.vlm_allowed is True
    assert decision.effective_backend_key == "auto"
    assert decision.allowed_backend_key == "smolvlm"
    assert decision.backend_chain[:2] == ["smolvlm", "qwen"]
    assert decision.budget["max_allowed_latency_seconds"] == 12
    assert decision.budget["max_allowed_rss_mb"] == 4096
    assert "env_camera_policy_json" in decision.policy_sources


def test_camera_metadata_override_has_precedence_over_env_disable() -> None:
    camera_id = "cam-special"

    with patch.object(settings, "vlm_disable_for_camera_ids", camera_id), patch.object(
        settings,
        "semantic_descriptor_backend",
        "simple",
    ):
        decision = resolve_vlm_policy(
            requested_backend=settings.semantic_descriptor_backend,
            event_type_hint="manual_review_required",
            camera_id=camera_id,
            camera_metadata={
                "vlm_policy": {
                    "enabled": True,
                    "backend": "qwen",
                    "max_allowed_latency_seconds": 7,
                }
            },
        )

    assert decision.vlm_allowed is True
    assert decision.effective_backend_key == "qwen"
    assert decision.allowed_backend_key == "qwen"
    assert decision.budget["max_allowed_latency_seconds"] == 7
    assert decision.policy_sources[-1] == "camera_metadata"


def test_camera_override_can_force_simple_even_for_eligible_event() -> None:
    decision = resolve_vlm_policy(
        requested_backend="auto",
        event_type_hint="identity_conflict",
        camera_id="cam-disabled",
        camera_metadata={"vlm_policy": {"force_simple": True}},
    )

    assert decision.backend_chain == ["simple"]
    assert decision.vlm_allowed is False
    assert decision.reason == "vlm_policy_simple_backend"
