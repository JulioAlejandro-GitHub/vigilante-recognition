from __future__ import annotations

from unittest.mock import patch

from app.config import settings
from app.services.vlm_degradation_service import vlm_degradation_state
from app.services.vlm_policy_service import evaluate_attempt_budget, resolve_vlm_policy


def test_budget_can_be_configured_per_backend() -> None:
    with patch.object(settings, "vlm_max_allowed_rss_mb", 4096.0), patch.object(
        settings,
        "qwen_max_allowed_rss_mb",
        10240.0,
    ), patch.object(settings, "smolvlm_max_allowed_rss_mb", 6144.0):
        decision = resolve_vlm_policy(
            requested_backend="auto",
            event_type_hint="manual_review_required",
            camera_id="cam-budget",
        )

    assert decision.budget["max_allowed_rss_mb"] == 4096.0
    assert decision.budget["backend_budgets"]["qwen"]["max_allowed_rss_mb"] == 10240.0
    assert decision.budget["backend_budgets"]["smolvlm"]["max_allowed_rss_mb"] == 6144.0

    qwen_allowed, qwen_reasons, qwen_trace = evaluate_attempt_budget(
        backend_key="qwen",
        attempt={"duration_ms": 500, "runtime_after_inference_rss_mb": 9000.0},
        policy_decision=decision,
    )
    smol_allowed, smol_reasons, smol_trace = evaluate_attempt_budget(
        backend_key="smolvlm",
        attempt={"duration_ms": 500, "runtime_after_inference_rss_mb": 9000.0},
        policy_decision=decision,
    )

    assert qwen_allowed is True
    assert qwen_reasons == []
    assert qwen_trace["budget_scope"] == "backend"
    assert qwen_trace["rss_budget_source"] == "qwen_max_allowed_rss_mb"
    assert qwen_trace["max_allowed_rss_mb"] == 10240.0

    assert smol_allowed is False
    assert smol_reasons == ["vlm_memory_budget_exceeded"]
    assert smol_trace["budget_scope"] == "backend"
    assert smol_trace["rss_budget_source"] == "smolvlm_max_allowed_rss_mb"
    assert smol_trace["observed_rss_mb"] == 9000.0
    assert smol_trace["max_allowed_rss_mb"] == 6144.0


def test_camera_policy_can_override_backend_memory_budget() -> None:
    decision = resolve_vlm_policy(
        requested_backend="auto",
        event_type_hint="manual_review_required",
        camera_id="cam-budget",
        camera_metadata={
            "vlm_policy": {
                "enabled": True,
                "backend": "auto",
                "qwen_max_rss_mb": 12288,
                "smolvlm_max_rss_mb": 8192,
            }
        },
    )

    assert decision.budget["backend_budgets"]["qwen"]["max_allowed_rss_mb"] == 12288
    assert decision.budget["backend_budgets"]["smolvlm"]["max_allowed_rss_mb"] == 8192


def test_backend_specific_env_budget_survives_generic_camera_rss_budget() -> None:
    with patch.object(settings, "qwen_max_allowed_rss_mb", 12288.0), patch.object(
        settings,
        "smolvlm_max_allowed_rss_mb",
        10240.0,
    ):
        decision = resolve_vlm_policy(
            requested_backend="auto",
            event_type_hint="manual_review_required",
            camera_id="cam-budget",
            camera_metadata={
                "vlm_policy": {
                    "enabled": True,
                    "backend": "auto",
                    "max_rss_mb": 4096,
                }
            },
        )

    allowed, reasons, trace = evaluate_attempt_budget(
        backend_key="qwen",
        attempt={"duration_ms": 500, "runtime_after_inference_rss_mb": 9000.0},
        policy_decision=decision,
    )

    assert decision.budget["max_allowed_rss_mb"] == 4096
    assert decision.budget["backend_budgets"]["qwen"]["max_allowed_rss_mb"] == 12288.0
    assert allowed is True
    assert reasons == []
    assert trace["configured_global_max_allowed_rss_mb"] == 4096.0
    assert trace["configured_backend_max_allowed_rss_mb"] == 12288.0
    assert trace["max_allowed_rss_mb"] == 12288.0


def test_memory_budget_rejection_trace_opens_backend_circuit() -> None:
    with patch.object(settings, "vlm_max_allowed_rss_mb", 4096.0), patch.object(
        settings,
        "smolvlm_max_allowed_rss_mb",
        0.0,
    ):
        decision = resolve_vlm_policy(
            requested_backend="smolvlm",
            event_type_hint="manual_review_required",
            camera_id="cam-budget",
        )
        allowed, reasons, trace = evaluate_attempt_budget(
            backend_key="smolvlm",
            attempt={"duration_ms": 500, "runtime_after_inference_rss_mb": 7603.14},
            policy_decision=decision,
        )
        health = vlm_degradation_state.record_failure("smolvlm", reason=reasons[0])

    assert allowed is False
    assert reasons == ["vlm_memory_budget_exceeded"]
    assert trace["status"] == "exceeded"
    assert trace["observed_rss_mb"] == 7603.14
    assert trace["max_allowed_rss_mb"] == 4096.0
    assert health["circuit_open"] is True
    assert health["recent_failure_reasons"] == ["vlm_memory_budget_exceeded"]
