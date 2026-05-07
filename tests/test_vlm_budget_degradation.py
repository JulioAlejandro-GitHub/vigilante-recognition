from __future__ import annotations

import time
from unittest.mock import patch

from app.config import settings
from app.consumer import load_fixture_message
from app.services.semantic_backends.base import (
    SemanticBackendContext,
    SemanticBackendOutput,
    SemanticDescriptorBackend,
)
from app.services.semantic_descriptor_service import SemanticDescriptorService


class PolicyTraceBackend(SemanticDescriptorBackend):
    def __init__(
        self,
        *,
        key: str,
        backend_name: str,
        sleep_seconds: float = 0.0,
        trace: dict | None = None,
    ) -> None:
        self.key = key
        self.backend_name = backend_name
        self.sleep_seconds = sleep_seconds
        self.trace = trace or {}

    def generate_descriptor(
        self,
        *,
        image_path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=_minimal_descriptor("gray" if self.key != "simple" else "blue"),
            confidence=0.82,
            raw_summary=f"{self.key} descriptor",
            trace=self.trace,
        )


def _minimal_descriptor(color: str) -> dict:
    return {
        "subject_type": "person",
        "top_clothing": {"category": "upper_garment", "color": color, "pattern": "solid"},
        "bottom_clothing": {"category": "lower_garment", "color": color, "pattern": "solid"},
        "dominant_colors": [color],
        "accessories": [],
        "carried_object": "unknown",
        "body_build": "average",
        "pose_direction": "front",
        "scene_observation_quality": {"level": "medium", "notes": "stable"},
        "descriptor_confidence": 0.82,
        "raw_summary": f"person wearing {color}",
    }


def _service() -> SemanticDescriptorService:
    return SemanticDescriptorService(
        backends={
            "qwen": PolicyTraceBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                sleep_seconds=0.08,
            ),
            "smolvlm": PolicyTraceBackend(
                key="smolvlm",
                backend_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            ),
            "simple": PolicyTraceBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
            ),
        }
    )


def test_slow_preferred_backend_is_rejected_by_latency_budget_and_degrades() -> None:
    fixture = load_fixture_message("tests/fixtures/frame_manual_review_required.json")

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen",
    ), patch.object(
        settings,
        "vlm_degradation_policy",
        "preferred_then_secondary_then_simple",
    ), patch.object(
        settings,
        "vlm_max_allowed_latency_seconds",
        0.05,
    ):
        descriptor = _service().generate(
            frame_ref=fixture.frame_ref,
            event_type_hint="manual_review_required",
            camera_id=fixture.camera_id,
        )

    trace = descriptor.descriptor["semantic_backend_trace"]
    attempts = trace["attempts"]

    assert descriptor.generated is True
    assert descriptor.backend == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen", "smolvlm"]
    assert attempts[0]["status"] == "rejected_by_budget"
    assert attempts[0]["reason"] == "vlm_latency_budget_exceeded"
    assert attempts[0]["budget"]["status"] == "exceeded"
    assert attempts[1]["status"] == "success"
    assert trace["semantic_backend_fallback_used"] is True
    assert trace["semantic_backend_error"] == "vlm_latency_budget_exceeded"
    assert trace["vlm_policy_trace"]["allowed_backend_key"] == "qwen"
    assert trace["vlm_policy_trace"]["budget"]["max_allowed_latency_seconds"] == 0.05


def test_memory_budget_rejects_vlm_and_falls_back_to_simple() -> None:
    fixture = load_fixture_message("tests/fixtures/frame_manual_review_required.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": PolicyTraceBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                trace={"runtime_after_inference_rss_mb": 9999.0},
            ),
            "simple": PolicyTraceBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
            ),
        }
    )

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen",
    ), patch.object(settings, "vlm_max_allowed_rss_mb", 128.0), patch.object(
        settings,
        "qwen_max_allowed_rss_mb",
        0.0,
    ):
        descriptor = service.generate(
            frame_ref=fixture.frame_ref,
            event_type_hint="manual_review_required",
            camera_id=fixture.camera_id,
        )

    trace = descriptor.descriptor["semantic_backend_trace"]
    attempts = trace["attempts"]

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen", "simple"]
    assert attempts[0]["status"] == "rejected_by_budget"
    assert attempts[0]["reason"] == "vlm_memory_budget_exceeded"
    assert attempts[0]["budget"]["observed_rss_mb"] == 9999.0
    assert attempts[-1]["status"] == "success"
    assert descriptor.descriptor["semantic_backend_allowed_key"] == "qwen"
    assert descriptor.descriptor["semantic_backend_fallback_used"] is True


def test_policy_trace_records_camera_event_and_final_backend() -> None:
    fixture = load_fixture_message("tests/fixtures/frame_manual_review_required.json")

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "auto",
    ):
        descriptor = _service().generate(
            frame_ref=fixture.frame_ref,
            event_type_hint="manual_review_required",
            camera_id=fixture.camera_id,
            camera_metadata={"site": "test"},
        )

    trace = descriptor.descriptor["semantic_backend_trace"]
    policy = trace["vlm_policy_trace"]

    assert descriptor.generated is True
    assert trace["event_type"] == "manual_review_required"
    assert trace["camera_id"] == fixture.camera_id
    assert trace["semantic_backend_requested"] == "auto"
    assert trace["semantic_backend_allowed_key"] == "qwen"
    assert trace["semantic_backend_selected_key"] == "qwen"
    assert policy["camera_id"] == fixture.camera_id
    assert policy["event_type"] == "manual_review_required"
    assert policy["budget"]["max_concurrent_inferences"] == 1
