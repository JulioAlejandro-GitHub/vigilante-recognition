from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.config import settings
from app.consumer import load_fixture_message
from app.services.semantic_backends.base import (
    SemanticBackendContext,
    SemanticBackendOutput,
    SemanticDescriptorBackend,
)
from app.services.semantic_descriptor_service import SemanticDescriptorService
from app.services.vlm_execution_policy_service import validate_vlm_runtime_config


class TraceSemanticBackend(SemanticDescriptorBackend):
    def __init__(self, *, key: str, backend_name: str) -> None:
        self.key = key
        self.backend_name = backend_name

    def generate_descriptor(
        self,
        *,
        image_path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor={
                "subject_type": "person",
                "top_clothing": {
                    "category": "jacket",
                    "color": "gray",
                    "pattern": "solid",
                },
                "bottom_clothing": {
                    "category": "pants",
                    "color": "blue",
                    "pattern": "solid",
                },
                "dominant_colors": ["gray", "blue"],
                "accessories": [],
                "carried_object": "unknown",
                "body_build": "average",
                "pose_direction": "front",
                "scene_observation_quality": {"level": "medium", "notes": "test"},
                "descriptor_confidence": 0.76,
                "raw_summary": "person with gray jacket",
            },
            confidence=0.76,
            raw_summary="person with gray jacket",
            raw_response='{"subject_type":"person"}',
            trace={
                "model_name": self.backend_name,
                "device": "mps",
                "requested_device": "auto",
                "dtype": "float16",
                "runtime": "isolated_subprocess",
                "raw_output_chars": 25,
                "image_original_size": {"width": 320, "height": 240},
                "image_inference_size": {"width": 256, "height": 192},
                "image_resized": True,
                "model_load_elapsed_ms": 1000,
                "runtime_inference_elapsed_ms": 220,
                "runtime_after_inference_rss_mb": 2048.0,
            },
        )


def test_vlm_runtime_config_defaults_are_safe_and_operational() -> None:
    validation = validate_vlm_runtime_config()

    assert validation["valid"] is True
    assert validation["policy"]["global_backend_default"] == "simple"
    assert validation["policy"]["auto_preferred_backend"] == "qwen"
    assert validation["policy"]["requested_device"] == "auto"
    assert validation["policy"]["timeout_seconds"] == 60
    assert validation["policy"]["max_new_tokens"] == 192
    assert validation["policy"]["max_image_edge"] == 384
    assert validation["policy"]["max_allowed_rss_mb"] == 8192.0
    assert validation["policy"]["qwen_max_allowed_rss_mb"] == 12288.0
    assert validation["policy"]["smolvlm_max_allowed_rss_mb"] == 10240.0
    assert validation["policy"]["serialization_guard_enabled"] is True


def test_requirements_vlm_declares_real_runtime_dependencies() -> None:
    requirements = Path("requirements-vlm.txt").read_text(encoding="utf-8")

    for package in [
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "num2words",
        "psutil",
    ]:
        assert package in requirements


def test_real_backend_trace_contains_operational_metrics_and_contract_fields() -> None:
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": TraceSemanticBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
            ),
            "simple": TraceSemanticBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
            ),
        }
    )

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen",
    ), patch.object(settings, "vlm_timeout_seconds", 13), patch.object(
        settings,
        "vlm_max_new_tokens",
        64,
    ), patch.object(
        settings,
        "vlm_max_image_edge",
        384,
    ), patch.object(
        settings,
        "vlm_device",
        "auto",
    ):
        descriptor = service.generate(
            frame_ref=fixture.frame_ref,
            source_frame_ref="s3://vigilante-frames/frames/cam01/test.jpg",
            event_type_hint="human_presence_no_face",
        )

    trace = descriptor.descriptor["semantic_backend_trace"]
    attempt = trace["attempts"][0]

    assert descriptor.generated is True
    assert descriptor.source_frame_ref == "s3://vigilante-frames/frames/cam01/test.jpg"
    assert descriptor.descriptor["source_frame_ref"] == "s3://vigilante-frames/frames/cam01/test.jpg"
    assert trace["execution_policy"]["serialization_guard"] == "single_inflight_request_per_backend_subprocess"
    assert trace["timeout_applied_seconds"] == 13
    assert trace["descriptor_valid"] is True
    assert trace["total_duration_ms"] >= 0
    assert attempt["status"] == "success"
    assert attempt["timeout_applied_seconds"] == 13
    assert attempt["max_new_tokens"] == 64
    assert attempt["max_image_edge"] == 384
    assert attempt["device"] == "mps"
    assert attempt["image_inference_size"] == {"width": 256, "height": 192}
    assert attempt["descriptor_valid"] is True
    assert attempt["descriptor_output_chars"] > 0
    assert attempt["raw_output_chars"] == 25
