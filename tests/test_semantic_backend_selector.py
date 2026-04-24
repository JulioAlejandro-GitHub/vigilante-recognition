from __future__ import annotations

from unittest.mock import patch

from app.config import settings
from app.consumer import load_fixture_message
from app.services.semantic_backends.base import (
    SemanticBackendContext,
    SemanticBackendError,
    SemanticBackendOutput,
    SemanticDescriptorBackend,
)
from app.services.semantic_descriptor_service import SemanticDescriptorService


class StubSemanticBackend(SemanticDescriptorBackend):
    def __init__(
        self,
        *,
        key: str,
        backend_name: str,
        descriptor: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.key = key
        self.backend_name = backend_name
        self.descriptor = descriptor or {}
        self.error = error

    def generate_descriptor(
        self,
        *,
        image_path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        if self.error is not None:
            raise self.error
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=self.descriptor,
            confidence=self.descriptor.get("descriptor_confidence"),
            raw_summary=self.descriptor.get("raw_summary"),
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
        "descriptor_confidence": 0.8,
        "raw_summary": f"person wearing {color}",
    }


def test_selector_uses_simple_backend_when_real_vlm_is_disabled():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService()

    with patch.object(settings, "semantic_use_real_vlm", False), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen_vl",
    ):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert attempts[0]["backend_key"] == "qwen_vl"
    assert attempts[0]["status"] == "skipped"
    assert attempts[-1]["backend_key"] == "simple"
    assert attempts[-1]["status"] == "success"


def test_selector_falls_back_to_secondary_vlm_when_primary_backend_fails():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen_vl": StubSemanticBackend(
                key="qwen_vl",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                error=SemanticBackendError("model_load_failed:RuntimeError"),
            ),
            "smolvlm": StubSemanticBackend(
                key="smolvlm",
                backend_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                descriptor=_minimal_descriptor("gray"),
            ),
            "simple": StubSemanticBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
                descriptor=_minimal_descriptor("blue"),
            ),
        }
    )

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen_vl",
    ):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "success"
    assert attempts[1]["backend_key"] == "smolvlm"


def test_selector_falls_back_to_simple_when_all_real_backends_fail():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen_vl": StubSemanticBackend(
                key="qwen_vl",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                error=SemanticBackendError("model_load_failed:RuntimeError"),
            ),
            "smolvlm": StubSemanticBackend(
                key="smolvlm",
                backend_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                error=SemanticBackendError("vlm_output_invalid_json"),
            ),
            "simple": StubSemanticBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
                descriptor=_minimal_descriptor("gray"),
            ),
        }
    )

    with patch.object(settings, "semantic_use_real_vlm", True), patch.object(
        settings,
        "semantic_descriptor_backend",
        "qwen_vl",
    ):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen_vl", "smolvlm", "simple"]
    assert attempts[-1]["status"] == "success"
