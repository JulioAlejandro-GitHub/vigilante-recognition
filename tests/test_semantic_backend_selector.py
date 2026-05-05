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
        "qwen",
    ):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert attempts[0]["backend_key"] == "qwen"
    assert attempts[0]["status"] == "skipped"
    assert attempts[0]["reason"] == "qwen_vl_disabled"
    assert attempts[-1]["backend_key"] == "simple"
    assert attempts[-1]["status"] == "success"
    assert descriptor.descriptor["semantic_backend_requested"] == "qwen"
    assert descriptor.descriptor["semantic_backend_selected"] == "simple_color_signature_v1"
    assert descriptor.descriptor["semantic_backend_fallback_used"] is True


def test_auto_selector_falls_back_to_secondary_vlm_when_primary_backend_fails():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
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
        "auto",
    ), patch.object(settings, "vlm_auto_preferred_backend", "qwen"):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "success"
    assert attempts[1]["backend_key"] == "smolvlm"
    assert descriptor.descriptor["semantic_backend_fallback_used"] is True
    assert descriptor.descriptor["semantic_backend_error"] == "model_load_failed:RuntimeError"


def test_auto_selector_falls_back_to_simple_when_all_real_backends_fail():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
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
        "auto",
    ), patch.object(settings, "vlm_auto_preferred_backend", "qwen"):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen", "smolvlm", "simple"]
    assert attempts[-1]["status"] == "success"
    assert descriptor.descriptor["semantic_backend_error"] == "vlm_output_invalid_json"


def test_auto_selector_records_timeout_details_and_uses_secondary_vlm():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                error=SemanticBackendError(
                    "backend_timeout",
                    details={"stage": "runtime", "timeout_seconds": 3},
                ),
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
        "auto",
    ), patch.object(settings, "vlm_auto_preferred_backend", "qwen"):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert attempts[0]["reason"] == "backend_timeout"
    assert attempts[0]["stage"] == "runtime"
    assert attempts[0]["timeout_seconds"] == 3
    assert attempts[1]["status"] == "success"


def test_forced_qwen_degrades_directly_to_simple_when_fallback_is_enabled():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
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
        "qwen",
    ), patch.object(settings, "semantic_enable_fallback", True):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen", "simple"]
    assert descriptor.descriptor["semantic_backend_fallback_used"] is True


def test_forced_qwen_failure_is_explicit_when_fallback_is_disabled():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                error=SemanticBackendError("model_load_failed:RuntimeError"),
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
        "qwen",
    ), patch.object(settings, "semantic_enable_fallback", False):
        descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is False
    assert descriptor.rejection_reasons == ["model_load_failed:RuntimeError"]
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert [attempt["backend_key"] for attempt in attempts] == ["qwen"]
    assert descriptor.descriptor["semantic_backend_error"] == "model_load_failed:RuntimeError"


def test_vlm_is_skipped_when_event_type_policy_does_not_allow_context():
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen": StubSemanticBackend(
                key="qwen",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                descriptor=_minimal_descriptor("red"),
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
        "auto",
    ), patch.object(settings, "vlm_auto_preferred_backend", "qwen"), patch.object(
        settings,
        "vlm_enable_for_event_types",
        "manual_review_required",
    ):
        descriptor = service.generate(
            frame_ref=fixture.frame_ref,
            event_type_hint="human_presence_no_face",
        )

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    attempts = descriptor.descriptor["generation_trace"]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["skipped", "skipped", "success"]
    assert attempts[0]["reason"] == "vlm_event_type_not_enabled:human_presence_no_face"
    assert descriptor.descriptor["semantic_backend_trace"]["semantic_backend_activation_allowed"] is False
