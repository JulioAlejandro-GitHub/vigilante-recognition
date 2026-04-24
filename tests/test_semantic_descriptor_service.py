import json
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


class StubSemanticBackend(SemanticDescriptorBackend):
    def __init__(self, *, key: str, backend_name: str, descriptor: dict):
        self.key = key
        self.backend_name = backend_name
        self.descriptor = descriptor

    def generate_descriptor(
        self,
        *,
        image_path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=self.descriptor,
            confidence=self.descriptor.get("descriptor_confidence"),
            raw_summary=self.descriptor.get("raw_summary"),
        )


def test_semantic_descriptor_generation_for_low_quality_fixture():
    service = SemanticDescriptorService()
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")

    descriptor = service.generate(frame_ref=fixture.frame_ref)

    assert descriptor.generated is True
    assert descriptor.backend == "simple_color_signature_v1"
    assert descriptor.descriptor["appearance"]["dominant_palette"]
    assert descriptor.descriptor["silhouette"]["frame_aspect_ratio"] in {"portrait", "square", "landscape"}
    assert descriptor.descriptor["descriptor_schema_version"] == "semantic_descriptor_v2"
    assert descriptor.descriptor["descriptor_backend"] == "simple_color_signature_v1"
    assert descriptor.descriptor["scene_observation_quality"]["level"] in {"low", "medium", "high"}
    assert descriptor.source_frame_ref.endswith("face_low_quality.jpg")


def test_semantic_descriptor_similarity_prefers_same_image_signature():
    service = SemanticDescriptorService()
    no_face_fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    recurrent_fixture = load_fixture_message("tests/fixtures/frame_recurrent_unresolved.json")
    identified_fixture = load_fixture_message("tests/fixtures/frame_ingested_identified.json")

    no_face_descriptor = service.generate(frame_ref=no_face_fixture.frame_ref)
    recurrent_descriptor = service.generate(frame_ref=recurrent_fixture.frame_ref)
    identified_descriptor = service.generate(frame_ref=identified_fixture.frame_ref)

    same_signature_similarity = service.compare(no_face_descriptor, recurrent_descriptor)
    different_signature_similarity = service.compare(no_face_descriptor, identified_descriptor)

    assert same_signature_similarity >= 0.95
    assert same_signature_similarity > different_signature_similarity


def test_semantic_descriptor_normalizes_vlm_style_output():
    raw_descriptor = json.loads(
        Path("tests/fixtures/semantic_vlm_raw_response.json").read_text(encoding="utf-8")
    )
    fixture = load_fixture_message("tests/fixtures/frame_ingested_no_face.json")
    service = SemanticDescriptorService(
        backends={
            "qwen_vl": StubSemanticBackend(
                key="qwen_vl",
                backend_name="Qwen/Qwen2.5-VL-3B-Instruct",
                descriptor=raw_descriptor,
            ),
            "smolvlm": StubSemanticBackend(
                key="smolvlm",
                backend_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                descriptor=raw_descriptor,
            ),
            "simple": StubSemanticBackend(
                key="simple",
                backend_name="simple_color_signature_v1",
                descriptor=raw_descriptor,
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
    assert descriptor.backend == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert descriptor.descriptor["top_clothing"]["category"] == "hoodie"
    assert descriptor.descriptor["top_clothing"]["color"] == "red"
    assert descriptor.descriptor["bottom_clothing"]["category"] == "jeans"
    assert descriptor.descriptor["bottom_clothing"]["color"] == "blue"
    assert descriptor.descriptor["dominant_colors"] == ["red", "blue"]
    assert descriptor.descriptor["accessories"] == ["backpack", "cap"]
    assert descriptor.descriptor["pose_direction"] == "left"
    assert descriptor.descriptor["scene_observation_quality"]["level"] == "medium"
    assert descriptor.descriptor["descriptor_confidence"] == 0.77
