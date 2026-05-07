from __future__ import annotations

from pathlib import Path

import pytest

from app.services.semantic_backends import (
    QwenVLSemanticBackend,
    SemanticBackendContext,
    SemanticBackendError,
    VlmRuntimeResult,
)


class StubRunner:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text

    def generate_text(self, *, image_path: Path, prompt: str, timeout_seconds: int) -> VlmRuntimeResult:
        return VlmRuntimeResult(
            raw_text=self.raw_text,
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            device="mps",
            requested_device="auto",
            dtype_name="float16",
        )


def _context(image_path: Path) -> SemanticBackendContext:
    return SemanticBackendContext(
        frame_ref=str(image_path),
        image_path=image_path,
        timeout_seconds=10,
    )


def test_qwen_backend_recovers_fenced_json_and_exposes_parser_trace() -> None:
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        runner=StubRunner(
            """```json
            {
              "subject_type": "person",
              "top_clothing": {"category": "jacket", "color": "gray", "pattern": "solid"},
              "bottom_clothing": {"category": "pants", "color": "black", "pattern": "solid"},
              "dominant_colors": ["gray", "black"],
              "accessories": [],
              "carried_object": "unknown",
              "body_build": "average",
              "pose_direction": "front",
              "scene_observation_quality": {"level": "medium", "notes": "partial frame"},
              "descriptor_confidence": 0.74,
              "raw_summary": "person with gray jacket"
            }
            ```"""
        ),
    )

    output = backend.generate_descriptor(image_path=image_path, context=_context(image_path))

    assert output.descriptor["top_clothing"]["category"] == "jacket"
    assert output.trace["parse_strategy_used"] == "fenced_json"
    assert output.trace["json_recovered"] is True
    assert output.trace["raw_output_preview"].startswith("```json")
    assert output.trace["missing_fields"] == []


def test_qwen_backend_recovers_json_with_text_before_and_after() -> None:
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        runner=StubRunner(
            """
            The concise JSON descriptor is below.
            {"subject_type":"person","top_clothing":"blue hoodie","raw_summary":"person in blue hoodie"}
            Done.
            """
        ),
    )

    output = backend.generate_descriptor(image_path=image_path, context=_context(image_path))

    assert output.descriptor["top_clothing"]["category"] == "hoodie"
    assert output.descriptor["top_clothing"]["color"] == "blue"
    assert output.trace["parse_strategy_used"] == "extracted_json_object"
    assert output.trace["json_recovered"] is True
    assert "bottom_clothing" in output.trace["missing_fields"]


def test_qwen_backend_fails_explicitly_when_output_cannot_be_recovered() -> None:
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        runner=StubRunner("No structured JSON object is present in this answer."),
    )

    with pytest.raises(SemanticBackendError) as excinfo:
        backend.generate_descriptor(image_path=image_path, context=_context(image_path))

    assert excinfo.value.reason == "vlm_output_invalid_json"
    assert excinfo.value.details["parse_stage"] == "failed"
    assert excinfo.value.details["json_recovered"] is False
    assert excinfo.value.details["raw_output_chars"] > 0
