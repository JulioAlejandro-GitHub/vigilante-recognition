from __future__ import annotations

from pathlib import Path

import pytest

from app.services.semantic_backends import (
    QwenVLSemanticBackend,
    SemanticBackendContext,
    SemanticBackendError,
    SmolVlmSemanticBackend,
    VlmRuntimeError,
    VlmRuntimeResult,
)


class StubRunner:
    def __init__(
        self,
        *,
        result: VlmRuntimeResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_text(self, *, image_path: Path, prompt: str, timeout_seconds: int) -> VlmRuntimeResult:
        self.calls.append(
            {
                "image_path": image_path,
                "prompt": prompt,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_qwen_backend_parses_json_and_exposes_runtime_trace():
    runner = StubRunner(
        result=VlmRuntimeResult(
            raw_text='```json {"subject_type":"person","top_clothing":{"category":"hoodie","color":"red","pattern":"solid"},"bottom_clothing":{"category":"jeans","color":"blue","pattern":"solid"},"dominant_colors":["red","blue"],"accessories":["backpack"],"carried_object":"unknown","body_build":"average","pose_direction":"left","scene_observation_quality":{"level":"medium","notes":"clear enough"},"descriptor_confidence":0.81,"raw_summary":"person with red hoodie"} ```',
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            device="mps",
            requested_device="auto",
            dtype_name="float16",
        )
    )
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device_preference="auto",
        runner=runner,
    )
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    context = SemanticBackendContext(
        frame_ref=str(image_path),
        image_path=image_path,
        timeout_seconds=12,
    )

    output = backend.generate_descriptor(image_path=image_path, context=context)

    assert output.descriptor["top_clothing"]["category"] == "hoodie"
    assert output.confidence == 0.81
    assert output.trace["device"] == "mps"
    assert output.trace["runtime"] == "isolated_subprocess"
    assert runner.calls[0]["timeout_seconds"] == 12
    assert "Do not infer identity or sensitive attributes" in str(runner.calls[0]["prompt"])


def test_qwen_backend_surfaces_timeout_reason_from_runner():
    runner = StubRunner(
        error=VlmRuntimeError(
            "backend_timeout",
            details={"stage": "runtime", "timeout_seconds": 9},
        )
    )
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device_preference="auto",
        runner=runner,
    )
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    context = SemanticBackendContext(
        frame_ref=str(image_path),
        image_path=image_path,
        timeout_seconds=9,
    )

    with pytest.raises(SemanticBackendError, match="backend_timeout") as excinfo:
        backend.generate_descriptor(image_path=image_path, context=context)

    assert excinfo.value.details["stage"] == "runtime"
    assert excinfo.value.details["timeout_seconds"] == 9


def test_smolvlm_backend_parses_json_and_exposes_runtime_trace():
    runner = StubRunner(
        result=VlmRuntimeResult(
            raw_text='{"subject_type":"person","top_clothing":{"category":"jacket","color":"gray","pattern":"solid"},"bottom_clothing":{"category":"pants","color":"black","pattern":"solid"},"dominant_colors":["gray","black"],"accessories":[],"carried_object":"unknown","body_build":"average","pose_direction":"front","scene_observation_quality":{"level":"medium","notes":"partial frame"},"descriptor_confidence":0.74,"raw_summary":"person with gray jacket"}',
            model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            device="cpu",
            requested_device="cpu",
            dtype_name="float32",
            extra={
                "max_new_tokens": 96,
                "max_image_edge": 512,
                "image_resized": False,
            },
        )
    )
    backend = SmolVlmSemanticBackend(
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device_preference="cpu",
        runner=runner,
    )
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    context = SemanticBackendContext(
        frame_ref=str(image_path),
        image_path=image_path,
        timeout_seconds=7,
        max_new_tokens=96,
        max_image_edge=512,
    )

    output = backend.generate_descriptor(image_path=image_path, context=context)

    assert output.descriptor["top_clothing"]["category"] == "jacket"
    assert output.confidence == 0.74
    assert output.trace["model_name"] == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    assert output.trace["device"] == "cpu"
    assert output.trace["prompt_policy_version"] == "forensic_observation_json_v1"
    assert output.trace["max_new_tokens"] == 96
    assert runner.calls[0]["timeout_seconds"] == 7


def test_transformers_backend_reuses_runner_instance_between_calls():
    runner = StubRunner(
        result=VlmRuntimeResult(
            raw_text='{"subject_type":"person","top_clothing":{"category":"shirt","color":"white","pattern":"solid"},"bottom_clothing":{"category":"pants","color":"black","pattern":"solid"},"dominant_colors":["white","black"],"accessories":[],"carried_object":"unknown","body_build":"average","pose_direction":"front","scene_observation_quality":{"level":"medium","notes":"stable"},"descriptor_confidence":0.7,"raw_summary":"person with white shirt"}',
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            requested_device="cpu",
            dtype_name="float32",
        )
    )
    backend = QwenVLSemanticBackend(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device_preference="cpu",
        runner=runner,
    )
    image_path = Path("tests/fixtures/images/face_low_quality.jpg")
    context = SemanticBackendContext(
        frame_ref=str(image_path),
        image_path=image_path,
        timeout_seconds=5,
    )

    backend.generate_descriptor(image_path=image_path, context=context)
    backend.generate_descriptor(image_path=image_path, context=context)

    assert len(runner.calls) == 2
    assert runner.calls[0]["timeout_seconds"] == runner.calls[1]["timeout_seconds"] == 5
