from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.domain.entities import FaceDetectionResult


class SemanticBackendError(RuntimeError):
    pass


@dataclass
class SemanticBackendContext:
    frame_ref: str
    image_path: Path
    face_detection: FaceDetectionResult | None = None
    requested_backend: str | None = None
    timeout_seconds: int = 45
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticBackendOutput:
    backend_name: str
    descriptor: dict[str, Any]
    confidence: float | None = None
    raw_summary: str | None = None
    raw_response: Any | None = None


class SemanticDescriptorBackend(ABC):
    key: str
    backend_name: str
    supports_real_vlm: bool = False

    @abstractmethod
    def generate_descriptor(
        self,
        *,
        image_path: Path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        raise NotImplementedError


class TransformersImageTextSemanticBackend(SemanticDescriptorBackend):
    supports_real_vlm = True

    def __init__(self, *, key: str, model_name: str) -> None:
        self.key = key
        self.backend_name = model_name
        self._processor = None
        self._model = None
        self._device = "cpu"

    def generate_descriptor(
        self,
        *,
        image_path: Path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        prompt = self._build_prompt(context=context)
        raw_text = self._run_inference(image_path=image_path, prompt=prompt)
        parsed = self._extract_json_payload(raw_text)
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=parsed,
            confidence=self._coerce_confidence(parsed.get("descriptor_confidence")),
            raw_summary=str(parsed.get("raw_summary", "")).strip() or None,
            raw_response=self._truncate_response(raw_text),
        )

    def _run_inference(self, *, image_path: Path, prompt: str) -> str:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise SemanticBackendError("optional_vlm_dependencies_missing") from exc

        if self._processor is None or self._model is None:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.backend_name,
                    trust_remote_code=True,
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.backend_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                    self._device = "cuda"
            except Exception as exc:
                raise SemanticBackendError(f"model_load_failed:{type(exc).__name__}") from exc

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            if self._device != "cpu":
                inputs = {
                    key: value.to(self._device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            output_ids = self._model.generate(**inputs, max_new_tokens=320)
            raw_text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as exc:
            raise SemanticBackendError(f"model_inference_failed:{type(exc).__name__}") from exc

        return raw_text

    def _build_prompt(self, *, context: SemanticBackendContext) -> str:
        face_state = "unknown"
        if context.face_detection is not None:
            if context.face_detection.usable:
                face_state = "usable_face"
            elif context.face_detection.detected:
                face_state = "low_quality_face"
            else:
                face_state = "face_not_detected"

        return (
            "You are generating a structured surveillance appearance descriptor for a single visible human subject.\n"
            "Return JSON only. Do not include markdown, explanations, or any text outside JSON.\n"
            "Describe only directly observable visual cues.\n"
            "Do not infer identity, ethnicity, nationality, age, religion, disability, socioeconomic status, "
            "emotion, profession, or criminal intent.\n"
            "If a value is not clearly visible, use 'unknown' or an empty list.\n"
            "Schema:\n"
            "{\n"
            '  "subject_type": "person",\n'
            '  "top_clothing": {"category": "unknown", "color": "unknown", "pattern": "unknown"},\n'
            '  "bottom_clothing": {"category": "unknown", "color": "unknown", "pattern": "unknown"},\n'
            '  "dominant_colors": ["unknown"],\n'
            '  "accessories": [],\n'
            '  "carried_object": "unknown",\n'
            '  "body_build": "unknown",\n'
            '  "pose_direction": "unknown",\n'
            '  "scene_observation_quality": {"level": "unknown", "notes": "unknown"},\n'
            '  "descriptor_confidence": 0.0,\n'
            '  "raw_summary": "short neutral visual summary"\n'
            "}\n"
            f"Context: face_state={face_state}. Prefer neutral terms like upper_garment, pants, backpack, cap, front, left, right.\n"
            "Keep raw_summary to one short sentence."
        )

    def _extract_json_payload(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise SemanticBackendError("vlm_output_invalid_json") from exc
            if isinstance(parsed, dict):
                return parsed

        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            candidate = brace_match.group(1)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise SemanticBackendError("vlm_output_invalid_json") from exc
            if isinstance(parsed, dict):
                return parsed

        raise SemanticBackendError("vlm_output_missing_json_object")

    def _coerce_confidence(self, value: Any) -> float | None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _truncate_response(self, raw_text: str, *, limit: int = 500) -> str:
        compact = " ".join(raw_text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
