from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.domain.entities import FaceDetectionResult
from app.services.semantic_backends.model_loader import (
    ProcessIsolatedTransformersRunner,
    VlmRuntimeError,
)


class SemanticBackendError(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


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
    trace: dict[str, Any] = field(default_factory=dict)


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

    def __init__(
        self,
        *,
        key: str,
        model_name: str,
        device_preference: str = "auto",
        runner: Any | None = None,
        max_new_tokens: int = 320,
    ) -> None:
        self.key = key
        self.backend_name = model_name
        self._runner = runner or ProcessIsolatedTransformersRunner(
            backend_key=key,
            model_name=model_name,
            device_preference=device_preference,
            max_new_tokens=max_new_tokens,
        )

    def generate_descriptor(
        self,
        *,
        image_path: Path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        prompt = self._build_prompt(context=context)
        try:
            runtime_result = self._runner.generate_text(
                image_path=image_path,
                prompt=prompt,
                timeout_seconds=context.timeout_seconds,
            )
        except VlmRuntimeError as exc:
            raise SemanticBackendError(exc.reason, details=exc.details) from exc

        raw_text = runtime_result.raw_text
        try:
            parsed = self._extract_json_payload(raw_text)
        except SemanticBackendError as exc:
            raise SemanticBackendError(
                exc.reason,
                details={**runtime_result.trace_payload(), **exc.details},
            ) from exc

        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=parsed,
            confidence=self._coerce_confidence(parsed.get("descriptor_confidence")),
            raw_summary=str(parsed.get("raw_summary", "")).strip() or None,
            raw_response=self._truncate_response(raw_text),
            trace=runtime_result.trace_payload(),
        )

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
            "Analyze one visible human subject in the image and return JSON only.\n"
            "Describe direct visual observables only.\n"
            "Do not infer identity or sensitive attributes such as ethnicity, nationality, age, religion, disability, socioeconomic status, profession, emotion, or intent.\n"
            "Use concise neutral labels and keep unknown values as 'unknown' or [].\n"
            "Keep raw_summary to one short neutral sentence.\n"
            "Return exactly this JSON shape:\n"
            "{\n"
            '  "subject_type": "person",\n'
            '  "top_clothing": {"category": "unknown", "color": "unknown", "pattern": "unknown"},\n'
            '  "bottom_clothing": {"category": "unknown", "color": "unknown", "pattern": "unknown"},\n'
            '  "dominant_colors": [],\n'
            '  "accessories": [],\n'
            '  "carried_object": "unknown",\n'
            '  "body_build": "unknown",\n'
            '  "pose_direction": "unknown",\n'
            '  "scene_observation_quality": {"level": "unknown", "notes": "unknown"},\n'
            '  "descriptor_confidence": 0.0,\n'
            '  "raw_summary": "short neutral visual summary"\n'
            "}\n"
            f"Context: face_state={face_state}. Prefer generic clothing labels such as upper_garment, jacket, hoodie, shirt, pants, jeans, skirt, backpack, cap, front, left, right."
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
