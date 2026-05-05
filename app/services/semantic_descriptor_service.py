from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import time
from typing import Any

from app.config import settings
from app.domain.entities import FaceDetectionResult, SemanticDescriptorResult
from app.services.semantic_backends import (
    QwenVLSemanticBackend,
    SemanticBackendContext,
    SemanticBackendError,
    SemanticDescriptorBackend,
    SimpleSemanticDescriptorBackend,
    SmolVlmSemanticBackend,
)


class SemanticDescriptorService:
    SIMPLE_BACKEND_KEY = "simple"
    QWEN_BACKEND_KEY = "qwen"
    SMOLVLM_BACKEND_KEY = "smolvlm"
    AUTO_BACKEND_KEY = "auto"
    REAL_BACKEND_KEYS = {QWEN_BACKEND_KEY, "qwen_vl", SMOLVLM_BACKEND_KEY}
    TRACE_VERSION = "semantic_backend_trace_v1"
    ACTIVATION_POLICY_VERSION = "semantic_vlm_activation_policy_v1"

    def __init__(
        self,
        *,
        backends: dict[str, SemanticDescriptorBackend] | None = None,
    ) -> None:
        if backends is not None:
            self.backends = backends
            return

        qwen_backend = QwenVLSemanticBackend()
        self.backends = {
            self.SIMPLE_BACKEND_KEY: SimpleSemanticDescriptorBackend(),
            self.QWEN_BACKEND_KEY: qwen_backend,
            "qwen_vl": qwen_backend,
            self.SMOLVLM_BACKEND_KEY: SmolVlmSemanticBackend(),
        }

    def generate(
        self,
        *,
        frame_ref: str,
        source_frame_ref: str | None = None,
        face_detection: FaceDetectionResult | None = None,
        event_type_hint: str | None = None,
    ) -> SemanticDescriptorResult:
        frame_path = self._resolve_frame_path(frame_ref)
        published_source_frame_ref = frame_ref if source_frame_ref is None else source_frame_ref
        requested_backend = settings.semantic_descriptor_backend
        requested_key = self._normalize_backend_key(requested_backend)
        activation = self._vlm_activation_decision(
            event_type_hint=event_type_hint,
            face_detection=face_detection,
        )
        backend_chain = self._build_backend_chain(requested_backend=requested_backend)
        generation_trace: dict[str, Any] = {
            "trace_version": self.TRACE_VERSION,
            "policy_version": self.ACTIVATION_POLICY_VERSION,
            "prompt_policy_version": "forensic_observation_json_v1",
            "semantic_backend_requested": requested_backend,
            "semantic_backend_normalized": requested_key,
            "semantic_backend_selected": None,
            "semantic_backend_selected_key": None,
            "semantic_backend_fallback_used": False,
            "semantic_backend_error": None,
            "semantic_backend_candidate_chain": backend_chain,
            "semantic_backend_activation_allowed": activation["allowed"],
            "semantic_backend_activation_reason": activation["reason"],
            "semantic_backend_event_type_hint": event_type_hint,
            "semantic_backend_enabled_event_types": activation["enabled_event_types"],
            "timeout_seconds": settings.effective_vlm_timeout_seconds,
            "max_new_tokens": settings.vlm_max_new_tokens,
            "max_image_edge": settings.vlm_max_image_edge,
            "requested_device": settings.effective_vlm_device,
            "requested_backend": requested_backend,
            "fallback_enabled": settings.semantic_enable_fallback,
            "selected_backend": None,
            "selected_backend_key": None,
            "attempts": [],
        }
        if frame_path is None:
            generation_trace["semantic_backend_error"] = "frame_ref_not_found"
            return SemanticDescriptorResult(
                backend=self._backend_name_for(self.SIMPLE_BACKEND_KEY),
                source_frame_ref=published_source_frame_ref,
                rejection_reasons=["frame_ref_not_found"],
                descriptor={
                    "semantic_backend_requested": requested_backend,
                    "semantic_backend_selected": None,
                    "semantic_backend_fallback_used": False,
                    "semantic_backend_error": "frame_ref_not_found",
                    "semantic_backend_trace": generation_trace,
                    "generation_trace": generation_trace,
                },
            )

        rejection_reasons: list[str] = []
        for backend_key in backend_chain:
            backend = self._get_backend(backend_key)
            if backend is None:
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend_key,
                        "status": "skipped",
                        "reason": "backend_not_registered",
                    }
                )
                continue

            if backend_key in self.REAL_BACKEND_KEYS:
                enabled, disabled_reason = self._real_backend_enabled(backend_key)
                if not enabled:
                    generation_trace["attempts"].append(
                        {
                            "backend_key": backend_key,
                            "backend_name": backend.backend_name,
                            "status": "skipped",
                            "reason": disabled_reason,
                            "model_name": self._model_name_for(backend_key),
                        }
                    )
                    continue

            if backend_key in self.REAL_BACKEND_KEYS and not activation["allowed"]:
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "skipped",
                        "reason": activation["reason"],
                        "model_name": self._model_name_for(backend_key),
                        "event_type_hint": event_type_hint,
                    }
                )
                continue

            context = SemanticBackendContext(
                frame_ref=frame_ref,
                image_path=frame_path,
                face_detection=face_detection,
                requested_backend=requested_backend,
                timeout_seconds=settings.effective_vlm_timeout_seconds,
                max_new_tokens=settings.vlm_max_new_tokens,
                max_image_edge=settings.vlm_max_image_edge,
                event_type_hint=event_type_hint,
            )
            attempt_started_at = time.perf_counter()
            try:
                backend_output = backend.generate_descriptor(image_path=frame_path, context=context)
                duration_ms = self._duration_ms(attempt_started_at)
                descriptor, signature, confidence = self._normalize_descriptor(
                    raw_descriptor=backend_output.descriptor,
                    backend_name=backend_output.backend_name,
                    frame_ref=published_source_frame_ref,
                    face_detection=face_detection,
                    confidence_override=backend_output.confidence,
                    raw_summary=backend_output.raw_summary,
                )
                generation_trace["semantic_backend_selected"] = backend_output.backend_name
                generation_trace["semantic_backend_selected_key"] = backend_key
                generation_trace["selected_backend"] = backend_output.backend_name
                generation_trace["selected_backend_key"] = backend_key
                generation_trace["attempts"].append(
                    {
                        **backend_output.trace,
                        "backend_key": backend_key,
                        "backend_name": backend_output.backend_name,
                        "status": "success",
                        "duration_ms": duration_ms,
                        "timeout_seconds": context.timeout_seconds,
                        "max_new_tokens": context.max_new_tokens,
                        "max_image_edge": context.max_image_edge,
                    }
                )
                fallback_used = self._fallback_used(
                    requested_key=requested_key,
                    selected_key=backend_key,
                    attempts=generation_trace["attempts"],
                )
                generation_trace["semantic_backend_fallback_used"] = fallback_used
                generation_trace["semantic_backend_error"] = self._last_failed_reason(
                    generation_trace["attempts"]
                )
                descriptor = self._attach_backend_trace(
                    descriptor=descriptor,
                    generation_trace=generation_trace,
                    requested_backend=requested_backend,
                    selected_backend=backend_output.backend_name,
                    fallback_used=fallback_used,
                )
                if backend_output.raw_response is not None:
                    descriptor["backend_response_preview"] = backend_output.raw_response
                return SemanticDescriptorResult(
                    generated=True,
                    backend=backend_output.backend_name,
                    descriptor=descriptor,
                    signature=signature,
                    confidence=confidence,
                    source_frame_ref=published_source_frame_ref,
                )
            except SemanticBackendError as exc:
                duration_ms = self._duration_ms(attempt_started_at)
                rejection_reasons.append(str(exc))
                failure_details = dict(exc.details)
                failure_details.setdefault("timeout_seconds", context.timeout_seconds)
                generation_trace["attempts"].append(
                    {
                        **failure_details,
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "failed",
                        "reason": str(exc),
                        "duration_ms": duration_ms,
                        "timeout_applied_seconds": context.timeout_seconds,
                        "max_new_tokens": context.max_new_tokens,
                        "max_image_edge": context.max_image_edge,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive path
                duration_ms = self._duration_ms(attempt_started_at)
                reason = f"unexpected_backend_error:{backend_key}:{type(exc).__name__}"
                rejection_reasons.append(reason)
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "failed",
                        "reason": reason,
                        "duration_ms": duration_ms,
                        "timeout_seconds": context.timeout_seconds,
                        "max_new_tokens": context.max_new_tokens,
                        "max_image_edge": context.max_image_edge,
                    }
                )

        generation_trace["semantic_backend_error"] = self._last_failed_reason(
            generation_trace["attempts"]
        )
        return SemanticDescriptorResult(
            backend=self._backend_name_for(self.SIMPLE_BACKEND_KEY),
            source_frame_ref=published_source_frame_ref,
            rejection_reasons=rejection_reasons or ["semantic_descriptor_unavailable"],
            descriptor={
                "semantic_backend_requested": requested_backend,
                "semantic_backend_selected": None,
                "semantic_backend_fallback_used": bool(rejection_reasons),
                "semantic_backend_error": generation_trace["semantic_backend_error"],
                "semantic_backend_trace": generation_trace,
                "generation_trace": generation_trace,
            },
        )

    def compare(
        self,
        current_descriptor: SemanticDescriptorResult | None,
        candidate_descriptor: SemanticDescriptorResult | None,
    ) -> float:
        if (
            current_descriptor is None
            or candidate_descriptor is None
            or not current_descriptor.generated
            or not candidate_descriptor.generated
        ):
            return 0.0

        current_signature = current_descriptor.signature
        candidate_signature = candidate_descriptor.signature
        if not current_signature or not candidate_signature:
            return 0.0

        score = 0.0
        score += 0.18 if current_signature.get("upper_region_color") == candidate_signature.get("upper_region_color") else 0.0
        score += 0.18 if current_signature.get("middle_region_color") == candidate_signature.get("middle_region_color") else 0.0
        score += 0.12 if current_signature.get("lower_region_color") == candidate_signature.get("lower_region_color") else 0.0
        score += 0.1 if current_signature.get("contrast_level") == candidate_signature.get("contrast_level") else 0.0
        score += 0.08 if current_signature.get("saturation_level") == candidate_signature.get("saturation_level") else 0.0
        score += 0.08 if current_signature.get("frame_aspect_ratio") == candidate_signature.get("frame_aspect_ratio") else 0.0
        score += 0.08 if current_signature.get("subject_scale") == candidate_signature.get("subject_scale") else 0.0
        score += 0.06 if current_signature.get("horizontal_position") == candidate_signature.get("horizontal_position") else 0.0
        score += 0.05 if current_signature.get("face_state") == candidate_signature.get("face_state") else 0.0

        current_palette = set(current_signature.get("dominant_palette", []))
        candidate_palette = set(candidate_signature.get("dominant_palette", []))
        if current_palette and candidate_palette:
            score += 0.15 * (len(current_palette & candidate_palette) / len(current_palette | candidate_palette))

        return round(min(1.0, score), 4)

    def _build_backend_chain(self, *, requested_backend: str | None) -> list[str]:
        requested_key = self._normalize_backend_key(requested_backend)
        if requested_key == self.SIMPLE_BACKEND_KEY:
            return [self.SIMPLE_BACKEND_KEY]

        if requested_key == self.AUTO_BACKEND_KEY:
            preferred_key = self._normalize_auto_preferred_backend(
                settings.vlm_auto_preferred_backend
            )
            secondary_key = (
                self.SMOLVLM_BACKEND_KEY
                if preferred_key == self.QWEN_BACKEND_KEY
                else self.QWEN_BACKEND_KEY
            )
            raw_chain: list[str | None] = [preferred_key]
            if settings.semantic_enable_fallback:
                raw_chain.extend([secondary_key, self.SIMPLE_BACKEND_KEY])
        elif requested_key in {self.QWEN_BACKEND_KEY, self.SMOLVLM_BACKEND_KEY}:
            raw_chain = [requested_key]
            if settings.semantic_enable_fallback:
                raw_chain.append(self.SIMPLE_BACKEND_KEY)
        else:
            raw_chain = [self.SIMPLE_BACKEND_KEY]

        chain: list[str] = []
        for backend_key in raw_chain:
            if backend_key and backend_key not in chain:
                chain.append(backend_key)
        return chain

    def _normalize_backend_key(self, requested_backend: str | None) -> str:
        normalized = (requested_backend or "").strip().lower()
        alias_map = {
            "auto": self.AUTO_BACKEND_KEY,
            "simple": self.SIMPLE_BACKEND_KEY,
            "simple_color_signature_v1": self.SIMPLE_BACKEND_KEY,
            "qwen": self.QWEN_BACKEND_KEY,
            "qwen_vl": self.QWEN_BACKEND_KEY,
            "qwen-vl": self.QWEN_BACKEND_KEY,
            "qwen/qwen2.5-vl-3b-instruct": self.QWEN_BACKEND_KEY,
            "qwen2.5-vl-3b-instruct": self.QWEN_BACKEND_KEY,
            "smolvlm": self.SMOLVLM_BACKEND_KEY,
            "smol_vlm": self.SMOLVLM_BACKEND_KEY,
            "smol-vlm": self.SMOLVLM_BACKEND_KEY,
            "huggingfacetb/smolvlm2-2.2b-instruct": self.SMOLVLM_BACKEND_KEY,
            "smolvlm2-2.2b-instruct": self.SMOLVLM_BACKEND_KEY,
        }
        return alias_map.get(normalized, self.SIMPLE_BACKEND_KEY)

    def _normalize_auto_preferred_backend(self, requested_backend: str | None) -> str:
        normalized = self._normalize_backend_key(requested_backend)
        if normalized in {self.QWEN_BACKEND_KEY, self.SMOLVLM_BACKEND_KEY}:
            return normalized
        return self.QWEN_BACKEND_KEY

    def _backend_name_for(self, backend_key: str) -> str:
        backend = self._get_backend(backend_key)
        if backend is None:
            return backend_key
        return backend.backend_name

    def _get_backend(self, backend_key: str) -> SemanticDescriptorBackend | None:
        backend = self.backends.get(backend_key)
        if backend is not None:
            return backend
        if backend_key == self.QWEN_BACKEND_KEY:
            return self.backends.get("qwen_vl")
        if backend_key == "qwen_vl":
            return self.backends.get(self.QWEN_BACKEND_KEY)
        return None

    def _real_backend_enabled(self, backend_key: str) -> tuple[bool, str | None]:
        normalized = self._normalize_backend_key(backend_key)
        if normalized == self.QWEN_BACKEND_KEY:
            if settings.effective_qwen_vl_enabled:
                return True, None
            return False, "qwen_vl_disabled"
        if normalized == self.SMOLVLM_BACKEND_KEY:
            if settings.effective_smolvlm_enabled:
                return True, None
            return False, "smolvlm_disabled"
        return True, None

    def _model_name_for(self, backend_key: str) -> str | None:
        normalized = self._normalize_backend_key(backend_key)
        if normalized == self.QWEN_BACKEND_KEY:
            return settings.effective_qwen_model_name
        if normalized == self.SMOLVLM_BACKEND_KEY:
            return settings.effective_smolvlm_model_name
        return None

    def _vlm_activation_decision(
        self,
        *,
        event_type_hint: str | None,
        face_detection: FaceDetectionResult | None,
    ) -> dict[str, Any]:
        enabled_event_types = settings.vlm_event_type_policy
        if not enabled_event_types:
            return {
                "allowed": False,
                "reason": "vlm_event_policy_empty",
                "enabled_event_types": [],
            }

        normalized_event_type = self._normalize_label(event_type_hint, default="")
        if "*" in enabled_event_types or "all" in enabled_event_types:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_all",
                "enabled_event_types": enabled_event_types,
            }
        if not normalized_event_type:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_no_event_type_hint",
                "enabled_event_types": enabled_event_types,
            }
        if normalized_event_type in enabled_event_types:
            return {
                "allowed": True,
                "reason": f"vlm_event_type_enabled:{normalized_event_type}",
                "enabled_event_types": enabled_event_types,
            }
        if "person_detected" in enabled_event_types and face_detection is not None:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_person_detected",
                "enabled_event_types": enabled_event_types,
            }
        if "face_detected" in enabled_event_types and face_detection and face_detection.detected:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_face_detected",
                "enabled_event_types": enabled_event_types,
            }
        if "face_usable" in enabled_event_types and face_detection and face_detection.usable:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_face_usable",
                "enabled_event_types": enabled_event_types,
            }
        if "no_face" in enabled_event_types and face_detection and not face_detection.usable:
            return {
                "allowed": True,
                "reason": "vlm_event_policy_no_face",
                "enabled_event_types": enabled_event_types,
            }
        return {
            "allowed": False,
            "reason": f"vlm_event_type_not_enabled:{normalized_event_type}",
            "enabled_event_types": enabled_event_types,
        }

    def _duration_ms(self, started_at: float) -> int:
        return int(round((time.perf_counter() - started_at) * 1000))

    def _fallback_used(
        self,
        *,
        requested_key: str,
        selected_key: str,
        attempts: list[dict[str, Any]],
    ) -> bool:
        if selected_key == self.SIMPLE_BACKEND_KEY and requested_key != self.SIMPLE_BACKEND_KEY:
            return True
        if requested_key in {self.QWEN_BACKEND_KEY, self.SMOLVLM_BACKEND_KEY}:
            return selected_key != requested_key
        return any(
            attempt.get("status") in {"failed", "skipped"}
            and attempt.get("backend_key") in self.REAL_BACKEND_KEYS
            for attempt in attempts
        )

    def _last_failed_reason(self, attempts: list[dict[str, Any]]) -> str | None:
        for attempt in reversed(attempts):
            if attempt.get("status") == "failed" and attempt.get("reason"):
                return str(attempt["reason"])
        return None

    def _attach_backend_trace(
        self,
        *,
        descriptor: dict[str, Any],
        generation_trace: dict[str, Any],
        requested_backend: str,
        selected_backend: str,
        fallback_used: bool,
    ) -> dict[str, Any]:
        trace_snapshot = deepcopy(generation_trace)
        descriptor["semantic_backend_requested"] = requested_backend
        descriptor["semantic_backend_selected"] = selected_backend
        descriptor["semantic_backend_fallback_used"] = fallback_used
        descriptor["semantic_backend_error"] = generation_trace.get("semantic_backend_error")
        descriptor["semantic_backend_trace"] = trace_snapshot
        descriptor["generation_trace"] = trace_snapshot
        return descriptor

    def _normalize_descriptor(
        self,
        *,
        raw_descriptor: dict[str, Any],
        backend_name: str,
        frame_ref: str,
        face_detection: FaceDetectionResult | None,
        confidence_override: float | None,
        raw_summary: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        descriptor = dict(raw_descriptor or {})
        appearance = self._as_dict(descriptor.get("appearance"))
        silhouette = self._as_dict(descriptor.get("silhouette"))
        raw_observation_quality = (
            descriptor.get("scene_observation_quality")
            or descriptor.get("observation_quality")
            or {}
        )
        observation_quality = self._as_dict(raw_observation_quality)
        if not observation_quality and isinstance(raw_observation_quality, str):
            observation_quality = {"level": raw_observation_quality}

        top_clothing = self._normalize_clothing(
            descriptor.get("top_clothing"),
            fallback_color=appearance.get("upper_region_color"),
            default_category="upper_garment",
        )
        bottom_clothing = self._normalize_clothing(
            descriptor.get("bottom_clothing"),
            fallback_color=appearance.get("lower_region_color"),
            default_category="lower_garment",
        )
        dominant_colors = self._normalize_color_list(
            descriptor.get("dominant_colors") or appearance.get("dominant_palette")
        )
        if not dominant_colors:
            dominant_colors = self._ordered_unique(
                [top_clothing["color"], appearance.get("middle_region_color"), bottom_clothing["color"]]
            )

        accessories = self._normalize_string_list(
            descriptor.get("accessories")
            or (descriptor.get("accessories_detail") or {}).get("visible_accessories")
        )
        carried_object = self._normalize_label(
            descriptor.get("carried_object")
            or (descriptor.get("accessories_detail") or {}).get("carried_object"),
            default="unknown",
        )
        body_build = self._normalize_label(descriptor.get("body_build"), default="average")
        pose_direction = self._normalize_pose_direction(descriptor.get("pose_direction"))
        scene_level = self._normalize_quality_level(
            observation_quality.get("level")
            or observation_quality.get("quality_level")
            or observation_quality.get("descriptor_quality")
        )
        descriptor_confidence = self._coerce_confidence(
            descriptor.get("descriptor_confidence")
            or observation_quality.get("descriptor_confidence")
            or confidence_override
        )
        if descriptor_confidence == 0.0 and confidence_override is not None:
            descriptor_confidence = confidence_override
        descriptor_confidence = round(min(1.0, max(0.0, descriptor_confidence or 0.0)), 4)

        frame_aspect_ratio = self._normalize_label(
            silhouette.get("frame_aspect_ratio"),
            default="unknown",
        )
        subject_scale = self._normalize_label(
            silhouette.get("subject_scale"),
            default="unknown",
        )
        horizontal_position = self._normalize_label(
            silhouette.get("horizontal_position"),
            default="center",
        )
        contrast_level = self._normalize_quality_level(
            appearance.get("contrast_level"),
            default="medium",
        )
        saturation_level = self._normalize_quality_level(
            appearance.get("saturation_level"),
            default="medium",
        )
        subject_type = self._normalize_label(descriptor.get("subject_type"), default="person")
        middle_region_color = self._normalize_color(
            appearance.get("middle_region_color") or (dominant_colors[1] if len(dominant_colors) > 1 else top_clothing["color"])
        )
        face_state = self._categorize_face_state(face_detection=face_detection)

        observation_quality_payload = {
            "descriptor_confidence": descriptor_confidence,
            "face_detected": bool(face_detection and face_detection.detected),
            "face_usable": bool(face_detection and face_detection.usable),
            "source_region": self._normalize_label(
                observation_quality.get("source_region"),
                default="full_frame",
            ),
            "image_size": observation_quality.get("image_size")
            or (face_detection.image_size if face_detection else None),
            "level": scene_level,
            "notes": self._normalize_label(observation_quality.get("notes"), default="unknown"),
        }

        signature = {
            "subject_type": subject_type,
            "upper_region_color": top_clothing["color"],
            "middle_region_color": middle_region_color,
            "lower_region_color": bottom_clothing["color"],
            "dominant_palette": dominant_colors,
            "contrast_level": contrast_level,
            "saturation_level": saturation_level,
            "frame_aspect_ratio": frame_aspect_ratio,
            "horizontal_position": horizontal_position,
            "subject_scale": subject_scale,
            "face_state": face_state,
            "pose_direction": pose_direction,
            "body_build": body_build,
            "accessories": accessories,
            "carried_object": carried_object,
        }

        normalized_descriptor = {
            "descriptor_schema_version": "semantic_descriptor_v2",
            "subject_type": subject_type,
            "top_clothing": top_clothing,
            "bottom_clothing": bottom_clothing,
            "dominant_colors": dominant_colors,
            "accessories": accessories,
            "carried_object": carried_object,
            "body_build": body_build,
            "pose_direction": pose_direction,
            "scene_observation_quality": observation_quality_payload,
            "descriptor_backend": backend_name,
            "descriptor_confidence": descriptor_confidence,
            "raw_summary": self._normalize_summary(descriptor.get("raw_summary") or raw_summary),
            "appearance": {
                "dominant_palette": dominant_colors,
                "upper_region_color": top_clothing["color"],
                "middle_region_color": middle_region_color,
                "lower_region_color": bottom_clothing["color"],
                "contrast_level": contrast_level,
                "saturation_level": saturation_level,
            },
            "silhouette": {
                "frame_aspect_ratio": frame_aspect_ratio,
                "subject_scale": subject_scale,
                "horizontal_position": horizontal_position,
            },
            "accessories_detail": {
                "visible_accessories": accessories,
                "carried_object": carried_object,
            },
            "observation_quality": observation_quality_payload,
            "signature": signature,
            "source_frame_ref": frame_ref,
        }
        return normalized_descriptor, signature, descriptor_confidence

    def _resolve_frame_path(self, frame_ref: str) -> Path | None:
        frame_path = Path(frame_ref)
        if frame_path.is_absolute() and frame_path.exists():
            return frame_path
        relative_path = Path.cwd() / frame_ref
        if relative_path.exists():
            return relative_path
        return None

    def _normalize_clothing(
        self,
        raw_value: Any,
        *,
        fallback_color: str | None,
        default_category: str,
    ) -> dict[str, str]:
        if isinstance(raw_value, dict):
            category = self._normalize_label(raw_value.get("category"), default=default_category)
            color = self._normalize_color(raw_value.get("color") or fallback_color)
            pattern = self._normalize_pattern(raw_value.get("pattern"))
            return {"category": category, "color": color, "pattern": pattern}

        if isinstance(raw_value, str):
            lowered = raw_value.lower()
            category = default_category
            for candidate in [
                "hoodie",
                "jacket",
                "coat",
                "shirt",
                "sweater",
                "dress",
                "jeans",
                "pants",
                "shorts",
                "skirt",
                "uniform",
            ]:
                if candidate in lowered:
                    category = candidate
                    break
            color = self._extract_color_from_text(lowered) or self._normalize_color(fallback_color)
            return {"category": category, "color": color, "pattern": "solid"}

        return {
            "category": default_category,
            "color": self._normalize_color(fallback_color),
            "pattern": "unknown",
        }

    def _normalize_color_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            parts = re.split(r"[;,/]| and ", value)
            return self._ordered_unique([self._normalize_color(part) for part in parts])
        if isinstance(value, list):
            return self._ordered_unique([self._normalize_color(item) for item in value])
        return []

    def _normalize_string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            parts = re.split(r"[;,/]| and ", value)
            return self._ordered_unique([self._normalize_label(part, default="unknown") for part in parts])
        if isinstance(value, list):
            return self._ordered_unique([self._normalize_label(item, default="unknown") for item in value])
        return []

    def _normalize_color(self, value: Any) -> str:
        lowered = self._normalize_label(value, default="unknown")
        color_aliases = {
            "dark blue": "blue",
            "light blue": "blue",
            "navy blue": "navy",
            "deep red": "red",
            "dark red": "red",
            "dark gray": "gray",
            "light gray": "gray",
            "dark green": "green",
            "light green": "green",
            "tan": "beige",
        }
        if lowered in color_aliases:
            return color_aliases[lowered]
        for color in [
            "black",
            "white",
            "gray",
            "red",
            "orange",
            "yellow",
            "green",
            "teal",
            "blue",
            "navy",
            "purple",
            "brown",
            "beige",
        ]:
            if color in lowered:
                return color
        return lowered

    def _extract_color_from_text(self, text: str) -> str | None:
        normalized = self._normalize_color(text)
        if normalized == "unknown":
            return None
        return normalized

    def _normalize_label(self, value: Any, *, default: str = "unknown") -> str:
        if value is None:
            return default
        cleaned = str(value).strip().lower().replace("-", "_")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return default
        return cleaned

    def _normalize_pattern(self, value: Any) -> str:
        normalized = self._normalize_label(value, default="unknown")
        if normalized in {"plain", "solid_color"}:
            return "solid"
        if "stripe" in normalized:
            return "striped"
        if "plaid" in normalized or "check" in normalized:
            return "plaid"
        if "graphic" in normalized or "logo" in normalized:
            return "graphic"
        return normalized

    def _normalize_pose_direction(self, value: Any) -> str:
        normalized = self._normalize_label(value, default="front")
        if normalized in {"slightly_left", "left_facing"}:
            return "left"
        if normalized in {"slightly_right", "right_facing"}:
            return "right"
        if normalized in {"forward", "frontal"}:
            return "front"
        return normalized

    def _normalize_quality_level(self, value: Any, *, default: str = "medium") -> str:
        normalized = self._normalize_label(value, default=default)
        if normalized in {"moderate", "mid"}:
            return "medium"
        if normalized in {"very_high"}:
            return "high"
        if normalized in {"very_low"}:
            return "low"
        return normalized

    def _normalize_summary(self, value: Any) -> str | None:
        if value is None:
            return None
        summary = " ".join(str(value).split()).strip()
        return summary or None

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _ordered_unique(self, values: list[str | None]) -> list[str]:
        ordered: list[str] = []
        for value in values:
            normalized = self._normalize_label(value, default="unknown")
            if normalized == "unknown" or normalized in ordered:
                continue
            ordered.append(normalized)
        return ordered[:4]

    def _as_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _categorize_face_state(self, *, face_detection: FaceDetectionResult | None) -> str:
        if face_detection is None:
            return "unknown"
        if face_detection.usable:
            return "usable_face"
        if face_detection.detected:
            return "low_quality_face"
        return "face_not_available"
