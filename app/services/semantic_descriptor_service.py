from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
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
    REAL_BACKEND_KEYS = {"qwen_vl", "smolvlm"}

    def __init__(
        self,
        *,
        backends: dict[str, SemanticDescriptorBackend] | None = None,
    ) -> None:
        self.backends = backends or {
            self.SIMPLE_BACKEND_KEY: SimpleSemanticDescriptorBackend(),
            "qwen_vl": QwenVLSemanticBackend(),
            "smolvlm": SmolVlmSemanticBackend(),
        }

    def generate(
        self,
        *,
        frame_ref: str,
        source_frame_ref: str | None = None,
        face_detection: FaceDetectionResult | None = None,
    ) -> SemanticDescriptorResult:
        frame_path = self._resolve_frame_path(frame_ref)
        published_source_frame_ref = frame_ref if source_frame_ref is None else source_frame_ref
        requested_backend = settings.semantic_descriptor_backend
        generation_trace = {
            "requested_backend": requested_backend,
            "fallback_enabled": settings.semantic_enable_fallback,
            "selected_backend": None,
            "selected_backend_key": None,
            "attempts": [],
        }
        if frame_path is None:
            return SemanticDescriptorResult(
                backend=self._backend_name_for(self.SIMPLE_BACKEND_KEY),
                source_frame_ref=published_source_frame_ref,
                rejection_reasons=["frame_ref_not_found"],
                descriptor={"generation_trace": generation_trace},
            )

        rejection_reasons: list[str] = []
        for backend_key in self._build_backend_chain(requested_backend=requested_backend):
            backend = self.backends.get(backend_key)
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

            if backend_key in self.REAL_BACKEND_KEYS and not settings.semantic_use_real_vlm:
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "skipped",
                        "reason": "real_vlm_disabled",
                    }
                )
                continue

            context = SemanticBackendContext(
                frame_ref=frame_ref,
                image_path=frame_path,
                face_detection=face_detection,
                requested_backend=requested_backend,
                timeout_seconds=settings.semantic_timeout_seconds,
            )
            try:
                backend_output = backend.generate_descriptor(image_path=frame_path, context=context)
                descriptor, signature, confidence = self._normalize_descriptor(
                    raw_descriptor=backend_output.descriptor,
                    backend_name=backend_output.backend_name,
                    frame_ref=published_source_frame_ref,
                    face_detection=face_detection,
                    confidence_override=backend_output.confidence,
                    raw_summary=backend_output.raw_summary,
                )
                generation_trace["selected_backend"] = backend_output.backend_name
                generation_trace["selected_backend_key"] = backend_key
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend_output.backend_name,
                        "status": "success",
                        **backend_output.trace,
                    }
                )
                descriptor["generation_trace"] = deepcopy(generation_trace)
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
                rejection_reasons.append(str(exc))
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "failed",
                        "reason": str(exc),
                        **exc.details,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive path
                reason = f"unexpected_backend_error:{backend_key}:{type(exc).__name__}"
                rejection_reasons.append(reason)
                generation_trace["attempts"].append(
                    {
                        "backend_key": backend_key,
                        "backend_name": backend.backend_name,
                        "status": "failed",
                        "reason": reason,
                    }
                )

        return SemanticDescriptorResult(
            backend=self._backend_name_for(self.SIMPLE_BACKEND_KEY),
            source_frame_ref=published_source_frame_ref,
            rejection_reasons=rejection_reasons or ["semantic_descriptor_unavailable"],
            descriptor={"generation_trace": generation_trace},
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
        fallback_key = self._fallback_key_for(requested_key) if settings.semantic_enable_fallback else None
        chain: list[str] = []
        for backend_key in [requested_key, fallback_key, self.SIMPLE_BACKEND_KEY]:
            if backend_key and backend_key not in chain:
                chain.append(backend_key)
        return chain

    def _normalize_backend_key(self, requested_backend: str | None) -> str:
        normalized = (requested_backend or "").strip().lower()
        alias_map = {
            "simple": self.SIMPLE_BACKEND_KEY,
            "simple_color_signature_v1": self.SIMPLE_BACKEND_KEY,
            "qwen_vl": "qwen_vl",
            "qwen/qwen2.5-vl-3b-instruct": "qwen_vl",
            "qwen2.5-vl-3b-instruct": "qwen_vl",
            "smolvlm": "smolvlm",
            "smol_vlm": "smolvlm",
            "huggingfacetb/smolvlm2-2.2b-instruct": "smolvlm",
            "smolvlm2-2.2b-instruct": "smolvlm",
        }
        return alias_map.get(normalized, self.SIMPLE_BACKEND_KEY)

    def _fallback_key_for(self, requested_key: str) -> str | None:
        if requested_key == "qwen_vl":
            return "smolvlm"
        if requested_key == "smolvlm":
            return self.SIMPLE_BACKEND_KEY
        return None

    def _backend_name_for(self, backend_key: str) -> str:
        backend = self.backends.get(backend_key)
        if backend is None:
            return backend_key
        return backend.backend_name

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
