from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.domain.entities import FaceDetectionResult, SemanticDescriptorResult


class SemanticDescriptorService:
    FUTURE_PRIMARY_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    FUTURE_FALLBACK_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def generate(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
    ) -> SemanticDescriptorResult:
        requested_backend = settings.semantic_descriptor_backend
        if requested_backend != "simple_color_signature_v1":
            return self._generate_simple(frame_ref=frame_ref, face_detection=face_detection, requested_backend=requested_backend)
        return self._generate_simple(frame_ref=frame_ref, face_detection=face_detection)

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

    def _generate_simple(
        self,
        *,
        frame_ref: str,
        face_detection: FaceDetectionResult | None = None,
        requested_backend: str | None = None,
    ) -> SemanticDescriptorResult:
        frame_path = self._resolve_frame_path(frame_ref)
        if frame_path is None:
            return SemanticDescriptorResult(
                backend="simple_color_signature_v1",
                source_frame_ref=frame_ref,
                rejection_reasons=["frame_ref_not_found"],
            )

        image = cv2.imread(str(frame_path))
        if image is None:
            return SemanticDescriptorResult(
                backend="simple_color_signature_v1",
                source_frame_ref=frame_ref,
                rejection_reasons=["frame_unreadable"],
            )

        image_height, image_width = image.shape[:2]
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        top_region = image[0 : max(1, image_height // 3), :]
        middle_region = image[max(1, image_height // 3) : max(2, (2 * image_height) // 3), :]
        lower_region = image[max(2, (2 * image_height) // 3) :, :]

        upper_region_color = self._label_color(top_region)
        middle_region_color = self._label_color(middle_region)
        lower_region_color = self._label_color(lower_region)
        dominant_palette = self._ordered_unique([upper_region_color, middle_region_color, lower_region_color])

        contrast_level = self._bucket_level(float(grayscale.std()), low=28.0, high=62.0)
        saturation_level = self._bucket_level(float(hsv[:, :, 1].mean()), low=45.0, high=105.0)
        frame_aspect_ratio = self._categorize_aspect_ratio(image_width=image_width, image_height=image_height)
        horizontal_position = self._categorize_horizontal_position(face_detection=face_detection, image_width=image_width)
        subject_scale = self._categorize_subject_scale(face_detection=face_detection, image_width=image_width, image_height=image_height)
        face_state = self._categorize_face_state(face_detection=face_detection)

        confidence = 0.62
        if face_detection and face_detection.detected:
            confidence += 0.12
        if face_detection and face_detection.usable:
            confidence += 0.1
        if dominant_palette:
            confidence += 0.08
        confidence = round(min(1.0, confidence), 4)

        signature = {
            "upper_region_color": upper_region_color,
            "middle_region_color": middle_region_color,
            "lower_region_color": lower_region_color,
            "dominant_palette": dominant_palette,
            "contrast_level": contrast_level,
            "saturation_level": saturation_level,
            "frame_aspect_ratio": frame_aspect_ratio,
            "horizontal_position": horizontal_position,
            "subject_scale": subject_scale,
            "face_state": face_state,
        }
        descriptor = {
            "backend": "simple_color_signature_v1",
            "appearance": {
                "dominant_palette": dominant_palette,
                "upper_region_color": upper_region_color,
                "middle_region_color": middle_region_color,
                "lower_region_color": lower_region_color,
                "contrast_level": contrast_level,
                "saturation_level": saturation_level,
            },
            "silhouette": {
                "frame_aspect_ratio": frame_aspect_ratio,
                "subject_scale": subject_scale,
                "horizontal_position": horizontal_position,
            },
            "accessories": {
                "visible_accessories": [],
                "carried_object": "unknown",
            },
            "observation_quality": {
                "descriptor_confidence": confidence,
                "face_detected": bool(face_detection and face_detection.detected),
                "face_usable": bool(face_detection and face_detection.usable),
                "source_region": "full_frame",
                "image_size": {"width": int(image_width), "height": int(image_height)},
            },
            "signature": signature,
            "source_frame_ref": frame_ref,
        }
        if requested_backend:
            descriptor["backend_requested"] = requested_backend

        return SemanticDescriptorResult(
            generated=True,
            backend="simple_color_signature_v1",
            descriptor=descriptor,
            signature=signature,
            confidence=confidence,
            source_frame_ref=frame_ref,
        )

    def _resolve_frame_path(self, frame_ref: str) -> Path | None:
        frame_path = Path(frame_ref)
        if frame_path.is_absolute() and frame_path.exists():
            return frame_path
        relative_path = Path.cwd() / frame_ref
        if relative_path.exists():
            return relative_path
        return None

    def _label_color(self, region) -> str:
        if region.size == 0:
            return "unknown"

        mean_bgr = region.reshape(-1, 3).mean(axis=0)
        blue, green, red = [float(channel) for channel in mean_bgr]
        hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        hue, saturation, value = [float(channel) for channel in hsv]

        if value < 40:
            return "black"
        if value > 210 and saturation < 35:
            return "white"
        if saturation < 35:
            return "gray"
        if hue < 10 or hue >= 170:
            return "red"
        if hue < 20:
            return "orange"
        if hue < 34:
            return "yellow"
        if hue < 50:
            return "green"
        if hue < 85:
            return "teal"
        if hue < 118:
            return "blue"
        if hue < 140:
            return "purple"
        if red > 120 and green > 95 and blue < 110:
            return "beige"
        if red > 90 and green > 60 and blue < 70:
            return "brown"
        return "brown" if red >= blue else "navy"

    def _ordered_unique(self, values: list[str]) -> list[str]:
        ordered: list[str] = []
        for value in values:
            if value not in ordered and value != "unknown":
                ordered.append(value)
        return ordered[:3]

    def _bucket_level(self, value: float, *, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value < high:
            return "medium"
        return "high"

    def _categorize_aspect_ratio(self, *, image_width: int, image_height: int) -> str:
        ratio = image_width / max(1, image_height)
        if ratio > 1.15:
            return "landscape"
        if ratio < 0.85:
            return "portrait"
        return "square"

    def _categorize_horizontal_position(
        self,
        *,
        face_detection: FaceDetectionResult | None,
        image_width: int,
    ) -> str:
        if face_detection and face_detection.bbox:
            center_x = face_detection.bbox["x"] + (face_detection.bbox["width"] / 2.0)
            normalized = center_x / max(1, image_width)
            if normalized < 0.4:
                return "left"
            if normalized > 0.6:
                return "right"
        return "center"

    def _categorize_subject_scale(
        self,
        *,
        face_detection: FaceDetectionResult | None,
        image_width: int,
        image_height: int,
    ) -> str:
        if face_detection and face_detection.bbox:
            face_area = face_detection.bbox["width"] * face_detection.bbox["height"]
            image_area = max(1, image_width * image_height)
            ratio = face_area / image_area
            if ratio >= 0.12:
                return "close_up"
            if ratio >= 0.04:
                return "upper_body"
            return "distant"
        return "unknown"

    def _categorize_face_state(self, *, face_detection: FaceDetectionResult | None) -> str:
        if face_detection is None:
            return "unknown"
        if face_detection.usable:
            return "usable_face"
        if face_detection.detected:
            return "low_quality_face"
        return "face_not_available"
