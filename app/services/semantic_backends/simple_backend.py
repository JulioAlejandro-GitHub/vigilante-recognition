from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.domain.entities import FaceDetectionResult
from app.services.semantic_backends.base import (
    SemanticBackendContext,
    SemanticBackendError,
    SemanticBackendOutput,
    SemanticDescriptorBackend,
)


class SimpleSemanticDescriptorBackend(SemanticDescriptorBackend):
    key = "simple"
    backend_name = "simple_color_signature_v1"

    def generate_descriptor(
        self,
        *,
        image_path: Path,
        context: SemanticBackendContext,
    ) -> SemanticBackendOutput:
        image = cv2.imread(str(image_path))
        if image is None:
            raise SemanticBackendError("frame_unreadable")

        image_height, image_width = image.shape[:2]
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        top_region = image[0 : max(1, image_height // 3), :]
        middle_region = image[max(1, image_height // 3) : max(2, (2 * image_height) // 3), :]
        lower_region = image[max(2, (2 * image_height) // 3) :, :]

        upper_region_color = self._label_color(top_region)
        middle_region_color = self._label_color(middle_region)
        lower_region_color = self._label_color(lower_region)
        dominant_palette = self._ordered_unique(
            [upper_region_color, middle_region_color, lower_region_color]
        )

        contrast_level = self._bucket_level(float(grayscale.std()), low=28.0, high=62.0)
        saturation_level = self._bucket_level(float(hsv[:, :, 1].mean()), low=45.0, high=105.0)
        frame_aspect_ratio = self._categorize_aspect_ratio(
            image_width=image_width,
            image_height=image_height,
        )
        horizontal_position = self._categorize_horizontal_position(
            face_detection=context.face_detection,
            image_width=image_width,
        )
        subject_scale = self._categorize_subject_scale(
            face_detection=context.face_detection,
            image_width=image_width,
            image_height=image_height,
        )

        confidence = 0.62
        if context.face_detection and context.face_detection.detected:
            confidence += 0.12
        if context.face_detection and context.face_detection.usable:
            confidence += 0.1
        if dominant_palette:
            confidence += 0.08
        confidence = round(min(1.0, confidence), 4)

        descriptor = {
            "subject_type": "person",
            "top_clothing": {
                "category": "upper_garment",
                "color": upper_region_color,
                "pattern": "solid",
            },
            "bottom_clothing": {
                "category": "lower_garment",
                "color": lower_region_color,
                "pattern": "solid",
            },
            "dominant_colors": dominant_palette,
            "accessories": [],
            "carried_object": "unknown",
            "body_build": "average",
            "pose_direction": "front",
            "scene_observation_quality": {
                "level": "high" if confidence >= 0.9 else "medium" if confidence >= 0.75 else "low",
                "face_detected": bool(context.face_detection and context.face_detection.detected),
                "face_usable": bool(context.face_detection and context.face_detection.usable),
                "source_region": "full_frame",
                "image_size": {"width": int(image_width), "height": int(image_height)},
            },
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
            "raw_summary": (
                f"person with {upper_region_color} upper clothing and {lower_region_color} lower clothing"
            ),
        }
        return SemanticBackendOutput(
            backend_name=self.backend_name,
            descriptor=descriptor,
            confidence=confidence,
            raw_summary=descriptor["raw_summary"],
        )

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
