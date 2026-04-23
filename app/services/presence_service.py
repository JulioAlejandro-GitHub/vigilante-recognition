from __future__ import annotations

from pathlib import Path

import cv2

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult, FaceMatchResult, PresenceDecision
from app.models import HumanTrack


class PresenceService:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")

    def inspect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float] | None = None,
    ) -> FaceDetectionResult:
        return self.detect_face(frame_ref=frame_ref, quality_metadata=quality_metadata or {})

    def decide(
        self,
        *,
        track: HumanTrack,
        face_detection: FaceDetectionResult,
        embedding_result: FaceEmbeddingResult | None = None,
        match_result: FaceMatchResult | None = None,
    ) -> PresenceDecision:
        decision_reason = ["human_track_confirmed"]
        serialized_detection = self._serialize_face_detection(face_detection)

        if not face_detection.usable:
            if face_detection.detected:
                decision_reason.append("face_quality_threshold_failed")
            else:
                decision_reason.extend(face_detection.rejection_reasons or ["face_not_detected"])

            return PresenceDecision(
                event_type="human_presence_no_face",
                severity="low",
                confidence=min(1.0, max(0.5, track.person_presence_score or 0.0)),
                decision_reason=decision_reason,
                payload={"face_detection": serialized_detection},
            )

        decision_reason.extend(
            [
                "face_detected",
                "face_quality_threshold_passed",
            ]
        )
        payload = {
            "face_detection": serialized_detection,
            "embedding_backend": embedding_result.backend if embedding_result else settings.embedding_backend,
            "embedding_dimensions": embedding_result.dimensions if embedding_result else 0,
            "identified": False,
        }

        if embedding_result and embedding_result.generated:
            decision_reason.append("embedding_generated")
        elif embedding_result:
            decision_reason.extend(embedding_result.rejection_reasons or ["embedding_not_generated"])

        if match_result and match_result.identified and match_result.best_match is not None:
            decision_reason.extend(
                [
                    "gallery_match_threshold_passed",
                    "second_best_margin_passed",
                    "known_identity_resolved",
                ]
            )
            payload.update(
                {
                    "identified": True,
                    "person_profile_id": match_result.best_match.person_profile_id,
                    "external_person_key": match_result.best_match.external_person_key,
                    "match_confidence": match_result.match_confidence,
                    "matching_strategy": match_result.matching_strategy,
                    "matched_person": {
                        "full_name": match_result.best_match.full_name,
                        "person_type": match_result.best_match.person_type,
                        "risk_level": match_result.best_match.risk_level,
                        "gallery_source": match_result.best_match.gallery_source,
                    },
                }
            )
            return PresenceDecision(
                event_type="face_detected_identified",
                severity="medium",
                confidence=match_result.match_confidence,
                decision_reason=decision_reason,
                payload=payload,
            )

        if match_result:
            decision_reason.extend(match_result.rejection_reasons or ["match_not_confident"])
            payload.update(
                {
                    "match_confidence": match_result.match_confidence,
                    "matching_strategy": match_result.matching_strategy,
                    "best_similarity": match_result.best_similarity,
                    "second_best_similarity": match_result.second_best_similarity,
                    "second_best_margin": match_result.second_best_margin,
                    "evaluated_candidates": match_result.evaluated_candidates,
                }
            )
            if match_result.best_match is not None:
                payload["best_candidate"] = {
                    "person_profile_id": match_result.best_match.person_profile_id,
                    "full_name": match_result.best_match.full_name,
                    "external_person_key": match_result.best_match.external_person_key,
                    "similarity": match_result.best_match.similarity,
                    "gallery_source": match_result.best_match.gallery_source,
                }

        return PresenceDecision(
            event_type="face_detected_unidentified",
            severity="medium",
            confidence=face_detection.quality_score,
            decision_reason=decision_reason,
            payload=payload,
        )

    def detect_face(
        self,
        *,
        frame_ref: str,
        quality_metadata: dict[str, float],
    ) -> FaceDetectionResult:
        frame_path = self._resolve_frame_path(frame_ref)
        if frame_path is None:
            return FaceDetectionResult(
                rejection_reasons=["frame_ref_not_found"],
                frame_quality_metadata=quality_metadata,
            )

        image = cv2.imread(str(frame_path))
        if image is None:
            return FaceDetectionResult(
                rejection_reasons=["frame_unreadable"],
                frame_quality_metadata=quality_metadata,
            )

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        image_height, image_width = grayscale.shape
        image_size = {"width": int(image_width), "height": int(image_height)}

        if len(faces) == 0:
            return FaceDetectionResult(
                detected=False,
                usable=False,
                image_size=image_size,
                rejection_reasons=["face_not_detected"],
                frame_quality_metadata=quality_metadata,
            )

        x, y, width, height = max(faces, key=lambda bbox: int(bbox[2]) * int(bbox[3]))
        face_crop = grayscale[y : y + height, x : x + width]
        quality_metrics = self._compute_quality_metrics(
            image_width=image_width,
            image_height=image_height,
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            face_crop=face_crop,
        )
        quality_score = round(
            (0.4 * quality_metrics["blur_score"])
            + (0.25 * quality_metrics["brightness_score"])
            + (0.2 * quality_metrics["size_score"])
            + (0.15 * quality_metrics["centered_score"]),
            4,
        )
        usable = quality_score >= settings.face_quality_threshold
        rejection_reasons = [] if usable else ["face_quality_threshold_failed"]

        return FaceDetectionResult(
            detected=True,
            usable=usable,
            quality_score=quality_score,
            bbox={
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
            image_size=image_size,
            rejection_reasons=rejection_reasons,
            quality_metrics=quality_metrics,
            frame_quality_metadata=quality_metadata,
        )

    def _resolve_frame_path(self, frame_ref: str) -> Path | None:
        frame_path = Path(frame_ref)
        if frame_path.is_absolute() and frame_path.exists():
            return frame_path
        relative_path = Path.cwd() / frame_ref
        if relative_path.exists():
            return relative_path
        return None

    def _compute_quality_metrics(
        self,
        *,
        image_width: int,
        image_height: int,
        x: int,
        y: int,
        width: int,
        height: int,
        face_crop,
    ) -> dict[str, float]:
        blur_variance = float(cv2.Laplacian(face_crop, cv2.CV_64F).var())
        blur_score = min(1.0, blur_variance / 350.0)

        brightness = float(face_crop.mean())
        brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)

        size_score = min(1.0, min(width, height) / 120.0)

        face_center_x = x + (width / 2.0)
        face_center_y = y + (height / 2.0)
        max_distance = ((image_width / 2.0) ** 2 + (image_height / 2.0) ** 2) ** 0.5
        distance_to_center = ((face_center_x - (image_width / 2.0)) ** 2 + (face_center_y - (image_height / 2.0)) ** 2) ** 0.5
        centered_score = max(0.0, 1.0 - (distance_to_center / max_distance))

        return {
            "blur_score": round(blur_score, 4),
            "brightness_score": round(brightness_score, 4),
            "size_score": round(size_score, 4),
            "centered_score": round(centered_score, 4),
        }

    def _serialize_face_detection(self, face_detection: FaceDetectionResult) -> dict[str, object]:
        return {
            "detected": face_detection.detected,
            "usable": face_detection.usable,
            "quality_score": face_detection.quality_score,
            "bbox": face_detection.bbox,
            "image_size": face_detection.image_size,
            "rejection_reasons": face_detection.rejection_reasons,
            "quality_metrics": face_detection.quality_metrics,
            "frame_quality_metadata": face_detection.frame_quality_metadata,
        }
