from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.config import Settings, settings
from app.services.camera_runtime_config_service import extract_camera_runtime_config, stable_config_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsightFaceTuningDefaults:
    model_name: str
    provider: str
    model_root: str | None
    det_size: str
    detection_threshold: float
    max_faces: int
    face_quality_threshold: float
    min_face_bbox_size: int
    min_face_area_ratio: float


@dataclass(frozen=True)
class InsightFaceEffectiveTuning:
    camera_id: str | None
    model_name: str
    provider: str
    model_root: str | None
    det_size: str
    detection_threshold: float
    max_faces: int
    face_quality_threshold: float
    min_face_bbox_size: int
    min_face_area_ratio: float
    config_source: str
    camera_override_applied: bool
    camera_override_key: str | None = None
    camera_override_errors: tuple[str, ...] = field(default_factory=tuple)
    camera_config_version: str | None = None
    camera_config_hash: str | None = None
    effective_config_hash: str | None = None

    def quality_thresholds_trace(self) -> dict[str, object]:
        return {
            "face_quality_threshold": self.face_quality_threshold,
            "min_face_bbox_size": self.min_face_bbox_size,
            "min_face_area_ratio": self.min_face_area_ratio,
        }

    def camera_trace(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "config_source": self.config_source,
            "face_tuning_source": self.config_source,
            "camera_config_version": self.camera_config_version,
            "camera_config_hash": self.camera_config_hash,
            "effective_config_hash": self.effective_config_hash,
            "camera_override_applied": self.camera_override_applied,
            "camera_override_key": self.camera_override_key,
            "camera_override_errors": list(self.camera_override_errors),
            "quality_thresholds": self.quality_thresholds_trace(),
        }

    def cache_signature(self) -> tuple[object, ...]:
        return (
            self.camera_id,
            self.det_size,
            self.detection_threshold,
            self.max_faces,
            self.face_quality_threshold,
            self.min_face_bbox_size,
            self.min_face_area_ratio,
            self.config_source,
        )


class CameraFaceTuningService:
    """Resolves explicit InsightFace tuning for one camera without mutating global settings."""

    _FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "det_size": ("det_size", "insightface_det_size"),
        "detection_threshold": (
            "detection_threshold",
            "det_thresh",
            "insightface_detection_threshold",
        ),
        "max_faces": ("max_faces", "insightface_max_faces"),
        "face_quality_threshold": (
            "face_quality_threshold",
            "quality_threshold",
            "min_quality_score",
        ),
        "min_face_bbox_size": (
            "min_face_bbox_size",
            "minimum_face_bbox_size",
            "min_bbox_size",
        ),
        "min_face_area_ratio": (
            "min_face_area_ratio",
            "minimum_face_area_ratio",
            "min_area_ratio",
        ),
    }

    _METADATA_PATHS: tuple[tuple[str, ...], ...] = (
        ("recognition", "face_tuning", "insightface"),
        ("recognition", "face_tuning"),
        ("recognition", "insightface"),
        ("insightface",),
        ("insightface_tuning",),
        ("face_tuning", "insightface"),
        ("face_recognition", "insightface"),
        ("camera_face_tuning", "insightface"),
    )

    def __init__(self, *, settings_obj: Settings | None = None) -> None:
        self.settings = settings_obj or settings

    def build_defaults(
        self,
        *,
        model_name: str | None = None,
        provider: str | None = None,
        model_root: str | None = None,
        det_size: str | None = None,
        detection_threshold: float | None = None,
        max_faces: int | None = None,
    ) -> InsightFaceTuningDefaults:
        return InsightFaceTuningDefaults(
            model_name=model_name or self.settings.insightface_model_name,
            provider=provider or self.settings.insightface_provider,
            model_root=model_root if model_root is not None else self.settings.insightface_model_root,
            det_size=det_size or self.settings.insightface_det_size,
            detection_threshold=(
                self.settings.insightface_detection_threshold
                if detection_threshold is None
                else detection_threshold
            ),
            max_faces=self.settings.insightface_max_faces if max_faces is None else max_faces,
            face_quality_threshold=self.settings.face_quality_threshold,
            min_face_bbox_size=self.settings.insightface_min_face_bbox_size,
            min_face_area_ratio=self.settings.insightface_min_face_area_ratio,
        )

    def resolve(
        self,
        *,
        camera_id: str | None,
        defaults: InsightFaceTuningDefaults,
        camera_metadata: Mapping[str, Any] | None = None,
    ) -> InsightFaceEffectiveTuning:
        invalid_errors: list[str] = []
        for candidate in self._override_candidates(
            camera_id=camera_id,
            camera_metadata=camera_metadata,
        ):
            override = self._normalize_override(candidate.override)
            if not override:
                continue
            tuning, errors = self._apply_override(
                camera_id=camera_id,
                defaults=defaults,
                override=override,
                source=candidate.source,
                override_key=candidate.override_key,
                camera_config_version=candidate.camera_config_version,
                camera_config_hash=candidate.camera_config_hash,
            )
            if not errors:
                return tuning
            invalid_errors.extend(f"{candidate.source}:{error}" for error in errors)
            logger.warning(
                "insightface_camera_override_invalid camera_id=%s source=%s override_key=%s errors=%s",
                camera_id or "<unknown>",
                candidate.source,
                candidate.override_key,
                ";".join(errors),
            )

        return InsightFaceEffectiveTuning(
            camera_id=camera_id,
            model_name=defaults.model_name,
            provider=defaults.provider,
            model_root=defaults.model_root,
            det_size=defaults.det_size,
            detection_threshold=defaults.detection_threshold,
            max_faces=defaults.max_faces,
            face_quality_threshold=defaults.face_quality_threshold,
            min_face_bbox_size=defaults.min_face_bbox_size,
            min_face_area_ratio=defaults.min_face_area_ratio,
            config_source="global",
            camera_override_applied=False,
            camera_override_errors=tuple(invalid_errors),
            effective_config_hash=_effective_tuning_hash(
                det_size=defaults.det_size,
                detection_threshold=defaults.detection_threshold,
                max_faces=defaults.max_faces,
                face_quality_threshold=defaults.face_quality_threshold,
                min_face_bbox_size=defaults.min_face_bbox_size,
                min_face_area_ratio=defaults.min_face_area_ratio,
                config_source="global",
            ),
        )

    def _override_candidates(
        self,
        *,
        camera_id: str | None,
        camera_metadata: Mapping[str, Any] | None,
    ) -> list["_TuningOverrideCandidate"]:
        candidates: list[_TuningOverrideCandidate] = []
        runtime_config = extract_camera_runtime_config(camera_metadata)
        if runtime_config.face_tuning:
            candidates.append(
                _TuningOverrideCandidate(
                    source=runtime_config.source_label,
                    override_key="camera_runtime_config.recognition.face_tuning",
                    override=runtime_config.face_tuning,
                    camera_config_version=runtime_config.camera_config_version,
                    camera_config_hash=runtime_config.config_hash or runtime_config.effective_config_hash,
                )
            )

        metadata_override = self._metadata_override(camera_metadata)
        if metadata_override is not None:
            override_key, override = metadata_override
            candidates.append(_TuningOverrideCandidate("camera_metadata", override_key, override))

        env_override = self._env_override(camera_id)
        if env_override is not None:
            override_key, override = env_override
            candidates.append(_TuningOverrideCandidate("camera_overrides_json", override_key, override))
        return candidates

    def _metadata_override(
        self,
        camera_metadata: Mapping[str, Any] | None,
    ) -> tuple[str, Mapping[str, Any]] | None:
        if not camera_metadata:
            return None
        for path in self._METADATA_PATHS:
            value: Any = camera_metadata
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    value = None
                    break
                value = value[key]
            if isinstance(value, Mapping):
                return ".".join(path), value
        return None

    def _env_override(self, camera_id: str | None) -> tuple[str, Mapping[str, Any]] | None:
        raw_json = (self.settings.insightface_camera_overrides_json or "").strip()
        if not raw_json or not camera_id:
            return None
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning(
                "insightface_camera_overrides_json_invalid error=%s",
                str(exc),
            )
            return ("INSIGHTFACE_CAMERA_OVERRIDES_JSON", {"__invalid__": str(exc)})
        if not isinstance(parsed, Mapping):
            logger.warning("insightface_camera_overrides_json_invalid error=expected_object")
            return ("INSIGHTFACE_CAMERA_OVERRIDES_JSON", {"__invalid__": "expected_object"})

        camera_id_key = str(camera_id)
        candidates = [camera_id_key, camera_id_key.lower(), camera_id_key.upper()]
        for key in candidates:
            value = parsed.get(key)
            if isinstance(value, Mapping):
                return key, value
            if value is not None:
                return key, {"__invalid__": "expected_object"}
        return None

    def _normalize_override(self, raw_override: Mapping[str, Any]) -> dict[str, Any]:
        if "__invalid__" in raw_override:
            return {"__invalid__": raw_override["__invalid__"]}

        normalized: dict[str, Any] = {}
        for canonical, aliases in self._FIELD_ALIASES.items():
            for alias in aliases:
                if alias in raw_override:
                    normalized[canonical] = raw_override[alias]
                    break
        return normalized

    def _apply_override(
        self,
        *,
        camera_id: str | None,
        defaults: InsightFaceTuningDefaults,
        override: Mapping[str, Any],
        source: str,
        override_key: str | None,
        camera_config_version: str | None = None,
        camera_config_hash: str | None = None,
    ) -> tuple[InsightFaceEffectiveTuning, list[str]]:
        errors: list[str] = []
        if "__invalid__" in override:
            errors.append(str(override["__invalid__"]))

        det_size = defaults.det_size
        detection_threshold = defaults.detection_threshold
        max_faces = defaults.max_faces
        face_quality_threshold = defaults.face_quality_threshold
        min_face_bbox_size = defaults.min_face_bbox_size
        min_face_area_ratio = defaults.min_face_area_ratio

        if "det_size" in override:
            det_size, error = self._coerce_det_size(override["det_size"])
            if error:
                errors.append(error)
        if "detection_threshold" in override:
            detection_threshold, error = self._coerce_probability(
                override["detection_threshold"],
                field_name="detection_threshold",
            )
            if error:
                errors.append(error)
        if "max_faces" in override:
            max_faces, error = self._coerce_non_negative_int(override["max_faces"], field_name="max_faces")
            if error:
                errors.append(error)
        if "face_quality_threshold" in override:
            face_quality_threshold, error = self._coerce_probability(
                override["face_quality_threshold"],
                field_name="face_quality_threshold",
            )
            if error:
                errors.append(error)
        if "min_face_bbox_size" in override:
            min_face_bbox_size, error = self._coerce_non_negative_int(
                override["min_face_bbox_size"],
                field_name="min_face_bbox_size",
            )
            if error:
                errors.append(error)
        if "min_face_area_ratio" in override:
            min_face_area_ratio, error = self._coerce_probability(
                override["min_face_area_ratio"],
                field_name="min_face_area_ratio",
            )
            if error:
                errors.append(error)

        if errors:
            return (
                InsightFaceEffectiveTuning(
                    camera_id=camera_id,
                    model_name=defaults.model_name,
                    provider=defaults.provider,
                    model_root=defaults.model_root,
                    det_size=defaults.det_size,
                    detection_threshold=defaults.detection_threshold,
                    max_faces=defaults.max_faces,
                    face_quality_threshold=defaults.face_quality_threshold,
                    min_face_bbox_size=defaults.min_face_bbox_size,
                    min_face_area_ratio=defaults.min_face_area_ratio,
                    config_source="global",
                    camera_override_applied=False,
                    camera_override_key=override_key,
                    camera_override_errors=tuple(errors),
                    camera_config_version=camera_config_version,
                    camera_config_hash=camera_config_hash,
                    effective_config_hash=_effective_tuning_hash(
                        det_size=defaults.det_size,
                        detection_threshold=defaults.detection_threshold,
                        max_faces=defaults.max_faces,
                        face_quality_threshold=defaults.face_quality_threshold,
                        min_face_bbox_size=defaults.min_face_bbox_size,
                        min_face_area_ratio=defaults.min_face_area_ratio,
                        config_source="global",
                    ),
                ),
                errors,
            )

        return (
            InsightFaceEffectiveTuning(
                camera_id=camera_id,
                model_name=defaults.model_name,
                provider=defaults.provider,
                model_root=defaults.model_root,
                det_size=det_size,
                detection_threshold=detection_threshold,
                max_faces=max_faces,
                face_quality_threshold=face_quality_threshold,
                min_face_bbox_size=min_face_bbox_size,
                min_face_area_ratio=min_face_area_ratio,
                config_source=source,
                camera_override_applied=True,
                camera_override_key=override_key,
                camera_config_version=camera_config_version,
                camera_config_hash=camera_config_hash,
                effective_config_hash=_effective_tuning_hash(
                    det_size=det_size,
                    detection_threshold=detection_threshold,
                    max_faces=max_faces,
                    face_quality_threshold=face_quality_threshold,
                    min_face_bbox_size=min_face_bbox_size,
                    min_face_area_ratio=min_face_area_ratio,
                    config_source=source,
                ),
            ),
            [],
        )

    def _coerce_det_size(self, value: Any) -> tuple[str, str | None]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            value = f"{value[0]},{value[1]}"
        if not isinstance(value, str):
            return "", "det_size_invalid"
        normalized = value.lower().replace("x", ",")
        try:
            raw_width, raw_height = normalized.split(",", 1)
            width = int(raw_width.strip())
            height = int(raw_height.strip())
        except (TypeError, ValueError):
            return "", "det_size_invalid"
        if width <= 0 or height <= 0:
            return "", "det_size_invalid"
        return f"{width},{height}", None

    def _coerce_probability(self, value: Any, *, field_name: str) -> tuple[float, str | None]:
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return 0.0, f"{field_name}_invalid"
        if normalized < 0.0 or normalized > 1.0:
            return 0.0, f"{field_name}_invalid"
        return round(normalized, 4), None

    def _coerce_non_negative_int(self, value: Any, *, field_name: str) -> tuple[int, str | None]:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return 0, f"{field_name}_invalid"
        if normalized < 0:
            return 0, f"{field_name}_invalid"
        return normalized, None


@dataclass(frozen=True)
class _TuningOverrideCandidate:
    source: str
    override_key: str | None
    override: Mapping[str, Any]
    camera_config_version: str | None = None
    camera_config_hash: str | None = None


def _effective_tuning_hash(
    *,
    det_size: str,
    detection_threshold: float,
    max_faces: int,
    face_quality_threshold: float,
    min_face_bbox_size: int,
    min_face_area_ratio: float,
    config_source: str,
) -> str | None:
    return stable_config_hash(
        {
            "det_size": det_size,
            "detection_threshold": detection_threshold,
            "max_faces": max_faces,
            "face_quality_threshold": face_quality_threshold,
            "min_face_bbox_size": min_face_bbox_size,
            "min_face_area_ratio": min_face_area_ratio,
            "config_source": config_source,
        }
    )
