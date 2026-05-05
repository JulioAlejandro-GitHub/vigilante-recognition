from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


RUNTIME_CONFIG_METADATA_KEY = "camera_runtime_config"
API_CAMERA_CONFIG_SOURCE = "api.camera.metadata"
API_CAMERA_POLICY_SOURCE = "api_camera_metadata"


@dataclass(frozen=True)
class CameraRuntimeConfig:
    present: bool = False
    camera_id: str | None = None
    config_source: str = "not_provided"
    camera_config_version: str | None = None
    config_hash: str | None = None
    effective_config_hash: str | None = None
    recognition_enabled: bool | None = None
    face_tuning: dict[str, Any] = field(default_factory=dict)
    vlm_policy: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def source_label(self) -> str:
        if self.config_source == API_CAMERA_CONFIG_SOURCE:
            return API_CAMERA_POLICY_SOURCE
        if self.present:
            return "camera_runtime_config"
        return "not_provided"

    def trace_payload(self, *, camera_id: str | None = None) -> dict[str, Any]:
        return {
            "camera_id": camera_id or self.camera_id,
            "config_source": self.config_source,
            "camera_config_version": self.camera_config_version,
            "camera_config_hash": self.config_hash,
            "effective_config_hash": self.effective_config_hash,
            "camera_override_applied": bool(self.face_tuning or self.vlm_policy or self.recognition_enabled is not None),
            "recognition_enabled": self.recognition_enabled,
            "face_tuning_source": self.source_label if self.face_tuning else None,
            "vlm_policy_source": self.source_label if self.vlm_policy else None,
            "errors": list(self.errors),
        }


def extract_camera_runtime_config(camera_metadata: Mapping[str, Any] | None) -> CameraRuntimeConfig:
    metadata = dict(camera_metadata or {})
    raw = metadata.get(RUNTIME_CONFIG_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return CameraRuntimeConfig()

    errors: list[str] = []
    recognition = raw.get("recognition")
    if recognition is None:
        recognition = {}
    if not isinstance(recognition, Mapping):
        errors.append("recognition_invalid")
        recognition = {}

    face_tuning = recognition.get("face_tuning")
    if face_tuning is None:
        face_tuning = {}
    if not isinstance(face_tuning, Mapping):
        errors.append("face_tuning_invalid")
        face_tuning = {}

    vlm_policy = recognition.get("vlm_policy")
    if vlm_policy is None:
        vlm_policy = {}
    if not isinstance(vlm_policy, Mapping):
        errors.append("vlm_policy_invalid")
        vlm_policy = {}

    return CameraRuntimeConfig(
        present=True,
        camera_id=_optional_text(raw.get("camera_id")),
        config_source=_optional_text(raw.get("config_source")) or _optional_text(raw.get("source")) or API_CAMERA_CONFIG_SOURCE,
        camera_config_version=_optional_text(raw.get("camera_config_version")) or _optional_text(raw.get("config_version")),
        config_hash=_optional_text(raw.get("config_hash")),
        effective_config_hash=_optional_text(raw.get("effective_config_hash")),
        recognition_enabled=_optional_bool(recognition.get("enabled")),
        face_tuning=dict(face_tuning),
        vlm_policy=dict(vlm_policy),
        errors=tuple(errors),
    )


def stable_config_hash(value: Mapping[str, Any] | None) -> str | None:
    if not value:
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None
