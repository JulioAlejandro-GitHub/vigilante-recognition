from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_recognition_event(
    *,
    event_type: str,
    camera_id: str,
    track_id: str,
    subject_id: str,
    severity: str,
    confidence: float,
    decision_reason: list[str],
    frame_ref: str | None,
    evidence_refs: list[str] | None = None,
    payload_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    camera_id_value = str(camera_id)
    track_id_value = str(track_id)
    subject_id_value = str(subject_id)
    resolved_evidence_refs = evidence_refs if evidence_refs is not None else ([frame_ref] if frame_ref else [])
    payload = {
        "severity": severity,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "evidence_refs": resolved_evidence_refs,
        "requires_human_review": False,
    }
    if payload_details:
        payload.update(payload_details)
    payload["evidence_refs"] = resolved_evidence_refs

    return {
        "event_id": f"evt_{track_id_value}_{event_type}",
        "event_type": event_type,
        "event_version": "1.0",
        "occurred_at": now,
        "emitted_at": now,
        "source": {
            "component": "vigilante-recognition",
            "instance": "local-worker",
            "version": "0.1.0",
        },
        "payload": payload,
        "context": {
            "camera_id": camera_id_value,
            "track_id": track_id_value,
            "subject_id": subject_id_value,
            "idempotency_key": f"{track_id_value}:{event_type}",
        },
    }
