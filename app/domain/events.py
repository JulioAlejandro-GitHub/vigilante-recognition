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
    frame_ref: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": f"evt_{track_id}_{event_type}",
        "event_type": event_type,
        "event_version": "1.0",
        "occurred_at": now,
        "emitted_at": now,
        "source": {
            "component": "vigilante-recognition",
            "instance": "local-worker",
            "version": "0.1.0",
        },
        "payload": {
            "severity": severity,
            "confidence": confidence,
            "decision_reason": decision_reason,
            "evidence_refs": [frame_ref],
            "requires_human_review": False,
        },
        "context": {
            "camera_id": camera_id,
            "track_id": track_id,
            "subject_id": subject_id,
            "idempotency_key": f"{track_id}:{event_type}",
        },
    }
