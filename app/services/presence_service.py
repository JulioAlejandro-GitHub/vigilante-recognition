from __future__ import annotations

from app.config import settings
from app.domain.entities import PresenceDecision
from app.models import HumanTrack


class PresenceService:
    def decide(self, track: HumanTrack) -> PresenceDecision:
        if track.person_presence_score >= 0.6:
            return PresenceDecision(
                event_type="human_presence_detected",
                severity="medium",
                confidence=min(1.0, track.person_presence_score),
                decision_reason=[
                    "presence_score_threshold_reached",
                    "human_track_confirmed",
                ],
            )
        return PresenceDecision(
            event_type="human_presence_no_face",
            severity="low",
            confidence=max(0.5, track.person_presence_score),
            decision_reason=[
                "track_closed_without_face",
                "presence_confirmed_without_face",
            ],
        )
