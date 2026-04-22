from __future__ import annotations

from app.domain.entities import FrameIngestedMessage
from app.infra.repository import RecognitionRepository


class TrackService:
    def __init__(self, repo: RecognitionRepository) -> None:
        self.repo = repo

    def open_track_from_frame(self, message: FrameIngestedMessage):
        camera_id = self.repo.resolve_camera_id(message.camera_id)
        subject = self.repo.create_subject(first_seen_at=message.captured_at)
        track = self.repo.create_track(
            camera_id=camera_id,
            subject_id=subject.observed_subject_id,
            started_at=message.captured_at,
            external_track_key=message.context.get("correlation_id"),
        )
        self.repo.update_track_presence(track)
        return subject, track
