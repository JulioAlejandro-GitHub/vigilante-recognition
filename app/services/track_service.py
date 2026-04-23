from __future__ import annotations

from datetime import datetime

from app.config import settings
from app.domain.entities import FaceDetectionResult, FrameIngestedMessage
from app.infra.repository import RecognitionRepository


class TrackService:
    def __init__(self, repo: RecognitionRepository) -> None:
        self.repo = repo

    def open_track_from_frame(self, message: FrameIngestedMessage):
        persisted_camera_id = message.camera_uuid
        external_track_key = message.context.get("correlation_id")
        track = self.repo.find_track_by_camera_and_external_key(
            camera_id=persisted_camera_id,
            external_track_key=external_track_key,
        )
        if track is None:
            subject = self.repo.create_subject(first_seen_at=message.captured_at, camera_id=persisted_camera_id)
            track = self.repo.create_track(
                camera_id=persisted_camera_id,
                subject_id=subject.observed_subject_id,
                started_at=message.captured_at,
                external_track_key=external_track_key,
            )
        else:
            subject = self.repo.get_subject(track.observed_subject_id)
            if subject is None:
                subject = self.repo.create_subject(first_seen_at=message.captured_at, camera_id=persisted_camera_id)
                self.repo.attach_subject_to_track(track, subject_id=subject.observed_subject_id)
            else:
                self.repo.touch_subject(subject, seen_at=message.captured_at, camera_id=persisted_camera_id)
        self.repo.update_track_presence(track)
        return subject, track

    def confirm_basic_presence(self, track):
        for _ in range(max(0, settings.presence_confirmation_frames - 1)):
            track = self.repo.update_track_presence(track)
        return track

    def register_face_observation(
        self,
        *,
        track,
        face_detection: FaceDetectionResult,
        frame_ref: str,
        detected_at: datetime,
    ):
        return self.repo.update_track_face_observation(
            track,
            face_detection=face_detection,
            frame_ref=frame_ref,
            detected_at=detected_at,
        )
