from __future__ import annotations

from datetime import datetime

from app.config import settings
from app.domain.entities import FaceDetectionResult, FaceEmbeddingResult, FaceMatchResult, FrameIngestedMessage
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

    def register_face_match(
        self,
        *,
        track,
        match_result: FaceMatchResult,
        matched_at: datetime,
    ):
        return self.repo.update_track_match_result(
            track,
            match_result=match_result,
            matched_at=matched_at,
        )

    def update_subject_face_profile(
        self,
        *,
        subject,
        camera_id,
        observed_at: datetime,
        frame_ref: str,
        face_detection: FaceDetectionResult,
        embedding_result: FaceEmbeddingResult | None = None,
        match_result: FaceMatchResult | None = None,
    ):
        return self.repo.update_subject_face_profile(
            subject,
            camera_id=camera_id,
            observed_at=observed_at,
            frame_ref=frame_ref,
            face_detection=face_detection,
            embedding_result=embedding_result,
            match_result=match_result,
        )

    def link_track_to_subject(
        self,
        *,
        track,
        source_subject,
        target_subject,
        resolved_at: datetime,
        payload: dict,
    ):
        self.repo.mark_subject_continuity(
            source_subject,
            outcome="correlated",
            resolved_at=resolved_at,
            payload=payload,
        )
        self.repo.attach_subject_to_track(track, subject_id=target_subject.observed_subject_id)
        target_subject = self.repo.touch_subject(
            target_subject,
            seen_at=resolved_at,
            camera_id=track.camera_id,
            increment_recurrence=True,
        )
        return target_subject, track
