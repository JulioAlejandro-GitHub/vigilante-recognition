from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.sqlalchemy_url, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    with engine.begin() as connection:
        connection.execute(text("SELECT 1"))
        constraint_definition = connection.execute(
            text(
                """
                SELECT pg_get_constraintdef(c.oid)
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = 'recognition'
                  AND t.relname = 'recognition_event'
                  AND c.conname = 'chk_recognition_event_type'
                """
            )
        ).scalar_one_or_none()
        if constraint_definition and "case_suggestion_created" not in constraint_definition:
            connection.execute(text("ALTER TABLE recognition.recognition_event DROP CONSTRAINT IF EXISTS chk_recognition_event_type"))
            connection.execute(
                text(
                    """
                    ALTER TABLE recognition.recognition_event
                    ADD CONSTRAINT chk_recognition_event_type CHECK (
                        event_type = ANY (
                            ARRAY[
                                'human_track_opened',
                                'human_track_updated',
                                'human_track_closed',
                                'human_presence_detected',
                                'human_presence_no_face',
                                'face_quality_failed',
                                'face_detected_unidentified',
                                'face_detected_identified',
                                'candidate_match_created',
                                'observed_subject_created',
                                'observed_subject_updated',
                                'cross_camera_subject_correlated',
                                'identity_conflict',
                                'manual_review_required',
                                'recurrent_unresolved_subject',
                                'case_suggestion_created'
                            ]::text[]
                        )
                    )
                    """
                )
            )


def get_session():
    return SessionLocal()
