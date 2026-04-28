from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.storage.frame_resolver import FrameResolver
from app.storage.s3_frame_resolver import S3FrameResolver


def build_frame_resolver(*, frame_search_roots: list[Path] | None = None) -> FrameResolver:
    return FrameResolver(
        search_roots=frame_search_roots,
        remote_resolver=S3FrameResolver(
            endpoint=settings.storage_s3_endpoint,
            access_key=settings.storage_s3_access_key,
            secret_key=settings.storage_s3_secret_key,
            secure=settings.storage_s3_secure,
            cache_dir=settings.storage_s3_cache_dir,
            connect_timeout_seconds=settings.storage_s3_connect_timeout_seconds,
            read_timeout_seconds=settings.storage_s3_read_timeout_seconds,
        ),
    )
