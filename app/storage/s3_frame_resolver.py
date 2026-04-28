from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from app.storage.frame_resolver import FrameResolutionError
from app.storage.temp_file_manager import FrameCachePathManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3FrameUri:
    bucket: str
    object_key: str
    original_uri: str


class S3FrameResolver:
    """Resolve s3://bucket/key or minio://bucket/key into a local cached file."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        cache_dir: Path | str = ".runtime/ingestion/frame-cache",
        client: Any | None = None,
        connect_timeout_seconds: float = 3.0,
        read_timeout_seconds: float = 15.0,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.cache_paths = FrameCachePathManager(cache_dir)
        self._client = client

    def resolve_reference(self, reference: str) -> Path:
        parsed = parse_s3_frame_uri(reference)
        target_path = self.cache_paths.path_for_object(bucket=parsed.bucket, object_key=parsed.object_key)
        if target_path.is_file():
            logger.info(
                "remote_frame_cache_hit endpoint=%s bucket=%s object_key=%s cached_path=%s",
                self.endpoint,
                parsed.bucket,
                parsed.object_key,
                target_path,
            )
            return target_path.resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".download-", dir=target_path.parent, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            self._get_client().fget_object(parsed.bucket, parsed.object_key, str(temp_path))
            temp_path.replace(target_path)
        except Exception as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            reason = _categorize_download_error(exc)
            logger.warning(
                "remote_frame_download_failed endpoint=%s bucket=%s object_key=%s reason=%s error_type=%s",
                self.endpoint,
                parsed.bucket,
                parsed.object_key,
                reason,
                type(exc).__name__,
            )
            raise FrameResolutionError(
                frame_refs=[reference],
                attempted_paths=[target_path],
                attempted_locations=[reference],
                reason=reason,
                details={
                    "bucket": parsed.bucket,
                    "object_key": parsed.object_key,
                    "endpoint": self.endpoint,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "storage_error_code": getattr(exc, "code", None),
                },
            ) from exc

        logger.info(
            "remote_frame_resolved endpoint=%s bucket=%s object_key=%s cached_path=%s",
            self.endpoint,
            parsed.bucket,
            parsed.object_key,
            target_path,
        )
        return target_path.resolve()

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from minio import Minio
            from urllib3 import PoolManager, Timeout
        except ImportError as exc:
            raise RuntimeError("S3 frame resolution requires the 'minio' package") from exc

        self._client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
            http_client=PoolManager(
                timeout=Timeout(
                    connect=self.connect_timeout_seconds,
                    read=self.read_timeout_seconds,
                ),
                retries=False,
            ),
        )
        return self._client


def parse_s3_frame_uri(reference: str) -> S3FrameUri:
    parsed = urlparse(reference)
    if parsed.scheme not in {"s3", "minio"}:
        raise FrameResolutionError(
            frame_refs=[reference],
            attempted_locations=[reference],
            reason="unsupported_remote_frame_uri",
        )
    if parsed.query or parsed.fragment:
        raise FrameResolutionError(
            frame_refs=[reference],
            attempted_locations=[reference],
            reason="invalid_remote_frame_uri",
            details={"error": "query_or_fragment_not_supported"},
        )
    bucket = parsed.netloc.strip()
    object_key = unquote(parsed.path.lstrip("/"))
    if not bucket or not object_key or object_key.endswith("/"):
        raise FrameResolutionError(
            frame_refs=[reference],
            attempted_locations=[reference],
            reason="invalid_remote_frame_uri",
            details={"error": "expected s3://bucket/object-key"},
        )
    return S3FrameUri(bucket=bucket, object_key=object_key, original_uri=reference)


def _categorize_download_error(exc: Exception) -> str:
    code = str(getattr(exc, "code", "") or "")
    error_type = type(exc).__name__.lower()
    message = str(exc).lower()
    if code in {"NoSuchBucket", "NoSuchBucketPolicy"}:
        return "remote_bucket_not_found"
    if code in {"NoSuchKey", "NoSuchObject", "NoSuchUpload"}:
        return "remote_object_not_found"
    if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidToken", "ExpiredToken"}:
        return "remote_storage_auth_failed"
    if "timeout" in error_type or "timeout" in message:
        return "remote_storage_timeout"
    if any(marker in error_type for marker in ["endpoint", "maxretry", "newconnection", "name_resolution", "connection"]):
        return "remote_storage_endpoint_unreachable"
    if any(marker in message for marker in ["connection refused", "name resolution", "failed to establish"]):
        return "remote_storage_endpoint_unreachable"
    return "remote_frame_download_failed"
