from __future__ import annotations

from pathlib import Path

import pytest

from app.storage.frame_resolver import FrameResolutionError
from app.storage.s3_frame_resolver import S3FrameResolver, parse_s3_frame_uri


class _FakeStorageError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _FakeEndpointError(Exception):
    pass


def test_parse_s3_frame_uri_accepts_bucket_and_object_key() -> None:
    parsed = parse_s3_frame_uri("s3://vigilante-frames/frames/cam01/frame%201.jpg")

    assert parsed.bucket == "vigilante-frames"
    assert parsed.object_key == "frames/cam01/frame 1.jpg"


def test_parse_s3_frame_uri_rejects_malformed_uri() -> None:
    with pytest.raises(FrameResolutionError) as excinfo:
        parse_s3_frame_uri("s3:///missing-bucket.jpg")

    assert excinfo.value.reason == "invalid_remote_frame_uri"


def test_s3_frame_resolver_downloads_frame_to_cache(tmp_path) -> None:
    client = _FakeS3Client(
        {
            ("vigilante-frames", "frames/cam01/frame.jpg"): b"\xff\xd8remote-frame\xff\xd9",
        }
    )
    resolver = S3FrameResolver(
        endpoint="localhost:9000",
        access_key="minio",
        secret_key="minio123",
        cache_dir=tmp_path / "cache",
        client=client,
    )

    resolved_path = resolver.resolve_reference("s3://vigilante-frames/frames/cam01/frame.jpg")

    assert resolved_path.is_file()
    assert resolved_path.read_bytes() == b"\xff\xd8remote-frame\xff\xd9"
    assert resolved_path == (tmp_path / "cache" / "vigilante-frames" / "frames" / "cam01" / "frame.jpg").resolve()
    assert [(bucket, key) for bucket, key, _ in client.downloads] == [
        ("vigilante-frames", "frames/cam01/frame.jpg")
    ]


def test_s3_frame_resolver_uses_cached_file_without_redownloading(tmp_path) -> None:
    cached = tmp_path / "cache" / "vigilante-frames" / "frames" / "cam01" / "frame.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    client = _FakeS3Client({})
    resolver = S3FrameResolver(
        endpoint="localhost:9000",
        access_key="minio",
        secret_key="minio123",
        cache_dir=tmp_path / "cache",
        client=client,
    )

    resolved_path = resolver.resolve_reference("s3://vigilante-frames/frames/cam01/frame.jpg")

    assert resolved_path == cached.resolve()
    assert client.downloads == []


def test_s3_frame_resolver_reports_missing_object(tmp_path) -> None:
    resolver = S3FrameResolver(
        endpoint="localhost:9000",
        access_key="minio",
        secret_key="minio123",
        cache_dir=tmp_path / "cache",
        client=_FakeS3Client({}, error=_FakeStorageError("NoSuchKey", "not found")),
    )

    with pytest.raises(FrameResolutionError) as excinfo:
        resolver.resolve_reference("s3://vigilante-frames/frames/missing.jpg")

    assert excinfo.value.reason == "remote_object_not_found"
    assert excinfo.value.details["storage_error_code"] == "NoSuchKey"


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [
        (_FakeStorageError("NoSuchBucket", "bucket missing"), "remote_bucket_not_found"),
        (_FakeStorageError("AccessDenied", "denied"), "remote_storage_auth_failed"),
        (TimeoutError("read timeout"), "remote_storage_timeout"),
        (_FakeEndpointError("connection refused"), "remote_storage_endpoint_unreachable"),
    ],
)
def test_s3_frame_resolver_classifies_storage_failures(tmp_path, error, expected_reason) -> None:
    resolver = S3FrameResolver(
        endpoint="localhost:9000",
        access_key="minio",
        secret_key="minio123",
        cache_dir=tmp_path / "cache",
        client=_FakeS3Client({}, error=error),
    )

    with pytest.raises(FrameResolutionError) as excinfo:
        resolver.resolve_reference("s3://vigilante-frames/frames/frame.jpg")

    assert excinfo.value.reason == expected_reason


class _FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], bytes], *, error: Exception | None = None) -> None:
        self.objects = objects
        self.error = error
        self.downloads: list[tuple[str, str, str]] = []

    def fget_object(self, bucket: str, object_key: str, file_path: str) -> None:
        if self.error is not None:
            raise self.error
        self.downloads.append((bucket, object_key, file_path))
        try:
            body = self.objects[(bucket, object_key)]
        except KeyError as exc:
            raise _FakeStorageError("NoSuchKey", "not found") from exc
        Path(file_path).write_bytes(body)
