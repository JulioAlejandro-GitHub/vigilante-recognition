from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.domain.entities import FrameIngestedMessage, SemanticDescriptorResult
from app.services.canonical_frame_ref_service import CanonicalFrameRefService


CAMERA_ID = "11111111-1111-1111-1111-111111111111"


def test_canonical_frame_ref_prefers_frame_ref_over_cache_path(tmp_path: Path) -> None:
    cached_path = str(tmp_path / "cache" / "bucket" / "frame.jpg")
    message = _message(
        frame_ref=cached_path,
        frame_uri="s3://vigilante-frames/frames/cam01/frame.jpg",
        metadata={"original_frame_ref": "s3://vigilante-frames/frames/cam01/frame.jpg"},
        canonical_frame_ref="s3://vigilante-frames/frames/cam01/frame.jpg",
        cached_path=cached_path,
    )

    resolution = CanonicalFrameRefService().resolve(message)

    assert resolution.frame_ref == "s3://vigilante-frames/frames/cam01/frame.jpg"
    assert resolution.source == "message.canonical_frame_ref"


def test_canonical_frame_ref_uses_frame_uri_when_frame_ref_missing(tmp_path: Path) -> None:
    cached_path = str(tmp_path / "cache" / "bucket" / "frame.jpg")
    message = _message(
        frame_ref=cached_path,
        frame_uri="minio://vigilante-frames/frames/cam01/frame.jpg",
        metadata={"original_frame_uri": "minio://vigilante-frames/frames/cam01/frame.jpg"},
        cached_path=cached_path,
    )

    resolution = CanonicalFrameRefService().resolve(message)

    assert resolution.frame_ref == "minio://vigilante-frames/frames/cam01/frame.jpg"
    assert resolution.source == "payload.metadata.original_frame_uri"


def test_canonical_frame_ref_does_not_use_internal_cache_as_fallback() -> None:
    cached_path = ".runtime/ingestion/frame-cache/vigilante-frames/frame.jpg"
    message = _message(frame_ref=cached_path, cached_path=cached_path)

    resolution = CanonicalFrameRefService().resolve(message)

    assert resolution.frame_ref is None
    assert resolution.fallback_reason == "only_internal_cache_ref_available"


def test_canonicalize_semantic_descriptor_rewrites_top_level_and_descriptor_source() -> None:
    cached_path = ".runtime/ingestion/frame-cache/vigilante-frames/frame.jpg"
    result = SemanticDescriptorResult(
        generated=True,
        backend="simple_color_signature_v1",
        source_frame_ref=cached_path,
        descriptor={
            "descriptor_backend": "simple_color_signature_v1",
            "source_frame_ref": cached_path,
        },
    )

    canonical = CanonicalFrameRefService().canonicalize_semantic_descriptor(
        result,
        canonical_frame_ref="s3://vigilante-frames/frames/cam01/frame.jpg",
    )

    assert canonical.source_frame_ref == "s3://vigilante-frames/frames/cam01/frame.jpg"
    assert canonical.descriptor["source_frame_ref"] == "s3://vigilante-frames/frames/cam01/frame.jpg"


def _message(
    *,
    frame_ref: str | None = None,
    frame_uri: str | None = None,
    metadata: dict | None = None,
    canonical_frame_ref: str | None = None,
    cached_path: str | None = None,
) -> FrameIngestedMessage:
    captured_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return FrameIngestedMessage(
        event_id="evt_test",
        event_type="frame.ingested",
        event_version="1.0",
        occurred_at=captured_at,
        payload={
            "camera_id": CAMERA_ID,
            "captured_at": captured_at,
            "frame_ref": frame_ref,
            "frame_uri": frame_uri,
            "metadata": metadata or {},
        },
        context={"correlation_id": "corr_test"},
        canonical_frame_ref=canonical_frame_ref,
        cached_path=cached_path,
    )
