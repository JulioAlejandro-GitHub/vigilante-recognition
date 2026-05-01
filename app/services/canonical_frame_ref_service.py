from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.domain.entities import FrameIngestedMessage, SemanticDescriptorResult


SHARED_FRAME_SCHEMES = {"s3", "minio"}


@dataclass(frozen=True)
class CanonicalFrameRefResolution:
    frame_ref: str | None
    source: str | None = None
    fallback_reason: str | None = None


class CanonicalFrameRefService:
    """Resolve the portable frame reference to publish outside recognition."""

    def resolve(self, message: FrameIngestedMessage) -> CanonicalFrameRefResolution:
        candidates = self._candidates(message)

        for source, candidate in candidates:
            if candidate and self._is_shared_reference(candidate) and not self.is_internal_cache_ref(candidate, message):
                return CanonicalFrameRefResolution(frame_ref=candidate, source=source)

        for source, candidate in candidates:
            if candidate and not self.is_internal_cache_ref(candidate, message):
                return CanonicalFrameRefResolution(
                    frame_ref=candidate,
                    source=source,
                    fallback_reason="no_shared_remote_frame_ref_available",
                )

        return CanonicalFrameRefResolution(
            frame_ref=None,
            fallback_reason="only_internal_cache_ref_available",
        )

    def canonicalize_semantic_descriptor(
        self,
        result: SemanticDescriptorResult,
        *,
        canonical_frame_ref: str | None,
    ) -> SemanticDescriptorResult:
        if not canonical_frame_ref:
            return result

        descriptor = dict(result.descriptor or {})
        descriptor["source_frame_ref"] = canonical_frame_ref
        return result.model_copy(
            update={
                "source_frame_ref": canonical_frame_ref,
                "descriptor": descriptor,
            }
        )

    def is_internal_cache_ref(self, reference: str, message: FrameIngestedMessage | None = None) -> bool:
        if not reference:
            return False

        if message is not None:
            internal_refs = {
                message.cached_path,
                (message.payload.metadata or {}).get("cached_path"),
                (message.payload.metadata or {}).get("_recognition_cached_path"),
            }
            if reference in {ref for ref in internal_refs if isinstance(ref, str) and ref}:
                return True

        parsed = urlparse(reference)
        if parsed.scheme and parsed.scheme != "file":
            return False

        normalized = reference.replace("\\", "/")
        if "/.runtime/ingestion/frame-cache/" in normalized or normalized.startswith(".runtime/ingestion/frame-cache/"):
            return True

        try:
            path = Path(reference).expanduser().resolve(strict=False)
            cache_dir = Path(settings.storage_s3_cache_dir).expanduser().resolve(strict=False)
            return path == cache_dir or cache_dir in path.parents
        except (OSError, RuntimeError, ValueError):
            return False

    def _candidates(self, message: FrameIngestedMessage) -> list[tuple[str, str | None]]:
        metadata = dict(message.payload.metadata or {})
        return [
            ("message.canonical_frame_ref", message.canonical_frame_ref),
            ("payload.metadata.original_frame_ref", metadata.get("original_frame_ref")),
            ("payload.frame_ref", message.payload.frame_ref),
            ("payload.metadata.original_frame_uri", metadata.get("original_frame_uri")),
            ("payload.frame_uri", message.payload.frame_uri),
        ]

    def _is_shared_reference(self, reference: str) -> bool:
        parsed = urlparse(reference)
        return parsed.scheme in SHARED_FRAME_SCHEMES and bool(parsed.netloc and parsed.path.strip("/"))
