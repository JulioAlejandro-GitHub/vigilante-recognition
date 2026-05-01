from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from app.domain.entities import FrameIngestedMessage
from app.services.canonical_frame_ref_service import CanonicalFrameRefService


class FrameResolutionError(FileNotFoundError):
    def __init__(
        self,
        *,
        frame_refs: list[str],
        attempted_paths: list[Path] | None = None,
        attempted_locations: list[str] | None = None,
        reason: str = "frame_not_found",
        details: dict | None = None,
    ) -> None:
        attempted_paths = attempted_paths or []
        attempted_locations = attempted_locations or []
        attempted = ", ".join(str(path) for path in attempted_paths) or "none"
        if attempted_locations:
            attempted = f"{attempted}; remote={', '.join(attempted_locations)}"
        refs = ", ".join(frame_refs) or "none"
        super().__init__(f"Could not resolve frame from refs [{refs}]. Reason: {reason}. Tried: {attempted}")
        self.frame_refs = frame_refs
        self.attempted_paths = attempted_paths
        self.attempted_locations = attempted_locations
        self.reason = reason
        self.details = details or {}


class RemoteFrameResolver(Protocol):
    def resolve_reference(self, reference: str) -> Path:
        """Download or materialize a remote frame reference and return a local path."""


@dataclass(frozen=True)
class _ResolutionAttempt:
    attempted_paths: list[Path]
    attempted_locations: list[str]
    errors: list[dict]


class LocalFrameResolver:
    """Resolve local frame paths from frame_uri first, then frame_ref."""

    def __init__(self, *, search_roots: list[Path] | None = None) -> None:
        self.search_roots = [Path(root).expanduser() for root in search_roots or []]

    def resolve(self, message: FrameIngestedMessage) -> Path:
        references = self._references(message)
        attempted_paths: list[Path] = []
        for reference in references:
            local_path = self._to_local_path(reference)
            if local_path is None:
                continue
            for candidate in self._candidate_paths(local_path):
                attempted_paths.append(candidate)
                if candidate.is_file():
                    return candidate.resolve()
        raise FrameResolutionError(frame_refs=references, attempted_paths=attempted_paths)

    def with_resolved_frame_ref(self, message: FrameIngestedMessage) -> FrameIngestedMessage:
        resolved_path = self.resolve(message)
        resolved_frame_ref = str(resolved_path)
        if message.frame_ref == resolved_frame_ref:
            return message

        canonical_frame_ref = CanonicalFrameRefService().resolve(message).frame_ref
        payload_metadata = dict(message.payload.metadata or {})
        if message.payload.frame_ref:
            payload_metadata.setdefault("original_frame_ref", message.payload.frame_ref)
        if message.payload.frame_uri:
            payload_metadata.setdefault("original_frame_uri", message.payload.frame_uri)

        resolved_payload = message.payload.model_copy(
            update={
                "frame_ref": resolved_frame_ref,
                "frame_uri": message.payload.frame_uri or resolved_frame_ref,
                "metadata": payload_metadata,
            }
        )
        return message.model_copy(
            update={
                "payload": resolved_payload,
                "canonical_frame_ref": canonical_frame_ref,
                "cached_path": resolved_frame_ref,
            }
        )

    def _references(self, message: FrameIngestedMessage) -> list[str]:
        refs = []
        if message.payload.frame_uri:
            refs.append(message.payload.frame_uri)
        if message.payload.frame_ref:
            refs.append(message.payload.frame_ref)
        return list(dict.fromkeys(refs))

    def _candidate_paths(self, local_path: Path) -> list[Path]:
        expanded = local_path.expanduser()
        if expanded.is_absolute():
            return [expanded]

        candidates = [Path.cwd() / expanded]
        candidates.extend(root / expanded for root in self.search_roots)
        return candidates

    def _to_local_path(self, reference: str) -> Path | None:
        parsed = urlparse(reference)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme in {"s3", "minio", "http", "https"}:
            return None
        return Path(reference)


class FrameResolver:
    """Resolve frame.ingested references from local files or S3-compatible storage."""

    def __init__(
        self,
        *,
        search_roots: list[Path] | None = None,
        local_resolver: LocalFrameResolver | None = None,
        remote_resolver: RemoteFrameResolver | None = None,
    ) -> None:
        self.local_resolver = local_resolver or LocalFrameResolver(search_roots=search_roots)
        self.remote_resolver = remote_resolver

    def resolve(self, message: FrameIngestedMessage) -> Path:
        references = self._references(message)
        attempt = _ResolutionAttempt(attempted_paths=[], attempted_locations=[], errors=[])

        for reference in references:
            if _is_remote_reference(reference):
                if self.remote_resolver is None:
                    attempt.attempted_locations.append(reference)
                    attempt.errors.append(
                        {
                            "reference": reference,
                            "reason": "remote_frame_resolver_not_configured",
                        }
                    )
                    continue
                try:
                    return self.remote_resolver.resolve_reference(reference)
                except FrameResolutionError as exc:
                    attempt.attempted_paths.extend(exc.attempted_paths)
                    attempt.attempted_locations.extend(exc.attempted_locations or [reference])
                    attempt.errors.append(
                        {
                            "reference": reference,
                            "reason": exc.reason,
                            "details": exc.details,
                        }
                    )
                    continue

            local_path = self.local_resolver._to_local_path(reference)
            if local_path is None:
                continue
            for candidate in self.local_resolver._candidate_paths(local_path):
                attempt.attempted_paths.append(candidate)
                if candidate.is_file():
                    return candidate.resolve()

        reason = attempt.errors[-1]["reason"] if attempt.errors else "frame_not_found"
        raise FrameResolutionError(
            frame_refs=references,
            attempted_paths=attempt.attempted_paths,
            attempted_locations=attempt.attempted_locations,
            reason=str(reason),
            details={"errors": attempt.errors} if attempt.errors else {},
        )

    def with_resolved_frame_ref(self, message: FrameIngestedMessage) -> FrameIngestedMessage:
        resolved_path = self.resolve(message)
        resolved_frame_ref = str(resolved_path)
        if message.frame_ref == resolved_frame_ref:
            return message

        canonical_frame_ref = CanonicalFrameRefService().resolve(message).frame_ref
        payload_metadata = dict(message.payload.metadata or {})
        if message.payload.frame_ref:
            payload_metadata.setdefault("original_frame_ref", message.payload.frame_ref)
        if message.payload.frame_uri:
            payload_metadata.setdefault("original_frame_uri", message.payload.frame_uri)

        resolved_payload = message.payload.model_copy(
            update={
                "frame_ref": resolved_frame_ref,
                "frame_uri": message.payload.frame_uri or resolved_frame_ref,
                "metadata": payload_metadata,
            }
        )
        return message.model_copy(
            update={
                "payload": resolved_payload,
                "canonical_frame_ref": canonical_frame_ref,
                "cached_path": resolved_frame_ref,
            }
        )

    def _references(self, message: FrameIngestedMessage) -> list[str]:
        refs = []
        if message.payload.frame_uri:
            refs.append(message.payload.frame_uri)
        if message.payload.frame_ref:
            refs.append(message.payload.frame_ref)
        return list(dict.fromkeys(refs))


def _is_remote_reference(reference: str) -> bool:
    return urlparse(reference).scheme in {"s3", "minio"}
