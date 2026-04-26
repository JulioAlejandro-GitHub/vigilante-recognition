from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from app.domain.entities import FrameIngestedMessage


class FrameResolutionError(FileNotFoundError):
    def __init__(self, *, frame_refs: list[str], attempted_paths: list[Path]) -> None:
        attempted = ", ".join(str(path) for path in attempted_paths) or "none"
        refs = ", ".join(frame_refs) or "none"
        super().__init__(f"Could not resolve frame from refs [{refs}]. Tried: {attempted}")
        self.frame_refs = frame_refs
        self.attempted_paths = attempted_paths


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

        payload_metadata = dict(message.payload.metadata or {})
        payload_metadata.setdefault("original_frame_ref", message.frame_ref)
        if message.payload.frame_uri:
            payload_metadata.setdefault("original_frame_uri", message.payload.frame_uri)

        resolved_payload = message.payload.model_copy(
            update={
                "frame_ref": resolved_frame_ref,
                "frame_uri": message.payload.frame_uri or resolved_frame_ref,
                "metadata": payload_metadata,
            }
        )
        return message.model_copy(update={"payload": resolved_payload})

    def _references(self, message: FrameIngestedMessage) -> list[str]:
        refs = []
        if message.payload.frame_uri:
            refs.append(message.payload.frame_uri)
        refs.append(message.frame_ref)
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

