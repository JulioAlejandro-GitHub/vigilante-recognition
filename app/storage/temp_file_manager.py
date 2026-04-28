from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


class FrameCachePathManager:
    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).expanduser()

    def path_for_object(self, *, bucket: str, object_key: str) -> Path:
        bucket_part = self._sanitize_part(bucket)
        key_parts = [
            self._sanitize_part(part)
            for part in PurePosixPath(object_key).parts
            if part not in {"", ".", "..", "/"}
        ]
        if not key_parts:
            key_parts = ["frame"]
        return self.root_dir / bucket_part / Path(*key_parts)

    def _sanitize_part(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return sanitized or "_"
