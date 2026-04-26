from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileCheckpointStore:
    """Local byte-offset checkpoints keyed by source path."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def get_offset(self, source_path: Path | str) -> int:
        record = self._state().get("sources", {}).get(self._source_key(source_path), {})
        try:
            return max(0, int(record.get("offset", 0)))
        except (TypeError, ValueError):
            return 0

    def mark_consumed(self, source_path: Path | str, *, offset: int, line_number: int | None = None) -> None:
        state = self._state()
        sources = state.setdefault("sources", {})
        sources[self._source_key(source_path)] = {
            "offset": max(0, int(offset)),
            "line_number": line_number,
            "updated_at": self._now(),
        }
        self._write_state(state)

    def reset(self, source_path: Path | str) -> None:
        state = self._state()
        state.setdefault("sources", {}).pop(self._source_key(source_path), None)
        self._write_state(state)

    def _state(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"sources": {}}
        try:
            with self.path.open("r", encoding="utf-8") as checkpoint_file:
                loaded = json.load(checkpoint_file)
        except (OSError, json.JSONDecodeError):
            return {"sources": {}}
        if not isinstance(loaded, dict):
            return {"sources": {}}
        loaded.setdefault("sources", {})
        return loaded

    def _write_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as checkpoint_file:
            json.dump(state, checkpoint_file, indent=2, sort_keys=True)
            checkpoint_file.write("\n")
        temp_path.replace(self.path)

    def _source_key(self, source_path: Path | str) -> str:
        return str(Path(source_path).expanduser().resolve(strict=False))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
