from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileEventDeduper:
    """Local processed-event registry keyed by frame.ingested event_id."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def has_processed(self, event_id: str) -> bool:
        return event_id in self._state().get("processed_event_ids", {})

    def mark_processed(self, event_id: str, *, source_path: Path | str, line_number: int | None = None) -> None:
        state = self._state()
        processed = state.setdefault("processed_event_ids", {})
        processed[event_id] = {
            "source_path": self._source_path(source_path),
            "line_number": line_number,
            "processed_at": self._now(),
        }
        self._write_state(state)

    def reset(self) -> None:
        self._write_state({"processed_event_ids": {}})

    def _state(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"processed_event_ids": {}}
        try:
            with self.path.open("r", encoding="utf-8") as dedupe_file:
                loaded = json.load(dedupe_file)
        except (OSError, json.JSONDecodeError):
            return {"processed_event_ids": {}}
        if not isinstance(loaded, dict):
            return {"processed_event_ids": {}}
        loaded.setdefault("processed_event_ids", {})
        return loaded

    def _write_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as dedupe_file:
            json.dump(state, dedupe_file, indent=2, sort_keys=True)
            dedupe_file.write("\n")
        temp_path.replace(self.path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _source_path(self, source_path: Path | str) -> str:
        value = str(source_path)
        if value.startswith("rabbitmq:"):
            return value
        return str(Path(value).expanduser().resolve(strict=False))
