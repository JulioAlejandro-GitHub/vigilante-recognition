from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RejectedEventStore:
    """Append-only local DLQ for invalid or non-processable ingestion events."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        reason: str,
        source_path: Path | str,
        line_number: int | None = None,
        offset: int | None = None,
        event_id: str | None = None,
        event_type: str | None = None,
        details: dict[str, Any] | None = None,
        raw_line: str | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "rejected_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_path": self._source_path(source_path),
            "line_number": line_number,
            "offset": offset,
            "event_id": event_id,
            "event_type": event_type,
            "details": details or {},
        }
        if raw_line is not None:
            record["raw_line"] = raw_line[:4000]

        with self.path.open("a", encoding="utf-8") as rejected_file:
            rejected_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            rejected_file.write("\n")

    def _source_path(self, source_path: Path | str) -> str:
        value = str(source_path)
        if value.startswith("rabbitmq:"):
            return value
        return str(Path(value).expanduser().resolve(strict=False))
