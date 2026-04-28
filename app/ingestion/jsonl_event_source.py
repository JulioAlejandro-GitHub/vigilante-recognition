from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class InvalidJsonlLineError(ValueError):
    def __init__(self, *, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"Invalid JSONL event at {path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason


@dataclass(frozen=True)
class JsonlEvent:
    line_number: int
    payload: dict[str, Any]


class JsonlEventSource:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def iter_events(self) -> Iterator[JsonlEvent]:
        with self.path.open("r", encoding="utf-8") as event_file:
            for line_number, raw_line in enumerate(event_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InvalidJsonlLineError(
                        path=self.path,
                        line_number=line_number,
                        reason=exc.msg,
                    ) from exc
                if not isinstance(payload, dict):
                    raise InvalidJsonlLineError(
                        path=self.path,
                        line_number=line_number,
                        reason="event must be a JSON object",
                    )
                yield JsonlEvent(line_number=line_number, payload=payload)

