from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_RECOMMENDATIONS_FILENAME = "recommendations.jsonl"


class JsonlRuntimeRecommendationStore:
    """Append-only JSONL store for generated operational recommendations."""

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_max_mb: float = 10.0,
        retention_files: int = 5,
    ) -> None:
        self.path = resolve_runtime_recommendations_path(path)
        self.rotate_max_bytes = max(0, int(float(rotate_max_mb) * 1024 * 1024))
        self.retention_files = max(0, int(retention_files))
        self._lock = Lock()

    def append(self, recommendation: dict[str, Any]) -> None:
        encoded = (
            json.dumps(recommendation, ensure_ascii=False, sort_keys=True, default=str)
            + "\n"
        ).encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(incoming_bytes=len(encoded))
            with self.path.open("ab") as handle:
                handle.write(encoded)

    def append_many(self, recommendations: Iterable[dict[str, Any]]) -> int:
        count = 0
        for recommendation in recommendations:
            self.append(recommendation)
            count += 1
        return count

    def iter_records(self) -> Iterable[dict[str, Any]]:
        for path in self.recommendation_files():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "runtime_recommendations_invalid_jsonl path=%s line=%s",
                                path,
                                line_number,
                            )
                            continue
                        if isinstance(parsed, dict):
                            yield parsed
            except FileNotFoundError:
                continue

    def recommendation_files(self) -> list[Path]:
        rotated = sorted(
            self.path.parent.glob(f"{self.path.stem}.*{self.path.suffix}"),
            key=lambda item: (item.stat().st_mtime, item.name),
        )
        files = [path for path in rotated if path.is_file()]
        if self.path.exists() and self.path.is_file():
            files.append(self.path)
        return files

    def _rotate_if_needed(self, *, incoming_bytes: int) -> None:
        if self.rotate_max_bytes <= 0 or not self.path.exists():
            return
        current_size = self.path.stat().st_size
        if current_size + incoming_bytes <= self.rotate_max_bytes:
            return

        timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        suffix = time.time_ns() % 1_000_000_000
        rotated_path = self.path.with_name(
            f"{self.path.stem}.{timestamp}.{suffix}{self.path.suffix}"
        )
        self.path.rename(rotated_path)
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        if self.retention_files <= 0:
            for path in self.path.parent.glob(f"{self.path.stem}.*{self.path.suffix}"):
                if path.is_file():
                    path.unlink(missing_ok=True)
            return

        rotated = sorted(
            [
                path
                for path in self.path.parent.glob(f"{self.path.stem}.*{self.path.suffix}")
                if path.is_file()
            ],
            key=lambda item: (item.stat().st_mtime, item.name),
        )
        for path in rotated[: max(0, len(rotated) - self.retention_files)]:
            path.unlink(missing_ok=True)


def resolve_runtime_recommendations_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() in {".jsonl", ".ndjson", ".log"}:
        return resolved
    return resolved / DEFAULT_RUNTIME_RECOMMENDATIONS_FILENAME
