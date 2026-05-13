from __future__ import annotations

import logging
import sys
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
PRIMARY_ID_KEYS = (
    "event_id",
    "camera_id",
    "subject_id",
    "track_id",
    "run_id",
    "recommendation_id",
    "case_id",
    "media_id",
)

_watcher_lock = Lock()
_watcher_thread: Thread | None = None
_watcher_stop = Event()
_watcher_path: Path | None = None
_last_applied_file_level: str | None = None


def configure_logging(
    level: str = "INFO",
    *,
    runtime_level_path: str | Path | None = None,
    watch: bool = True,
    poll_seconds: float = 2.0,
) -> None:
    resolved_level = set_log_level(level, source="startup", announce=False)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=resolved_level, format=LOG_FORMAT, stream=sys.stdout)
    else:
        _apply_level(resolved_level)
    logging.getLogger(__name__).info("log_level_configured level=%s", logging.getLevelName(resolved_level))
    if runtime_level_path is not None:
        apply_runtime_log_level_file(runtime_level_path, source="startup_file", missing_ok=True)
        if watch:
            start_runtime_log_level_watcher(runtime_level_path, poll_seconds=poll_seconds)


def set_log_level(level: str, *, source: str = "runtime", announce: bool = True) -> int:
    resolved_level = normalize_log_level(level)
    _apply_level(resolved_level)
    if announce:
        logging.getLogger(__name__).info(
            "log_level_changed level=%s source=%s",
            logging.getLevelName(resolved_level),
            source,
        )
    return resolved_level


def normalize_log_level(level: str) -> int:
    normalized = str(level or "").strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level!r}")
    return int(getattr(logging, normalized))


def current_log_level_name() -> str:
    return logging.getLevelName(logging.getLogger().getEffectiveLevel())


def write_runtime_log_level(path: str | Path, level: str, *, source: str = "runtime_file") -> str:
    resolved_level = logging.getLevelName(normalize_log_level(level))
    level_path = Path(path)
    level_path.parent.mkdir(parents=True, exist_ok=True)
    level_path.write_text(f"{resolved_level}\n", encoding="utf-8")
    set_log_level(resolved_level, source=source)
    global _last_applied_file_level
    _last_applied_file_level = resolved_level
    return resolved_level


def apply_runtime_log_level_file(
    path: str | Path,
    *,
    source: str = "runtime_file",
    missing_ok: bool = True,
) -> str | None:
    level_path = Path(path)
    try:
        raw_level = level_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if not raw_level:
        return None
    resolved_level = logging.getLevelName(normalize_log_level(raw_level))
    global _last_applied_file_level
    if resolved_level == _last_applied_file_level and current_log_level_name() == resolved_level:
        return resolved_level
    set_log_level(resolved_level, source=source)
    _last_applied_file_level = resolved_level
    return resolved_level


def start_runtime_log_level_watcher(path: str | Path, *, poll_seconds: float = 2.0) -> None:
    global _watcher_path, _watcher_thread
    level_path = Path(path)
    level_path.parent.mkdir(parents=True, exist_ok=True)
    poll = max(0.2, float(poll_seconds or 2.0))
    with _watcher_lock:
        if _watcher_thread is not None and _watcher_thread.is_alive() and _watcher_path == level_path:
            return
        _watcher_stop.clear()
        _watcher_path = level_path
        _watcher_thread = Thread(
            target=_watch_runtime_log_level,
            name="runtime-log-level-watcher",
            args=(level_path, poll),
            daemon=True,
        )
        _watcher_thread.start()


def stop_runtime_log_level_watcher() -> None:
    _watcher_stop.set()


def primary_id(payload: Mapping[str, Any] | None) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        return ("id", "")
    for container in (payload, _mapping(payload.get("context")), _mapping(payload.get("payload"))):
        for key in PRIMARY_ID_KEYS:
            value = container.get(key)
            if value not in (None, ""):
                return (key, str(value))
    return ("id", "")


def event_log_fields(event: Mapping[str, Any]) -> dict[str, str]:
    payload = _mapping(event.get("payload"))
    context = _mapping(event.get("context"))
    metadata = _mapping(payload.get("metadata"))
    return {
        "event_id": str(event.get("event_id") or context.get("event_id") or payload.get("event_id") or ""),
        "event_type": str(event.get("event_type") or payload.get("event_type") or ""),
        "camera_id": str(context.get("camera_id") or payload.get("camera_id") or ""),
        "track_id": str(context.get("track_id") or payload.get("track_id") or ""),
        "subject_id": str(context.get("subject_id") or payload.get("subject_id") or ""),
        "run_id": str(context.get("run_id") or _mapping(metadata.get("pipeline")).get("run_id") or ""),
    }


def compact_value(value: Any, *, max_chars: int = 96) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    head = max(8, max_chars - 17)
    return f"{text[:head]}...{text[-12:]}"


def _apply_level(level: int) -> None:
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


def _watch_runtime_log_level(path: Path, poll_seconds: float) -> None:
    last_mtime_ns: int | None = None
    while not _watcher_stop.wait(poll_seconds):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if last_mtime_ns == stat.st_mtime_ns:
            continue
        last_mtime_ns = stat.st_mtime_ns
        try:
            apply_runtime_log_level_file(path, source="runtime_file", missing_ok=True)
        except ValueError as exc:
            logging.getLogger(__name__).warning(
                "log_level_runtime_file_invalid path=%s error=%s",
                path,
                exc,
            )
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "log_level_runtime_file_unreadable path=%s error_type=%s",
                path,
                type(exc).__name__,
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
