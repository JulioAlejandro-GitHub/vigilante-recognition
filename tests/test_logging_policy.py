from __future__ import annotations

import logging

from app.logging import apply_runtime_log_level_file, current_log_level_name, set_log_level
from app.publisher import EventPublisher


def _recognition_event() -> dict:
    return {
        "event_id": "evt_recognition_001",
        "event_type": "manual_review_required",
        "context": {
            "camera_id": "11111111-1111-1111-1111-111111111111",
            "track_id": "track-1",
            "subject_id": "subject-1",
            "run_id": "run-1",
        },
        "payload": {
            "camera_runtime_config_trace": {"backend_chain": ["qwen", "smolvlm", "simple"]},
            "semantic_backend_trace": {"attempts": [{"raw_response": "x" * 2000}]},
            "large_nested_payload": {"text": "y" * 2000},
        },
    }


def test_recognition_event_info_is_compact_and_debug_keeps_payload(caplog) -> None:
    publisher = EventPublisher()

    with caplog.at_level(logging.INFO):
        publisher.publish(_recognition_event())

    info_text = "\n".join(record.getMessage() for record in caplog.records if record.levelno == logging.INFO)
    assert "recognition_event_ready event_id=evt_recognition_001" in info_text
    assert "event_type=manual_review_required" in info_text
    assert "camera_runtime_config_trace" not in info_text
    assert "semantic_backend_trace" not in info_text
    assert "large_nested_payload" not in info_text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        publisher.publish(_recognition_event())

    debug_text = "\n".join(record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG)
    assert "camera_runtime_config_trace" in debug_text
    assert "semantic_backend_trace" in debug_text
    assert "large_nested_payload" in debug_text


def test_runtime_log_level_file_changes_level_without_restart(tmp_path) -> None:
    previous_level = current_log_level_name()
    level_path = tmp_path / "log-level"
    try:
        level_path.write_text("DEBUG\n", encoding="utf-8")
        assert apply_runtime_log_level_file(level_path) == "DEBUG"
        assert current_log_level_name() == "DEBUG"

        level_path.write_text("INFO\n", encoding="utf-8")
        assert apply_runtime_log_level_file(level_path) == "INFO"
        assert current_log_level_name() == "INFO"
    finally:
        set_log_level(previous_level, source="test_restore", announce=False)
