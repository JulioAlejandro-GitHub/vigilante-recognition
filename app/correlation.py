from __future__ import annotations

from typing import Any


def extract_run_id(value: Any) -> str | None:
    event = _event_dict(value)
    context = _dict(event.get("context"))
    payload = _dict(event.get("payload"))
    metadata = _dict(payload.get("metadata"))

    candidates = [
        context.get("run_id"),
        context.get("smoke_run_id"),
        metadata.get("run_id"),
        _dict(metadata.get("pipeline")).get("run_id"),
        _dict(metadata.get("correlation")).get("run_id"),
        _dict(metadata.get("smoke")).get("run_id"),
    ]
    for candidate in candidates:
        cleaned = _clean(candidate)
        if cleaned:
            return cleaned
    return None


def source_correlation_payload(message: Any) -> dict[str, Any]:
    event = _event_dict(message)
    event_id = _clean(event.get("event_id"))
    payload = _dict(event.get("payload"))
    frame_ref = _clean(
        getattr(message, "canonical_frame_ref", None)
        or payload.get("frame_ref")
        or payload.get("frame_uri")
        or getattr(message, "frame_ref", None)
    )
    run_id = extract_run_id(message)

    correlation: dict[str, Any] = {}
    if run_id:
        correlation["run_id"] = run_id
    if event_id:
        correlation["source_event_id"] = event_id
        correlation["source_frame_event_id"] = event_id
    if frame_ref:
        correlation["source_frame_ref"] = frame_ref
    return correlation


def enrich_recognition_event_with_correlation(event: dict[str, Any], correlation: dict[str, Any]) -> dict[str, Any]:
    if not correlation:
        return event

    payload = _dict(event.setdefault("payload", {}))
    context = _dict(event.setdefault("context", {}))

    run_id = _clean(correlation.get("run_id"))
    source_event_id = _clean(correlation.get("source_event_id") or correlation.get("source_frame_event_id"))
    source_frame_event_id = _clean(correlation.get("source_frame_event_id") or source_event_id)
    source_frame_ref = _clean(correlation.get("source_frame_ref"))

    if run_id:
        context["run_id"] = run_id
    if source_event_id:
        context["source_event_id"] = source_event_id
        payload["source_event_id"] = source_event_id
    if source_frame_event_id:
        context["source_frame_event_id"] = source_frame_event_id
        payload["source_frame_event_id"] = source_frame_event_id
    if source_frame_ref:
        payload["source_frame_ref"] = source_frame_ref

    payload["correlation"] = {**_dict(payload.get("correlation")), **correlation}
    event["payload"] = payload
    event["context"] = context
    return event


def _event_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
