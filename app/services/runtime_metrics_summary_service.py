from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable

REAL_SEMANTIC_BACKENDS = {"qwen", "qwen_vl", "smolvlm"}


class RuntimeMetricsSummaryService:
    SUMMARY_VERSION = "runtime_metrics_summary_v1"

    def summarize(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "schema_version": self.SUMMARY_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": 0,
            "by_camera": {},
            "by_face_backend": {},
            "by_semantic_backend": {},
            "by_event_type": {},
        }

        for record in records:
            if not isinstance(record, dict):
                continue
            summary["total_events"] += 1
            camera_id = str(record.get("camera_id") or "unknown")
            event_type = str(record.get("event_type") or "unknown")
            self._accumulate_operational_bucket(
                summary["by_camera"],
                camera_id,
                record,
            )
            self._accumulate_operational_bucket(
                summary["by_event_type"],
                event_type,
                record,
            )
            self._accumulate_face_backend(summary["by_face_backend"], record)
            self._accumulate_semantic_backend(summary["by_semantic_backend"], record)

        total_events = int(summary["total_events"])
        summary["by_camera"] = self._finalize_operational_buckets(summary["by_camera"])
        summary["by_event_type"] = self._finalize_operational_buckets(
            summary["by_event_type"]
        )
        summary["by_face_backend"] = self._finalize_face_backend_buckets(
            summary["by_face_backend"],
            total_events=total_events,
        )
        summary["by_semantic_backend"] = self._finalize_semantic_backend_buckets(
            summary["by_semantic_backend"],
            total_events=total_events,
        )
        return summary

    def summarize_store(self, store: Any) -> dict[str, Any]:
        return self.summarize(store.iter_records())

    def _accumulate_operational_bucket(
        self,
        buckets: dict[str, dict[str, Any]],
        key: str,
        record: dict[str, Any],
    ) -> None:
        bucket = buckets.setdefault(key, self._new_operational_bucket())
        bucket["processed_events"] += 1

        face = _as_dict(record.get("face"))
        semantic = _as_dict(record.get("semantic"))
        budget = _as_dict(record.get("budget"))

        if face.get("detected") is True:
            bucket["face_detected_count"] += 1
        if face.get("usable") is True:
            bucket["face_usable_count"] += 1
        if face.get("low_quality") is True:
            bucket["low_quality_face_count"] += 1
        _append_number(bucket["_face_latencies_ms"], face.get("detect_elapsed_ms"))

        semantic_backend = _semantic_backend_key(semantic)
        if semantic_backend:
            bucket["_semantic_backend_counts"][semantic_backend] += 1
            bucket["semantic_selected_count"] += 1
        if semantic.get("descriptor_valid") is True:
            bucket["semantic_success_count"] += 1
        if semantic.get("fallback_used") is True:
            bucket["semantic_fallback_count"] += 1
        if semantic.get("fallback_used") is True and semantic_backend == "simple":
            bucket["fallback_to_simple_count"] += 1
        if semantic.get("real_backend_attempted") is True:
            bucket["vlm_attempted_count"] += 1
        if (
            semantic.get("descriptor_valid") is True
            and semantic_backend in REAL_SEMANTIC_BACKENDS
        ):
            bucket["vlm_success_count"] += 1
        if semantic.get("parser_strategy") or semantic.get("parser_error"):
            bucket["parser_seen_count"] += 1
        if semantic.get("json_recovered") is True:
            bucket["parser_recovered_count"] += 1
        if _budget_rejected(budget):
            bucket["budget_rejected_count"] += 1
        _append_number(bucket["_vlm_durations_ms"], semantic.get("total_duration_ms"))

    def _accumulate_face_backend(
        self,
        buckets: dict[str, dict[str, Any]],
        record: dict[str, Any],
    ) -> None:
        face = _as_dict(record.get("face"))
        selected = _clean_key(face.get("selected")) or "unknown"
        requested = _clean_key(face.get("requested"))

        bucket = buckets.setdefault(selected, self._new_face_backend_bucket())
        bucket["selected_count"] += 1
        if face.get("fallback_used") is True:
            bucket["fallback_count"] += 1
        if face.get("usable") is True:
            bucket["usable_count"] += 1
        _append_number(bucket["_latencies_ms"], face.get("detect_elapsed_ms"))

        if requested:
            requested_bucket = buckets.setdefault(requested, self._new_face_backend_bucket())
            requested_bucket["requested_count"] += 1
            if face.get("fallback_used") is True and requested != selected:
                requested_bucket["fallback_away_count"] += 1

    def _accumulate_semantic_backend(
        self,
        buckets: dict[str, dict[str, Any]],
        record: dict[str, Any],
    ) -> None:
        semantic = _as_dict(record.get("semantic"))
        budget = _as_dict(record.get("budget"))
        selected = _semantic_backend_key(semantic) or "none"
        requested = _clean_key(semantic.get("effective_request")) or _clean_key(
            semantic.get("requested")
        )

        bucket = buckets.setdefault(selected, self._new_semantic_backend_bucket())
        bucket["selected_count"] += 1
        if semantic.get("descriptor_valid") is True:
            bucket["success_count"] += 1
        if semantic.get("fallback_used") is True:
            bucket["fallback_count"] += 1
        _append_number(bucket["_durations_ms"], semantic.get("total_duration_ms"))

        if requested:
            buckets.setdefault(requested, self._new_semantic_backend_bucket())[
                "requested_count"
            ] += 1

        parser_backend = _clean_key(semantic.get("parser_backend_key")) or selected
        parser_bucket = buckets.setdefault(parser_backend, self._new_semantic_backend_bucket())
        if semantic.get("json_recovered") is True:
            parser_bucket["parser_recovered_count"] += 1
        if _invalid_json_error(semantic.get("parser_error")):
            parser_bucket["invalid_json_count"] += 1

        budget_backend = _clean_key(budget.get("backend_key")) or selected
        budget_bucket = buckets.setdefault(budget_backend, self._new_semantic_backend_bucket())
        if _budget_rejected(budget):
            budget_bucket["budget_rejected_count"] += 1
        _append_number(budget_bucket["_observed_rss_mb"], budget.get("observed_rss_mb"))

    def _finalize_operational_buckets(
        self,
        buckets: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        finalized = {}
        for key, bucket in sorted(buckets.items()):
            processed = bucket["processed_events"]
            semantic_counts = bucket.pop("_semantic_backend_counts")
            face_latencies = bucket.pop("_face_latencies_ms")
            vlm_durations = bucket.pop("_vlm_durations_ms")
            bucket["face_detected_rate"] = _rate(bucket["face_detected_count"], processed)
            bucket["face_usable_rate"] = _rate(bucket["face_usable_count"], processed)
            bucket["low_quality_face_rate"] = _rate(
                bucket["low_quality_face_count"],
                processed,
            )
            bucket["semantic_backend_most_used"] = _most_common(semantic_counts)
            bucket["semantic_success_rate"] = _rate(
                bucket["semantic_success_count"],
                bucket["semantic_selected_count"],
            )
            bucket["vlm_success_rate"] = _rate(
                bucket["vlm_success_count"],
                bucket["vlm_attempted_count"],
            )
            bucket["fallback_to_simple_rate"] = _rate(
                bucket["fallback_to_simple_count"],
                processed,
            )
            bucket["semantic_fallback_rate"] = _rate(
                bucket["semantic_fallback_count"],
                processed,
            )
            bucket["parser_recovery_rate"] = _rate(
                bucket["parser_recovered_count"],
                bucket["parser_seen_count"],
            )
            bucket["face_detect_latency_ms"] = _latency_summary(face_latencies)
            bucket["vlm_duration_ms"] = _latency_summary(vlm_durations)
            finalized[key] = bucket
        return finalized

    def _finalize_face_backend_buckets(
        self,
        buckets: dict[str, dict[str, Any]],
        *,
        total_events: int,
    ) -> dict[str, dict[str, Any]]:
        finalized = {}
        for key, bucket in sorted(buckets.items()):
            selected_count = bucket["selected_count"]
            latencies = bucket.pop("_latencies_ms")
            bucket["usage_rate"] = _rate(selected_count, total_events)
            bucket["fallback_rate"] = _rate(bucket["fallback_count"], selected_count)
            bucket["fallback_away_rate"] = _rate(
                bucket["fallback_away_count"],
                bucket["requested_count"],
            )
            bucket["usable_rate"] = _rate(bucket["usable_count"], selected_count)
            bucket["detect_latency_ms"] = _latency_summary(latencies)
            finalized[key] = bucket
        return finalized

    def _finalize_semantic_backend_buckets(
        self,
        buckets: dict[str, dict[str, Any]],
        *,
        total_events: int,
    ) -> dict[str, dict[str, Any]]:
        finalized = {}
        for key, bucket in sorted(buckets.items()):
            selected_count = bucket["selected_count"]
            durations = bucket.pop("_durations_ms")
            rss_values = bucket.pop("_observed_rss_mb")
            bucket["usage_rate"] = _rate(selected_count, total_events)
            bucket["success_rate"] = _rate(bucket["success_count"], selected_count)
            bucket["fallback_rate"] = _rate(bucket["fallback_count"], selected_count)
            bucket["duration_ms"] = _latency_summary(durations)
            bucket["observed_rss_mb"] = _value_summary(rss_values)
            finalized[key] = bucket
        return finalized

    def _new_operational_bucket(self) -> dict[str, Any]:
        return {
            "processed_events": 0,
            "face_detected_count": 0,
            "face_usable_count": 0,
            "low_quality_face_count": 0,
            "semantic_selected_count": 0,
            "semantic_success_count": 0,
            "semantic_fallback_count": 0,
            "fallback_to_simple_count": 0,
            "vlm_attempted_count": 0,
            "vlm_success_count": 0,
            "parser_seen_count": 0,
            "parser_recovered_count": 0,
            "budget_rejected_count": 0,
            "_semantic_backend_counts": Counter(),
            "_face_latencies_ms": [],
            "_vlm_durations_ms": [],
        }

    def _new_face_backend_bucket(self) -> dict[str, Any]:
        return {
            "selected_count": 0,
            "requested_count": 0,
            "fallback_count": 0,
            "fallback_away_count": 0,
            "usable_count": 0,
            "_latencies_ms": [],
        }

    def _new_semantic_backend_bucket(self) -> dict[str, Any]:
        return {
            "selected_count": 0,
            "requested_count": 0,
            "success_count": 0,
            "fallback_count": 0,
            "parser_recovered_count": 0,
            "invalid_json_count": 0,
            "budget_rejected_count": 0,
            "_durations_ms": [],
            "_observed_rss_mb": [],
        }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _append_number(values: list[float], value: Any) -> None:
    coerced = _coerce_float(value)
    if coerced is not None:
        values.append(coerced)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_key(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned:
        return None
    if cleaned in {"simple_color_signature_v1", "simple_opencv_haar"}:
        return "simple"
    if cleaned in {"qwen_vl", "qwen-vl"}:
        return "qwen"
    if cleaned in {"smol_vlm", "smol-vlm"}:
        return "smolvlm"
    return cleaned


def _semantic_backend_key(semantic: dict[str, Any]) -> str | None:
    return _clean_key(semantic.get("selected_key")) or _clean_key(semantic.get("selected"))


def _budget_rejected(budget: dict[str, Any]) -> bool:
    status = _clean_key(budget.get("status"))
    if status in {"rejected", "exceeded"}:
        return True
    return bool(budget.get("rejection_reason") or budget.get("rejection_reasons"))


def _invalid_json_error(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).lower()
    return "invalid_json" in normalized or "missing_json" in normalized


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 2),
        "p50": round(_percentile(ordered, 50), 2),
        "p95": round(_percentile(ordered, 95), 2),
    }


def _value_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None}
    return {"count": len(values), "mean": round(mean(values), 2), "max": round(max(values), 2)}


def _percentile(ordered_values: list[float], percentile: int) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (percentile / 100) * (len(ordered_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = rank - lower
    return ordered_values[lower] + ((ordered_values[upper] - ordered_values[lower]) * weight)


def _most_common(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]
