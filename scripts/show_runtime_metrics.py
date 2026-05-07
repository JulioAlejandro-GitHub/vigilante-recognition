#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from app.services.runtime_metrics_store import JsonlRuntimeMetricsStore  # noqa: E402
from app.services.runtime_metrics_summary_service import (  # noqa: E402
    RuntimeMetricsSummaryService,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show local vigilante-recognition runtime metrics summaries.",
    )
    parser.add_argument(
        "--path",
        default=settings.runtime_metrics_path,
        help="JSONL file or directory. Defaults to RUNTIME_METRICS_PATH.",
    )
    parser.add_argument(
        "--section",
        choices=["all", "camera", "backend", "event-type"],
        default="all",
        help="Summary section to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw summary JSON.",
    )
    args = parser.parse_args()

    store = JsonlRuntimeMetricsStore(
        args.path,
        rotate_max_mb=settings.runtime_metrics_rotate_max_mb,
        retention_files=settings.runtime_metrics_retention_files,
    )
    summary = RuntimeMetricsSummaryService().summarize_store(store)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    _print_summary(summary, section=args.section)


def _print_summary(summary: dict[str, Any], *, section: str) -> None:
    print(
        f"Runtime metrics: events={summary.get('total_events', 0)} "
        f"generated_at={summary.get('generated_at')}"
    )
    if section in {"all", "camera"}:
        _print_camera_summary(summary.get("by_camera", {}))
    if section in {"all", "backend"}:
        _print_face_backend_summary(summary.get("by_face_backend", {}))
        _print_semantic_backend_summary(summary.get("by_semantic_backend", {}))
    if section in {"all", "event-type"}:
        _print_event_type_summary(summary.get("by_event_type", {}))


def _print_camera_summary(rows: dict[str, Any]) -> None:
    print("\nBy camera")
    if not rows:
        print("  no data")
        return
    for camera_id, row in rows.items():
        face_latency = row.get("face_detect_latency_ms", {})
        vlm_duration = row.get("vlm_duration_ms", {})
        print(
            "  "
            f"{camera_id}: events={row.get('processed_events')} "
            f"face_detected={_pct(row.get('face_detected_rate'))} "
            f"face_usable={_pct(row.get('face_usable_rate'))} "
            f"low_quality={_pct(row.get('low_quality_face_rate'))} "
            f"face_p50_ms={face_latency.get('p50')} "
            f"face_p95_ms={face_latency.get('p95')} "
            f"semantic_top={row.get('semantic_backend_most_used')} "
            f"vlm_success={_pct(row.get('vlm_success_rate'))} "
            f"fallback_simple={_pct(row.get('fallback_to_simple_rate'))} "
            f"parser_recovery={_pct(row.get('parser_recovery_rate'))} "
            f"budget_rejected={row.get('budget_rejected_count')} "
            f"vlm_p50_ms={vlm_duration.get('p50')} "
            f"vlm_p95_ms={vlm_duration.get('p95')}"
        )


def _print_face_backend_summary(rows: dict[str, Any]) -> None:
    print("\nFace backends")
    if not rows:
        print("  no data")
        return
    for backend, row in rows.items():
        latency = row.get("detect_latency_ms", {})
        print(
            "  "
            f"{backend}: selected={row.get('selected_count')} "
            f"requested={row.get('requested_count')} "
            f"usage={_pct(row.get('usage_rate'))} "
            f"fallback={_pct(row.get('fallback_rate'))} "
            f"fallback_away={_pct(row.get('fallback_away_rate'))} "
            f"usable={_pct(row.get('usable_rate'))} "
            f"mean_ms={latency.get('mean')} "
            f"p50_ms={latency.get('p50')} "
            f"p95_ms={latency.get('p95')}"
        )


def _print_semantic_backend_summary(rows: dict[str, Any]) -> None:
    print("\nSemantic backends")
    if not rows:
        print("  no data")
        return
    for backend, row in rows.items():
        duration = row.get("duration_ms", {})
        rss = row.get("observed_rss_mb", {})
        print(
            "  "
            f"{backend}: selected={row.get('selected_count')} "
            f"requested={row.get('requested_count')} "
            f"success={row.get('success_count')} "
            f"fallback={row.get('fallback_count')} "
            f"parser_recovered={row.get('parser_recovered_count')} "
            f"invalid_json={row.get('invalid_json_count')} "
            f"budget_rejected={row.get('budget_rejected_count')} "
            f"mean_ms={duration.get('mean')} "
            f"p50_ms={duration.get('p50')} "
            f"p95_ms={duration.get('p95')} "
            f"rss_mean_mb={rss.get('mean')} "
            f"rss_max_mb={rss.get('max')}"
        )


def _print_event_type_summary(rows: dict[str, Any]) -> None:
    print("\nBy event_type")
    if not rows:
        print("  no data")
        return
    for event_type, row in rows.items():
        vlm_duration = row.get("vlm_duration_ms", {})
        print(
            "  "
            f"{event_type}: events={row.get('processed_events')} "
            f"vlm_attempted={row.get('vlm_attempted_count')} "
            f"vlm_success={_pct(row.get('vlm_success_rate'))} "
            f"semantic_success={_pct(row.get('semantic_success_rate'))} "
            f"fallback_simple={_pct(row.get('fallback_to_simple_rate'))} "
            f"semantic_fallback={_pct(row.get('semantic_fallback_rate'))} "
            f"parser_recovery={_pct(row.get('parser_recovery_rate'))} "
            f"budget_rejected={row.get('budget_rejected_count')} "
            f"semantic_top={row.get('semantic_backend_most_used')} "
            f"vlm_p50_ms={vlm_duration.get('p50')} "
            f"vlm_p95_ms={vlm_duration.get('p95')}"
        )


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


if __name__ == "__main__":
    main()
