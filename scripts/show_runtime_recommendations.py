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
from app.services.runtime_recommendation_service import (  # noqa: E402
    RuntimeRecommendationService,
)
from app.services.runtime_recommendation_store import (  # noqa: E402
    JsonlRuntimeRecommendationStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show local vigilante-recognition runtime recommendations.",
    )
    parser.add_argument(
        "--metrics-path",
        default=settings.runtime_metrics_path,
        help="Runtime metrics JSONL file or directory. Defaults to RUNTIME_METRICS_PATH.",
    )
    parser.add_argument(
        "--recommendations-path",
        default=settings.runtime_recommendations_path,
        help=(
            "Recommendation JSONL file or directory. Defaults to "
            "RUNTIME_RECOMMENDATIONS_PATH."
        ),
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=settings.runtime_recommendations_window_hours,
        help="Metrics lookback window. Use 0 to read all metrics.",
    )
    parser.add_argument(
        "--min-events-per-camera",
        type=int,
        default=settings.runtime_recommendations_min_events_per_camera,
        help="Minimum metrics events required before actionable recommendations.",
    )
    parser.add_argument(
        "--camera-id",
        help="Only show one camera.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Append generated recommendations to RUNTIME_RECOMMENDATIONS_PATH.",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Show persisted recommendation records instead of regenerating from metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit persisted records when --from-store is used.",
    )
    parser.add_argument(
        "--actionable-only",
        action="store_true",
        help="Hide status rows such as insufficient evidence or no changes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    args = parser.parse_args()

    metrics_store = JsonlRuntimeMetricsStore(
        args.metrics_path,
        rotate_max_mb=settings.runtime_metrics_rotate_max_mb,
        retention_files=settings.runtime_metrics_retention_files,
    )
    recommendation_store = JsonlRuntimeRecommendationStore(
        args.recommendations_path,
        rotate_max_mb=settings.runtime_recommendations_rotate_max_mb,
        retention_files=settings.runtime_recommendations_retention_files,
    )
    service = RuntimeRecommendationService(
        enabled=True,
        metrics_store=metrics_store,
        recommendation_store=recommendation_store,
        min_events_per_camera=args.min_events_per_camera,
        window_hours=args.window_hours,
        default_face_tuning={
            "face_backend": settings.face_backend,
            "det_size": settings.insightface_det_size,
            "detection_threshold": settings.insightface_detection_threshold,
            "max_faces": settings.insightface_max_faces,
            "face_quality_threshold": settings.face_quality_threshold,
            "min_face_bbox_size": settings.insightface_min_face_bbox_size,
            "min_face_area_ratio": settings.insightface_min_face_area_ratio,
            "config_source": "settings_defaults",
        },
        default_vlm_policy={
            "backend": settings.semantic_descriptor_backend,
            "preferred_backend": settings.vlm_auto_preferred_backend,
            "secondary_backend": settings.vlm_secondary_backend,
            "enabled_event_types": settings.vlm_event_type_policy,
            "disabled_event_types": [],
            "degradation_policy": settings.vlm_degradation_policy,
            "max_allowed_latency_seconds": settings.vlm_max_allowed_latency_seconds,
            "max_allowed_rss_mb": settings.vlm_max_allowed_rss_mb,
            "qwen_max_allowed_rss_mb": settings.qwen_max_allowed_rss_mb,
            "smolvlm_max_allowed_rss_mb": settings.smolvlm_max_allowed_rss_mb,
            "max_concurrent_inferences": settings.vlm_max_concurrent_inferences,
            "config_source": "settings_defaults",
        },
    )

    if args.from_store:
        records = service.read_persisted(
            camera_id=args.camera_id,
            limit=args.limit if args.limit > 0 else None,
        )
        if args.actionable_only:
            records = [record for record in records if record.get("actionable")]
        payload = {
            "schema_version": "runtime_recommendations_store_view_v1",
            "recommendations": records,
            "recommendation_count": len(records),
        }
    else:
        payload = service.generate(
            persist=args.persist,
            include_status=not args.actionable_only,
            camera_id=args.camera_id,
            force=True,
        )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.from_store:
        _print_recommendations(
            payload.get("recommendations", []),
            header=(
                f"Persisted runtime recommendations: "
                f"recommendations={payload.get('recommendation_count', 0)}"
            ),
        )
        return

    _print_generated(payload)


def _print_generated(payload: dict[str, Any]) -> None:
    window = payload.get("window_summary", {})
    print(
        "Runtime recommendations: "
        f"cameras={payload.get('camera_count', 0)} "
        f"recommendations={payload.get('recommendation_count', 0)} "
        f"actionable={payload.get('actionable_recommendation_count', 0)} "
        f"persisted={payload.get('persisted_count', 0)} "
        f"generated_at={payload.get('generated_at')}"
    )
    print(
        "Window: "
        f"hours={window.get('window_hours')} "
        f"records_in_window={window.get('metrics_records_in_window')} "
        f"first_event_at={window.get('first_event_at')} "
        f"last_event_at={window.get('last_event_at')}"
    )
    _print_recommendations(payload.get("recommendations", []), header="")


def _print_recommendations(recommendations: list[dict[str, Any]], *, header: str) -> None:
    if header:
        print(header)
    if not recommendations:
        print("\nNo recommendations.")
        return

    current_camera = None
    for recommendation in recommendations:
        camera_id = recommendation.get("camera_id")
        if camera_id != current_camera:
            current_camera = camera_id
            print(f"\nCamera {camera_id}")

        evidence = recommendation.get("evidence", {})
        print(
            "  "
            f"[{recommendation.get('severity')}] "
            f"{recommendation.get('recommendation_type')}: "
            f"{recommendation.get('title')}"
        )
        print(f"    reason: {recommendation.get('reason')}")
        print(f"    confidence: {recommendation.get('confidence')}")
        print(f"    current: {_compact_json(recommendation.get('current_value'))}")
        print(f"    suggested: {_compact_json(recommendation.get('suggested_value'))}")
        print(f"    evidence: {_compact_json(evidence)}")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
