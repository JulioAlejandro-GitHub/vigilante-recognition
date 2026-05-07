#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.domain.entities import FaceDetectionResult
from app.services.semantic_descriptor_service import SemanticDescriptorService
from app.services.vlm_execution_policy_service import validate_vlm_runtime_config
from app.services.vlm_metrics_service import (
    compare_vlm_trace_summaries,
    extract_vlm_trace_summary,
)


def main() -> int:
    args = _parse_args()
    output_path = Path(args.write_json).expanduser() if args.write_json else None
    backends = _selected_backends(args.backend)
    results: list[dict[str, Any]] = []

    for backend in backends:
        started_at = time.perf_counter()
        result = _run_backend_validation(backend=backend, args=args)
        result["wall_duration_ms"] = _duration_ms(started_at)
        results.append(result)

    comparison = compare_vlm_trace_summaries([result["summary"] for result in results])
    payload = {
        "validation_schema_version": "vlm_runtime_validation_v1",
        "config": validate_vlm_runtime_config(),
        "results": results,
        "comparison": comparison,
    }

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(encoded)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")

    if args.require_real:
        failed = [
            result
            for result in results
            if not result["summary"].get("generated")
            or result["summary"].get("fallback_used")
            or result["summary"].get("selected_backend_key") not in {"qwen", "smolvlm"}
        ]
        if failed:
            return 2
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real Qwen/SmolVLM2 semantic runtime and emit comparable metrics.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=["qwen", "smolvlm", "auto", "both"],
        default=None,
        help="Backend to validate. Repeatable. Use 'both' for qwen and smolvlm.",
    )
    parser.add_argument(
        "--image",
        default="tests/fixtures/images/face_low_quality.jpg",
        help="Local image path used for inference.",
    )
    parser.add_argument(
        "--source-frame-ref",
        default=None,
        help="Canonical source_frame_ref to publish in the descriptor. Defaults to s3://vigilante-frames/vlm-validation/<image-name>.",
    )
    parser.add_argument("--event-type", default="manual_review_required")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-image-edge", type=int, default=None)
    parser.add_argument("--auto-preferred-backend", choices=["qwen", "smolvlm"], default=None)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the same backend N times in one process to measure warm runner reuse.",
    )
    parser.add_argument("--write-json", default=".runtime/vlm/last_validation.json")
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Exit non-zero if any requested real backend falls back or fails.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable simple fallback for this validation run.",
    )
    return parser.parse_args()


def _selected_backends(raw_backends: list[str] | None) -> list[str]:
    selected = raw_backends or ["both"]
    result: list[str] = []
    for backend in selected:
        expanded = ["qwen", "smolvlm"] if backend == "both" else [backend]
        for item in expanded:
            if item not in result:
                result.append(item)
    return result


def _run_backend_validation(*, backend: str, args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    source_frame_ref = args.source_frame_ref or f"s3://vigilante-frames/vlm-validation/{image_path.name}"

    _apply_runtime_overrides(backend=backend, args=args)
    service = SemanticDescriptorService()
    face_detection = FaceDetectionResult(
        detected=True,
        usable=False,
        image_size=None,
        rejection_reasons=["validation_runtime_fixture"],
    )
    runs: list[dict[str, Any]] = []
    for run_index in range(max(1, int(args.repeat))):
        run_started_at = time.perf_counter()
        descriptor = service.generate(
            frame_ref=str(image_path),
            source_frame_ref=source_frame_ref,
            face_detection=face_detection,
            event_type_hint=args.event_type,
        )
        summary = extract_vlm_trace_summary(descriptor)
        runs.append(
            {
                "run_index": run_index + 1,
                "wall_duration_ms": _duration_ms(run_started_at),
                "summary": summary,
                "descriptor": _descriptor_payload(descriptor),
            }
        )

    last_run = runs[-1]
    return {
        "backend": backend,
        "summary": last_run["summary"],
        "descriptor": last_run["descriptor"],
        "runs": runs,
    }


def _apply_runtime_overrides(*, backend: str, args: argparse.Namespace) -> None:
    settings.semantic_descriptor_backend = backend
    settings.semantic_enable_fallback = not args.no_fallback
    settings.qwen_vl_enabled = backend in {"qwen", "auto"}
    settings.smolvlm_enabled = backend in {"smolvlm", "auto"}

    if backend == "qwen":
        settings.smolvlm_enabled = False
    elif backend == "smolvlm":
        settings.qwen_vl_enabled = False
    elif backend == "auto":
        settings.qwen_vl_enabled = True
        settings.smolvlm_enabled = True

    if args.device:
        settings.vlm_device = args.device
        settings.semantic_device = ""
    if args.timeout is not None:
        settings.vlm_timeout_seconds = args.timeout
        settings.semantic_timeout_seconds = None
    if args.max_new_tokens is not None:
        settings.vlm_max_new_tokens = args.max_new_tokens
    if args.max_image_edge is not None:
        settings.vlm_max_image_edge = args.max_image_edge
    if args.auto_preferred_backend:
        settings.vlm_auto_preferred_backend = args.auto_preferred_backend


def _duration_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _descriptor_payload(descriptor) -> dict[str, Any]:
    return {
        "backend": descriptor.backend,
        "generated": descriptor.generated,
        "confidence": descriptor.confidence,
        "source_frame_ref": descriptor.source_frame_ref,
        "raw_summary": descriptor.descriptor.get("raw_summary"),
        "semantic_backend_trace": descriptor.descriptor.get("semantic_backend_trace"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
