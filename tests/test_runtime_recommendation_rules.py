from __future__ import annotations

from app.services.runtime_recommendation_rules import evaluate_camera_recommendations


GENERATED_AT = "2026-05-07T12:00:00+00:00"
WINDOW = {
    "window_hours": 24,
    "window_start": "2026-05-06T12:00:00+00:00",
    "window_end": GENERATED_AT,
    "metrics_records_in_window": 20,
}


def test_recommendation_rules_report_insufficient_evidence() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(processed_events=3),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    assert recommendations[0]["recommendation_type"] == "status"
    assert recommendations[0]["suggested_value"] == "insufficient_evidence"
    assert recommendations[0]["actionable"] is False


def test_recommendation_rules_suggest_face_quality_tuning() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(
            face_detected_rate=1.0,
            face_usable_rate=0.30,
            low_quality_face_rate=0.70,
        ),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    face_recommendation = _first_type(recommendations, "face_tuning")
    assert face_recommendation["title"] == "Ajustar face_quality_threshold"
    assert face_recommendation["suggested_value"]["face_quality_threshold"] == 0.65
    assert "low_quality_face_rate" in face_recommendation["metrics_used"]
    assert face_recommendation["auto_apply"] is False


def test_recommendation_rules_suggest_conservative_vlm_policy_on_high_fallback() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(
            vlm_attempted_count=8,
            fallback_to_simple_count=7,
            fallback_to_simple_per_vlm_attempt_rate=0.875,
            event_types={
                "human_presence_no_face": {
                    "processed_events": 8,
                    "vlm_attempted_count": 8,
                }
            },
        ),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    policy_recommendation = _first_type(recommendations, "event_policy")
    assert policy_recommendation["severity"] == "high"
    assert policy_recommendation["suggested_value"]["enable_for_event_types"] == [
        "manual_review_required",
        "identity_conflict",
        "recurrent_unresolved_subject",
        "case_suggestion_created",
    ]
    assert policy_recommendation["suggested_value"]["disable_for_event_types"] == [
        "human_presence_no_face"
    ]


def test_recommendation_rules_suggest_budget_action_for_smolvlm_rejections() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(
            semantic_backends={
                "smolvlm": {
                    "attempted_count": 6,
                    "success_rate": 0.0,
                    "fallback_away_rate": 1.0,
                    "budget_rejected_count": 4,
                    "budget_rejection_rate": 0.6667,
                    "observed_rss_mb": {"count": 4, "mean": 19000, "max": 20000},
                }
            },
        ),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    budget_recommendation = _first_type(recommendations, "budget")
    assert budget_recommendation["title"] == "Evitar smolvlm por rechazos de budget"
    assert budget_recommendation["suggested_value"]["disabled_backend_candidate"] == "smolvlm"
    assert budget_recommendation["evidence"]["budget_rejected_count"] == 4


def test_recommendation_rules_suggest_qwen_when_qwen_is_good() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(
            semantic_backends={
                "qwen": {
                    "attempted_count": 8,
                    "success_count": 8,
                    "success_rate": 1.0,
                    "fallback_away_rate": 0.0,
                    "budget_rejection_rate": 0.0,
                    "parser_seen_count": 8,
                    "parser_recovery_rate": 0.25,
                    "invalid_json_rate": 0.0,
                },
                "smolvlm": {
                    "attempted_count": 5,
                    "success_count": 2,
                    "success_rate": 0.4,
                    "fallback_away_rate": 0.6,
                    "budget_rejection_rate": 0.0,
                    "invalid_json_rate": 0.0,
                },
            },
        ),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    qwen_recommendation = _first_type(recommendations, "vlm_policy")
    assert qwen_recommendation["suggested_value"]["preferred_backend"] == "qwen"
    assert qwen_recommendation["evidence"]["backend_success_rate"] == 1.0


def test_recommendation_rules_report_healthy_camera_without_forced_change() -> None:
    recommendations = evaluate_camera_recommendations(
        _camera_metrics(
            face_detected_rate=1.0,
            face_usable_rate=0.95,
            low_quality_face_rate=0.05,
            semantic_backends={},
        ),
        generated_at=GENERATED_AT,
        window_summary=WINDOW,
        min_events_per_camera=5,
    )

    assert len(recommendations) == 1
    assert recommendations[0]["title"] == "Sin cambios recomendados"
    assert recommendations[0]["suggested_value"] == "keep_current_configuration"
    assert recommendations[0]["actionable"] is False


def _camera_metrics(**overrides):
    base = {
        "camera_id": "cam-rules",
        "processed_events": 20,
        "face_detected_rate": 1.0,
        "face_usable_rate": 0.95,
        "low_quality_face_rate": 0.05,
        "small_face_rejection_rate": 0.0,
        "face_rejection_reasons": {},
        "face_detect_latency_ms": {"count": 20, "mean": 10, "p50": 10, "p95": 20},
        "face_backends": {},
        "semantic_backends": {},
        "event_types": {},
        "vlm_attempted_count": 0,
        "fallback_to_simple_count": 0,
        "fallback_to_simple_rate": 0.0,
        "fallback_to_simple_per_vlm_attempt_rate": 0.0,
        "semantic_fallback_rate": 0.0,
        "budget_rejected_count": 0,
        "current_face_tuning": {
            "det_size": "640,640",
            "detection_threshold": 0.5,
            "face_quality_threshold": 0.75,
            "min_face_bbox_size": 0,
            "min_face_area_ratio": 0.0,
        },
        "current_vlm_policy": {
            "backend": "auto",
            "preferred_backend": "smolvlm",
            "secondary_backend": "qwen",
            "smolvlm_max_allowed_rss_mb": 10240,
            "qwen_max_allowed_rss_mb": 12288,
        },
    }
    base.update(overrides)
    return base


def _first_type(recommendations, recommendation_type):
    return next(
        recommendation
        for recommendation in recommendations
        if recommendation["recommendation_type"] == recommendation_type
    )
