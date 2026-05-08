from __future__ import annotations

from copy import deepcopy
from typing import Any

RECOMMENDATION_SCHEMA_VERSION = "runtime_recommendation_v1"
RECOMMENDATION_RULESET_VERSION = "runtime_recommendation_rules_v1"

LOW_FACE_DETECTED_RATE = 0.50
LOW_FACE_USABLE_RATE = 0.55
HIGH_LOW_QUALITY_FACE_RATE = 0.35
FACE_LATENCY_REASONABLE_P95_MS = 250.0
SMALL_FACE_REJECTION_RATE = 0.25

MIN_BACKEND_ATTEMPTS = 3
GOOD_BACKEND_SUCCESS_RATE = 0.80
LOW_BACKEND_FALLBACK_RATE = 0.20
HIGH_BACKEND_FALLBACK_RATE = 0.60
HIGH_BUDGET_REJECTION_RATE = 0.30
HIGH_INVALID_JSON_RATE = 0.30
SAFE_RSS_INCREASE_FACTOR = 1.25
SUGGESTED_RSS_HEADROOM_FACTOR = 1.15
MAX_SUGGESTED_RSS_MB = 24576

HIGH_VALUE_EVENT_TYPES = [
    "manual_review_required",
    "identity_conflict",
    "recurrent_unresolved_subject",
    "case_suggestion_created",
]


def evaluate_camera_recommendations(
    camera_metrics: dict[str, Any],
    *,
    generated_at: str,
    window_summary: dict[str, Any],
    min_events_per_camera: int,
    include_status: bool = True,
) -> list[dict[str, Any]]:
    """Apply explicit if/then rules to one camera metrics aggregate."""

    processed_events = int(camera_metrics.get("processed_events") or 0)
    camera_window = {
        **window_summary,
        "camera_events": processed_events,
        "min_events_per_camera": int(min_events_per_camera),
    }

    if processed_events < min_events_per_camera:
        if not include_status:
            return []
        return [
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=camera_window,
                recommendation_type="status",
                severity="info",
                title="Evidencia insuficiente",
                reason=(
                    "La camara no alcanza el volumen minimo configurado para sugerir "
                    "cambios operativos."
                ),
                evidence={
                    "processed_events": processed_events,
                    "min_events_per_camera": min_events_per_camera,
                },
                current_value=None,
                suggested_value="insufficient_evidence",
                confidence=0.2,
                metrics_used=["processed_events"],
                actionable=False,
            )
        ]

    recommendations: list[dict[str, Any]] = []
    recommendations.extend(
        _face_tuning_recommendations(
            camera_metrics,
            generated_at=generated_at,
            window_summary=camera_window,
        )
    )
    recommendations.extend(
        _vlm_policy_recommendations(
            camera_metrics,
            generated_at=generated_at,
            window_summary=camera_window,
        )
    )

    if not recommendations and include_status:
        recommendations.append(
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=camera_window,
                recommendation_type="status",
                severity="info",
                title="Sin cambios recomendados",
                reason=(
                    "Las tasas de rostro, fallback, parser, budget y latencia estan "
                    "dentro de los umbrales simples de este rule set."
                ),
                evidence={
                    "processed_events": processed_events,
                    "face_detected_rate": camera_metrics.get("face_detected_rate"),
                    "face_usable_rate": camera_metrics.get("face_usable_rate"),
                    "semantic_backend_most_used": camera_metrics.get(
                        "semantic_backend_most_used"
                    ),
                    "fallback_to_simple_rate": camera_metrics.get(
                        "fallback_to_simple_rate"
                    ),
                    "budget_rejected_count": camera_metrics.get(
                        "budget_rejected_count"
                    ),
                },
                current_value={
                    "face_tuning": camera_metrics.get("current_face_tuning"),
                    "vlm_policy": camera_metrics.get("current_vlm_policy"),
                },
                suggested_value="keep_current_configuration",
                confidence=0.75,
                metrics_used=[
                    "face_detected_rate",
                    "face_usable_rate",
                    "fallback_to_simple_rate",
                    "budget_rejected_count",
                ],
                actionable=False,
            )
        )

    return recommendations


def _face_tuning_recommendations(
    camera_metrics: dict[str, Any],
    *,
    generated_at: str,
    window_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    face_detected_rate = _float(camera_metrics.get("face_detected_rate"))
    face_usable_rate = _float(camera_metrics.get("face_usable_rate"))
    low_quality_rate = _float(camera_metrics.get("low_quality_face_rate"))
    face_latency = _dict(camera_metrics.get("face_detect_latency_ms"))
    p95_ms = _optional_float(face_latency.get("p95"))
    current = _dict(camera_metrics.get("current_face_tuning"))

    if face_detected_rate < LOW_FACE_DETECTED_RATE:
        latency_has_margin = p95_ms is None or p95_ms <= FACE_LATENCY_REASONABLE_P95_MS
        suggested: dict[str, Any] = {}
        if latency_has_margin:
            suggested["det_size"] = _increase_det_size(current.get("det_size"))
        else:
            suggested["detection_threshold"] = _decrease_probability(
                current.get("detection_threshold"),
                step=0.05,
                floor=0.2,
            )

        recommendations.append(
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=window_summary,
                recommendation_type="face_tuning",
                severity="high" if face_detected_rate < 0.25 else "medium",
                title=(
                    "Subir det_size de InsightFace"
                    if latency_has_margin
                    else "Revisar detection_threshold antes de subir det_size"
                ),
                reason=(
                    "La tasa de face_detected es baja. La latencia facial deja margen "
                    "para probar mayor resolucion de deteccion."
                    if latency_has_margin
                    else "La tasa de face_detected es baja, pero la latencia p95 ya es "
                    "alta para subir resolucion sin revisar primero el threshold."
                ),
                evidence={
                    "face_detected_rate": face_detected_rate,
                    "face_detect_latency_p95_ms": p95_ms,
                    "thresholds": {
                        "low_face_detected_rate": LOW_FACE_DETECTED_RATE,
                        "reasonable_p95_ms": FACE_LATENCY_REASONABLE_P95_MS,
                    },
                },
                current_value={
                    "det_size": current.get("det_size"),
                    "detection_threshold": current.get("detection_threshold"),
                },
                suggested_value=suggested,
                confidence=0.72 if latency_has_margin else 0.58,
                metrics_used=["face_detected_rate", "face_detect_latency_ms.p95"],
            )
        )

    if (
        face_detected_rate >= 0.80
        and (
            face_usable_rate < LOW_FACE_USABLE_RATE
            or low_quality_rate >= HIGH_LOW_QUALITY_FACE_RATE
        )
    ):
        small_face_rate = _float(camera_metrics.get("small_face_rejection_rate"))
        if small_face_rate >= SMALL_FACE_REJECTION_RATE:
            suggested_value = {
                "min_face_bbox_size": _decrease_int(
                    current.get("min_face_bbox_size"),
                    ratio=0.8,
                ),
                "min_face_area_ratio": _decrease_probability(
                    current.get("min_face_area_ratio"),
                    step=0.005,
                    floor=0.0,
                ),
            }
            title = "Revisar filtros de tamano minimo de rostro"
            reason = (
                "Hay muchas detecciones descartadas por senales de rostro pequeno. "
                "Conviene revisar min_face_bbox_size y min_face_area_ratio para esta camara."
            )
            metrics_used = [
                "face_detected_rate",
                "face_usable_rate",
                "low_quality_face_rate",
                "small_face_rejection_rate",
            ]
        else:
            suggested_value = {
                "face_quality_threshold": _decrease_probability(
                    current.get("face_quality_threshold"),
                    step=0.10,
                    floor=0.35,
                )
            }
            title = "Ajustar face_quality_threshold"
            reason = (
                "La camara detecta rostros, pero una fraccion alta queda como no usable. "
                "Probar un umbral de calidad menor puede recuperar candidatos sin tocar "
                "la configuracion global."
            )
            metrics_used = [
                "face_detected_rate",
                "face_usable_rate",
                "low_quality_face_rate",
            ]

        recommendations.append(
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=window_summary,
                recommendation_type="face_tuning",
                severity="high" if face_usable_rate < 0.30 else "medium",
                title=title,
                reason=reason,
                evidence={
                    "face_detected_rate": face_detected_rate,
                    "face_usable_rate": face_usable_rate,
                    "low_quality_face_rate": low_quality_rate,
                    "small_face_rejection_rate": small_face_rate,
                    "rejection_reasons": camera_metrics.get(
                        "face_rejection_reasons", {}
                    ),
                    "thresholds": {
                        "low_face_usable_rate": LOW_FACE_USABLE_RATE,
                        "high_low_quality_face_rate": HIGH_LOW_QUALITY_FACE_RATE,
                    },
                },
                current_value={
                    "face_quality_threshold": current.get("face_quality_threshold"),
                    "min_face_bbox_size": current.get("min_face_bbox_size"),
                    "min_face_area_ratio": current.get("min_face_area_ratio"),
                },
                suggested_value=suggested_value,
                confidence=0.78,
                metrics_used=metrics_used,
            )
        )

    face_backends = _dict(camera_metrics.get("face_backends"))
    insightface = _dict(face_backends.get("insightface"))
    if (
        int(insightface.get("requested_count") or 0) >= MIN_BACKEND_ATTEMPTS
        and _float(insightface.get("fallback_away_rate")) >= HIGH_BACKEND_FALLBACK_RATE
    ):
        recommendations.append(
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=window_summary,
                recommendation_type="face_tuning",
                severity="medium",
                title="Evitar InsightFace temporalmente en esta camara",
                reason=(
                    "InsightFace fue solicitado repetidamente y termino cayendo a otro "
                    "backend con demasiada frecuencia."
                ),
                evidence={
                    "insightface_requested_count": insightface.get("requested_count"),
                    "insightface_fallback_away_rate": insightface.get(
                        "fallback_away_rate"
                    ),
                    "insightface_error_count": insightface.get("error_count"),
                },
                current_value={"face_backend": current.get("face_backend") or "auto"},
                suggested_value={"face_backend": "simple"},
                confidence=0.7,
                metrics_used=[
                    "face_backends.insightface.requested_count",
                    "face_backends.insightface.fallback_away_rate",
                ],
            )
        )

    return recommendations


def _vlm_policy_recommendations(
    camera_metrics: dict[str, Any],
    *,
    generated_at: str,
    window_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    semantic_backends = _dict(camera_metrics.get("semantic_backends"))
    current_policy = _dict(camera_metrics.get("current_vlm_policy"))
    qwen = _dict(semantic_backends.get("qwen"))
    smolvlm = _dict(semantic_backends.get("smolvlm"))

    for backend_key, backend in [("qwen", qwen), ("smolvlm", smolvlm)]:
        recommendations.extend(
            _backend_budget_recommendations(
                camera_metrics,
                backend_key=backend_key,
                backend=backend,
                current_policy=current_policy,
                generated_at=generated_at,
                window_summary=window_summary,
            )
        )
        recommendations.extend(
            _backend_parser_recommendations(
                camera_metrics,
                backend_key=backend_key,
                backend=backend,
                current_policy=current_policy,
                generated_at=generated_at,
                window_summary=window_summary,
            )
        )

    vlm_attempted_count = int(camera_metrics.get("vlm_attempted_count") or 0)
    fallback_to_simple_per_vlm = _float(
        camera_metrics.get("fallback_to_simple_per_vlm_attempt_rate")
    )
    if (
        vlm_attempted_count >= MIN_BACKEND_ATTEMPTS
        and fallback_to_simple_per_vlm >= HIGH_BACKEND_FALLBACK_RATE
    ):
        low_value_event_types = _low_value_event_types(camera_metrics)
        if low_value_event_types:
            recommendation_type = "event_policy"
            title = "Limitar VLM a eventos de mayor valor"
            suggested_value = {
                "enable_for_event_types": HIGH_VALUE_EVENT_TYPES,
                "disable_for_event_types": low_value_event_types,
            }
            reason = (
                "La camara degrada a simple en la mayoria de intentos VLM y parte del "
                "volumen viene de eventos de menor valor operacional."
            )
            metrics_used = [
                "vlm_attempted_count",
                "fallback_to_simple_per_vlm_attempt_rate",
                "event_types",
            ]
        else:
            recommendation_type = "vlm_policy"
            title = "Forzar simple temporalmente"
            suggested_value = {"backend": "simple", "force_simple": True}
            reason = (
                "VLM casi siempre termina en fallback simple para esta camara. Mantener "
                "simple reduce latencia y consumo hasta corregir el backend real."
            )
            metrics_used = [
                "vlm_attempted_count",
                "fallback_to_simple_per_vlm_attempt_rate",
            ]

        recommendations.append(
            _recommendation(
                camera_metrics,
                generated_at=generated_at,
                window_summary=window_summary,
                recommendation_type=recommendation_type,
                severity="high" if fallback_to_simple_per_vlm >= 0.80 else "medium",
                title=title,
                reason=reason,
                evidence={
                    "vlm_attempted_count": vlm_attempted_count,
                    "fallback_to_simple_per_vlm_attempt_rate": fallback_to_simple_per_vlm,
                    "semantic_fallback_rate": camera_metrics.get(
                        "semantic_fallback_rate"
                    ),
                    "low_value_event_types": low_value_event_types,
                },
                current_value=current_policy,
                suggested_value=suggested_value,
                confidence=0.76,
                metrics_used=metrics_used,
            )
        )

    preferred = _preferred_backend_recommendation(
        camera_metrics,
        current_policy=current_policy,
        qwen=qwen,
        smolvlm=smolvlm,
        generated_at=generated_at,
        window_summary=window_summary,
    )
    if preferred is not None:
        recommendations.append(preferred)

    return recommendations


def _backend_budget_recommendations(
    camera_metrics: dict[str, Any],
    *,
    backend_key: str,
    backend: dict[str, Any],
    current_policy: dict[str, Any],
    generated_at: str,
    window_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    attempted_count = int(backend.get("attempted_count") or 0)
    budget_rejected_count = int(backend.get("budget_rejected_count") or 0)
    budget_rate = _float(backend.get("budget_rejection_rate"))
    if (
        attempted_count < MIN_BACKEND_ATTEMPTS
        or budget_rejected_count < 2
        or budget_rate < HIGH_BUDGET_REJECTION_RATE
    ):
        return []

    observed_rss = _dict(backend.get("observed_rss_mb"))
    observed_rss_max = _optional_float(observed_rss.get("max"))
    current_backend_budget = _optional_float(
        current_policy.get(f"{backend_key}_max_allowed_rss_mb")
    )
    suggested_value: dict[str, Any]
    title: str
    reason: str
    severity = "high" if budget_rate >= 0.60 else "medium"

    if (
        observed_rss_max is not None
        and current_backend_budget is not None
        and current_backend_budget > 0
        and observed_rss_max <= current_backend_budget * SAFE_RSS_INCREASE_FACTOR
        and observed_rss_max * SUGGESTED_RSS_HEADROOM_FACTOR <= MAX_SUGGESTED_RSS_MB
    ):
        suggested_budget = int(
            round(observed_rss_max * SUGGESTED_RSS_HEADROOM_FACTOR)
        )
        suggested_value = {f"{backend_key}_max_allowed_rss_mb": suggested_budget}
        title = f"Ajustar budget RSS de {backend_key}"
        reason = (
            "El backend tiene rechazos por budget, pero el RSS observado queda cerca "
            "del limite actual. Se puede evaluar un aumento acotado."
        )
    else:
        alternative = "qwen" if backend_key == "smolvlm" else "smolvlm"
        suggested_value = {
            "backend": "auto",
            "preferred_backend": alternative,
            "degradation_policy": "preferred_then_simple",
            "disabled_backend_candidate": backend_key,
        }
        title = f"Evitar {backend_key} por rechazos de budget"
        reason = (
            "El backend acumula rechazos por budget en esta camara. Antes de subir "
            "memoria sin margen claro, conviene evitarlo o restringirlo."
        )

    return [
        _recommendation(
            camera_metrics,
            generated_at=generated_at,
            window_summary=window_summary,
            recommendation_type="budget",
            severity=severity,
            title=title,
            reason=reason,
            evidence={
                "backend": backend_key,
                "attempted_count": attempted_count,
                "budget_rejected_count": budget_rejected_count,
                "budget_rejection_rate": budget_rate,
                "observed_rss_mb": observed_rss,
                "current_backend_budget_mb": current_backend_budget,
                "thresholds": {
                    "high_budget_rejection_rate": HIGH_BUDGET_REJECTION_RATE,
                    "safe_rss_increase_factor": SAFE_RSS_INCREASE_FACTOR,
                },
            },
            current_value=current_policy,
            suggested_value=suggested_value,
            confidence=0.82,
            metrics_used=[
                f"semantic_backends.{backend_key}.attempted_count",
                f"semantic_backends.{backend_key}.budget_rejection_rate",
                f"semantic_backends.{backend_key}.observed_rss_mb.max",
            ],
        )
    ]


def _backend_parser_recommendations(
    camera_metrics: dict[str, Any],
    *,
    backend_key: str,
    backend: dict[str, Any],
    current_policy: dict[str, Any],
    generated_at: str,
    window_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    parser_seen = int(backend.get("parser_seen_count") or 0)
    invalid_json_rate = _float(backend.get("invalid_json_rate"))
    if parser_seen < MIN_BACKEND_ATTEMPTS or invalid_json_rate < HIGH_INVALID_JSON_RATE:
        return []

    alternative = "qwen" if backend_key == "smolvlm" else "smolvlm"
    return [
        _recommendation(
            camera_metrics,
            generated_at=generated_at,
            window_summary=window_summary,
            recommendation_type="vlm_policy",
            severity="medium",
            title=f"Evitar {backend_key} por JSON invalido recurrente",
            reason=(
                "El parser observa demasiadas salidas JSON invalidas para este backend "
                "en esta camara."
            ),
            evidence={
                "backend": backend_key,
                "parser_seen_count": parser_seen,
                "invalid_json_count": backend.get("invalid_json_count"),
                "invalid_json_rate": invalid_json_rate,
                "parser_recovery_rate": backend.get("parser_recovery_rate"),
            },
            current_value=current_policy,
            suggested_value={
                "backend": "auto",
                "preferred_backend": alternative,
                "degradation_policy": "preferred_then_simple",
                "disabled_backend_candidate": backend_key,
            },
            confidence=0.68,
            metrics_used=[
                f"semantic_backends.{backend_key}.parser_seen_count",
                f"semantic_backends.{backend_key}.invalid_json_rate",
            ],
        )
    ]


def _preferred_backend_recommendation(
    camera_metrics: dict[str, Any],
    *,
    current_policy: dict[str, Any],
    qwen: dict[str, Any],
    smolvlm: dict[str, Any],
    generated_at: str,
    window_summary: dict[str, Any],
) -> dict[str, Any] | None:
    qwen_good = _backend_is_good(qwen)
    smol_good = _backend_is_good(smolvlm)
    if not qwen_good and not smol_good:
        return None

    if qwen_good and (
        not smol_good or _backend_score(qwen) >= _backend_score(smolvlm) + 0.10
    ):
        preferred = "qwen"
        backend = qwen
        alternative = smolvlm
    elif smol_good and (
        not qwen_good or _backend_score(smolvlm) >= _backend_score(qwen) + 0.10
    ):
        preferred = "smolvlm"
        backend = smolvlm
        alternative = qwen
    else:
        return None

    title = (
        f"Mantener {preferred} como backend preferido"
        if current_policy.get("preferred_backend") == preferred
        else f"Preferir {preferred} para esta camara"
    )
    return _recommendation(
        camera_metrics,
        generated_at=generated_at,
        window_summary=window_summary,
        recommendation_type="vlm_policy",
        severity="low",
        title=title,
        reason=(
            f"{preferred} muestra buena tasa de exito, bajo fallback y parser estable "
            "en esta camara."
        ),
        evidence={
            "preferred_backend": preferred,
            "backend_attempted_count": backend.get("attempted_count"),
            "backend_success_rate": backend.get("success_rate"),
            "backend_fallback_away_rate": backend.get("fallback_away_rate"),
            "backend_budget_rejection_rate": backend.get("budget_rejection_rate"),
            "backend_parser_recovery_rate": backend.get("parser_recovery_rate"),
            "backend_invalid_json_rate": backend.get("invalid_json_rate"),
            "alternative_success_rate": alternative.get("success_rate"),
            "alternative_fallback_away_rate": alternative.get("fallback_away_rate"),
        },
        current_value=current_policy,
        suggested_value={"backend": "auto", "preferred_backend": preferred},
        confidence=0.74,
        metrics_used=[
            f"semantic_backends.{preferred}.attempted_count",
            f"semantic_backends.{preferred}.success_rate",
            f"semantic_backends.{preferred}.fallback_away_rate",
            f"semantic_backends.{preferred}.parser_recovery_rate",
            f"semantic_backends.{preferred}.invalid_json_rate",
        ],
    )


def _backend_is_good(backend: dict[str, Any]) -> bool:
    return (
        int(backend.get("attempted_count") or 0) >= MIN_BACKEND_ATTEMPTS
        and _float(backend.get("success_rate")) >= GOOD_BACKEND_SUCCESS_RATE
        and _float(backend.get("fallback_away_rate")) <= LOW_BACKEND_FALLBACK_RATE
        and _float(backend.get("budget_rejection_rate")) <= 0.10
        and _float(backend.get("invalid_json_rate")) <= HIGH_INVALID_JSON_RATE
    )


def _backend_score(backend: dict[str, Any]) -> float:
    return round(
        (
            _float(backend.get("success_rate"))
            - _float(backend.get("fallback_away_rate"))
            - _float(backend.get("budget_rejection_rate"))
            - (_float(backend.get("invalid_json_rate")) * 0.5)
        ),
        4,
    )


def _low_value_event_types(camera_metrics: dict[str, Any]) -> list[str]:
    result = []
    for event_type, row in _dict(camera_metrics.get("event_types")).items():
        if event_type in HIGH_VALUE_EVENT_TYPES:
            continue
        if int(_dict(row).get("vlm_attempted_count") or 0) > 0:
            result.append(str(event_type))
    return sorted(result)


def _recommendation(
    camera_metrics: dict[str, Any],
    *,
    generated_at: str,
    window_summary: dict[str, Any],
    recommendation_type: str,
    severity: str,
    title: str,
    reason: str,
    evidence: dict[str, Any],
    current_value: Any,
    suggested_value: Any,
    confidence: float,
    metrics_used: list[str],
    actionable: bool = True,
) -> dict[str, Any]:
    camera_id = str(camera_metrics.get("camera_id") or "unknown")
    stable_key = _stable_key(
        camera_id=camera_id,
        recommendation_type=recommendation_type,
        title=title,
        suggested_value=suggested_value,
    )
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "rule_set_version": RECOMMENDATION_RULESET_VERSION,
        "recommendation_id": f"{generated_at}:{stable_key}",
        "camera_id": camera_id,
        "recommendation_type": recommendation_type,
        "severity": severity,
        "title": title,
        "reason": reason,
        "evidence": deepcopy(evidence),
        "current_value": deepcopy(current_value),
        "suggested_value": deepcopy(suggested_value),
        "confidence": round(float(confidence), 4),
        "window_summary": deepcopy(window_summary),
        "metrics_used": list(metrics_used),
        "generated_at": generated_at,
        "actionable": actionable,
        "auto_apply": False,
    }


def _increase_det_size(value: Any) -> str:
    current = _parse_det_size(value)
    if current is None:
        return "960,960"
    width, height = current
    steps = [320, 480, 640, 800, 960, 1280]
    target_width = next((step for step in steps if step > width), min(width * 2, 1280))
    target_height = next((step for step in steps if step > height), min(height * 2, 1280))
    return f"{target_width},{target_height}"


def _parse_det_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("x", ",")
    try:
        raw_width, raw_height = normalized.split(",", 1)
        width = int(raw_width.strip())
        height = int(raw_height.strip())
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _decrease_probability(value: Any, *, step: float, floor: float) -> float:
    current = _optional_float(value)
    if current is None:
        current = 0.75
    return round(max(float(floor), current - step), 4)


def _decrease_int(value: Any, *, ratio: float) -> int:
    try:
        current = int(value)
    except (TypeError, ValueError):
        return 0
    if current <= 0:
        return 0
    return max(0, int(round(current * ratio)))


def _stable_key(
    *,
    camera_id: str,
    recommendation_type: str,
    title: str,
    suggested_value: Any,
) -> str:
    normalized_title = (
        title.lower().replace(" ", "_").replace("/", "_").replace(":", "")
    )
    normalized_suggestion = str(suggested_value).lower().replace(" ", "")
    normalized_suggestion = "".join(
        char for char in normalized_suggestion if char.isalnum() or char in {"_", "-"}
    )
    return f"{camera_id}:{recommendation_type}:{normalized_title}:{normalized_suggestion[:80]}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
