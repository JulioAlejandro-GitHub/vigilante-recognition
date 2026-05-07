from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


PARSER_VERSION = "vlm_output_parser_v1"


@dataclass(frozen=True)
class VlmOutputParseResult:
    descriptor: dict[str, Any]
    trace: dict[str, Any]


class VlmOutputParserError(ValueError):
    def __init__(self, reason: str, *, trace: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.trace = trace


class VlmOutputParserService:
    EXPECTED_FIELDS = [
        "subject_type",
        "top_clothing",
        "bottom_clothing",
        "dominant_colors",
        "accessories",
        "carried_object",
        "body_build",
        "pose_direction",
        "scene_observation_quality",
        "descriptor_confidence",
        "raw_summary",
    ]
    FIELD_ALIASES = {
        "upper_clothing": "top_clothing",
        "upper_body_clothing": "top_clothing",
        "top": "top_clothing",
        "lower_clothing": "bottom_clothing",
        "lower_body_clothing": "bottom_clothing",
        "bottom": "bottom_clothing",
        "colors": "dominant_colors",
        "dominant_palette": "dominant_colors",
        "visible_accessories": "accessories",
        "held_object": "carried_object",
        "object_carried": "carried_object",
        "direction": "pose_direction",
        "observation_quality": "scene_observation_quality",
        "quality": "scene_observation_quality",
        "confidence": "descriptor_confidence",
        "summary": "raw_summary",
        "description": "raw_summary",
        "visual_summary": "raw_summary",
    }
    DESCRIPTOR_WRAPPER_KEYS = [
        "semantic_descriptor",
        "descriptor",
        "visual_descriptor",
        "appearance_descriptor",
        "person_descriptor",
        "subject_descriptor",
        "person",
        "subject",
    ]
    DESCRIPTOR_LIST_WRAPPER_KEYS = [
        "descriptors",
        "subjects",
        "people",
        "persons",
    ]
    COLORS = [
        "black",
        "white",
        "gray",
        "grey",
        "red",
        "orange",
        "yellow",
        "green",
        "teal",
        "blue",
        "navy",
        "purple",
        "brown",
        "beige",
        "tan",
    ]
    CLOTHING_CATEGORIES = [
        "hoodie",
        "jacket",
        "coat",
        "shirt",
        "sweater",
        "dress",
        "jeans",
        "pants",
        "shorts",
        "skirt",
        "uniform",
        "upper_garment",
        "lower_garment",
    ]

    def parse(self, raw_text: str) -> VlmOutputParseResult:
        raw_output = "" if raw_text is None else str(raw_text)
        base_trace = {
            "parser_version": PARSER_VERSION,
            "raw_output_chars": len(raw_output),
            "raw_output_preview": self._preview(raw_output),
        }
        base_trace["raw_output_preview_chars"] = len(base_trace["raw_output_preview"])

        candidates = self._candidate_payloads(raw_output)
        parse_attempts: list[dict[str, Any]] = []
        best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        last_error = "vlm_output_missing_json_object"

        for candidate in candidates:
            for method_name, parser in self._parse_methods(candidate["payload"]):
                strategy = self._strategy_name(candidate["strategy"], method_name)
                attempt_trace = {
                    "candidate_strategy": candidate["strategy"],
                    "parse_method": method_name,
                    "parse_strategy_used": strategy,
                    "candidate_chars": len(candidate["payload"]),
                }
                try:
                    parsed = parser()
                except Exception as exc:
                    last_error = "vlm_output_invalid_json"
                    attempt_trace.update(
                        {
                            "status": "failed",
                            "parser_error": f"{type(exc).__name__}:{str(exc)[:160]}",
                        }
                    )
                    parse_attempts.append(attempt_trace)
                    continue

                if not isinstance(parsed, dict):
                    last_error = "vlm_output_json_not_object"
                    attempt_trace.update(
                        {
                            "status": "failed",
                            "parser_error": f"parsed_type:{type(parsed).__name__}",
                        }
                    )
                    parse_attempts.append(attempt_trace)
                    continue

                descriptor_source, source_trace = self._descriptor_source(parsed)
                signal_score = self._semantic_signal_score(descriptor_source)
                if signal_score <= 0:
                    last_error = "vlm_output_missing_semantic_fields"
                    attempt_trace.update(
                        {
                            "status": "failed",
                            "parser_error": "no_expected_semantic_fields",
                            "parsed_json_keys": sorted(str(key) for key in parsed.keys())[:30],
                            **source_trace,
                        }
                    )
                    parse_attempts.append(attempt_trace)
                    continue

                descriptor, normalization_trace = self._normalize_descriptor(descriptor_source)
                score = (
                    (signal_score * 1000)
                    + (self._strategy_priority(strategy) * 10)
                    + min(100, len(json.dumps(descriptor, sort_keys=True)))
                )
                result_trace = {
                    **base_trace,
                    "parse_stage": "structure_normalized",
                    "parse_strategy_used": strategy,
                    "json_recovered": candidate["strategy"] != "direct" or method_name != "json",
                    "parsed_json_keys": sorted(str(key) for key in parsed.keys())[:30],
                    "parse_attempts": parse_attempts[-10:] + [{**attempt_trace, "status": "success"}],
                    **source_trace,
                    **normalization_trace,
                }
                if best is None or score > best[0]:
                    best = (score, descriptor, result_trace)
                parse_attempts.append({**attempt_trace, "status": "success"})

        if best is not None:
            return VlmOutputParseResult(descriptor=best[1], trace=best[2])

        trace = {
            **base_trace,
            "parse_stage": "failed",
            "parse_strategy_used": None,
            "json_recovered": False,
            "normalized_fields": [],
            "missing_fields": list(self.EXPECTED_FIELDS),
            "parse_attempts": parse_attempts[-12:],
            "parser_error": last_error,
        }
        raise VlmOutputParserError(last_error, trace=trace)

    def _candidate_payloads(self, raw_text: str) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(strategy: str, payload: str) -> None:
            cleaned = payload.strip()
            if not cleaned:
                return
            key = cleaned
            if key in seen:
                return
            seen.add(key)
            candidates.append({"strategy": strategy, "payload": cleaned})

        cleaned = raw_text.strip()
        add("direct", cleaned)

        for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL):
            add("fenced_json", match.group(1))

        for payload in self._balanced_json_objects(cleaned):
            add("balanced_json_object", payload)
        for payload in self._balanced_json_objects(self._normalize_quotes(cleaned)):
            add("balanced_json_object", payload)

        return candidates

    def _parse_methods(self, payload: str) -> list[tuple[str, Callable[[], Any]]]:
        return [
            ("json", lambda: json.loads(payload)),
            ("python_literal", lambda: self._jsonable(ast.literal_eval(payload))),
            ("tolerant_json_cleanup", lambda: json.loads(self._clean_jsonish_payload(payload))),
        ]

    def _strategy_name(self, candidate_strategy: str, method_name: str) -> str:
        if candidate_strategy == "direct" and method_name == "json":
            return "direct_json"
        if candidate_strategy == "fenced_json" and method_name == "json":
            return "fenced_json"
        if candidate_strategy == "balanced_json_object" and method_name == "json":
            return "extracted_json_object"
        if method_name == "python_literal":
            return f"{candidate_strategy}_python_literal_eval"
        return f"{candidate_strategy}_tolerant_json_cleanup"

    def _strategy_priority(self, strategy: str) -> int:
        priorities = {
            "direct_json": 50,
            "fenced_json": 45,
            "extracted_json_object": 40,
        }
        if strategy.endswith("_python_literal_eval"):
            return 30
        if strategy.endswith("_tolerant_json_cleanup"):
            return 20
        return priorities.get(strategy, 10)

    def _balanced_json_objects(self, text: str) -> list[str]:
        objects: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        quote_char = ""
        escaped = False

        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote_char:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
                continue

            if char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : index + 1])
                    start = None

        return objects

    def _clean_jsonish_payload(self, payload: str) -> str:
        cleaned = self._normalize_quotes(payload.strip())
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r"\bNone\b", "null", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = re.sub(
            r"([{,])\s*([A-Za-z_][A-Za-z0-9_ -]*)\s*:",
            lambda match: f'{match.group(1)} "{match.group(2).strip()}":',
            cleaned,
        )
        cleaned = re.sub(
            r":\s*([A-Za-z_][A-Za-z0-9_ -]*)(\s*[,}])",
            self._quote_bare_json_value,
            cleaned,
        )
        return cleaned

    def _quote_bare_json_value(self, match: re.Match[str]) -> str:
        value = match.group(1).strip()
        if value.lower() in {"true", "false", "null"}:
            return f": {value.lower()}{match.group(2)}"
        return f': "{value}"{match.group(2)}'

    def _normalize_descriptor(self, parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raw, alias_fields = self._apply_aliases(parsed)
        normalized_fields: list[str] = list(alias_fields)
        missing_fields: list[str] = []

        subject_type = self._clean_label(raw.get("subject_type"), default="person")
        self._track_missing_or_normalized(
            raw,
            "subject_type",
            subject_type,
            "person",
            missing_fields,
            normalized_fields,
        )

        top_clothing = self._normalize_clothing(
            raw.get("top_clothing"),
            field_name="top_clothing",
            default_category="upper_garment",
            missing_fields=missing_fields,
            normalized_fields=normalized_fields,
        )
        bottom_clothing = self._normalize_clothing(
            raw.get("bottom_clothing"),
            field_name="bottom_clothing",
            default_category="lower_garment",
            missing_fields=missing_fields,
            normalized_fields=normalized_fields,
        )
        dominant_colors = self._normalize_color_list(raw.get("dominant_colors"))
        if not dominant_colors:
            missing_fields.append("dominant_colors")
            dominant_colors = self._ordered_unique([top_clothing["color"], bottom_clothing["color"]])
            normalized_fields.append("dominant_colors")

        accessories = self._normalize_string_list(raw.get("accessories"))
        if "accessories" not in raw:
            missing_fields.append("accessories")
            normalized_fields.append("accessories")

        carried_object = self._clean_label(raw.get("carried_object"), default="unknown")
        self._track_missing_or_normalized(
            raw,
            "carried_object",
            carried_object,
            "unknown",
            missing_fields,
            normalized_fields,
        )
        body_build = self._clean_label(raw.get("body_build"), default="unknown")
        self._track_missing_or_normalized(
            raw,
            "body_build",
            body_build,
            "unknown",
            missing_fields,
            normalized_fields,
        )
        pose_direction = self._clean_label(raw.get("pose_direction"), default="unknown")
        self._track_missing_or_normalized(
            raw,
            "pose_direction",
            pose_direction,
            "unknown",
            missing_fields,
            normalized_fields,
        )
        scene_observation_quality = self._normalize_observation_quality(
            raw.get("scene_observation_quality"),
            missing_fields=missing_fields,
            normalized_fields=normalized_fields,
        )
        descriptor_confidence = self._coerce_confidence(raw.get("descriptor_confidence"))
        if "descriptor_confidence" not in raw:
            missing_fields.append("descriptor_confidence")
            normalized_fields.append("descriptor_confidence")
        raw_summary = self._clean_summary(raw.get("raw_summary"), default="unknown")
        self._track_missing_or_normalized(
            raw,
            "raw_summary",
            raw_summary,
            "unknown",
            missing_fields,
            normalized_fields,
        )

        appearance = self._normalize_appearance(raw.get("appearance"), top_clothing, bottom_clothing, dominant_colors)
        signature = {
            "subject_type": subject_type,
            "upper_region_color": appearance["upper_region_color"],
            "middle_region_color": appearance["middle_region_color"],
            "lower_region_color": appearance["lower_region_color"],
            "dominant_palette": dominant_colors,
            "contrast_level": appearance["contrast_level"],
            "saturation_level": appearance["saturation_level"],
            "pose_direction": pose_direction,
            "body_build": body_build,
            "accessories": accessories,
            "carried_object": carried_object,
        }
        if "appearance" not in raw:
            normalized_fields.append("appearance")
        if "signature" not in raw:
            normalized_fields.append("signature")

        descriptor = {
            "subject_type": subject_type,
            "top_clothing": top_clothing,
            "bottom_clothing": bottom_clothing,
            "dominant_colors": dominant_colors,
            "accessories": accessories,
            "carried_object": carried_object,
            "body_build": body_build,
            "pose_direction": pose_direction,
            "scene_observation_quality": scene_observation_quality,
            "descriptor_confidence": descriptor_confidence,
            "raw_summary": raw_summary,
            "appearance": appearance,
            "signature": signature,
        }
        extra_keys = sorted(str(key) for key in parsed.keys() if str(key) not in set(self.EXPECTED_FIELDS))
        return descriptor, {
            "normalized_fields": self._ordered_unique(normalized_fields, limit=60),
            "missing_fields": self._ordered_unique(missing_fields, limit=60),
            "parser_extra_keys": extra_keys[:30],
        }

    def _descriptor_source(self, parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        direct_score = self._semantic_signal_score(parsed)
        if direct_score > 0:
            return parsed, {
                "descriptor_source_path": "$",
                "descriptor_json_keys": sorted(str(key) for key in parsed.keys())[:30],
            }

        lowered_key_map = {str(key).strip().lower(): key for key in parsed}
        for wrapper_key in self.DESCRIPTOR_WRAPPER_KEYS:
            raw_key = lowered_key_map.get(wrapper_key)
            if raw_key is None:
                continue
            value = parsed.get(raw_key)
            if not isinstance(value, dict):
                continue
            if self._semantic_signal_score(value) <= 0:
                continue
            return value, {
                "descriptor_source_path": f"$.{raw_key}",
                "descriptor_json_keys": sorted(str(key) for key in value.keys())[:30],
                "normalized_fields": [str(raw_key)],
            }

        for wrapper_key in self.DESCRIPTOR_LIST_WRAPPER_KEYS:
            raw_key = lowered_key_map.get(wrapper_key)
            if raw_key is None:
                continue
            value = parsed.get(raw_key)
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                if self._semantic_signal_score(item) <= 0:
                    continue
                return item, {
                    "descriptor_source_path": f"$.{raw_key}[{index}]",
                    "descriptor_json_keys": sorted(str(key) for key in item.keys())[:30],
                    "normalized_fields": [str(raw_key)],
                }

        return parsed, {
            "descriptor_source_path": "$",
            "descriptor_json_keys": sorted(str(key) for key in parsed.keys())[:30],
        }

    def _apply_aliases(self, parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        raw = dict(parsed)
        normalized_fields: list[str] = []
        lowered_key_map = {str(key).strip().lower(): key for key in raw}
        for canonical in self.EXPECTED_FIELDS:
            if canonical in raw or canonical not in lowered_key_map:
                continue
            raw[canonical] = raw[lowered_key_map[canonical]]
            normalized_fields.append(canonical)
        for alias, canonical in self.FIELD_ALIASES.items():
            if canonical in raw or alias not in lowered_key_map:
                continue
            raw[canonical] = raw[lowered_key_map[alias]]
            normalized_fields.append(canonical)
        return raw, normalized_fields

    def _semantic_signal_score(self, parsed: dict[str, Any]) -> int:
        raw, _ = self._apply_aliases(parsed)
        score = 0
        for field in self.EXPECTED_FIELDS:
            value = raw.get(field)
            if value not in (None, "", [], {}):
                score += 1
        return score

    def _normalize_clothing(
        self,
        raw_value: Any,
        *,
        field_name: str,
        default_category: str,
        missing_fields: list[str],
        normalized_fields: list[str],
    ) -> dict[str, str]:
        if isinstance(raw_value, dict):
            category = self._clean_label(raw_value.get("category"), default=default_category)
            color = self._normalize_color(raw_value.get("color"))
            pattern = self._clean_label(raw_value.get("pattern"), default="unknown")
            for nested_field, value, default in [
                ("category", category, default_category),
                ("color", color, "unknown"),
                ("pattern", pattern, "unknown"),
            ]:
                if nested_field not in raw_value:
                    missing_fields.append(f"{field_name}.{nested_field}")
                    normalized_fields.append(field_name)
                elif value == default and raw_value.get(nested_field) in (None, "", [], {}):
                    normalized_fields.append(field_name)
            return {"category": category, "color": color, "pattern": pattern}

        if isinstance(raw_value, str) and raw_value.strip():
            text = raw_value.strip()
            normalized_fields.append(field_name)
            return {
                "category": self._category_from_text(text) or default_category,
                "color": self._color_from_text(text) or "unknown",
                "pattern": self._pattern_from_text(text),
            }

        missing_fields.append(field_name)
        normalized_fields.append(field_name)
        return {"category": default_category, "color": "unknown", "pattern": "unknown"}

    def _normalize_observation_quality(
        self,
        value: Any,
        *,
        missing_fields: list[str],
        normalized_fields: list[str],
    ) -> dict[str, str]:
        if isinstance(value, dict):
            level = self._clean_label(
                value.get("level") or value.get("quality_level") or value.get("descriptor_quality"),
                default="unknown",
            )
            notes = self._clean_summary(value.get("notes"), default="unknown")
            if "level" not in value and "quality_level" not in value and "descriptor_quality" not in value:
                missing_fields.append("scene_observation_quality.level")
                normalized_fields.append("scene_observation_quality")
            if "notes" not in value:
                missing_fields.append("scene_observation_quality.notes")
                normalized_fields.append("scene_observation_quality")
            return {"level": level, "notes": notes}

        if isinstance(value, str) and value.strip():
            normalized_fields.append("scene_observation_quality")
            return {"level": self._clean_label(value, default="unknown"), "notes": "unknown"}

        missing_fields.append("scene_observation_quality")
        normalized_fields.append("scene_observation_quality")
        return {"level": "unknown", "notes": "unknown"}

    def _normalize_appearance(
        self,
        value: Any,
        top_clothing: dict[str, str],
        bottom_clothing: dict[str, str],
        dominant_colors: list[str],
    ) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        upper = self._normalize_color(raw.get("upper_region_color") or top_clothing["color"])
        lower = self._normalize_color(raw.get("lower_region_color") or bottom_clothing["color"])
        middle = self._normalize_color(
            raw.get("middle_region_color")
            or (dominant_colors[1] if len(dominant_colors) > 1 else upper)
        )
        return {
            "dominant_palette": dominant_colors,
            "upper_region_color": upper,
            "middle_region_color": middle,
            "lower_region_color": lower,
            "contrast_level": self._clean_label(raw.get("contrast_level"), default="unknown"),
            "saturation_level": self._clean_label(raw.get("saturation_level"), default="unknown"),
        }

    def _track_missing_or_normalized(
        self,
        raw: dict[str, Any],
        field_name: str,
        normalized_value: Any,
        default_value: Any,
        missing_fields: list[str],
        normalized_fields: list[str],
    ) -> None:
        if field_name not in raw:
            missing_fields.append(field_name)
            normalized_fields.append(field_name)
            return
        if normalized_value == default_value and raw.get(field_name) in (None, "", [], {}):
            normalized_fields.append(field_name)
            return
        if isinstance(raw.get(field_name), str) and raw[field_name] != normalized_value:
            normalized_fields.append(field_name)

    def _normalize_color_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = re.split(r"[;,/]| and ", value)
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = []
        return self._ordered_unique([self._normalize_color(item) for item in raw_values])

    def _normalize_string_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = re.split(r"[;,/]| and ", value)
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = []
        return self._ordered_unique([self._clean_label(item, default="unknown") for item in raw_values])

    def _normalize_color(self, value: Any) -> str:
        text = self._clean_label(value, default="unknown")
        aliases = {"grey": "gray", "tan": "beige", "dark blue": "navy", "navy blue": "navy"}
        text = aliases.get(text, text)
        for color in self.COLORS:
            normalized_color = aliases.get(color, color)
            if color.replace("_", " ") in text.replace("_", " "):
                return normalized_color
        return text

    def _color_from_text(self, value: str) -> str | None:
        color = self._normalize_color(value)
        return None if color == "unknown" else color

    def _category_from_text(self, value: str) -> str | None:
        text = self._clean_label(value, default="")
        for category in self.CLOTHING_CATEGORIES:
            if category in text:
                return category
        return None

    def _pattern_from_text(self, value: str) -> str:
        text = self._clean_label(value, default="")
        if "stripe" in text:
            return "striped"
        if "plaid" in text or "check" in text:
            return "plaid"
        if "logo" in text or "graphic" in text:
            return "graphic"
        if "solid" in text or "plain" in text:
            return "solid"
        return "unknown"

    def _clean_label(self, value: Any, *, default: str) -> str:
        if value is None:
            return default
        text = self._clean_scalar(value, default=default).lower().replace("-", "_")
        text = " ".join(text.split())
        return text or default

    def _clean_summary(self, value: Any, *, default: str) -> str:
        return self._clean_scalar(value, default=default, limit=240)

    def _clean_scalar(self, value: Any, *, default: str, limit: int = 120) -> str:
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return default
        text = str(value).replace("\x00", " ")
        text = self._normalize_quotes(text)
        text = re.sub(r"[\x01-\x1f\x7f]", " ", text)
        text = " ".join(text.strip().strip("`'\"").split())
        if not text:
            return default
        return text[:limit]

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(item) for item in value]
        return value

    def _normalize_quotes(self, value: str) -> str:
        return (
            value.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

    def _preview(self, raw_text: str, *, limit: int = 500) -> str:
        compact = " ".join(raw_text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _ordered_unique(self, values: list[Any], *, limit: int = 8) -> list[str]:
        ordered: list[str] = []
        for value in values:
            text = self._clean_label(value, default="unknown")
            if text == "unknown" or text in ordered:
                continue
            ordered.append(text)
        return ordered[:limit]
