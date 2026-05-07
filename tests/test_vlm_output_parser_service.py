from __future__ import annotations

import json

import pytest

from app.services.vlm_output_parser_service import (
    VlmOutputParserError,
    VlmOutputParserService,
)


def _descriptor_payload(**overrides):
    payload = {
        "subject_type": "person",
        "top_clothing": {"category": "hoodie", "color": "red", "pattern": "solid"},
        "bottom_clothing": {"category": "jeans", "color": "blue", "pattern": "solid"},
        "dominant_colors": ["red", "blue"],
        "accessories": ["backpack"],
        "carried_object": "unknown",
        "body_build": "average",
        "pose_direction": "left",
        "scene_observation_quality": {"level": "medium", "notes": "clear enough"},
        "descriptor_confidence": 0.81,
        "raw_summary": "person with red hoodie",
    }
    payload.update(overrides)
    return payload


def test_parser_accepts_direct_json() -> None:
    parser = VlmOutputParserService()

    result = parser.parse(json.dumps(_descriptor_payload()))

    assert result.descriptor["top_clothing"]["category"] == "hoodie"
    assert result.descriptor["descriptor_confidence"] == 0.81
    assert result.trace["parse_strategy_used"] == "direct_json"
    assert result.trace["json_recovered"] is False
    assert result.trace["missing_fields"] == []
    assert result.trace["raw_output_chars"] > 0


def test_parser_recovers_fenced_json() -> None:
    parser = VlmOutputParserService()
    raw_output = "```json\n" + json.dumps(_descriptor_payload()) + "\n```"

    result = parser.parse(raw_output)

    assert result.descriptor["bottom_clothing"]["color"] == "blue"
    assert result.trace["parse_strategy_used"] == "fenced_json"
    assert result.trace["json_recovered"] is True


def test_parser_recovers_json_with_surrounding_text() -> None:
    parser = VlmOutputParserService()
    raw_output = (
        "Here is the observation:\n"
        + json.dumps(_descriptor_payload(accessories=[]))
        + "\nNo other inference is needed."
    )

    result = parser.parse(raw_output)

    assert result.descriptor["subject_type"] == "person"
    assert result.trace["parse_strategy_used"] == "extracted_json_object"
    assert result.trace["json_recovered"] is True
    assert result.trace["raw_output_preview"].startswith("Here is the observation")


def test_parser_recovers_jsonish_single_quotes_and_trailing_commas() -> None:
    parser = VlmOutputParserService()
    raw_output = """
    {
      'subject_type': 'person',
      'top_clothing': 'red hoodie',
      'bottom_clothing': {'category': 'pants', 'color': 'black',},
      'raw_summary': 'person with red hoodie',
    }
    """

    result = parser.parse(raw_output)

    assert result.descriptor["top_clothing"] == {
        "category": "hoodie",
        "color": "red",
        "pattern": "unknown",
    }
    assert result.descriptor["bottom_clothing"]["color"] == "black"
    assert result.trace["json_recovered"] is True
    assert "dominant_colors" in result.trace["missing_fields"]


def test_parser_fails_with_clear_trace_when_output_is_not_recoverable() -> None:
    parser = VlmOutputParserService()

    with pytest.raises(VlmOutputParserError) as excinfo:
        parser.parse("The person appears in the frame, but no structured object follows.")

    assert excinfo.value.reason == "vlm_output_invalid_json"
    assert excinfo.value.trace["parse_stage"] == "failed"
    assert excinfo.value.trace["json_recovered"] is False
    assert excinfo.value.trace["parser_error"] == "vlm_output_invalid_json"


def test_parser_normalizes_partial_descriptor_with_missing_fields_trace() -> None:
    parser = VlmOutputParserService()

    result = parser.parse('{"subject_type":"person","top_clothing":"green jacket"}')

    assert result.descriptor["top_clothing"]["category"] == "jacket"
    assert result.descriptor["top_clothing"]["color"] == "green"
    assert result.descriptor["bottom_clothing"] == {
        "category": "lower_garment",
        "color": "unknown",
        "pattern": "unknown",
    }
    assert result.descriptor["raw_summary"] == "unknown"
    assert "bottom_clothing" in result.trace["missing_fields"]
    assert "raw_summary" in result.trace["missing_fields"]
    assert "appearance" in result.trace["normalized_fields"]
    assert "signature" in result.trace["normalized_fields"]


def test_parser_recovers_descriptor_wrapped_in_semantic_descriptor_key() -> None:
    parser = VlmOutputParserService()
    raw_output = json.dumps(
        {
            "semantic_descriptor": _descriptor_payload(
                top_clothing="gray jacket",
                bottom_clothing="black pants",
            ),
            "notes": "ignored wrapper metadata",
        }
    )

    result = parser.parse(raw_output)

    assert result.descriptor["top_clothing"]["category"] == "jacket"
    assert result.descriptor["bottom_clothing"]["color"] == "black"
    assert result.trace["descriptor_source_path"] == "$.semantic_descriptor"
    assert result.trace["parsed_json_keys"] == ["notes", "semantic_descriptor"]
    assert "notes" not in result.trace["parser_extra_keys"]
