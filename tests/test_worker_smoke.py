from unittest.mock import MagicMock, patch

from app.consumer import load_fixture_message
from app.worker import process_fixture


def test_fixture_loads():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert msg.event_type == "frame.ingested"
    assert msg.camera_id == "cam_101"
    assert msg.frame_ref.endswith("person_001.jpg")


@patch("app.worker.get_session")
def test_process_fixture_smoke(mock_get_session):
    # Mocking the session ensures we don't need a real Postgres DB to run smoke tests
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event is not None
    assert "event_type" in event
    # Only 1 frame was processed, so threshold wasn't reached, therefore no_face.
    assert event["event_type"] == "human_presence_no_face"

    mock_session.commit.assert_called_once()
