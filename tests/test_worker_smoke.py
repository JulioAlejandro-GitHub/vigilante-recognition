from unittest.mock import MagicMock, patch

from app.consumer import load_fixture_message
from app.worker import process_fixture


def test_fixture_loads():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert msg.event_type == "frame.ingested"
    assert msg.camera_id == "cam_101"
    assert msg.frame_ref.endswith("person_001.jpg")


@patch("app.worker.RecognitionRepository")
@patch("app.worker.get_session")
def test_process_fixture_smoke(mock_get_session, mock_repo_class):
    # Mocking the session ensures we don't need a real Postgres DB to run smoke tests
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session

    mock_repo_instance = MagicMock()
    from uuid import uuid4
    mock_repo_instance.resolve_camera_id.return_value = str(uuid4())
    # Ensure properties exist so the pipeline runs through
    mock_repo_instance.create_subject.return_value.observed_subject_id = str(uuid4())
    mock_repo_instance.create_track.return_value.human_track_id = str(uuid4())
    mock_repo_instance.create_track.return_value.camera_id = str(uuid4())
    mock_repo_instance.create_track.return_value.person_presence_score = 0.0
    mock_repo_class.return_value = mock_repo_instance

    event = process_fixture("tests/fixtures/frame_ingested_example.json")

    assert event is not None
    assert "event_type" in event
    # score increment is 0.0 initially so it's a no_face
    assert event["event_type"] == "human_presence_no_face"

    mock_session.commit.assert_called_once()
