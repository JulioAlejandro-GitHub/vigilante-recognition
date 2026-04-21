from app.consumer import load_fixture_message


def test_fixture_loads():
    msg = load_fixture_message("tests/fixtures/frame_ingested_example.json")
    assert msg.event_type == "frame.ingested"
    assert msg.camera_id == "cam_101"
    assert msg.frame_ref.endswith("person_001.jpg")
