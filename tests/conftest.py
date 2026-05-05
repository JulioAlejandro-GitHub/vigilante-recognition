import pytest

from app.services.vlm_degradation_service import vlm_degradation_state


@pytest.fixture(autouse=True)
def reset_vlm_degradation_state():
    vlm_degradation_state.reset()
    yield
    vlm_degradation_state.reset()
