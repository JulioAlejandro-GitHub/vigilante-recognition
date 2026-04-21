from __future__ import annotations

import json
from pathlib import Path

from app.domain.entities import FrameIngestedMessage


def load_fixture_message(path: str) -> FrameIngestedMessage:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrameIngestedMessage(**data)
