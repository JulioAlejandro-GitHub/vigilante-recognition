from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventPublisher:
    """Stub simple para el primer slice.

    En esta fase solo deja el evento en outbox y lo loggea.
    La publicación real a RabbitMQ se puede implementar en el siguiente paso.
    """

    def publish(self, event: dict[str, Any]) -> None:
        logger.info("recognition_event_ready %s", json.dumps(event, ensure_ascii=False))
