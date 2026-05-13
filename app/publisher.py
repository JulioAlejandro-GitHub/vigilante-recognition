from __future__ import annotations

import json
import logging
from typing import Any

from app.logging import event_log_fields

logger = logging.getLogger(__name__)


class EventPublisher:
    """Stub simple para el primer slice.

    En esta fase solo deja el evento en outbox y lo loggea.
    La publicación real a RabbitMQ se puede implementar en el siguiente paso.
    """

    def publish(self, event: dict[str, Any]) -> None:
        fields = event_log_fields(event)
        logger.info(
            "recognition_event_ready event_id=%s event_type=%s camera_id=%s track_id=%s subject_id=%s run_id=%s",
            fields["event_id"],
            fields["event_type"],
            fields["camera_id"],
            fields["track_id"],
            fields["subject_id"],
            fields["run_id"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "recognition_event_payload event_id=%s payload=%s",
                fields["event_id"],
                json.dumps(event, ensure_ascii=False, sort_keys=True, default=str),
            )
