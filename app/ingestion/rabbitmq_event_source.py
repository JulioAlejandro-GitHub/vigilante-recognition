from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

from app.messaging.topology import FrameIngestedTopology, declare_frame_ingested_topology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RabbitMqDelivery:
    body: bytes
    delivery_tag: int
    headers: dict[str, Any]
    redelivered: bool = False
    properties: Any | None = None


class RabbitMqEventSource:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        virtual_host: str,
        topology: FrameIngestedTopology,
        prefetch_count: int = 10,
        idle_timeout_seconds: float | None = 1.0,
        connection_factory=None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.topology = topology
        self.prefetch_count = max(1, int(prefetch_count))
        self.idle_timeout_seconds = idle_timeout_seconds
        self._connection_factory = connection_factory
        self._connection = None
        self._channel = None

    def iter_deliveries(self, *, max_messages: int | None = None) -> Iterator[RabbitMqDelivery]:
        consumed = 0

        while max_messages is None or consumed < max_messages:
            channel = self._ensure_channel()
            try:
                for method, properties, body in channel.consume(
                    queue=self.topology.recognition_queue,
                    inactivity_timeout=self.idle_timeout_seconds,
                    auto_ack=False,
                ):
                    if method is None:
                        if max_messages is None:
                            break
                        return
                    headers = dict(getattr(properties, "headers", None) or {})
                    yield RabbitMqDelivery(
                        body=body,
                        delivery_tag=method.delivery_tag,
                        headers=headers,
                        redelivered=bool(getattr(method, "redelivered", False)),
                        properties=properties,
                    )
                    consumed += 1
                    if max_messages is not None and consumed >= max_messages:
                        return
            except Exception as exc:
                if not self._is_recoverable_consume_error(exc):
                    raise
                logger.warning(
                    "rabbitmq_consumer_reconnecting queue=%s error_type=%s error=%s",
                    self.topology.recognition_queue,
                    type(exc).__name__,
                    str(exc),
                )
                self._discard_connection()
                if max_messages is not None:
                    return
            if max_messages is not None:
                return
            logger.debug(
                "rabbitmq_consumer_idle queue=%s inactivity_timeout=%s",
                self.topology.recognition_queue,
                self.idle_timeout_seconds,
            )

    def ack(self, delivery: RabbitMqDelivery) -> None:
        self._ensure_channel().basic_ack(delivery_tag=delivery.delivery_tag)

    def reject_to_dlq(self, delivery: RabbitMqDelivery) -> None:
        self._ensure_channel().basic_reject(delivery_tag=delivery.delivery_tag, requeue=False)

    def nack(self, delivery: RabbitMqDelivery, *, requeue: bool) -> None:
        self._ensure_channel().basic_nack(delivery_tag=delivery.delivery_tag, requeue=requeue)

    def retry(self, delivery: RabbitMqDelivery, *, retry_count: int) -> None:
        headers = dict(delivery.headers or {})
        headers["x-retry-count"] = retry_count
        self._ensure_channel().basic_publish(
            exchange=self.topology.exchange,
            routing_key=self.topology.routing_key,
            body=delivery.body,
            properties=self._retry_properties(delivery, headers=headers),
            mandatory=True,
        )

    def close(self) -> None:
        self._discard_connection()

    def _discard_connection(self) -> None:
        channel = self._channel
        connection = self._connection
        self._channel = None
        self._connection = None
        if channel is not None:
            cancel = getattr(channel, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:
                    logger.debug(
                        "rabbitmq_channel_cancel_ignored queue=%s error_type=%s error=%s",
                        self.topology.recognition_queue,
                        type(exc).__name__,
                        str(exc),
                    )
        if connection is not None and getattr(connection, "is_closed", False) is False:
            try:
                connection.close()
            except Exception as exc:
                logger.debug(
                    "rabbitmq_connection_close_ignored queue=%s error_type=%s error=%s",
                    self.topology.recognition_queue,
                    type(exc).__name__,
                    str(exc),
                )

    def _ensure_channel(self):
        if self._channel is not None and getattr(self._channel, "is_open", True):
            return self._channel

        self._connection = self._build_connection()
        self._channel = self._connection.channel()
        declare_frame_ingested_topology(self._channel, self.topology)
        self._channel.basic_qos(prefetch_count=self.prefetch_count)
        logger.info(
            "rabbitmq_consumer_ready host=%s port=%s vhost=%s queue=%s exchange=%s routing_key=%s prefetch_count=%s",
            self.host,
            self.port,
            self.virtual_host,
            self.topology.recognition_queue,
            self.topology.exchange,
            self.topology.routing_key,
            self.prefetch_count,
        )
        return self._channel

    def _build_connection(self):
        if self._connection_factory is not None:
            return self._connection_factory()

        try:
            import pika
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("RabbitMQ consumer mode requires the 'pika' package. Install requirements.txt.") from exc

        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=30,
        )
        return pika.BlockingConnection(parameters)

    def _is_recoverable_consume_error(self, exc: Exception) -> bool:
        if isinstance(exc, ValueError) and "Timeout closed before call" in str(exc):
            return True

        exc_type = type(exc)
        module_name = getattr(exc_type, "__module__", "")
        type_name = getattr(exc_type, "__name__", "")
        if module_name.startswith("pika.") and any(
            token in type_name
            for token in [
                "AMQPConnection",
                "ChannelClosed",
                "ChannelWrongState",
                "ConnectionClosed",
                "StreamLost",
            ]
        ):
            return True
        return False

    def _retry_properties(self, delivery: RabbitMqDelivery, *, headers: dict[str, Any]):
        try:
            import pika
        except ImportError:
            return delivery.properties

        properties = delivery.properties
        return pika.BasicProperties(
            app_id=getattr(properties, "app_id", None),
            content_type=getattr(properties, "content_type", "application/json"),
            delivery_mode=getattr(properties, "delivery_mode", 2),
            message_id=getattr(properties, "message_id", None),
            type=getattr(properties, "type", "frame.ingested"),
            headers=headers,
        )
