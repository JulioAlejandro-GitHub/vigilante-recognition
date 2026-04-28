from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameIngestedTopology:
    exchange: str = "vigilante.frames"
    routing_key: str = "frame.ingested"
    recognition_queue: str = "vigilante.recognition.frame_ingested"
    dead_letter_exchange: str = "vigilante.frames.dlx"
    dead_letter_queue: str = "vigilante.recognition.frame_ingested.dlq"
    dead_letter_routing_key: str = "frame.ingested.dlq"
    exchange_type: str = "topic"
    dead_letter_exchange_type: str = "direct"


DEFAULT_FRAME_INGESTED_TOPOLOGY = FrameIngestedTopology()


def declare_frame_ingested_topology(channel, topology: FrameIngestedTopology = DEFAULT_FRAME_INGESTED_TOPOLOGY) -> None:
    channel.exchange_declare(
        exchange=topology.exchange,
        exchange_type=topology.exchange_type,
        durable=True,
    )
    channel.exchange_declare(
        exchange=topology.dead_letter_exchange,
        exchange_type=topology.dead_letter_exchange_type,
        durable=True,
    )
    channel.queue_declare(
        queue=topology.recognition_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": topology.dead_letter_exchange,
            "x-dead-letter-routing-key": topology.dead_letter_routing_key,
        },
    )
    channel.queue_bind(
        queue=topology.recognition_queue,
        exchange=topology.exchange,
        routing_key=topology.routing_key,
    )
    channel.queue_declare(queue=topology.dead_letter_queue, durable=True)
    channel.queue_bind(
        queue=topology.dead_letter_queue,
        exchange=topology.dead_letter_exchange,
        routing_key=topology.dead_letter_routing_key,
    )
