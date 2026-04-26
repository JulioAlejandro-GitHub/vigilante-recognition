from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.entities import FrameIngestedMessage
from app.ingestion.frame_ingested_loader import FrameIngestedJsonlLoader
from app.storage.frame_resolver import LocalFrameResolver


@dataclass(frozen=True)
class ProcessIngestionOutboxResult:
    source_path: Path
    processed: int
    emitted_events: list[dict] = field(default_factory=list)


def process_ingestion_outbox(
    jsonl_path: Path | str,
    *,
    processor: Callable[[FrameIngestedMessage], dict],
    frame_search_roots: list[Path] | None = None,
) -> ProcessIngestionOutboxResult:
    path = Path(jsonl_path)
    loader = FrameIngestedJsonlLoader(
        path,
        frame_resolver=LocalFrameResolver(search_roots=frame_search_roots),
    )
    emitted_events: list[dict] = []
    for loaded_event in loader.iter_messages():
        emitted_events.append(processor(loaded_event.message))
    return ProcessIngestionOutboxResult(
        source_path=path,
        processed=len(emitted_events),
        emitted_events=emitted_events,
    )
