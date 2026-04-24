from __future__ import annotations

from app.config import settings
from app.services.semantic_backends.base import TransformersImageTextSemanticBackend


class SmolVlmSemanticBackend(TransformersImageTextSemanticBackend):
    def __init__(self, *, model_name: str | None = None) -> None:
        super().__init__(
            key="smolvlm",
            model_name=model_name or settings.semantic_vlm_fallback_model,
        )
