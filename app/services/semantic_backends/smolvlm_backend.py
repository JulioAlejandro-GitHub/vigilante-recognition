from __future__ import annotations

from app.config import settings
from app.services.semantic_backends.base import TransformersImageTextSemanticBackend


class SmolVlmSemanticBackend(TransformersImageTextSemanticBackend):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        device_preference: str | None = None,
        runner=None,
    ) -> None:
        super().__init__(
            key="smolvlm",
            model_name=model_name or settings.semantic_vlm_fallback_model,
            device_preference=device_preference or settings.semantic_device,
            runner=runner,
        )
