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
            model_name=model_name or settings.effective_smolvlm_model_name,
            device_preference=device_preference or settings.effective_vlm_device,
            runner=runner,
            max_new_tokens=settings.vlm_max_new_tokens,
            max_image_edge=settings.vlm_max_image_edge,
            serialization_guard_enabled=settings.vlm_serialization_guard_enabled,
        )
