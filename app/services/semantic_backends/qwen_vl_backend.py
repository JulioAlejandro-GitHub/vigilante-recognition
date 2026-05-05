from __future__ import annotations

from app.config import settings
from app.services.semantic_backends.base import TransformersImageTextSemanticBackend


class QwenVLSemanticBackend(TransformersImageTextSemanticBackend):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        device_preference: str | None = None,
        runner=None,
    ) -> None:
        super().__init__(
            key="qwen",
            model_name=model_name or settings.effective_qwen_model_name,
            device_preference=device_preference or settings.effective_vlm_device,
            runner=runner,
            max_new_tokens=settings.vlm_max_new_tokens,
            max_image_edge=settings.vlm_max_image_edge,
        )
