from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from app.services.semantic_backends.device_utils import (
    DeviceSelection,
    resolve_torch_dtype,
    select_device,
)


@dataclass
class VlmRuntimeResult:
    raw_text: str
    model_name: str
    device: str
    requested_device: str
    dtype_name: str
    runtime: str = "isolated_subprocess"
    extra: dict[str, Any] = field(default_factory=dict)

    def trace_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "requested_device": self.requested_device,
            "dtype": self.dtype_name,
            "runtime": self.runtime,
            **self.extra,
        }


class VlmRuntimeError(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class ProcessIsolatedTransformersRunner:
    def __init__(
        self,
        *,
        backend_key: str,
        model_name: str,
        device_preference: str = "auto",
        max_new_tokens: int = 320,
    ) -> None:
        self.backend_key = backend_key
        self.model_name = model_name
        self.device_preference = device_preference
        self.max_new_tokens = max_new_tokens
        self._ctx = get_context("spawn")
        self._parent_conn: Connection | None = None
        self._process = None
        self._request_counter = 0
        atexit.register(self.close)

    def generate_text(
        self,
        *,
        image_path: Path,
        prompt: str,
        timeout_seconds: int,
    ) -> VlmRuntimeResult:
        self._ensure_process()
        assert self._parent_conn is not None

        request_id = self._next_request_id()
        try:
            self._parent_conn.send(
                {
                    "command": "generate",
                    "request_id": request_id,
                    "image_path": str(image_path),
                    "prompt": prompt,
                }
            )
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._restart_process()
            raise VlmRuntimeError(
                "backend_process_exited",
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "send",
                    "exception_type": type(exc).__name__,
                },
            ) from exc

        if not self._parent_conn.poll(timeout_seconds):
            self._restart_process()
            raise VlmRuntimeError(
                "backend_timeout",
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "runtime",
                    "timeout_seconds": timeout_seconds,
                },
            )

        try:
            response = self._parent_conn.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._restart_process()
            raise VlmRuntimeError(
                "backend_process_exited",
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "receive",
                    "exception_type": type(exc).__name__,
                },
            ) from exc

        if response.get("request_id") != request_id:
            raise VlmRuntimeError(
                "backend_protocol_error",
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "receive",
                },
            )

        if response.get("status") == "error":
            raise VlmRuntimeError(
                str(response.get("reason", "backend_runtime_error")),
                details=response.get("details") or {},
            )

        runtime = response.get("runtime") or {}
        return VlmRuntimeResult(
            raw_text=str(response.get("raw_text", "")),
            model_name=str(runtime.get("model_name") or self.model_name),
            device=str(runtime.get("device") or "cpu"),
            requested_device=str(runtime.get("requested_device") or self.device_preference),
            dtype_name=str(runtime.get("dtype") or "float32"),
            runtime=str(runtime.get("runtime") or "isolated_subprocess"),
            extra={
                key: value
                for key, value in runtime.items()
                if key not in {"model_name", "device", "requested_device", "dtype", "runtime"}
            },
        )

    def close(self) -> None:
        if self._process is None:
            return
        if self._parent_conn is not None:
            try:
                self._parent_conn.send({"command": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._terminate_process()

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._start_process()

    def _start_process(self) -> None:
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
        parent_conn, child_conn = self._ctx.Pipe()
        self._parent_conn = parent_conn
        self._process = self._ctx.Process(
            target=_transformers_runtime_process,
            kwargs={
                "connection": child_conn,
                "backend_key": self.backend_key,
                "model_name": self.model_name,
                "device_preference": self.device_preference,
                "max_new_tokens": self.max_new_tokens,
            },
            daemon=True,
        )
        self._process.start()
        child_conn.close()

    def _restart_process(self) -> None:
        self._terminate_process()
        self._start_process()

    def _terminate_process(self) -> None:
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._parent_conn = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            self._process = None

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"{self.backend_key}-{self._request_counter}"


def _transformers_runtime_process(
    *,
    connection: Connection,
    backend_key: str,
    model_name: str,
    device_preference: str,
    max_new_tokens: int,
) -> None:
    session = _TransformersSession(
        backend_key=backend_key,
        model_name=model_name,
        device_preference=device_preference,
        max_new_tokens=max_new_tokens,
    )
    try:
        while True:
            request = connection.recv()
            if request.get("command") == "shutdown":
                return
            if request.get("command") != "generate":
                connection.send(
                    {
                        "status": "error",
                        "request_id": request.get("request_id"),
                        "reason": "backend_protocol_error",
                        "details": {
                            "backend_key": backend_key,
                            "model_name": model_name,
                            "requested_device": device_preference,
                            "stage": "runtime",
                        },
                    }
                )
                continue

            try:
                raw_text = session.generate_text(
                    image_path=Path(str(request["image_path"])),
                    prompt=str(request["prompt"]),
                )
                connection.send(
                    {
                        "status": "ok",
                        "request_id": request["request_id"],
                        "raw_text": raw_text,
                        "runtime": session.runtime_payload(),
                    }
                )
            except VlmRuntimeError as exc:
                connection.send(
                    {
                        "status": "error",
                        "request_id": request.get("request_id"),
                        "reason": exc.reason,
                        "details": exc.details,
                    }
                )
    except EOFError:
        return
    finally:
        try:
            connection.close()
        except OSError:
            pass


class _TransformersSession:
    def __init__(
        self,
        *,
        backend_key: str,
        model_name: str,
        device_preference: str,
        max_new_tokens: int,
    ) -> None:
        self.backend_key = backend_key
        self.model_name = model_name
        self.device_preference = device_preference
        self.max_new_tokens = max_new_tokens
        self._selection: DeviceSelection | None = None
        self._torch = None
        self._image_module = None
        self._processor = None
        self._model = None

    def generate_text(self, *, image_path: Path, prompt: str) -> str:
        self._ensure_loaded()
        assert self._selection is not None
        assert self._processor is not None
        assert self._model is not None
        assert self._image_module is not None
        assert self._torch is not None

        try:
            image = self._image_module.open(image_path).convert("RGB")
        except Exception as exc:
            raise VlmRuntimeError(
                f"model_inference_failed:{type(exc).__name__}",
                details=self._error_details(stage="image_open", selection=self._selection, exc=exc),
            ) from exc

        try:
            rendered_prompt = self._processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            rendered_prompt = prompt

        try:
            try:
                inputs = self._processor(
                    text=[rendered_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = self._processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )
            inputs = self._move_inputs_to_device(inputs)
            output_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
            generated_ids = output_ids
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                try:
                    generated_ids = [
                        output_row[input_row.shape[-1] :]
                        for input_row, output_row in zip(input_ids, output_ids)
                    ]
                except Exception:
                    generated_ids = output_ids
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            raw_text = str(decoded[0] if decoded else "").strip()
            if not raw_text:
                fallback_decoded = self._processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                raw_text = str(fallback_decoded[0] if fallback_decoded else "").strip()
        except Exception as exc:
            raise VlmRuntimeError(
                f"model_inference_failed:{type(exc).__name__}",
                details=self._error_details(stage="inference", selection=self._selection, exc=exc),
            ) from exc

        return raw_text

    def runtime_payload(self) -> dict[str, Any]:
        assert self._selection is not None
        return {
            "backend_key": self.backend_key,
            "model_name": self.model_name,
            "requested_device": self._selection.requested,
            "device": self._selection.resolved,
            "dtype": self._selection.dtype_name,
            "runtime": "isolated_subprocess",
        }

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None and self._selection is not None:
            return

        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise VlmRuntimeError(
                "optional_vlm_dependencies_missing",
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "import",
                    "exception_type": type(exc).__name__,
                },
            ) from exc

        self._torch = torch
        self._image_module = Image

        try:
            self._selection = select_device(self.device_preference, torch_module=torch)
        except (ValueError, RuntimeError) as exc:
            raise VlmRuntimeError(
                str(exc),
                details={
                    "backend_key": self.backend_key,
                    "model_name": self.model_name,
                    "requested_device": self.device_preference,
                    "stage": "device_selection",
                },
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            model_class = self._resolve_model_class(
                auto_model_class=AutoModelForImageTextToText,
            )
            load_kwargs = {
                "torch_dtype": resolve_torch_dtype(torch, self._selection.dtype_name),
            }
            try:
                self._model = model_class.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    **load_kwargs,
                )
            except TypeError:
                self._model = model_class.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                )
            self._model = self._model.to(self._selection.resolved)
            self._model.eval()
        except Exception as exc:
            raise VlmRuntimeError(
                f"model_load_failed:{type(exc).__name__}",
                details=self._error_details(stage="load", selection=self._selection, exc=exc),
            ) from exc

    def _move_inputs_to_device(self, inputs: Any) -> dict[str, Any]:
        assert self._selection is not None
        assert self._torch is not None
        dtype = resolve_torch_dtype(self._torch, self._selection.dtype_name)
        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            if hasattr(value, "to"):
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    moved[key] = value.to(device=self._selection.resolved, dtype=dtype)
                else:
                    moved[key] = value.to(self._selection.resolved)
            else:
                moved[key] = value
        return moved

    def _resolve_model_class(self, *, auto_model_class: Any) -> Any:
        if self.backend_key != "qwen_vl":
            return auto_model_class

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            return auto_model_class

    def _error_details(
        self,
        *,
        stage: str,
        selection: DeviceSelection | None,
        exc: Exception,
    ) -> dict[str, Any]:
        return {
            "backend_key": self.backend_key,
            "model_name": self.model_name,
            "requested_device": selection.requested if selection else self.device_preference,
            "device": selection.resolved if selection else "unknown",
            "dtype": selection.dtype_name if selection else "unknown",
            "stage": stage,
            "exception_type": type(exc).__name__,
        }
