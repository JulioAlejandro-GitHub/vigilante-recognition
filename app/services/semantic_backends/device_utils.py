from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    resolved: str
    dtype_name: str
    reason: str


def normalize_device_preference(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    alias_map = {
        "": "auto",
        "automatic": "auto",
        "gpu": "cuda",
        "metal": "mps",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported_device_preference:{normalized}")
    return normalized


def select_device(preferred: str | None, *, torch_module: Any | None = None) -> DeviceSelection:
    requested = normalize_device_preference(preferred)
    availability = _device_availability(torch_module=torch_module)

    if requested == "auto":
        if availability["cuda"]:
            return DeviceSelection(
                requested="auto",
                resolved="cuda",
                dtype_name=_cuda_dtype_name(torch_module),
                reason="auto_cuda_available",
            )
        if availability["mps"]:
            return DeviceSelection(
                requested="auto",
                resolved="mps",
                dtype_name="float16",
                reason="auto_mps_available",
            )
        return DeviceSelection(
            requested="auto",
            resolved="cpu",
            dtype_name="float32",
            reason="auto_cpu_fallback",
        )

    if requested == "cpu":
        return DeviceSelection(
            requested="cpu",
            resolved="cpu",
            dtype_name="float32",
            reason="explicit_cpu",
        )

    if requested == "cuda" and availability["cuda"]:
        return DeviceSelection(
            requested="cuda",
            resolved="cuda",
            dtype_name=_cuda_dtype_name(torch_module),
            reason="explicit_cuda",
        )

    if requested == "mps" and availability["mps"]:
        return DeviceSelection(
            requested="mps",
            resolved="mps",
            dtype_name="float16",
            reason="explicit_mps",
        )

    raise RuntimeError(f"device_unavailable:{requested}")


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    try:
        return getattr(torch_module, dtype_name)
    except AttributeError as exc:
        raise RuntimeError(f"unsupported_torch_dtype:{dtype_name}") from exc


def _device_availability(*, torch_module: Any | None) -> dict[str, bool]:
    if torch_module is None:
        return {"cuda": False, "mps": False}

    cuda_available = bool(
        getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)()
    )

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)())
    mps_built = bool(getattr(mps_backend, "is_built", lambda: mps_available)())

    return {
        "cuda": cuda_available,
        "mps": mps_available and mps_built,
    }


def _cuda_dtype_name(torch_module: Any | None) -> str:
    if torch_module is None:
        return "float16"
    is_bf16_supported = getattr(getattr(torch_module, "cuda", None), "is_bf16_supported", None)
    if callable(is_bf16_supported) and is_bf16_supported():
        return "bfloat16"
    return "float16"
