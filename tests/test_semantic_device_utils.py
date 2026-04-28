from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.semantic_backends.device_utils import select_device


def _torch_stub(
    *,
    cuda_available: bool = False,
    cuda_bf16: bool = False,
    mps_available: bool = False,
    mps_built: bool = True,
):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            is_bf16_supported=lambda: cuda_bf16,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: mps_available,
                is_built=lambda: mps_built,
            )
        ),
    )


def test_select_device_prefers_cuda_in_auto_mode():
    selection = select_device(
        "auto",
        torch_module=_torch_stub(cuda_available=True, cuda_bf16=True, mps_available=True),
    )

    assert selection.resolved == "cuda"
    assert selection.dtype_name == "bfloat16"
    assert selection.reason == "auto_cuda_available"


def test_select_device_uses_mps_when_cuda_is_unavailable():
    selection = select_device(
        "auto",
        torch_module=_torch_stub(cuda_available=False, mps_available=True),
    )

    assert selection.resolved == "mps"
    assert selection.dtype_name == "float16"
    assert selection.reason == "auto_mps_available"


def test_select_device_falls_back_to_cpu_when_no_accelerator_exists():
    selection = select_device(
        "auto",
        torch_module=_torch_stub(cuda_available=False, mps_available=False),
    )

    assert selection.resolved == "cpu"
    assert selection.dtype_name == "float32"
    assert selection.reason == "auto_cpu_fallback"


def test_select_device_rejects_unavailable_explicit_device():
    with pytest.raises(RuntimeError, match="device_unavailable:mps"):
        select_device(
            "mps",
            torch_module=_torch_stub(cuda_available=False, mps_available=False),
        )
