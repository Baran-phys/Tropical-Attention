"""Max-plus GEMM backend for Tropical Attention v2.0.0.

The fast path uses TensorBFS/tropical-gemm when its PyTorch helper or published
native extension is installed. The PyTorch implementation remains the fallback
so the repository is usable without Rust/CUDA build artifacts.
"""

from __future__ import annotations

import importlib
import math
import os
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

import torch

__version__ = "2.0.0"

_BACKEND_ENV = "TROPICAL_ATTENTION_MAXPLUS_BACKEND"
_AUTO = "auto"
_TORCH = "torch"
_TROPICAL_GEMM = "tropical_gemm"


class TropicalGemmUnavailable(RuntimeError):
    """Raised when the tropical-gemm backend is explicitly requested but unusable."""


def torch_maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference max-plus matrix product using native PyTorch operations.

    Computes ``C[..., i, j] = max_k(A[..., i, k] + B[..., k, j])`` with normal
    PyTorch broadcasting over leading dimensions.
    """

    _validate_maxplus_inputs(a, b)
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).amax(dim=-2)


def maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a max-plus matrix product with the fastest available backend.

    By default this uses ``tropical_gemm.pytorch`` when available for compatible
    float tensors, including its batched and CUDA DLPack paths. If that helper is
    absent, it uses the published top-level ``tropical_gemm`` extension on CPU.
    Set ``TROPICAL_ATTENTION_MAXPLUS_BACKEND=torch`` to force the PyTorch
    reference path, or ``...=tropical_gemm`` to fail loudly when the external
    backend cannot be used.
    """

    _validate_maxplus_inputs(a, b)
    backend = _backend_preference()
    if backend == _TORCH:
        return torch_maxplus_matmul(a, b)

    try:
        result = _tropical_gemm_maxplus_matmul(a, b)
    except Exception as exc:
        if backend == _TROPICAL_GEMM:
            raise TropicalGemmUnavailable(
                "tropical-gemm was requested but failed for this input"
            ) from exc
        result = None

    if result is not None:
        return result

    if backend == _TROPICAL_GEMM:
        raise TropicalGemmUnavailable(
            "tropical-gemm is unavailable or does not support this tensor shape, "
            "dtype, or device"
        )

    return torch_maxplus_matmul(a, b)


def maxplus_backend_status() -> str:
    """Return a compact status string useful in smoke tests and reports."""

    mode = _backend_preference()
    try:
        pt = _load_tropical_gemm_pytorch()
    except Exception as exc:
        try:
            _load_tropical_gemm_core()
        except Exception as core_exc:
            return f"{mode}: tropical-gemm unavailable ({core_exc})"
        return f"{mode}: tropical-gemm core available, PyTorch helper unavailable ({exc})"

    cuda = bool(getattr(pt, "GPU_AVAILABLE", False))
    has_batched = hasattr(pt, "tropical_maxplus_matmul_batched")
    return f"{mode}: tropical-gemm available (cuda={cuda}, batched={has_batched})"


def reset_backend_cache() -> None:
    """Clear cached imports. Primarily useful for tests that monkeypatch modules."""

    _load_tropical_gemm_pytorch.cache_clear()
    _load_tropical_gemm_core.cache_clear()


def _validate_maxplus_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("maxplus_matmul expects torch.Tensor inputs")
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError(
            f"maxplus_matmul expects tensors with at least 2 dimensions, got {a.shape} and {b.shape}"
        )
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"max-plus inner dimensions must match, got {a.shape[-1]} and {b.shape[-2]}"
        )
    if a.device != b.device:
        raise ValueError(f"max-plus inputs must be on the same device, got {a.device} and {b.device}")
    try:
        torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except RuntimeError as exc:
        raise ValueError(
            f"max-plus leading dimensions are not broadcastable: {a.shape[:-2]} and {b.shape[:-2]}"
        ) from exc


def _backend_preference() -> str:
    raw = os.getenv(_BACKEND_ENV, _AUTO).strip().lower().replace("-", "_")
    aliases = {
        "": _AUTO,
        _AUTO: _AUTO,
        "pytorch": _TORCH,
        _TORCH: _TORCH,
        "rust": _TROPICAL_GEMM,
        "simd": _TROPICAL_GEMM,
        "cuda": _TROPICAL_GEMM,
        _TROPICAL_GEMM: _TROPICAL_GEMM,
    }
    if raw not in aliases:
        raise ValueError(
            f"unknown {_BACKEND_ENV}={raw!r}; expected auto, torch, or tropical_gemm"
        )
    return aliases[raw]


@lru_cache(maxsize=1)
def _load_tropical_gemm_pytorch():
    return importlib.import_module("tropical_gemm.pytorch")


@lru_cache(maxsize=1)
def _load_tropical_gemm_core():
    return importlib.import_module("tropical_gemm")


def _tropical_gemm_maxplus_matmul(
    a: torch.Tensor, b: torch.Tensor
) -> Optional[torch.Tensor]:
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64):
        return None

    is_cuda = a.device.type == "cuda"
    try:
        result = _tropical_gemm_pytorch_maxplus_matmul(a, b, is_cuda)
    except ModuleNotFoundError:
        result = None
    if result is not None:
        return result

    if is_cuda:
        return None

    return _tropical_gemm_core_maxplus_matmul(a, b)


def _tropical_gemm_pytorch_maxplus_matmul(
    a: torch.Tensor, b: torch.Tensor, is_cuda: bool
) -> Optional[torch.Tensor]:
    pt = _load_tropical_gemm_pytorch()

    if a.dim() == 2 and b.dim() == 2:
        if is_cuda:
            if a.dtype != torch.float32 or not hasattr(pt, "tropical_maxplus_matmul_gpu"):
                return None
            return pt.tropical_maxplus_matmul_gpu(a.contiguous(), b.contiguous())
        if hasattr(pt, "tropical_maxplus_matmul"):
            return pt.tropical_maxplus_matmul(a.contiguous(), b.contiguous())
        return None

    batched_fn = getattr(pt, "tropical_maxplus_matmul_batched", None)
    if batched_fn is None:
        return None
    if a.dtype != torch.float32:
        # The current batched CPU binding casts to f32 internally, and the CUDA
        # batched DLPack path is also f32-only.
        return None

    return _call_batched_tropical_gemm(batched_fn, a, b)


def _tropical_gemm_core_maxplus_matmul(
    a: torch.Tensor, b: torch.Tensor
) -> Optional[torch.Tensor]:
    core = _load_tropical_gemm_core()
    if not _core_has_maxplus_argmax(core, a.dtype):
        return None

    if a.dim() == 2 and b.dim() == 2:
        return _CoreMaxPlus2DFunction.apply(a.contiguous(), b.contiguous())

    lead_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k = a.shape[-2:]
    n = b.shape[-1]

    a_batched = _expand_to_leading(a, lead_shape).reshape(-1, m, k).contiguous()
    b_batched = _expand_to_leading(b, lead_shape).reshape(-1, k, n).contiguous()
    out = _CoreMaxPlusBatchedFunction.apply(a_batched, b_batched)
    return out.reshape(*lead_shape, m, n)


def _call_batched_tropical_gemm(
    batched_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    lead_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k = a.shape[-2:]
    n = b.shape[-1]

    a_batched = _expand_to_leading(a, lead_shape).reshape(-1, m, k).contiguous()
    b_batched = _expand_to_leading(b, lead_shape).reshape(-1, k, n).contiguous()

    batch = math.prod(lead_shape) if lead_shape else 1
    if batch == 1 and not lead_shape:
        # Should be handled by the 2D path, but keep the shape contract explicit.
        return batched_fn(a_batched, b_batched).reshape(m, n)

    out = batched_fn(a_batched, b_batched)
    return out.reshape(*lead_shape, m, n)


def _expand_to_leading(x: torch.Tensor, lead_shape: torch.Size) -> torch.Tensor:
    x_lead = x.shape[:-2]
    pad = len(lead_shape) - len(x_lead)
    if pad < 0:
        raise ValueError(f"cannot expand {x.shape} to leading shape {lead_shape}")
    view_shape = (1,) * pad + tuple(x.shape)
    return x.reshape(view_shape).expand(*lead_shape, *x.shape[-2:])


def _core_has_maxplus_argmax(core, dtype: torch.dtype) -> bool:
    funcs = _core_dtype_funcs(core, dtype)
    return all(hasattr(core, name) for name in funcs.values())


def _core_dtype_funcs(core, dtype: torch.dtype) -> dict[str, str]:
    if dtype == torch.float64:
        return {
            "forward": "maxplus_matmul_with_argmax_f64",
            "backward_a": "backward_a_f64",
            "backward_b": "backward_b_f64",
        }
    return {
        "forward": "maxplus_matmul_with_argmax",
        "backward_a": "backward_a",
        "backward_b": "backward_b",
    }


def _numpy_dtype(dtype: torch.dtype):
    return np.float64 if dtype == torch.float64 else np.float32


def _as_numpy_2d(x: torch.Tensor, dtype: torch.dtype):
    arr = x.detach().cpu().numpy()
    np_dtype = _numpy_dtype(dtype)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


class _CoreMaxPlus2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        core = _load_tropical_gemm_core()
        funcs = _core_dtype_funcs(core, a.dtype)
        m, k = a.shape
        n = b.shape[1]

        c_flat, argmax_flat = getattr(core, funcs["forward"])(
            _as_numpy_2d(a, a.dtype), _as_numpy_2d(b, b.dtype)
        )
        c = np.asarray(c_flat, dtype=_numpy_dtype(a.dtype)).reshape(m, n)
        argmax = np.asarray(argmax_flat, dtype=np.int32).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax))
        ctx.k = k
        ctx.m = m
        ctx.n = n
        ctx.dtype = a.dtype

        return torch.from_numpy(c).to(device=a.device, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        core = _load_tropical_gemm_core()
        funcs = _core_dtype_funcs(core, ctx.dtype)

        grad_c_np = _as_numpy_2d(grad_c, ctx.dtype)
        argmax_np = argmax.numpy().astype(np.int32, copy=False)

        grad_a_flat = getattr(core, funcs["backward_a"])(grad_c_np, argmax_np, ctx.k)
        grad_b_flat = getattr(core, funcs["backward_b"])(grad_c_np, argmax_np, ctx.k)

        grad_a = torch.from_numpy(
            np.asarray(grad_a_flat, dtype=_numpy_dtype(ctx.dtype)).reshape(ctx.m, ctx.k)
        ).to(device=grad_c.device, dtype=grad_c.dtype)
        grad_b = torch.from_numpy(
            np.asarray(grad_b_flat, dtype=_numpy_dtype(ctx.dtype)).reshape(ctx.k, ctx.n)
        ).to(device=grad_c.device, dtype=grad_c.dtype)

        return grad_a, grad_b


class _CoreMaxPlusBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        core = _load_tropical_gemm_core()
        funcs = _core_dtype_funcs(core, a.dtype)
        batch, m, k = a.shape
        n = b.shape[2]
        np_dtype = _numpy_dtype(a.dtype)

        outputs = []
        argmaxes = []
        forward = getattr(core, funcs["forward"])
        for idx in range(batch):
            c_flat, argmax_flat = forward(
                _as_numpy_2d(a[idx], a.dtype), _as_numpy_2d(b[idx], b.dtype)
            )
            outputs.append(np.asarray(c_flat, dtype=np_dtype).reshape(m, n))
            argmaxes.append(np.asarray(argmax_flat, dtype=np.int64).reshape(m, n))

        ctx.save_for_backward(torch.from_numpy(np.stack(argmaxes, axis=0)))
        ctx.k = k
        return torch.from_numpy(np.stack(outputs, axis=0)).to(device=a.device, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax_cpu,) = ctx.saved_tensors
        argmax = argmax_cpu.to(device=grad_c.device, dtype=torch.long)
        batch, m, n = grad_c.shape
        k = ctx.k

        grad_a = torch.zeros(batch, m, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_a.scatter_add_(2, argmax, grad_c)

        argmax_t = argmax.transpose(1, 2)
        grad_c_t = grad_c.transpose(1, 2)
        grad_b_t = torch.zeros(batch, n, k, device=grad_c.device, dtype=grad_c.dtype)
        grad_b_t.scatter_add_(2, argmax_t, grad_c_t)
        grad_b = grad_b_t.transpose(1, 2)

        return grad_a, grad_b
