import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch._higher_order_ops.utils import _set_compilation_env
from torch.fx.experimental.proxy_tensor import _temp_remove_pre_dispatch_torch_function_mode

from quantum_attn import config
from quantum_attn.utils import checks

quantum_attn_ops = torch.ops.quantum_attn


def _dynamically_quantize_fp8(t: torch.Tensor, *, reduction_dim=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(torch.float32).eps
    q_max = torch.finfo(torch.float8_e4m3fn).max
    scale = t.abs().amax(reduction_dim, keepdim=True).mul(1.0 / q_max).clamp_min(eps)
    t_fp8 = (t / scale).clamp(-q_max, q_max).to(torch.float8_e4m3fn)
    return t_fp8, scale.squeeze(reduction_dim).to(torch.float32)


def dynamically_quantize_fp8(t: torch.Tensor, *, reduction_dim=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_dynamo_compiling():
        return _dynamically_quantize_fp8(t, reduction_dim=reduction_dim)

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                out = torch.compile(
                    _dynamically_quantize_fp8, backend="inductor", fullgraph=True, dynamic=config.dynamo.dynamic
                )(
                    t,
                    reduction_dim=reduction_dim,
                )
                return out


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    # if query.dtype != key.dtype or query.dtype != value.dtype:
    #     raise ValueError(
    #         f"Expected query, key, and value to have the same dtype, "
    #         f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
    #         f"and value.dtype: {value.dtype} instead."
    #     )
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"Expected query to have dtype torch.float16, torch.bfloat16, or torch.float8_e4m3fn, "
            f"but got query.dtype: {query.dtype} instead."
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )

    if query.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"Expected query, key, and value to have dtype torch.float16, torch.bfloat16, or torch.float8_e4m3fn, "
            f"but got query.dtype: {query.dtype} instead."
        )

    if query.device.type != "cuda":
        raise ValueError("Expected query, key, and value to be on a CUDA device")


_SUPPORTED_HEAD_DIMS = [64, 128, 256]


def _supported_head_dim(n: Union[int, torch.SymInt]) -> bool:
    """Returns true if the head dim is supported by FlexAttention"""
    return n in _SUPPORTED_HEAD_DIMS


def can_use_fp8_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[bool, str]:
    if not checks.cuda_capability_compare("ge", 9, 0, device=query.device):
        return False, "Minimum CUDA capability of 9.0 is required"

    if not checks.torch_version_compare("ge", "2.6.0"):
        return False, "Minimum PyTorch version of 2.6.0 is required"

    if not checks.has_triton_tma_support():
        return False, "Triton TMA support is required"

    if attn_mask is not None:
        return False, "NYI: attn_mask must be None"
    if dropout_p != 0.0:
        return False, "NYI: dropout_p must be 0.0"

    try:
        _validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    except ValueError as e:
        return False, str(e)

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False, "NYI: query, key, and value must be 4D tensors"
    if query.size(-3) != key.size(-3):
        return False, (
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.size(-3)} and Hkv={key.size(-3)}. "
            f"Try setting enable_gqa=True for GQA."
        )

    if not (_supported_head_dim(query.size(-1)) and _supported_head_dim(value.size(-1))):
        raise ValueError(
            f"NYI: Currently non power of 2 embedding dimension are not supported. "
            f"Got E={query.size(-1)} and Ev={value.size(-1)}."
        )

    if value.size(-1) != query.size(-1):
        return False, "NYI: query and value must have the same embedding dimension"

    return True, ""


def _fp8_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if (scale_q is None) != (scale_k is None) or (scale_q is None) != (scale_v is None):
        raise ValueError("scale_q and scale_k must be both provided or both not provided")

    if scale_q is None:
        if out_dtype is None:
            out_dtype = query.dtype
        query, scale_q = dynamically_quantize_fp8(query, reduction_dim=-1)
        key, scale_k = dynamically_quantize_fp8(key, reduction_dim=-1)
        value, scale_v = dynamically_quantize_fp8(value, reduction_dim=-2)
    else:
        if out_dtype is None:
            raise ValueError("out_dtype must be provided if scale_q, scale_k, and scale_v are provided")

    return quantum_attn_ops.fp8_attention_forward(
        query,
        key,
        value,
        scale_q,
        scale_k,
        scale_v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def fp8_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    supported, reason = can_use_fp8_attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    if not supported:
        raise ValueError(reason)

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        out = _fp8_attention_wrapper(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            out_dtype=out_dtype,
        )
        return out

    if config.attention.force_eager_fallback:
        out = _fp8_attention_wrapper(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            out_dtype=out_dtype,
        )
        return out

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("fp8_attention_forward requires dynamo support")
    if not torch._dynamo.is_inductor_supported():
        raise RuntimeError("fp8_attention_forward requires inductor support")

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                out = torch.compile(
                    _fp8_attention_wrapper, backend="inductor", fullgraph=True, dynamic=config.dynamo.dynamic
                )(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    scale_q=scale_q,
                    scale_k=scale_k,
                    scale_v=scale_v,
                    out_dtype=out_dtype,
                )
                return out
