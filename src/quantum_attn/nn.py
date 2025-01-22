import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch._higher_order_ops.utils import _set_compilation_env
from torch.fx.experimental.proxy_tensor import _temp_remove_pre_dispatch_torch_function_mode

from quantum_attn import config

quantum_attn_ops = torch.ops.quantum_attn


def _dynamically_quantize_fp8(t: torch.Tensor, *, reduction_dim=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(torch.float32).eps
    q_max = torch.finfo(torch.float8_e4m3fn).max
    scale = t.abs().amax(reduction_dim, keepdim=True).mul(1.0 / q_max).clamp_min(eps)
    t_fp8 = (t / scale).clamp(-q_max, q_max).to(torch.float8_e4m3fn)
    return t_fp8, scale.squeeze(reduction_dim)


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


def fp8_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    scale_q: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
) -> Tensor:
    # Some basic input validation
    _validate_sdpa_input(query, key, value)
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    if scale_q.dim() != 3 or scale_k.dim() != 3 or scale_v.dim() != 3:
        raise NotImplementedError("NYI: scale_q, scale_k, and scale_v must be 3D tensors")
    if query.size(-3) != key.size(-3):
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.size(-3)} and Hkv={key.size(-3)}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        out = quantum_attn_ops.fp8_attention_forward(
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
        )
        return out

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("fp8_attention_forward requires dynamo support")
    if not torch._dynamo.is_inductor_supported():
        raise RuntimeError("fp8_attention_forward requires inductor support")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass hop to it. So we wrap it in a dummy function.
    def _fp8_attention_wrapper(*args, **kwargs):
        return quantum_attn_ops.fp8_attention_forward(*args, **kwargs)

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
                )
                return out
