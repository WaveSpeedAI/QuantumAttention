from typing import Optional

import torch

from quantum_attn.nn import dynamically_quantize_fp8, fp8_attention_forward

__all__ = ["fp8_attn_func", "dynamic_fp8_attn_func", "dynamically_quantize_fp8"]


def fp8_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
    scale_q: torch.Tensor,
    scale_k: torch.Tensor,
) -> torch.Tensor:
    return fp8_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
    )


def dynamic_fp8_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
) -> torch.Tensor:
    query, scale_q = dynamically_quantize_fp8(query, reduction_dim=-1)
    key, scale_k = dynamically_quantize_fp8(key, reduction_dim=-1)

    return fp8_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
    )
