from typing import Optional

import torch
import torch.nn.functional as F

from quantum_attn.nn import (
    attention_forward,
    can_use_attention_forward,
    dynamically_quantize_fp8,
    fp8_attention_forward,
)

__all__ = [
    "attn_func",
    "attn_func_with_fallback",
    "fp8_attn_func",
    "fp8_attn_func_with_fallback",
    "dynamically_quantize_fp8",
]

torch_sdpa = F.scaled_dot_product_attention


def _define_composite_implicit_autograd_op(namespace, name, signature):
    def decorator(fn):
        torch.library.define(f"{namespace}::{name}", signature)
        torch.library.impl(f"{namespace}::{name}", ["CompositeImplicitAutograd"])(fn)

        ns = getattr(torch.ops, namespace)
        op = getattr(ns, name)
        return op

    return decorator


def _define_quantum_attn_composite_implicit_autograd_op(name, signature):
    return _define_composite_implicit_autograd_op("quantum_attn", name, signature)


def sdpa_dispatcher(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
    return tuple(filter(torch.is_tensor, (query, key, value, attn_mask)))


def attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
) -> torch.Tensor:
    return attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_define_quantum_attn_composite_implicit_autograd_op(
    "attn_func_with_fallback",
    "(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> Tensor",
)
def attn_func_with_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
) -> torch.Tensor:
    supported, reason = can_use_attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    if supported:
        return attn_func(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    return torch_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def fp8_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
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


@_define_quantum_attn_composite_implicit_autograd_op(
    "fp8_attn_func_with_fallback",
    "(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> Tensor",
)
def fp8_attn_func_with_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float = None,
) -> torch.Tensor:
    if can_use_attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )[0]:
        return fp8_attn_func(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    return torch_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
