from typing import Optional

import torch

aten = torch.ops.aten

# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    raise RuntimeError("Your PyTorch version is too old. Please upgrade to PyTorch >= 2.4.0")


def _attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return aten.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    ).contiguous()


@_torch_custom_op_wrapper("quantum_attn::attention_forward", mutates_args=(), device_types=("cuda",))
def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


@_torch_register_fake_wrapper("quantum_attn::attention_forward")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


def _fp8_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    out_dtype = value.dtype

    query = query.to(out_dtype)
    key = key.to(out_dtype)

    if scale_q is not None:
        assert scale_q.dim() == scale_k.dim()
        scale_q = scale_q.to(out_dtype)
        scale_k = scale_k.to(out_dtype)

        while scale_q.dim() < query.dim():
            scale_q = scale_q.unsqueeze(-1)
            scale_k = scale_k.unsqueeze(-1)

        query = query * scale_q
        key = key * scale_k

    return aten.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    ).contiguous()


@_torch_custom_op_wrapper("quantum_attn::fp8_attention_forward", mutates_args=(), device_types=("cuda",))
def fp8_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _fp8_attention_forward(
        query,
        key,
        value,
        scale_q,
        scale_k,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@_torch_register_fake_wrapper("quantum_attn::fp8_attention_forward")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return _fp8_attention_forward(
        query,
        key,
        value,
        scale_q,
        scale_k,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
