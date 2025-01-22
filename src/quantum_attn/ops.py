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


def _fp8_attention_forward(
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
) -> torch.Tensor:
    assert attn_mask is None, "attn_mask is not supported yet"
    assert dropout_p == 0.0, "dropout_p is not supported yet"

    device = query.device

    assert device.type == "cuda", "Must be CUDA tensors"
    assert torch.version.hip is None, "HIP is not supported"
    assert torch.cuda.get_device_capability(device.index) >= (9, 0), "Must be GPU with compute capability >= 9.0"

    assert torch._dynamo.is_dynamo_supported(), "dynamo is not available"
    assert torch._dynamo.is_inductor_supported(), "inductor is not available"

    if not torch.compiler.is_dynamo_compiling():
        raise RuntimeError("fp8_attention_forward must be called within torch.compile()")

    query = query * scale_q[..., None]
    key = key * scale_k[..., None]
    value = value * scale_v[..., None]

    return aten.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


@_torch_custom_op_wrapper("quantum_attn::fp8_attention_forward", mutates_args=(), device_types=("cuda",))
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
) -> torch.Tensor:
    return _fp8_attention_forward(
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


@_torch_register_fake_wrapper("quantum_attn::fp8_attention_forward")
def _(
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
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _fp8_attention_forward(
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
