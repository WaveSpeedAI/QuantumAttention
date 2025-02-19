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


_TK_TMA_SUPPORTED_HEAD_DIMS = [64, 128, 256]


def _tk_tma_supported_head_dim(n: Union[int, torch.SymInt]) -> bool:
    return n in _TK_TMA_SUPPORTED_HEAD_DIMS


def _validate_tk_tma_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    scale_q=None,
    scale_k=None,
) -> Tuple[bool, str]:
    if attn_mask is not None:
        return False, "NYI: attn_mask must be None"
    if dropout_p != 0.0:
        return False, "NYI: dropout_p must be 0.0"
    if scale is not None:
        return False, "NYI: scale must be None"
    if scale_q is None:
        if scale_k is not None:
            return False, "scale_k must be None if scale_q is None"
        if query.dtype != key.dtype or query.dtype != value.dtype:
            return (
                False,
                f"Expected query, key, and value to have the same dtype, but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, and value.dtype: {value.dtype} instead.",
            )
        if query.dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"Expected query, key, and value to have dtype torch.float16 or torch.bfloat16, but got query.dtype: {query.dtype} instead.",
            )
    else:
        if scale_k is None:
            return False, "scale_q must be None if scale_k is None"
        if query.dtype != torch.float8_e4m3fn:
            return (
                False,
                f"Expected query to have dtype torch.float8_e4m3fn, but got query.dtype: {query.dtype} instead.",
            )
        if scale_q.shape != query.shape[:-1]:
            return (
                False,
                f"Expected scale_q to have shape equal to query.shape[:-1], but got scale_q.shape: {scale_q.shape}, query.shape: {query.shape} instead.",
            )
        if scale_k.shape != key.shape[:-1]:
            return (
                False,
                f"Expected scale_k to have shape equal to key.shape[:-1], but got scale_k.shape: {scale_k.shape}, key.shape: {key.shape} instead.",
            )
    if query.dtype != key.dtype:
        return (
            False,
            f"Expected query and key to have the same dtype, but got query.dtype: {query.dtype}, key.dtype: {key.dtype} instead.",
        )
    if value.dtype not in (torch.float16, torch.bfloat16):
        return (
            False,
            f"Expected value to have dtype torch.float16 or torch.bfloat16, but got value.dtype: {value.dtype} instead.",
        )
    if query.device != key.device or query.device != value.device:
        return (
            False,
            f"Expected query, key, and value to have the same device type, but got query.device: {query.device}, key.device: {key.device}, and value.device: {value.device} instead.",
        )
    if query.device.type != "cuda":
        return False, "Expected query, key, and value to be on a CUDA device"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False, "NYI: query, key, and value must be 4D tensors"
    if key.size(-2) != value.size(-2):
        return (
            False,
            f"Expect key and value to have the same sequence length but got Sk={key.size(-2)} and Sv={value.size(-2)}.",
        )
    if value.size(-1) != query.size(-1):
        return False, "NYI: query and value must have the same embedding dimension"
    if query.size(-3) != key.size(-3):
        return (
            False,
            f"Expect query and key/value to have the same number of heads but got Hq={query.size(-3)} and Hkv={key.size(-3)}.",
        )
    if not _tk_tma_supported_head_dim(query.size(-1)):
        return False, f"Unsupported head dimension: {query.size(-1)}"

    return True, ""


_TRITON_TMA_SDPA_SUPPORTED_HEAD_DIMS = [64, 128, 256]


def _triton_tma_sdpa_supported_head_dim(n: Union[int, torch.SymInt]) -> bool:
    """Returns true if the head dim is supported by FlexAttention"""
    return n in _TRITON_TMA_SDPA_SUPPORTED_HEAD_DIMS


def _validate_triton_tma_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    scale_q=None,
    scale_k=None,
) -> Tuple[bool, str]:
    if attn_mask is not None:
        return False, "NYI: attn_mask must be None"
    if dropout_p != 0.0:
        return False, "NYI: dropout_p must be 0.0"
    if scale_q is None:
        if scale_k is not None:
            return False, "scale_k must be None if scale_q is None"
        if query.dtype != key.dtype or query.dtype != value.dtype:
            return (
                False,
                f"Expected query, key, and value to have the same dtype, but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, and value.dtype: {value.dtype} instead.",
            )
        if query.dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"Expected query, key, and value to have dtype torch.float16 or torch.bfloat16, but got query.dtype: {query.dtype} instead.",
            )
    else:
        if scale_k is None:
            return False, "scale_q must be None if scale_k is None"
        if query.dtype != torch.float8_e4m3fn:
            return (
                False,
                f"Expected query to have dtype torch.float8_e4m3fn, but got query.dtype: {query.dtype} instead.",
            )
        if scale_q.shape != query.shape[:-1]:
            return (
                False,
                f"Expected scale_q to have shape equal to query.shape[:-1], but got scale_q.shape: {scale_q.shape}, query.shape: {query.shape} instead.",
            )
        if scale_k.shape != key.shape[:-1]:
            return (
                False,
                f"Expected scale_k to have shape equal to key.shape[:-1], but got scale_k.shape: {scale_k.shape}, key.shape: {key.shape} instead.",
            )
    if query.dtype != key.dtype:
        return (
            False,
            f"Expected query and key to have the same dtype, but got query.dtype: {query.dtype}, key.dtype: {key.dtype} instead.",
        )
    if value.dtype not in (torch.float16, torch.bfloat16):
        return (
            False,
            f"Expected value to have dtype torch.float16 or torch.bfloat16, but got value.dtype: {value.dtype} instead.",
        )
    if query.device != key.device or query.device != value.device:
        return (
            False,
            f"Expected query, key, and value to have the same device type, but got query.device: {query.device}, key.device: {key.device}, and value.device: {value.device} instead.",
        )
    if query.device.type != "cuda":
        return False, "Expected query, key, and value to be on a CUDA device"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False, "NYI: query, key, and value must be 4D tensors"
    if key.size(-2) != value.size(-2):
        return (
            False,
            f"Expect key and value to have the same sequence length but got Sk={key.size(-2)} and Sv={value.size(-2)}.",
        )
    if value.size(-1) != query.size(-1):
        return False, "NYI: query and value must have the same embedding dimension"
    if query.size(-3) != key.size(-3):
        return (
            False,
            f"Expect query and key/value to have the same number of heads but got Hq={query.size(-3)} and Hkv={key.size(-3)}.",
        )
    if not _triton_tma_sdpa_supported_head_dim(query.size(-1)):
        return False, f"Unsupported head dimension: {query.size(-1)}"

    return True, ""


@torch.compiler.assume_constant_result
def _pre_check_can_use_tk_tma_attention_forward(device):
    if device.type != "cuda":
        return False, f"Expected device to be on a CUDA device, but got device: {device} instead."
    if not config.attention.enable_tk_tma_kernel:
        return False, "TK TMA kernel is disabled"
    if not checks.cuda_capability_compare("ge", 9, 0, device=device):
        return False, "Minimum CUDA capability of 9.0 is required"
    return True, ""


def can_use_tk_tma_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[bool, str]:
    supported, reason = _pre_check_can_use_tk_tma_attention_forward(device=query.device)
    if not supported:
        return False, reason

    valid, reason = _validate_tk_tma_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    if not valid:
        return False, reason

    return True, ""


@torch.compiler.assume_constant_result
def _pre_check_can_use_triton_tma_attention_forward(device):
    if device.type != "cuda":
        return False, f"Expected device to be on a CUDA device, but got device: {device} instead."
    if not config.attention.enable_triton_tma_kernel:
        return False, "Triton TMA kernel is disabled"
    if not checks.cuda_capability_compare("ge", 9, 0, device=device):
        return False, "Minimum CUDA capability of 9.0 is required"
    if not checks.torch_version_compare("ge", "2.7.0"):
        return False, "Minimum PyTorch version of 2.7.0 is required"
    if not checks.has_triton_tma_support():
        return False, "Triton TMA support is required"
    return True, ""


def can_use_triton_tma_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[bool, str]:
    supported, reason = _pre_check_can_use_triton_tma_attention_forward(device=query.device)
    if not supported:
        return False, reason

    valid, reason = _validate_triton_tma_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    if not valid:
        return False, reason

    return True, ""


@torch.compiler.assume_constant_result
def _skip_supported_check() -> bool:
    return config.attention.skip_supported_check


def can_use_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[bool, str]:
    if _skip_supported_check():
        return True, ""
    prefix_and_funcs = [
        ("tk_tma", can_use_tk_tma_attention_forward),
        ("triton_tma_sdpa", can_use_triton_tma_attention_forward),
    ]
    reasons = []
    for prefix, func in prefix_and_funcs:
        supported, reason = func(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
        if supported:
            return True, ""
        reasons.append((f"{prefix}: {reason}"))
    return False, " ".join(f"[{reason}]" for reason in reasons)


def _attention_forward_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    return quantum_attn_ops.attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("attention_forward requires dynamo support")
    if not torch._dynamo.is_inductor_supported():
        raise RuntimeError("attention_forward requires inductor support")

    supported, reason = can_use_attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    if not supported:
        raise ValueError(reason)

    # if scale is None:
    #     scale = 1.0 / math.sqrt(query.size(-1))

    from torch._subclasses.fake_tensor import is_fake

    if any(is_fake(x) for x in (query, key, value, attn_mask)):
        out = _attention_forward_wrapper(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        return out

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        out = _attention_forward_wrapper(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        return out

    if config.attention.force_eager_fallback:
        out = _attention_forward_wrapper(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        return out

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                out = torch.compile(
                    _attention_forward_wrapper,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=config.dynamo.dynamic,
                    mode=config.dynamo.mode,
                )(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
                return out


def _fp8_attention_forward_wrapper(
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
    scaling_method: Optional[str] = None,
) -> Tensor:
    if (scale_q is None) != (scale_k is None):
        raise ValueError("scale_q and scale_k must be both provided or both not provided")

    if scale_q is None:
        if scaling_method is None:
            scaling_method = "none"
        if scaling_method == "none":
            q_max = torch.finfo(torch.float8_e4m3fn).max
            query = query.clamp(-q_max, q_max).to(torch.float8_e4m3fn)
            key = key.clamp(-q_max, q_max).to(torch.float8_e4m3fn)
        elif scaling_method == "token-wise":
            reduction_dim = query.dim() - 1
            query, scale_q = dynamically_quantize_fp8(query, reduction_dim=reduction_dim)
            key, scale_k = dynamically_quantize_fp8(key, reduction_dim=reduction_dim)
        else:
            raise ValueError(f"Unsupported scaling_method: {scaling_method}")

    return quantum_attn_ops.fp8_attention_forward(
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
    scaling_method: Optional[str] = None,
) -> Tensor:
    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("fp8_attention_forward requires dynamo support")
    if not torch._dynamo.is_inductor_supported():
        raise RuntimeError("fp8_attention_forward requires inductor support")

    supported, reason = can_use_attention_forward(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    if not supported:
        raise ValueError(reason)

    # if scale is None:
    #     scale = 1.0 / math.sqrt(query.size(-1))

    from torch._subclasses.fake_tensor import is_fake

    if any(is_fake(x) for x in (query, key, value, attn_mask, scale_q, scale_k)):
        out = _fp8_attention_forward_wrapper(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            scale_q=scale_q,
            scale_k=scale_k,
            scaling_method=scaling_method,
        )
        return out

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        out = _fp8_attention_forward_wrapper(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            scale_q=scale_q,
            scale_k=scale_k,
            scaling_method=scaling_method,
        )
        return out

    if config.attention.force_eager_fallback:
        out = _fp8_attention_forward_wrapper(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            scale_q=scale_q,
            scale_k=scale_k,
            scaling_method=scaling_method,
        )
        return out

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                out = torch.compile(
                    _fp8_attention_forward_wrapper,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=config.dynamo.dynamic,
                    mode=config.dynamo.mode,
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
                    scaling_method=scaling_method,
                )
                return out
