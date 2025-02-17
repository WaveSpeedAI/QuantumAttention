try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

import torch

from . import config, nn, ops
from .quantum_attn_interface import (
    attn_func,
    attn_func_with_fallback,
    dynamically_quantize_fp8,
    fp8_attn_func,
    fp8_attn_func_with_fallback,
    fp8_token_wise_attn_func,
    fp8_token_wise_attn_func_with_fallback,
)

if torch._dynamo.is_inductor_supported():
    from . import inductor

__all__ = [
    "attn_func",
    "attn_func_with_fallback",
    "dynamically_quantize_fp8",
    "fp8_attn_func",
    "fp8_attn_func_with_fallback",
    "fp8_token_wise_attn_func",
    "fp8_token_wise_attn_func_with_fallback",
]
