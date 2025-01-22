try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from . import config, nn, ops
from .quantum_attn_interface import dynamic_fp8_attn_func, dynamically_quantize_fp8, fp8_attn_func

__all__ = [
    "fp8_attn_func",
    "dynamic_fp8_attn_func",
    "dynamically_quantize_fp8",
]
