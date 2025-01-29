import os  # noqa: C101
import sys

# import torch

_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
}


use_fast_accum = os.getenv("QUANTUM_ATTN_USE_FAST_ACCUM", "1") == "1"


class dynamo:
    dynamic = os.getenv("QUANTUM_ATTN_DYNAMIC") == "1"

    mode = os.getenv("QUANTUM_ATTN_MODE", "max-autotune-no-cudagraphs")


class triton:
    enable_fast_math = os.getenv("QUANTUM_ATTN_ENABLE_FAST_MATH", "1") == "1"

    allow_reduced_precision_compute = os.getenv("PARA_ATTN_ALLOW_REDUCED_PRECISION_COMPUTE") == "1"


class attention:
    force_eager_fallback = os.getenv("QUANTUM_ATTN_FORCE_EAGER_FALLBACK") == "1"

    enable_tk_tma_kernel = os.getenv("QUANTUM_ATTN_ENABLE_TK_TMA_KERNEL", "1") == "1"
    enable_triton_tma_kernel = os.getenv("QUANTUM_ATTN_ENABLE_TRITON_TMA_KERNEL", "1") == "1"


try:
    from torch.utils._config_module import install_config_module
except ImportError:
    # torch<2.2.0
    from torch._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
