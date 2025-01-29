import functools
import unittest

import torch
from torch._inductor import ir
from torch._inductor.kernel import mm_common
from torch._inductor.utils import ceildiv as cdiv
from torch._inductor.virtualized import V

from quantum_attn import config
from quantum_attn.utils import checks


def require_dense_memory(x, num_dims=None):
    strides = x.get_stride()
    try:
        strides_hint = [V.graph.sizevars.size_hint(s) for s in strides]
        stride_order = ir.get_stride_order(strides_hint)
    except Exception:
        # logger.warning(f"Failed to get size hint for strides: {strides}", exc_info=True)
        stride_order = list(reversed(range(len(x.get_size()))))
    required_strides = ir.FlexibleLayout.stride_ordered(x.get_size(), stride_order)
    if isinstance(x.layout, ir.FlexibleLayout):
        x.freeze_layout_with_same_order(required_strides)
    else:
        for i, (size, stride, required_stride) in enumerate(zip(x.layout.size, x.layout.stride, required_strides)):
            if num_dims is not None and i + num_dims < len(x.layout.size):
                continue
            if stride != required_stride and size != 1:
                x = ir.ExternKernel.copy_input(x)
                x.freeze_layout_with_same_order(required_strides)
                break
    return x


def acc_type(dtype):
    if dtype == torch.float16:
        if config.triton.allow_reduced_precision_compute and torch.version.hip is None:
            return "tl.float16"
        else:
            return "tl.float32"
    elif dtype == torch.bfloat16:
        return "tl.float32"
    else:
        return f"tl.{dtype}".replace("torch.", "")


@functools.cache
def get_device_shared_memory(device=0):
    try:
        from triton.runtime import driver

        if hasattr(driver, "active"):
            return driver.active.utils.get_device_properties(device)["max_shared_mem"]
        return driver.utils.get_device_properties(device)["max_shared_mem"]
    except Exception:
        return 1024**3


def mm_options(c, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None, optimize_block_size=True):
    with unittest.mock.patch.object(mm_common, "acc_type", acc_type):
        if checks.torch_version_compare("ge", "2.3.0"):
            options = mm_common.mm_options(c, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=b_prologue_cast_type)
        else:
            options = mm_common.mm_options(c, sym_k, layout, b_prologue_cast_type=b_prologue_cast_type)
    options["ENABLE_FAST_MATH"] = (
        config.triton.enable_fast_math
        and checks.has_triton_language("inline_asm_elementwise")
        and checks.is_nvidia_cuda()
    )
    device_capability = (
        torch.cuda.get_device_capability(layout.device.index or 0) if checks.is_nvidia_cuda() else (0, 0)
    )
    cuda_arch = device_capability[0] * 100 + device_capability[1] * 10
    options["CUDA_ARCH"] = cuda_arch
    cuda_version = checks.torch_cuda_version()
    cuda_version = cuda_version[0] * 1000 + cuda_version[1] * 10
    options["CUDA_VERSION"] = cuda_version
    options["USE_FAST_ACCUM"] = config.use_fast_accum

    return options


def reduce_block_size_for_cuda(BLOCK_M, BLOCK_N, m, n, device=None, b=1):
    if device is None:
        device_index = 0
    else:
        device_index = device.index or 0
    avail_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    if b * cdiv(m, BLOCK_M) * cdiv(n, BLOCK_N) < avail_sms:
        # if checks.cuda_capability_compare("ge", 9, 0, device=device):
        #     # Keep using WGMMA
        #     min_m = 64
        # else:
        #     min_m = 16
        min_m = 16
        while True:
            if BLOCK_M >= BLOCK_N:
                if BLOCK_M > min_m and b * cdiv(m, BLOCK_M // 2) * cdiv(n, BLOCK_N) <= avail_sms:
                    BLOCK_M //= 2
                    continue
                if BLOCK_N > 16 and b * cdiv(m, BLOCK_M) * cdiv(n, BLOCK_N // 2) <= avail_sms:
                    BLOCK_N //= 2
                    continue
            else:
                if BLOCK_N > 16 and b * cdiv(m, BLOCK_M) * cdiv(n, BLOCK_N // 2) <= avail_sms:
                    BLOCK_N //= 2
                    continue
                if BLOCK_M > min_m and b * cdiv(m, BLOCK_M // 2) * cdiv(n, BLOCK_N) <= avail_sms:
                    BLOCK_M //= 2
                    continue
            break
    return BLOCK_M, BLOCK_N
