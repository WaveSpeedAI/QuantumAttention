import sympy

import torch

if True:
    # Put this first to avoid circular import
    # ImportError: cannot import name 'CppGemmTemplate' from partially initialized module 'torch._inductor.codegen.cpp_gemm_template' (most likely due to a circular import)
    from torch._inductor.lowering import register_lowering

from torch._inductor import config as inductor_config, ir
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.select_algorithm import autotune_select_algorithm, ExternKernelChoice, TritonTemplate
from torch._inductor.utils import ceildiv as cdiv, is_dynamic, use_max_autotune
from torch._inductor.virtualized import V

from quantum_attn import config

from quantum_attn.utils import checks

from .mm_common import acc_type, get_device_shared_memory, mm_options, reduce_block_size_for_cuda, require_dense_memory
from .tk_attention import load_tk_attention_module

quantum_attn_ops = torch.ops.quantum_attn


def tk_attention_forward_kernel(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    assert attn_mask is None
    assert dropout_p == 0.0
    assert scale is None

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    module = load_tk_attention_module(dtype=value.dtype)
    out = module.attention_forward(query, key, value, is_causal)[0]
    return out


tk_attention_forward = ExternKernelChoice(
    tk_attention_forward_kernel,
    name="quantum_attn_tk_attention_forward",
    has_out_variant=False,
)


def persistent_attention_grid(b, h, s, d, meta):
    return (min(meta["NUM_SMS"], cdiv(s, meta["BLOCK_M"]) * b * h), 1, 1)


attention_forward_template = TritonTemplate(
    name="quantum_attn_attention_forward",
    grid=persistent_attention_grid,
    source=rf"""
import triton
import triton.language as tl

@triton.jit
def num_threads():
    return tl.extra.cuda.num_threads()

@triton.jit
def maximum(a, b):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "max.ftz.f16x2 $0, $1, $2;", "=r, r, r", [a, b], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "max.ftz.f16 $0, $1, $2;", "=h, h, h", [a, b], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "max.ftz.f32 $0, $1, $2;", "=f, f, f", [a, b], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = a if a > b else b
{{% endif %}}
    return x

@triton.jit
def maximum_(a, b):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "max.ftz.f16x2 $0, $1, $2;", "=r, r, r", [a, b], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "max.ftz.f16 $0, $1, $2;", "=h, h, h", [a, b], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "max.ftz.f32 $0, $1, $2;", "=f, f, f", [a, b], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = tl.maximum(a, b)
{{% endif %}}
    return x

@triton.jit
def add(a, b):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "add.ftz.f16x2 $0, $1, $2;", "=r, r, r", [a, b], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "add.ftz.f16 $0, $1, $2;", "=h, h, h", [a, b], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "add.ftz.f32 $0, $1, $2;", "=f, f, f", [a, b], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = a + b
{{% endif %}}
    return x

@triton.jit
def sub(a, b):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "sub.ftz.f16x2 $0, $1, $2;", "=r, r, r", [a, b], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "sub.ftz.f16 $0, $1, $2;", "=h, h, h", [a, b], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "sub.ftz.f32 $0, $1, $2;", "=f, f, f", [a, b], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = a - b
{{% endif %}}
    return x

@triton.jit
def mul(a, b):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "mul.ftz.f16x2 $0, $1, $2;", "=r, r, r", [a, b], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "mul.ftz.f16 $0, $1, $2;", "=h, h, h", [a, b], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "mul.ftz.f32 $0, $1, $2;", "=f, f, f", [a, b], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = a * b
{{% endif %}}
    return x

@triton.jit
def div(a, b):
    a_fp32 = a.to(tl.float32)
    b_fp32 = b.to(tl.float32)
{{% if ENABLE_FAST_MATH %}}
    x = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;", "=f, f, f", [a_fp32, b_fp32], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% else %}}
    x = a_fp32 / b_fp32
{{% endif %}}
    x = x.to(a.dtype)
    return x

@triton.jit
def fma(a, b, c):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if a.numel % (num_threads() * 2) == 0:
        x = tl.inline_asm_elementwise(
            "fma.rn.ftz.f16x2 $0, $1, $2, $3;", "=r, r, r, r", [a, b, c], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        x = tl.inline_asm_elementwise(
            "fma.rn.ftz.f16 $0, $1, $2, $3;", "=h, h, h, h", [a, b, c], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    x = tl.inline_asm_elementwise(
        "fma.rn.ftz.f32 $0, $1, $2, $3;", "=f, f, f, f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    x = a * b + c
{{% endif %}}
    return x

@triton.jit
def ex2(x):
{{% if ENABLE_FAST_MATH %}}
{{% if USE_FP16_COMPUTE %}}
    if x.numel % (num_threads() * 2) == 0:
        y = tl.inline_asm_elementwise(
            "ex2.approx.f16x2 $0, $1;", "=r, r", [x], dtype=tl.float16, is_pure=True, pack=2,
        )
    else:
        y = tl.inline_asm_elementwise(
            "ex2.approx.f16 $0, $1;", "=h, h", [x], dtype=tl.float16, is_pure=True, pack=1,
        )
{{% else %}}
    y = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;", "=f, f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )
{{% endif %}}
{{% else %}}
    y = {TritonOverrides.exp2("x.to(tl.float32)")}.to(x.dtype)
{{% endif %}}
    return y

@triton.jit
def dot(a, b, acc):
{{% if USE_FAST_ACCUM %}}
    acc = tl.dot(a, b, acc, out_dtype=acc.dtype)
{{% else %}}
    acc += tl.dot(a, b, out_dtype=acc.dtype)
{{% endif %}}
    return acc

@triton.jit
def _attn_fwd_inner(
{{% for i in range(TILES) %}}
                    acc_{{{{i}}}},
                    q_{{{{i}}}},
{{% endfor %}}
                    q_scale,
                    l_i, m_i,
                    K_desc_ptr, V_desc_ptr,
                    K_scale_block_ptr,
                    start_m, #
                    v_dtype,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    STAGE: tl.constexpr,
                    N_CTX_Q, N_CTX_K,
                    TILES: tl.constexpr,
                    EVEN_N: tl.constexpr,
                    QK_ACC_TYPE,
):
    # range of values handled by this stage
    if STAGE == 1:
        if BLOCK_N <= BLOCK_M:
            lo, hi = 0, start_m * BLOCK_M
        else:
            lo, hi = 0, start_m // (BLOCK_N // BLOCK_M) * BLOCK_N
    elif STAGE == 2:
        if BLOCK_N <= BLOCK_M:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
            lo = tl.multiple_of(lo, BLOCK_M)
        else:
            lo, hi = start_m // (BLOCK_N // BLOCK_M) * BLOCK_N, (start_m + 1) * BLOCK_M
            lo = tl.multiple_of(lo, BLOCK_N)
    # causal = False
    else:
        lo, hi = 0, N_CTX_K

{{% if IS_QUANTIZED %}}
    K_scale_block_ptr = tl.advance(K_scale_block_ptr, (lo,))
{{% endif %}}

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=QK_ACC_TYPE)
{{% for i in range(TILES) %}}
        k = tl._experimental_descriptor_load(
            K_desc_ptr, (start_n, BLOCK_K * {{{{i}}}}), (BLOCK_N, BLOCK_K), q_0.dtype
        )
        v_{{{{i}}}} = tl._experimental_descriptor_load(
            V_desc_ptr, (start_n, BLOCK_K * {{{{i}}}}), (BLOCK_N, BLOCK_K), v_dtype
        )
        qk = dot(q_{{{{i}}}}, k.T, qk)
{{% endfor %}}

{{% if IS_QUANTIZED %}}
        k_scale = tl.load(K_scale_block_ptr, boundary_check=(0,)).to(tl.float32)
        K_scale_block_ptr = tl.advance(K_scale_block_ptr, (BLOCK_N,))

        qk = qk * q_scale[:, None] * k_scale[None, :]
{{% if USE_FP16_COMPUTE %}}
        qk = qk.to(tl.float16)
{{% endif %}}
{{% else %}}
        qk = mul(qk, tl.full([1], {{{{SM_SCALE}}}} * 1.44269504, dtype=qk.dtype))
{{% endif %}}

        if EVEN_N:
            if STAGE == 2:
                offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tl.arange(0, BLOCK_N)
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(mask, qk, tl.full([1], -float("inf"), dtype=qk.dtype))
        else:
            offs_n = tl.arange(0, BLOCK_N)
            mask = (start_n + offs_n[None, :]) < N_CTX_K
            if STAGE == 2:
                offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                mask = mask & (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = tl.where(mask, qk, tl.full([1], -float("inf"), dtype=qk.dtype))
        m_ij = maximum_(m_i, tl.reduce(qk, 1, maximum))
        qk = sub(qk, m_ij[:, None])

{{% if FAST_SOFTMAX %}}

        numerator = ex2(qk)

        denominator = tl.reduce(numerator, 1, add)
        p = div(numerator, denominator[:, None])

{{% else %}}

        m_i_m_ij = sub(m_i, m_ij)
        alpha = ex2(m_i_m_ij)

{{% for i in range(TILES) %}}
        acc_{{{{i}}}} = mul(acc_{{{{i}}}}, alpha[:, None])
{{% endfor %}}

        p = ex2(qk)

        l_ij = tl.reduce(p, 1, add)

        # -- update m_i and l_i
        l_i  = fma(l_i, alpha, l_ij)
        m_i = m_ij

{{% endif %}}

        p = p.to(v_0.dtype)

        # -- update output accumulator --
{{% for i in range(TILES) %}}
        acc_{{{{i}}}} = dot(p, v_{{{{i}}}}, acc_{{{{i}}}})
{{% endfor %}}
    return (
{{% for i in range(TILES) %}}
        acc_{{{{i}}}},
{{% endfor %}}
        l_i, m_i,
    )

{{% if IS_QUANTIZED %}}
{{{{def_kernel("Q", "K", "V", "Q_scale", "K_scale")}}}}
{{% else %}}
{{{{def_kernel("Q", "K", "V")}}}}
{{% endif %}}
    Z = {{{{size("Q", 0)}}}}
    H = {{{{size("Q", 1)}}}}
    N_CTX_Q = {{{{size("Q", 2)}}}}
    N_CTX_K = {{{{size("K", 2)}}}}
    D = {{{{size("Q", 3)}}}}

    stride_qz = {{{{stride("Q", 0)}}}}
    stride_qh = {{{{stride("Q", 1)}}}}
    stride_qm = {{{{stride("Q", 2)}}}}
    stride_qk = {{{{stride("Q", 3)}}}}

    stride_kz = {{{{stride("K", 0)}}}}
    stride_kh = {{{{stride("K", 1)}}}}
    stride_kn = {{{{stride("K", 2)}}}}
    stride_kk = {{{{stride("K", 3)}}}}

    stride_vz = {{{{stride("V", 0)}}}}
    stride_vh = {{{{stride("V", 1)}}}}
    stride_vk = {{{{stride("V", 2)}}}}
    stride_vn = {{{{stride("V", 3)}}}}

{{% if IS_QUANTIZED %}}
    stride_q_scale_z = {{{{stride("Q_scale", 0)}}}}
    stride_q_scale_h = {{{{stride("Q_scale", 1)}}}}
    stride_q_scale_m = {{{{stride("Q_scale", 2)}}}}

    stride_k_scale_z = {{{{stride("K_scale", 0)}}}}
    stride_k_scale_h = {{{{stride("K_scale", 1)}}}}
    stride_k_scale_m = {{{{stride("K_scale", 2)}}}}
{{% endif %}}

    start_pid = tl.program_id(0)

    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    K_desc_ptr = workspace_base
    V_desc_ptr = workspace_base + TMA_SIZE

    num_programs_m = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    for pid in range(start_pid, num_programs_m * Z * H, NUM_SMS):
        start_m = pid % num_programs_m
        if STAGE & 2:
            if start_m // num_programs_m % 2 != 0:
                start_m = start_m * (start_m // num_programs_m + 1) - start_m % num_programs_m
        off_hz = pid // num_programs_m
        off_z = off_hz // H
        off_h = off_hz % H
        # off_z = off_z.to(tl.int64)
        # off_h = off_h.to(tl.int64)

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        # o_offset = off_z * stride_qz + off_h * stride_qh

        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX_Q, D),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

{{% if IS_QUANTIZED %}}
        q_scale_offset = off_z * stride_q_scale_z + off_h * stride_q_scale_h
        k_scale_offset = off_z * stride_k_scale_z + off_h * stride_k_scale_h

        Q_scale_block_ptr = tl.make_block_ptr(
            base=Q_scale + q_scale_offset,
            shape=(N_CTX_Q,),
            strides=(stride_q_scale_m,),
            offsets=(start_m * BLOCK_M,),
            block_shape=(BLOCK_M,),
            order=(0,),
        )
        K_scale_block_ptr = tl.make_block_ptr(
            base=K_scale + k_scale_offset,
            shape=(N_CTX_K,),
            strides=(stride_k_scale_m,),
            offsets=(0,),
            block_shape=(BLOCK_N,),
            order=(0,),
        )
{{% else %}}
        K_scale_block_ptr = None
{{% endif %}}

        if start_m < NUM_SMS:
            triton.language.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=K_desc_ptr,
                global_address=K + k_offset,
                load_size=[BLOCK_N, BLOCK_K],
                global_size=[N_CTX_K, D],
                element_ty=K.dtype.element_ty,
            )
            triton.language.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=V_desc_ptr,
                global_address=V + v_offset,
                load_size=[BLOCK_N, BLOCK_K],
                global_size=[N_CTX_K, D],
                element_ty=V.dtype.element_ty,
            )

            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(K_desc_ptr)
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(V_desc_ptr)

{{% if IS_QUANTIZED %}}
        q_scale = tl.load(Q_scale_block_ptr, boundary_check=(0,)).to(tl.float32)
        q_scale = q_scale * ({{{{SM_SCALE * 1.44269504}}}})
{{% else %}}
        q_scale = None
{{% endif %}}

{{% for i in range(TILES) %}}
        q_{{{{i}}}} = tl.load(Q_block_ptr, boundary_check=(0,))
{{% if i + 1 < TILES %}}
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_K))
{{% endif %}}
{{% endfor %}}

        # initialize pointer to m and l
{{% for i in range(TILES) %}}
        acc_{{{{i}}}} = tl.zeros([BLOCK_M, BLOCK_K], dtype=ACC_TYPE)
{{% endfor %}}
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=acc_0.dtype)
        l_i = tl.full([BLOCK_M], 1.0, dtype=acc_0.dtype)
        # load scales
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            (
{{% for i in range(TILES) %}}
                acc_{{{{i}}}},
{{% endfor %}}
                l_i, m_i,
            ) = _attn_fwd_inner(
{{% for i in range(TILES) %}}
                                acc_{{{{i}}}},
                                q_{{{{i}}}},
{{% endfor %}}
                                q_scale,
                                l_i, m_i, K_desc_ptr, V_desc_ptr,
                                K_scale_block_ptr,
                                start_m,
                                V.dtype.element_ty,
                                BLOCK_M, BLOCK_N, BLOCK_K,
                                4 - STAGE, N_CTX_Q, N_CTX_K,
                                TILES,
                                EVEN_N,
                                QK_ACC_TYPE,
                                )
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            tl.debug_barrier()
            (
{{% for i in range(TILES) %}}
                acc_{{{{i}}}},
{{% endfor %}}
                l_i, m_i,
            ) = _attn_fwd_inner(
{{% for i in range(TILES) %}}
                                acc_{{{{i}}}},
                                q_{{{{i}}}},
{{% endfor %}}
                                q_scale,
                                l_i, m_i, K_desc_ptr, V_desc_ptr,
                                K_scale_block_ptr,
                                start_m,
                                V.dtype.element_ty,
                                BLOCK_M, BLOCK_N, BLOCK_K,
                                2, N_CTX_Q, N_CTX_K,
                                TILES,
                                EVEN_N,
                                QK_ACC_TYPE,
                                )

        # epilogue
        start_m = pid % num_programs_m
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_hz = pid // num_programs_m
        off_z = off_hz // H
        off_h = off_hz % H
        # offs_m = offs_m.to(tl.int64)
        # off_z = off_z.to(tl.int64)
        # off_h = off_h.to(tl.int64)

        idx_m = offs_m[None, None, :, None]
        idx_z = tl.full([1, 1, 1, 1], off_z, dtype=idx_m.dtype)
        idx_h = tl.full([1, 1, 1, 1], off_h, dtype=idx_m.dtype)

{{% for i in range(TILES) %}}
        acc_{{{{i}}}} = div(acc_{{{{i}}}}, l_i[:, None])
        acc_{{{{i}}}} = acc_{{{{i}}}}[None, None, :, :]
        idx_d = tl.arange({{{{i}}}} * BLOCK_K, {{{{i + 1}}}} * BLOCK_K)[None, None, None, :]
        mask = (idx_z < Z) & (idx_h < H) & (idx_m < N_CTX_Q) & (idx_d < D)
        acc = acc_{{{{i}}}}
{{% if i == 0 %}}
        {{{{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc", "mask", indent_width=8)}}}}
{{% else %}}
        <STORE_OUTPUT>
{{% endif %}}
{{% endfor %}}
""",
)


def attention_heuristic_configs(
    head_dim,
    B,
    H,
    N_CTX_Q,
    N_CTX_K,
    is_causal=False,
    layout=None,
    optimize_block_size=True,
):
    import triton

    # https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
    # is_sm8x = torch.cuda.get_device_capability()[0] == 8

    if not checks.cuda_capability_compare("ge", 9, 0, device=layout.device):
        confs = ""
    else:
        if head_dim == 64:
            confs = "128.128.64.8.4 128.128.64.8.3 128.128.64.8.2 256.128.64.8.4 256.128.64.8.3 256.128.64.8.2"
        elif head_dim == 128:
            confs = "128.128.128.8.3 128.128.128.8.2 128.128.64.8.3 128.128.64.8.2"
        elif head_dim == 256:
            confs = "128.64.128.8.3 128.64.64.8.3 128.64.128.8.2 128.64.64.8.2"

    confs = [[int(x) for x in c.split(".")] for c in confs.split() if c]

    BLOCK_DMODEL = max(next_power_of_2(head_dim), 16)

    is_ge_sm90 = layout.device.type == "cuda" and checks.cuda_capability_compare("ge", 9, 0, device=layout.device)

    configs = []
    picked_confs = set()
    for c in confs:
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = c

        if optimize_block_size and checks.torch_version_compare("ge", "2.2.0"):
            n_ctx_q_hint = V.graph.sizevars.size_hint(N_CTX_Q, fallback=inductor_config.unbacked_symint_fallback)
            if n_ctx_q_hint <= 32:
                BLOCK_M = min(BLOCK_M, 32)
            elif n_ctx_q_hint <= 64:
                BLOCK_M = min(BLOCK_M, 64)
            # elif n_ctx_q_hint <= 96:
            #     BLOCK_M = min(BLOCK_M, 32)

            n_ctx_k_hint = V.graph.sizevars.size_hint(N_CTX_K, fallback=inductor_config.unbacked_symint_fallback)
            if n_ctx_k_hint <= 32:
                BLOCK_N = min(BLOCK_N, 32)
            elif n_ctx_k_hint <= 64:
                BLOCK_N = min(BLOCK_N, 64)
            # elif n_ctx_k_hint <= 96:
            #     BLOCK_N = min(BLOCK_N, 32)

            if layout.device.type == "cuda":
                b_hint = V.graph.sizevars.size_hint(B, fallback=inductor_config.unbacked_symint_fallback)
                h_hint = V.graph.sizevars.size_hint(H, fallback=inductor_config.unbacked_symint_fallback)
                BLOCK_M, _ = reduce_block_size_for_cuda(
                    BLOCK_M, 1, n_ctx_q_hint, 1, device=layout.device, b=b_hint * h_hint
                )

        while BLOCK_DMODEL % BLOCK_K != 0:
            BLOCK_K //= 2

        if BLOCK_M < 128 and is_ge_sm90 and not checks.triton_version_compare("ge", "3.1.0"):
            # https://github.com/triton-lang/triton/pull/4492
            # Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) && "mma -> mma layout conversion is only supported on Ampere"' failed.
            num_warps = min(num_warps, 4)

        key = (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        if key in picked_confs:
            continue
        picked_confs.add(key)

        configs.append(
            triton.Config(
                {
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_N": BLOCK_N,
                    "BLOCK_K": BLOCK_K,
                    "BLOCK_DMODEL": BLOCK_DMODEL,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )

    if not use_max_autotune():
        configs = configs[:1]

    return configs


def early_attention_config_prune(configs, query, key, value):
    query_dtype = query.get_dtype()
    key_dtype = key.get_dtype()
    value_dtype = value.get_dtype()
    device = query.get_device()

    assert device.type == "cuda"

    max_shared_memory = get_device_shared_memory(device.index)

    filtered_configs = []
    for c in configs:
        kw = c.kwargs
        BLOCK_M, BLOCK_N, BLOCK_DMODEL = kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_DMODEL"]
        num_stages = c.num_stages
        required_shared_memory = (
            BLOCK_N * num_stages * (key_dtype.itemsize + value_dtype.itemsize) + BLOCK_M * query_dtype.itemsize
        ) * BLOCK_DMODEL
        if required_shared_memory <= max_shared_memory:
            filtered_configs.append(c)
    return filtered_configs


def get_attention_layout(
    query,
    dtype,
):
    return ir.FixedLayout(
        query.get_device(),
        dtype,
        ir.convert_shape_to_inductor(query.get_size()),
    )


def generate_attention_template_choices(
    choices,
    query,
    key,
    value,
    scale_q=None,
    scale_k=None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    layout2=None,
    enable_max_autotune=False,
):
    from torch._inductor.utils import get_num_sms, get_tma_workspace_arg, TMA_DESCRIPTOR_SIZE

    query_size, key_size = (x.get_size() for x in (query, key))
    Lq = query_size[-1]
    Lq = V.graph.sizevars.evaluate_static_shape(Lq)
    N_CTX_Q, N_CTX_K = query_size[2], key_size[2]
    B, H = query_size[0], query_size[1]

    if scale is None:
        scale = float(1.0 / (Lq**0.5))

    key_t = ir.PermuteView.create(key, [0, 1, 3, 2])
    m1, n1, k1, layout1, mat1, mat2 = mm_args(query, key_t)
    stage = 3 if is_causal else 1

    args = [query, key, value]
    if scale_q is not None:
        args += [scale_q, scale_k]
    if attn_mask is not None:
        args.append(attn_mask)

    dynamic = is_dynamic(*args)

    triton_configs = []
    heuristic_configs = attention_heuristic_configs(
        Lq,
        B,
        H,
        N_CTX_Q,
        N_CTX_K,
        is_causal=is_causal,
        layout=layout2,
        optimize_block_size=not dynamic or enable_max_autotune,
    )
    triton_configs.extend(heuristic_configs)

    triton_configs = early_attention_config_prune(triton_configs, query, key, value)

    for fa_config in triton_configs:
        mm_options_ = mm_options(fa_config, m1, n1, k1, layout1)
        mm_options_["ACC_TYPE"] = acc_type(value.get_dtype())
        if scale_q is None:
            mm_options_["QK_ACC_TYPE"] = acc_type(value.get_dtype())
        else:
            mm_options_["QK_ACC_TYPE"] = "tl.float32"
        fast_softmax = not dynamic and mm_options_["BLOCK_N"] >= N_CTX_K
        even_n_symbolic = (
            # it isn't worth guarding on this
            sympy.gcd(N_CTX_K, mm_options_["BLOCK_N"])
            == mm_options_["BLOCK_N"]
        )

        attention_forward_template.maybe_append_choice(
            choices,
            input_nodes=args,
            layout=layout2,
            workspace_arg=get_tma_workspace_arg(
                num_tma_descriptors=2,
                device=query.get_device(),
            ),
            SM_SCALE=float(1.0 / (Lq**0.5)) if scale is None else scale,
            STAGE=stage,
            TILES=cdiv(Lq, mm_options_["BLOCK_K"]),
            EVEN_N=even_n_symbolic,
            NUM_STAGES=fa_config.num_stages,
            FAST_SOFTMAX=fast_softmax,
            USE_FP16_COMPUTE=mm_options_["ACC_TYPE"] == "tl.float16",
            TMA_SIZE=TMA_DESCRIPTOR_SIZE,
            NUM_SMS=get_num_sms(),
            IS_QUANTIZED=scale_q is not None,
            **mm_options_,
        )


@register_lowering(quantum_attn_ops.attention_forward.default, type_promotion_kind=None)
def tuned_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    layout=None,
    scale_q=None,
    scale_k=None,
):
    assert (scale_q is None) == (scale_k is None)

    k1 = V.graph.sizevars.evaluate_static_shape(query.get_size()[-1])
    n2 = V.graph.sizevars.evaluate_static_shape(value.get_size()[-1])

    use_tk_tma_kernel = (
        config.attention.enable_tk_tma_kernel
        and query.get_dtype() in (torch.float16, torch.bfloat16)
        and attn_mask is None
        and dropout_p == 0.0
        and scale is None
        and query.get_size()[-2] == key.get_size()[-2]
        and k1 == n2
        and k1 in (64, 128)
    )

    use_triton_tma_kernel = (
        config.attention.enable_triton_tma_kernel
        and query.get_dtype() in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        and checks.torch_version_compare("ge", "2.7.0")
        and checks.has_triton_tma_support()
        and attn_mask is None
        and dropout_p == 0.0
        and k1 == n2
        and k1 in (64, 128, 256)
    )

    query, key, value = (ir.ExternKernel.realize_input(x) for x in (query, key, value))
    if use_tk_tma_kernel:
        query, key, value = (require_dense_memory(x) for x in (query, key, value))
    elif use_triton_tma_kernel:
        query = require_dense_memory(query, num_dims=1)
        key, value = (require_dense_memory(x, num_dims=2) for x in (key, value))

    if scale_q is not None:
        scale_q, scale_k = (ir.ExternKernel.realize_input(x) for x in (scale_q, scale_k))
        scale_q, scale_k = (require_dense_memory(x, num_dims=1) for x in (scale_q, scale_k))
    for x in (query, key, value, scale_q, scale_k, attn_mask):
        if x is not None:
            x.freeze_layout()
    key_t = ir.PermuteView.create(key, [0, 1, 3, 2])
    m1, n1, k1, layout1, mat1, mat2 = mm_args(query, key_t)

    # if scale is None or math.isnan(
    #         scale):  # og_scale.as_float_unchecked() could be nan
    #     scale = float(1.0 / (k1**0.5))

    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
    }
    ordered_kwargs_for_cpp_kernel = [
        "dropout_p",
        "is_causal",
        "scale",
    ]

    if attn_mask is None:
        args = [query, key, value]
        if scale_q is not None:
            args += [scale_q, scale_k]
        kwargs["attn_mask"] = None
        ordered_kwargs_for_cpp_kernel.insert(0, "attn_mask")
    else:
        args = [query, key, value]
        if scale_q is not None:
            args += [scale_q, scale_k]
        args.append(attn_mask)

    if layout is None:
        layout2 = get_attention_layout(query, value.get_dtype())
    else:
        layout2 = layout

    choices = []
    if use_tk_tma_kernel:
        choices.append(
            tk_attention_forward.bind(
                args,
                layout=layout2,
                **kwargs,
            )
        )
    if use_triton_tma_kernel:
        generate_attention_template_choices(
            choices, *args, layout2=layout2, enable_max_autotune=use_max_autotune(), **kwargs
        )
    if not use_max_autotune():
        choices = choices[:1]
    return autotune_select_algorithm("quantum_attn_attention_forward", choices, args, layout2)


@register_lowering(quantum_attn_ops.fp8_attention_forward.default, type_promotion_kind=None)
def fp8_attention_forward(
    query,
    key,
    value,
    scale_q,
    scale_k,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    layout=None,
):
    return tuned_attention_forward(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        layout=layout,
        scale_q=scale_q,
        scale_k=scale_k,
    )
