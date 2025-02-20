import pytest

import quantum_attn
import torch
import torch.nn.functional as F
from quantum_attn import quantum_attn_interface
from torch.nn.attention import sdpa_kernel, SDPBackend

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


def flash_attention(query, key, value, is_causal=False):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


def cudnn_sdpa(query, key, value, is_causal=False):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


def vanilla_attention(query, key, value, is_causal=False):
    return quantum_attn_interface.attn_func(query, key, value, is_causal=is_causal)


def fp8_attention(query, key, value, is_causal=False):
    return quantum_attn_interface.fp8_attn_func(query, key, value, is_causal=is_causal)


def _test_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback, is_fp8=False):
    if is_causal and S_Q != S_KV:
        pytest.skip("Causal attention is only supported for S_Q == S_KV")

    if is_fp8:
        attn_func = fp8_attention
    else:
        attn_func = vanilla_attention

    torch.manual_seed(0)
    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)

    with quantum_attn.config.patch(
        {
            "attention.force_eager_fallback": force_eager_fallback,
        }
    ):
        try:
            attn_out = attn_func(query, key, value, is_causal=is_causal)
        except ValueError as e:
            if "Unsupported input" in str(e):
                pytest.skip(str(e))
            raise

    fa_out = flash_attention(query, key, value, is_causal=is_causal)

    rmse = torch.sqrt(F.mse_loss(attn_out, fa_out))
    print(f"RMSE: {rmse}")
    assert rmse < 1e-2, f"RMSE: {rmse}"


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize("S_Q", [1024, 999])
@pytest.mark.parametrize("S_KV", [1024, 999])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("force_eager_fallback", [False])
@torch.no_grad()
def test_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback):
    _test_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize("S_Q", [1024, 1000])
@pytest.mark.parametrize("S_KV", [1024, 1000])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("force_eager_fallback", [False])
@torch.no_grad()
def test_fp8_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback):
    _test_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback, is_fp8=True)


def _test_benchmark_attn_func(D, dtype, device, is_causal, is_fp8=False):
    import triton

    torch.manual_seed(0)

    B = 16
    H = 16
    S_Q = 8192
    S_KV = 8192

    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)

    def attention_fn():
        if is_fp8:
            fp8_attention(query, key, value, is_causal)
        else:
            vanilla_attention(query, key, value, is_causal)

    try:
        attention_fn()
    except ValueError as e:
        if "Unsupported input" in str(e):
            pytest.skip(str(e))
        raise

    def fa_fn():
        flash_attention(query, key, value, is_causal)

    def cudnn_sdpa_fn():
        cudnn_sdpa(query, key, value, is_causal)

    flops_per_matmul = 2 * B * H * S_Q * S_KV * D
    total_flops = 2 * flops_per_matmul

    if is_causal:
        total_flops //= 2

    ms_fa = triton.testing.do_bench(fa_fn)
    tflops_fa = total_flops * 1e-12 / (ms_fa * 1e-3)
    print(f"TFLOPS (Flash Attention): {tflops_fa:.2f}")

    if D <= 128:
        ms_cudnn_sdpa = triton.testing.do_bench(cudnn_sdpa_fn)
        tflops_cudnn_sdpa = total_flops * 1e-12 / (ms_cudnn_sdpa * 1e-3)
        print(f"TFLOPS (CUDNN SDPA): {tflops_cudnn_sdpa:.2f}")

    ms_quantum_attention = triton.testing.do_bench(attention_fn)
    tflops_quantum_attention = total_flops * 1e-12 / (ms_quantum_attention * 1e-3)
    print(f"TFLOPS (Quantum Attention): {tflops_quantum_attention:.2f}")


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_causal", [False, True])
@torch.no_grad()
def test_benchmark_attn_func(D, dtype, device, is_causal, is_fp8=False):
    _test_benchmark_attn_func(D, dtype, device, is_causal, is_fp8)


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_causal", [False, True])
@torch.no_grad()
def test_benchmark_fp8_attn_func(D, dtype, device, is_causal):
    _test_benchmark_attn_func(D, dtype, device, is_causal, is_fp8=True)
