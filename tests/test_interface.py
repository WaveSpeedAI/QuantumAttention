from contextlib import contextmanager

import pytest
import quantum_attn
import torch
import torch.nn.functional as F
import triton.profiler as proton
from quantum_attn import quantum_attn_interface

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


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
    torch.manual_seed(0)
    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)

    sdpa_out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
    )

    with quantum_attn.config.patch(
        {
            "attention.force_eager_fallback": force_eager_fallback,
        }
    ):
        qattn_out = quantum_attn_interface.fp8_attn_func(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    rmse = torch.sqrt(F.mse_loss(qattn_out, sdpa_out))
    print(f"RMSE: {rmse}")
    assert rmse < 1e-2, f"RMSE: {rmse}"


@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_causal", [False])
@torch.no_grad()
def test_benchmark_fp8_attn_func(D, dtype, device, is_causal):
    torch.manual_seed(0)

    @contextmanager
    def proton_context():
        proton.activate(0)
        try:
            yield
        finally:
            proton.deactivate(0)

    def bench_fn(reps, warmup_reps, fn, *args):
        for _ in range(warmup_reps):
            fn(*args)
        with proton_context():
            for _ in range(reps):
                fn(*args)

    def torch_sdpa(query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

    def fp8_attention(query, key, value):
        return quantum_attn_interface.fp8_attn_func(query, key, value)

    def bench(reps=1000, warmup_reps=10000):
        B = 2
        H = 8
        S_Q = 4096
        S_KV = 4096

        query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
        key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
        value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)

        bench_fn(reps, warmup_reps, torch_sdpa, query, key, value)
        bench_fn(reps, warmup_reps, fp8_attention, query, key, value)

    def show_profile(profile_name):
        import triton.profiler.viewer as proton_viewer

        metrics = ["time/ms"]
        file_name = f"{profile_name}.hatchet"
        proton_viewer.parse(metrics, file_name, depth=100)

    proton.start("attention", hook="triton")
    bench()
    proton.finalize()
    show_profile("attention")
