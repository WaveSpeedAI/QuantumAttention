import pytest
import quantum_attn

import torch
import torch.nn.functional as F
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
def test_dynamic_fp8_attn_func(B, H, S_Q, S_KV, D, dtype, device, is_causal, force_eager_fallback):
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
        qattn_out = quantum_attn_interface.dynamic_fp8_attn_func(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    rmse = torch.sqrt(F.mse_loss(qattn_out, sdpa_out))
    print(f"RMSE: {rmse}")
    print(f"qattn_out: {qattn_out}")
    print(f"sdpa_out: {sdpa_out}")
    assert rmse < 1e-2, f"RMSE: {rmse}"
