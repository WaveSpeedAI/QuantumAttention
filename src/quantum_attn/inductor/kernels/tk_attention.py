import functools
import os
from torch.utils.cpp_extension import load_inline
from . import tk_utils

TK_ATTENTION_SOURCE = """
#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (3);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (4);
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (2);
};

template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N; 
    const int hr;
};

template<int D, bool is_causal>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;

    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks   = (g.N + K::kv_height - 1) / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>));
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2; }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);

                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else { kv_iters = kv_blocks - 1; }

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {

            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);

            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }

            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                if (kv_idx == kv_iters-1 || kv_idx == kv_iters) {
                    const int q_blk = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) + warpid;
                        int k_blk = (kv_idx * (K::kv_height/kittens::TILE_ROW_DIM<bf16>));

                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }
            else {
                if (kv_idx == kv_iters && g.N % K::kv_height != 0) {
                    const int k_end = (g.N % K::kv_height) / kittens::TILE_ROW_DIM<bf16>;
                    const int k_rem = (g.N % K::kv_height) % kittens::TILE_ROW_DIM<bf16>;

                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  k_end) { neg_infty  (attn_subtile); }
                        else if (k_idx == k_end) { right_fill (attn_subtile, attn_subtile, kittens::TILE_ROW_DIM<bf16>-k_rem, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }

            row_max(max_vec, att_block, max_vec);

            if constexpr (D == 64) {
                mul(att_block, att_block,    1.44269504089f*0.125f);
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   {
                mul(att_block, att_block,    1.44269504089f*0.08838834764f);
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block);
            mul_row(o_reg, o_reg, max_vec_last_scaled);

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
        else                   { mul(norm_vec, norm_vec, -11.313708499f); }

        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        tma::store_async_wait();
    }
}

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

std::vector<torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.device() == k.device(), "Q and K tensors must be on the same device");
    TORCH_CHECK(q.device() == v.device(), "Q and V tensors must be on the same device");

    TORCH_CHECK(q.dim() == 4, "Q tensor must have 4 dimensions");
    TORCH_CHECK(k.dim() == 4, "K tensor must have 4 dimensions");
    TORCH_CHECK(v.dim() == 4, "V tensor must have 4 dimensions");

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto is_causal = causal;
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    torch::DeviceGuard device_guard(q.device());

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);

    // for the returned outputs
    torch::Tensor o     = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(head_dim)}, v.options().memory_format(at::MemoryFormat::Contiguous));

    torch::Tensor l_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(1)},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));


    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<64>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr)};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        constexpr int block_size_m = CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4;
        int num_m_blocks = (seq_len + block_size_m - 1) / block_size_m;
        dim3 grid(num_m_blocks, qo_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, true><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, false><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
    }

    if (head_dim == 128) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr)};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        constexpr int block_size_m = CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4;
        int num_m_blocks = (seq_len + block_size_m - 1) / block_size_m;
        dim3 grid(num_m_blocks, qo_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, true><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, false><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o, l_vec};
}
"""


@functools.cache
def load_tk_attention_module():
    old_torch_cuda_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        module = load_inline(
            name="quantum_attn_tk_attention",
            cpp_sources=[
                "std::vector<torch::Tensor> attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);"
            ],
            cuda_sources=[TK_ATTENTION_SOURCE],
            extra_cflags=["-std=c++20", "-DNDEBUG"],
            extra_cuda_cflags=[
                "-std=c++20",
                # "-U__CUDA_NO_HALF_OPERATORS__",
                # "-U__CUDA_NO_HALF_CONVERSIONS__",
                # "-U__CUDA_NO_HALF2_OPERATORS__",
                # "-U__CUDA_NO_HALF2_CONVERSIONS__",
                # "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                # "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                # "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
                "--use_fast_math",
                # "--ptxas-options=-v",  # printing out number of registers
                # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",  # printing out number of registers
                "-lineinfo",
                "-O3",
                "-DNDEBUG",
                "-DKITTENS_HOPPER",
            ],
            extra_ldflags=["-lcuda", "-lcudart"],
            extra_include_paths=[tk_utils.get_tk_include_dir()],
            functions=["attention_forward"],
            verbose=True,
        )
    finally:
        if old_torch_cuda_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST")
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_torch_cuda_arch_list
    return module
