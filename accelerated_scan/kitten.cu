//#include "../../../src/kittens.cuh"
#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

#define NUM_WORKERS 8 // sequence parallelism
#define DIMENSION 64 // dimension of keys and values (must be a multiple of 16)

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.


template <typename H = bf16>
__global__ void causal_attend_kernel(
    int n,
    const H* __restrict__ __q__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const float* __restrict__ __f__,
    H* __o__
) {
    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*DIMENSION);
    const bf16 *_q = reinterpret_cast<const bf16 *>(__q__) + block_start,
               *_k = reinterpret_cast<const bf16 *>(__k__) + block_start,
               *_v = reinterpret_cast<const bf16 *>(__v__) + block_start;
          bf16 *_o = reinterpret_cast<bf16 *>(__o__) + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x4<ducks::st_layout::swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x4<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_reg;

    int qo_blocks = n / (q_reg.rows*NUM_WORKERS);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {
        // each warp loads its own Q tile of 16x16
        auto q_index = q_blk*NUM_WORKERS + warpid;
        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);

        // zero flash attention O register.
        zero(o_reg);

        // iterate over k, v for these q's that have been loaded
        for(auto kv_blk = q_blk; kv_blk >= 0; kv_blk--) {
            int kv_warp_index = kv_blk*NUM_WORKERS + warpid;
            if (kv_warp_index <= q_index) { // ensure causality
                // each warp loads its own chunk of k, v into shared memory
                load(v_smem[warpid], _v + kv_warp_index*q_reg.num_elements, q_reg.cols);
                load(k_smem[warpid], _k + kv_warp_index*q_reg.num_elements, q_reg.cols);
            }
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = NUM_WORKERS-1; subtile >= 0; subtile--) {
                int kv_subtile_index = kv_blk*NUM_WORKERS + subtile;
                if (!(kv_subtile_index <= q_index)) { // ensure causality
                    continue;
                }
                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile
                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                if (kv_subtile_index == q_index) {
                    make_causal(att_block_mma, att_block_mma, kittens::base_types::constants<bf16>::zero());
                }

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

void
attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f, torch::Tensor o_small) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(f);
    CHECK_INPUT(o_small);
    
    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = NUM_WORKERS;

    unsigned long mem_size = 2*workers*sizeof(st_bf_1x4<ducks::st_layout::swizzle>);

    TORCH_CHECK(n % (workers*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");

    auto threads = workers * kittens::WARP_THREADS;
    //printf("[causal_attend] Requesting %lu bytes of memory for %d workers (%d threads)\n", mem_size, workers, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             causal_attend_kernel<T>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    causal_attend_kernel<T><<<batch*head,threads,mem_size>>>((int)n, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), f.data_ptr<float>(), o_small.data_ptr<T>());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

//#include "harness.impl"
    