//#include "../../../src/kittens.cuh"
#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.


template <typename H = bf16, int _height, int _width, int kNumWorkers = 8>
__global__ void causal_attend_kernel(
    int seqlen,
    const H* __restrict__ __q__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const float* __restrict__ __f__,
    H* __o__
) {
    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(seqlen*_width*TILE_DIM);
    const bf16 *_q = reinterpret_cast<const bf16 *>(__q__) + block_start,
               *_k = reinterpret_cast<const bf16 *>(__k__) + block_start,
               *_v = reinterpret_cast<const bf16 *>(__v__) + block_start;
          bf16 *_o = reinterpret_cast<bf16 *>(__o__) + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf<_height, _width, ducks::st_layout::swizzle> (&k_smem)[kNumWorkers] = al.allocate<st_bf<_height, _width, ducks::st_layout::swizzle>, kNumWorkers>();
    st_bf<_height, _width, ducks::st_layout::swizzle> (&v_smem)[kNumWorkers] = al.allocate<st_bf<_height, _width, ducks::st_layout::swizzle>, kNumWorkers>();

    // Initialize all of the register tiles.
    rt_bf<_height, _width> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl<_height, _height> att_block;
    rt_bf<_height, _height> att_block_mma;
    rt_fl<_height, _width> o_reg;

    const int qo_blocks = seqlen / (q_reg.rows*kNumWorkers);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {
        // each warp loads its own Q tile of 16x16
        auto q_index = q_blk*kNumWorkers + warpid;
        load(q_reg, _q + q_index*q_reg.num_elements, q_reg.cols);

        zero(o_reg); // zero flash attention O register.

        // iterate over k, v for these q's that have been loaded
        for(auto kv_blk = q_blk; kv_blk >= 0; kv_blk--) {
            int kv_warp_index = kv_blk*kNumWorkers + warpid;
            if (kv_warp_index <= q_index) { // ensure causality
                // each warp loads its own chunk of k, v into shared memory
                load(v_smem[warpid], _v + kv_warp_index*q_reg.num_elements, q_reg.cols);
                load(k_smem[warpid], _k + kv_warp_index*q_reg.num_elements, q_reg.cols);
            }
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = kNumWorkers-1; subtile >= 0; subtile--) {
                int kv_subtile_index = kv_blk*kNumWorkers + subtile;
                if (!(kv_subtile_index <= q_index)) { // ensure causality
                    continue;
                }
                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile
                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                if (kv_subtile_index == q_index) {
                    make_causal(att_block_mma, att_block_mma, 0);
                }

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf<_height, _width, ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*kNumWorkers + warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

void
attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f, torch::Tensor o_small) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(f);
    CHECK_INPUT(o_small);
    
    auto batch  = q.size(0);
    auto head   = q.size(1);
    auto seqlen = q.size(2);
    auto d      = q.size(3);
    auto dv     = v.size(3);
    bool k_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");

    using H = c10::BFloat16;
    constexpr int kHeight = 2; // tiles per sequence block, 4 means 4*16 = 64 sequence elements per warp
    constexpr int kWidth = 2; // tiles per vector, 4 means head dimension is 4*16 = 64
    TORCH_CHECK(d == 32, "q.size(3) and k.size(3) should be 32");
    TORCH_CHECK(dv == 32, "v.size(3) should be 32");

    constexpr int kNumWorkers = 16;

    unsigned long mem_size = 2*kNumWorkers*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>);

    TORCH_CHECK(seqlen % (kNumWorkers*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");

    auto threads = kNumWorkers * kittens::WARP_THREADS;
    //printf("[causal_attend] Requesting %lu bytes of memory for %d worker warps (%d threads)\n", mem_size, kNumWorkers, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(causal_attend_kernel<H, kHeight, kWidth, kNumWorkers>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    causal_attend_kernel<H, kHeight, kWidth, kNumWorkers><<<batch*head,threads,mem_size>>>((int)seqlen, q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), f.data_ptr<float>(), o_small.data_ptr<H>());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

//#include "harness.impl"
    
