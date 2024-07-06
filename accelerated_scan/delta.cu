#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.


template <typename H = bf16, int _height, int _width>
__device__ void tileprint_bf(rt<H, _height, _width> reg, char *name, int a, int b) {
    auto warpid        = kittens::warpid();
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            static_assert(reg.packed_per_thread == 4, "packed_per_thread must be 4");

            int row_top = laneid() / 4;
            int row_bottom = row_top + 8;
            int colL = laneid() % 4 * 2; // stride 4
            int colR = colL + 8;

            auto itemTL = __bfloat1622float2(reg.tiles[i][j].data[0]);
            auto itemTR = __bfloat1622float2(reg.tiles[i][j].data[2]);
            auto itemBL = __bfloat1622float2(reg.tiles[i][j].data[1]);
            auto itemBR = __bfloat1622float2(reg.tiles[i][j].data[3]);

            printf("%s kv=%d:%d warpid=%d laneid=%d top=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                name, a,b, warpid, laneid(), row_top, colL, itemTL.x, itemTL.y, colR, itemTR.x, itemTR.y);
            printf("%s kv=%d:%d warpid=%d laneid=%d bottom=%d colL=%d {%f,%f} colR=%d {%f,%f}\n",
                name, a,b, warpid, laneid(), row_bottom, colL, itemBL.x, itemBL.y, colR, itemBR.x, itemBR.y);
        }
    }
}


template<ducks::rt::all T>
__device__ static inline void sub_singlerow(T &dst, const T &lhs, const T &rhs, const int row_index) {
    const int row_top = laneid() / 4;
    const int row_bottom = row_top + 8;

    static_assert(dst.packed_per_tile == 4, "packed_per_thread must be 4");

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {            
            if (row_top == row_index) {
                dst.tiles[i][j].data[0] = lhs.tiles[i][j].data[0] - rhs.tiles[i][j].data[0];
                dst.tiles[i][j].data[2] = lhs.tiles[i][j].data[2] - rhs.tiles[i][j].data[2];
            }
            if (row_bottom == row_index) {
                dst.tiles[i][j].data[1] = lhs.tiles[i][j].data[1] - rhs.tiles[i][j].data[1];
                dst.tiles[i][j].data[3] = lhs.tiles[i][j].data[3] - rhs.tiles[i][j].data[3];
            }
        }
    }
}


template <typename H = c10::BFloat16, int _height, int _width, int kNumWarps = 8, typename T = bf16, typename T2 = bf16_2>
__global__ void decay_values_backward_kernel(
    int seqlen,
    const H* __restrict__ __d_out_w__,
    const H* __restrict__ __d_out_u__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const H* __restrict__ __beta__,
    H* __restrict__ __d_k__,
    H* __restrict__ __d_v__,
    H* __restrict__ __d_beta__,
    H* __restrict__ __w__,
    H* __restrict__ __u__
) {
    auto warpid           = kittens::warpid();
    auto block_start      = blockIdx.x*(seqlen*_width*TILE_DIM);
    auto beta_block_start = blockIdx.x*(seqlen*TILE_DIM); // width is 1 for beta
    const T *_d_out_w = reinterpret_cast<const T *>(__d_out_w__) + block_start,
            *_d_out_u = reinterpret_cast<const T *>(__d_out_u__) + block_start,
            *_k = reinterpret_cast<const T *>(__k__) + block_start,
            *_v = reinterpret_cast<const T *>(__v__) + block_start,
            *_beta = reinterpret_cast<const T *>(__beta__) + beta_block_start;
          T *_d_k = reinterpret_cast<T *>(__d_k__) + block_start,
            *_d_v = reinterpret_cast<T *>(__d_v__) + block_start,
            *_d_beta = reinterpret_cast<T *>(__d_beta__) + beta_block_start,
            *_w = reinterpret_cast<T *>(__w__) + block_start,
            *_u = reinterpret_cast<T *>(__u__) + block_start;
    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K, V and beta live in shared memory -- this is about all that will fit.
    // st_bf<_height, _width, ducks::st_layout::swizzle> (&k_smem)[kNumWarps] = al.allocate<st_bf<_height, _width, ducks::st_layout::swizzle>, kNumWarps>();
    // st_bf<_height, _width, ducks::st_layout::swizzle> (&v_smem)[kNumWarps] = al.allocate<st_bf<_height, _width, ducks::st_layout::swizzle>, kNumWarps>();
    // st_bf<_height, _width, ducks::st_layout::swizzle> (&beta_smem)[kNumWarps] = al.allocate<st_bf<_height, _width, ducks::st_layout::swizzle>, kNumWarps>();

    // Initialize all of the register tiles.
    rt<T2, _height, _width> k_reg, v_reg; // v_reg need to be swapped into col_l
    typename rt<T2, _height, _width>::col_vec beta_reg;
    rt_fl<_height, _height> mma;
    rt<T2, _height, _width> K;
    rt<T2, _height, _width> c_u, c_w;
    rt<T2, _height, _width> w_reg, u_reg;

    const int time_blocks = seqlen / (w_reg.rows*kNumWarps);
    //printf("seqlen: %d time_blocks: %d rows: %d\n", seqlen, time_blocks, w_reg.rows);

    // int time_warp_index = time_blocks*kNumWarps + warpid;
    int time_warp_index = warpid; // TODO

    load(k_reg, _k + time_warp_index*k_reg.num_elements, k_reg.cols); // k = k.clone()
    load(v_reg, _v + time_warp_index*v_reg.num_elements, v_reg.cols); // v = v.clone()
    load(beta_reg, _beta + time_warp_index); // beta = beta.clone()

    __syncthreads();

    zero(w_reg);
    zero(u_reg);
    zero(K);    
    mma_ABt(mma, k_reg, k_reg, mma); // K = k@k.T dimensions: (T,S)
    copy(K, mma); // convert to bf16 for mma_ABt
    mul_row(K, K, beta_reg); // K[t, :] *= beta[t]
    mul_row(k_reg, k_reg, beta_reg); // k[t, :] *= beta[t]
    mul_row(v_reg, v_reg, beta_reg); // v[t, :] *= beta[t]

    for (auto t = 0; t < w_reg.rows; t++) {
        __syncthreads();

        zero(mma);
        rt<T2, _height, _width, ducks::rt_layout::col> &w_reg_col = swap_layout_inplace(w_reg);
        mma_AB(mma, K, w_reg_col, mma);
        w_reg = swap_layout_inplace(w_reg_col);
        copy(c_w, mma);
        sub_singlerow(w_reg, k_reg, c_w, t); // w[t] = k[t] - c_w[t]

        __syncthreads();
    
        zero(mma);
        rt<T2, _height, _width, ducks::rt_layout::col> &u_reg_col = swap_layout_inplace(u_reg);
        mma_AB(mma, K, u_reg_col, mma); // (T,S) (S,D) -> (T,D)
        u_reg = swap_layout_inplace(u_reg_col);
        copy(c_u, mma);
        sub_singlerow(u_reg, v_reg, c_u, t); // u[t] = v[t] - c_u[t]
    }

    __syncthreads();

    store(_w, w_reg, w_reg.cols);
    store(_u, u_reg, u_reg.cols);
}

void
decay_values_backward(torch::Tensor d_out_w, torch::Tensor d_out_u, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                      torch::Tensor d_k, torch::Tensor d_v, torch::Tensor d_beta, torch::Tensor w, torch::Tensor u) {
    CHECK_INPUT(d_out_w);
    CHECK_INPUT(d_out_u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);
    CHECK_INPUT(d_k);
    CHECK_INPUT(d_v);
    CHECK_INPUT(d_beta);
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    
    auto batch_head = k.size(0);
    auto seqlen = k.size(1);
    auto d      = k.size(2);
    auto dv     = v.size(2);
    bool k_same = true;
    for(auto i = 0; i < 3; i++) { 
        k_same &= k.size(i) == v.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "K and V should be same size");

    // TORCH_CHECK(d_out_w.scalar_type() == c10::ScalarType::BFloat16, "d_out_w is a Bfloat");
    // TORCH_CHECK(d_out_u.scalar_type() == c10::ScalarType::BFloat16, "d_out_u is a Bfloat");
    // TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    // TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    // TORCH_CHECK(beta.scalar_type() == c10::ScalarType::BFloat16, "beta is a Bfloat");
    // TORCH_CHECK(d_k.scalar_type() == c10::ScalarType::BFloat16, "d_k is a Bfloat");
    // TORCH_CHECK(d_v.scalar_type() == c10::ScalarType::BFloat16, "d_v is a Bfloat");
    // TORCH_CHECK(d_beta.scalar_type() == c10::ScalarType::BFloat16, "d_beta is a Bfloat");
    // TORCH_CHECK(w.scalar_type() == c10::ScalarType::BFloat16, "w is a Bfloat");
    // TORCH_CHECK(u.scalar_type() == c10::ScalarType::BFloat16, "u is a Bfloat");

    using H = c10::BFloat16;
    using T = bf16;
    using T2 = bf16_2;
    // using H = float;
    // using T2 = float2;
    constexpr int kHeight = 1; // tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    constexpr int kWidth = 1; // tiles per vector, 2 means head dimension is 2*16 = 32
    TORCH_CHECK(d == kWidth * 16, "q.size(3) and k.size(3) should be kWidth*16");
    TORCH_CHECK(dv == kWidth * 16, "v.size(3) should be kWidth*16");

    constexpr int kNumWarps = 1;

    // storing k, v and beta
    unsigned long mem_size = kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) // k
                           + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) // v
                           + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) // beta
                           ;

    TORCH_CHECK(seqlen % (kNumWarps*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of warps times stored fragments");

    auto threads = kNumWarps * kittens::WARP_THREADS;
    //printf("[decay_values_backward] Requesting %lu bytes of memory for %d worker warps (%d threads)\n", mem_size, kNumWarps, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        decay_values_backward_kernel<H, kHeight, kWidth, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    decay_values_backward_kernel<H, kHeight, kWidth, kNumWarps, T, T2><<<batch_head,threads,mem_size>>>(
        (int)seqlen,
        d_out_w.data_ptr<H>(), d_out_u.data_ptr<H>(),
        k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(),
        d_k.data_ptr<H>(), d_v.data_ptr<H>(), d_beta.data_ptr<H>(),
        w.data_ptr<H>(), u.data_ptr<H>());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

//#include "harness.impl"
    
