#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.


template <typename H = bf16, int _height, int _width>
__device__ void tileprint(rt<H, _height, _width, ducks::rt_layout::row> reg, char *name) {
    auto laneid = kittens::laneid();
    static_assert(reg.height == 1 && reg.width == 1, "height and width must be 1");
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            static_assert(reg.packed_per_thread == 4, "packed_per_thread must be 4");

            int row_top = laneid / 4;
            int row_bottom = row_top + 8;
            int col_left = laneid % 4 * 2; // stride 4
            int col_right = col_left + 8;

            auto item_top_left = __bfloat1622float2(reg.tiles[i][j].data[0]);
            auto item_bottom_left = __bfloat1622float2(reg.tiles[i][j].data[1]);
            auto item_top_right = __bfloat1622float2(reg.tiles[i][j].data[2]);
            auto item_bottom_right = __bfloat1622float2(reg.tiles[i][j].data[3]);
            printf("lane=%02d "
                "%s[%02d,%02d] 0x=% .3f "
                "%s[,%02d] 0y=% .3f    "
                "%s[%02d,%02d] 1x=% .3f "
                "%s[,%02d] 1y=% .3f    "
                "%s[%02d,%02d] 2x=% .3f "
                "%s[,%02d] 2y=% .3f    "
                "%s[%02d,%02d] 3x=% .3f "
                "%s[,%02d] 3y=% .3f\n",
                laneid,
                name, row_top, col_left, item_top_left.x,
                name, col_left+1, item_top_left.y,
                name, row_bottom, col_left, item_bottom_left.x,
                name, col_left+1, item_bottom_left.y,
                name, row_top, col_right, item_top_right.x,
                name, col_right+1, item_top_right.y,
                name, row_bottom, col_right, item_bottom_right.x,
                name, col_right+1, item_bottom_right.y);
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



/**
 * @brief Set a constant to elements of the diagonal in a square register tile.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void set_diagonal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=1) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i || j < i) { // below or above the diagonal, ignore
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            } else { // on the diagonal, interesting!
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2]; // above diagonal, copy

                if (laneid() == 0 || laneid() == 9 || laneid() == 18 || laneid() == 27) {
                    // diagonal: every odd row
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                } else {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }

                if (laneid() == 4 || laneid() == 13 || laneid() == 22 || laneid() == 31) {
                    // diagonal: every even row
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                } else {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
            }
        }
    }
}


__device__ void vecprint(rv<bf16_2, 1, 2> reg, char *name) {
    auto warpid        = kittens::warpid();
    auto item0 = __bfloat1622float2(reg.data[0][0]);
    printf("warpid=%d tid=%d %s[0] = {%f,%f}\n", warpid, threadIdx.x, name, item0.x, item0.y);
    auto item1 = __bfloat1622float2(reg.data[0][1]);
    printf("warpid=%d tid=%d %s[1] = {%f,%f}\n", warpid, threadIdx.x, name, item1.x, item1.y);
}

__device__ void vecprint(rv<bf16_2, 1, 1> reg, char *name) {
    auto warpid        = kittens::warpid();
    auto item0 = __bfloat1622float2(reg.data[0][0]);
    printf("warpid=%d tid=%d %s[0] = {%f,%f}\n", warpid, threadIdx.x, name, item0.x, item0.y);
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
    rt_fl<_height, _height> mma_TT;
    rt_fl<_height, _width> mma_TD;
    rt<T2, _height, _height> K;
    rt<T2, _height, _width> c_u, c_w;
    rt<T2, _height, _width> w_reg, u_reg;

    const int time_blocks = seqlen / (w_reg.rows*kNumWarps);
    //printf("seqlen: %d time_blocks: %d rows: %d\n", seqlen, time_blocks, w_reg.rows);

    // int time_warp_index = time_blocks*kNumWarps + warpid;
    int time_warp_index = warpid; // TODO

    /*
     * load k, v, beta
     */

    load(k_reg, _k + time_warp_index*k_reg.num_elements, k_reg.cols); // k = k.clone()
    load(v_reg, _v + time_warp_index*v_reg.num_elements, v_reg.cols); // v = v.clone()
    load(beta_reg, _beta + time_warp_index); // beta = beta.clone()

    /*
     * decay_values_forward: compute w and u
     */

    __syncthreads();

    zero(w_reg);
    zero(u_reg);
    zero(K);    
    zero(mma_TT);
    mma_ABt(mma_TT, k_reg, k_reg, mma_TT); // K = k@k.T dimensions: (T,S)
    copy(K, mma_TT); // convert to bf16 for mma_ABt
    mul_row(K, K, beta_reg); // K[t, :] *= beta[t]
    mul_row(k_reg, k_reg, beta_reg); // k[t, :] *= beta[t]
    mul_row(v_reg, v_reg, beta_reg); // v[t, :] *= beta[t]

    for (auto t = 0; t < w_reg.rows; t++) {
        __syncthreads();

        zero(mma_TT);
        rt<T2, _height, _width, ducks::rt_layout::col> &w_reg_col = swap_layout_inplace(w_reg);
        mma_AB(mma_TT, K, w_reg_col, mma_TT);
        w_reg = swap_layout_inplace(w_reg_col);
        copy(c_w, mma_TT);
        sub_singlerow(w_reg, k_reg, c_w, t); // w[t] = k[t] - c_w[t]

        __syncthreads();
    
        zero(mma_TT);
        rt<T2, _height, _width, ducks::rt_layout::col> &u_reg_col = swap_layout_inplace(u_reg);
        mma_AB(mma_TT, K, u_reg_col, mma_TT); // (T,S) (S,D) -> (T,D)
        u_reg = swap_layout_inplace(u_reg_col);
        copy(c_u, mma_TT);
        sub_singlerow(u_reg, v_reg, c_u, t); // u[t] = v[t] - c_u[t]
    }

    __syncthreads();

    store(_w, w_reg, w_reg.cols);
    store(_u, u_reg, u_reg.cols);

    /*
     * decay_values_backward: compute d_{w_t} / d_{b_s}
     */

    rt<T2, _height, _width> bases_reg;
    rt<T2, _height, _height> decays_reg;
    rt<T2, _height, _height> pre_reg;

    zero(k_reg);
    load(k_reg, _k + time_warp_index*k_reg.num_elements, k_reg.cols); // k = k.clone(), again

    zero(K);
    zero(mma_TT);
    mma_ABt(mma_TT, k_reg, k_reg, mma_TT); // K = k@k.T dimensions: (T,S)
    copy(K, mma_TT); // convert to bf16 for mma_ABt
    make_causal(K, K, 0);
    set_diagonal(K, K, 0);

    zero(mma_TD);
    rt<T2, _height, _width, ducks::rt_layout::col> &w_reg_col = swap_layout_inplace(w_reg);
    mma_AB(mma_TD, K, w_reg_col, mma_TD); // bases_reg = K@w
    w_reg = swap_layout_inplace(w_reg_col);
    copy(bases_reg, mma_TD);

    sub(bases_reg, k_reg, bases_reg); // bases = k - K@w
    mul_row(K, K, beta_reg); // K[t, :] *= beta[t]

    zero(decays_reg);
    set_diagonal(decays_reg, decays_reg, 1);
    for (auto t = 0; t < decays_reg.rows; t++) {
        __syncthreads();

        zero(mma_TT);
        rt<T2, _height, _width, ducks::rt_layout::col> &decays_reg_col = swap_layout_inplace(decays_reg);
        mma_AB(mma_TT, K, decays_reg_col, mma_TT); // (T,S) (S,D) -> (T,D)
        decays_reg = swap_layout_inplace(decays_reg_col);
        copy(pre_reg, mma_TT);
        sub_singlerow(decays_reg, decays_reg, pre_reg, t); // decays[t] = (decays - K@decays)[t]
    }

    __syncthreads();

    zero(mma_TD);
    load(pre_reg, _d_out_w + time_warp_index*pre_reg.num_elements, pre_reg.cols); // pre_reg = d_out_w.clone()
    mma_ABt(mma_TD, pre_reg, bases_reg, mma_TD); // mma = d_out_w@bases.T
    copy(pre_reg, mma_TD); // pre_reg = d_out_w@bases.T
    mul(pre_reg, pre_reg, decays_reg); // pre_reg *= decays_reg

    typename rt<T2, _height, _height>::row_vec d_beta_reg;
    zero(d_beta_reg);
    col_sum(d_beta_reg, pre_reg); // d_beta = einsum('ts->s', pre_reg) 

    __syncthreads();

    store(_d_beta, d_beta_reg);
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
    
