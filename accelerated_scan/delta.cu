#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.


template <typename H = bf16, int _time, int _key>
__device__ void tileprint(rt<H, _time, _key, ducks::rt_layout::row> reg, char *name) {
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


template<typename op, ducks::rt::all T>
__device__ static inline void op_singlerow(T &dst, const T &lhs, const T &rhs, const int row_index) {
    const int row_top = laneid() / 4;
    const int row_bottom = row_top + 8;

    static_assert(dst.packed_per_tile == 4, "packed_per_tile must be 4");
    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {            
            if (row_top == row_index) {
                dst.tiles[i][j].data[0] = op::template op<dtype>(lhs.tiles[i][j].data[0], rhs.tiles[i][j].data[0]);
                dst.tiles[i][j].data[2] = op::template op<dtype>(lhs.tiles[i][j].data[2], rhs.tiles[i][j].data[2]);
            } else if (row_bottom == row_index) {
                dst.tiles[i][j].data[1] = op::template op<dtype>(lhs.tiles[i][j].data[1], rhs.tiles[i][j].data[1]);
                dst.tiles[i][j].data[3] = op::template op<dtype>(lhs.tiles[i][j].data[3], rhs.tiles[i][j].data[3]);
            }
        }
    }
}


template<ducks::rt::all RT>
__device__ static inline void reset_trailing_rows(RT &dst, const int row_index, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row_top = laneid() / 4;
    const int row_bottom = row_top + 8;

    static_assert(dst.packed_per_tile == 4, "packed_per_tile must be 4");

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {            
            if (row_top >= row_index) {
                dst.tiles[i][j].data[0].x = val;
                dst.tiles[i][j].data[0].y = val;
                dst.tiles[i][j].data[2].x = val;
                dst.tiles[i][j].data[2].y = val;
            }
            if (row_bottom >= row_index) {
                dst.tiles[i][j].data[1].x = val;
                dst.tiles[i][j].data[1].y = val;
                dst.tiles[i][j].data[3].x = val;
                dst.tiles[i][j].data[3].y = val;
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

__device__ void vecprint(rv<bf16_2, 8, 2> reg, char *name) {
    auto warpid        = kittens::warpid();
    
    #pragma unroll
    for(int i = 0; i < reg.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < reg.inner_dim; j++) {
            auto item = __bfloat1622float2(reg.data[i][j]);
            printf("warpid=%d tid=%d %s[%d][%d] = {%f,%f}\n", warpid, threadIdx.x, name, i, j, item.x, item.y);
        }
    }
}

template <typename D, typename ACCUM, int _time, int _key, int _value>
static __device__ inline void decay_values_forward(
    rt<ACCUM, _time, _time> &mma_TT,
    rt<ACCUM, _time, _key> &mma_TD,
    rt<D, _time, _time> &tt_reg,
    rt<D, _time, _time> &bKl_reg, // can be aliased to tt_reg
    rt<D, _time, _key> &k_reg,
    typename rt<D, _time, _key>::col_vec &beta_reg,
    rt<D, _time, _key> &w_reg,
    rt<D, _time, _value> &u_reg,
    rt<D, _time, _value> &v_reg,
    rt<D, _time, _value> &u_bases_reg, // can be aliased to v_reg
    rt<D, _time, _key> &bk_reg,
    rt<D, _time, _key> &tk_reg
) {
    using col = rt<D, _time, _key, ducks::rt_layout::col>;

    __syncthreads();

    mul_row(bk_reg, k_reg, beta_reg); // k = einsum('ntk,nt->ntk', k, beta)

    zero(mma_TT);
    mma_ABt(mma_TT, k_reg, k_reg, mma_TT); // tt = einsum('ntd,nsd->nts', k, k)
    copy(tt_reg, mma_TT);
    make_causal(tt_reg, tt_reg, 0);
    set_diagonal(tt_reg, tt_reg, 0); // tt = tt.tril(diagonal=-1)

    mul_row(bKl_reg, tt_reg, beta_reg); // tt = einsum('nts,nt->nts', tt, beta)

    zero(w_reg);
    zero(u_reg);
    copy(u_bases_reg, v_reg);
    mul_row(v_reg, v_reg, beta_reg); // v = einsum('ntw,nt->ntw', v, beta)

    for (auto t = 0; t < w_reg.rows; t++) {
        __syncthreads();

        {
            zero(mma_TD);
            col &w_reg_col = swap_layout_inplace(w_reg);
            mma_AB(mma_TD, bKl_reg, w_reg_col, mma_TD);
            w_reg = swap_layout_inplace(w_reg_col);
            copy(tk_reg, mma_TD);
            op_singlerow<base_ops::sub>(w_reg, bk_reg, tk_reg, t); // w[t] = bk[t] - tk[t]
        }

        __syncthreads();
    
        {
            zero(mma_TD);
            col &u_reg_col = swap_layout_inplace(u_reg);
            mma_AB(mma_TD, bKl_reg, u_reg_col, mma_TD); // (T,S) (S,D) -> (T,D)
            u_reg = swap_layout_inplace(u_reg_col);
            copy(tk_reg, mma_TD);
            op_singlerow<base_ops::sub>(u_reg, v_reg, tk_reg, t); // u[t] = bv[t] - tk[t]
        }
    }


    __syncthreads();
}


template <typename H, typename T, typename D, typename ACCUM, int _time, int _key, int _value, int kNumWarps = 1, int kChunkSize = 16>
__global__ void forward_kernel(
    const int num_chunks,
    const H* __restrict__ __q__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const H* __restrict__ __beta__,
    H* __restrict__ __w__,
    H* __restrict__ __u__,
    H* __restrict__ __y__
) {
    auto warpid           = kittens::warpid();
    auto block_start      = blockIdx.x*(num_chunks*kChunkSize*(_key*TILE_DIM));
    auto beta_block_start = blockIdx.x*(num_chunks*kChunkSize*1); // width is 1 for beta
    const T *_q = reinterpret_cast<const T *>(__q__) + block_start,
            *_k = reinterpret_cast<const T *>(__k__) + block_start,
            *_v = reinterpret_cast<const T *>(__v__) + block_start,
            *_beta = reinterpret_cast<const T *>(__beta__) + beta_block_start;
          T *_w = reinterpret_cast<T *>(__w__) + block_start,
            *_u = reinterpret_cast<T *>(__u__) + block_start,
            *_y = reinterpret_cast<T *>(__y__) + block_start;
    
    using col = rt<D, _time, _key, ducks::rt_layout::col>;
    /*
     * register allocations
     */
    rt<D, _time, _key> k_reg, w_reg, tk_reg, bk_reg;
    rt<D, _time, _value> v_reg, u_reg;
    typename rt<D, _time, _key>::col_vec beta_reg;
    rt<D, _time, _time> tt_reg;

    rt<ACCUM, _time, _time> mma_TT;
    rt<ACCUM, _time, _key> mma_TD;

    for (int time_block = 0; time_block < num_chunks / kNumWarps; time_block++) {
        int time_index = time_block * kNumWarps + warpid;

        /*
        * load k, v, beta
        */

        load(k_reg, _k + time_index*k_reg.num_elements, k_reg.cols); // k = k.clone()
        load(v_reg, _v + time_index*v_reg.num_elements, v_reg.cols); // v = v.clone()
        load(beta_reg, _beta + time_index*beta_reg.outer_dim*TILE_DIM); // beta = beta.clone()

        /*
        * decay_values_forward: compute w and u
        */

        decay_values_forward(mma_TT, mma_TD, tt_reg, tt_reg, k_reg, beta_reg, w_reg, u_reg, v_reg, v_reg, bk_reg, tk_reg);

        store(_w + time_index*u_reg.num_elements, w_reg, w_reg.cols);
        store(_u + time_index*u_reg.num_elements, u_reg, u_reg.cols);

        /*
        * attend to decayed values
        */

        auto &q_reg = w_reg;
        load(q_reg, _q + time_index*q_reg.num_elements, q_reg.cols); // q = q.clone()

        zero(mma_TT);
        mma_ABt(mma_TT, q_reg, k_reg, mma_TT); // mma_TT = einsum('nsk,ntk->nst', q, k)
        copy(tt_reg, mma_TT);
        make_causal(tt_reg, tt_reg, 0); // tt.tril_()

        zero(mma_TD);
        col &u_reg_col = swap_layout_inplace(u_reg);
        mma_AB(mma_TD, tt_reg, u_reg_col, mma_TD); // mma_TD = einsum('nst,ntw->ntw', tt, u)
        auto &y_reg = w_reg;
        copy(y_reg, mma_TD);

        store(_y + time_index*y_reg.num_elements, y_reg, y_reg.cols);
    }

    __syncthreads();

    if (warpid == 0) {
        loop<T, D, ACCUM, _time, _key, _value, kChunkSize>(_q, _k, _w, _u, _y, num_chunks);
    }
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value, int kChunkSize>
__device__ static void loop(
    const T *_q,
    const T *_k,
    const T *_w,
    const T *_u,
    T *_y,
    const int num_chunks
) {
    rt<D, _value, _key> state;
    rt<ACCUM, _value, _key> mma_state;
    rt<ACCUM, _time, _value> mma_value;
    rt<D, _time, _key> q, k, w;
    rt<D, _time, _value> u, y;
    rt<ACCUM, _time, _time> mma_qk;
    rt<D, _time, _time> qk;

    int chunk = 0;

    load(k, _k + chunk*k.num_elements, k.cols);
    load(u, _u + chunk*u.num_elements, u.cols);
    zero(mma_state);

    //mma_AtBt kt,vk->tv col row

    //mma_AtBt xt,sx->ts
    //mma_AtB  tv,tk->vk col col
    //mma_ABt vk,tk->vt row row
    //mma_ABt tk,vk->tv row row
    //mma_ABt st,tv->sv row row
    //mma_AB st,tv ->sv row col

    for (chunk = 1; chunk < num_chunks; chunk++) {
        auto &k_col = swap_layout_inplace(k);
        auto &u_col = swap_layout_inplace(u);
        mma_AtB(mma_state, u_col, k_col, mma_state);
        k = swap_layout_inplace(k_col);
        u = swap_layout_inplace(u_col);
        copy(state, mma_state);

        load(w, _w + chunk*w.num_elements, w.cols);

        zero(mma_value);
        mma_ABt(mma_value, w, state, mma_value); // u_old
        copy(u, mma_value); // u = u_old

        load(q, _q + chunk*q.num_elements, q.cols);
        load(k, _k + chunk*k.num_elements, k.cols);

        zero(mma_qk);
        mma_ABt(mma_qk, q, k, mma_qk); // qk = einsum('nsk,ntk->nst', q, k)
        copy(qk, mma_qk);
        make_causal(qk, qk, 0);

        load(y, _y + chunk*y.num_elements, y.cols);
        {
            zero(mma_value);
            auto &u_col1 = swap_layout_inplace(u);
            mma_AB(mma_value, qk, u_col1, mma_value); // y_prev
            u = swap_layout_inplace(u_col1);
            auto &y_prev = w; // repurpose w
            copy(y_prev, mma_value);
            sub(y, y, y_prev);

            zero(mma_value);
            mma_ABt(mma_value, q, state, mma_value); // y_cur
            auto &y_cur = w; // repurpose w again
            copy(y_cur, mma_value);
            add(y, y, y_cur);
        }
        store(_y + chunk*y.num_elements, y, y.cols);

        auto &u1 = y; // repurpose y
        load(u1, _u + chunk*u1.num_elements, u1.cols);
        sub(u, u1, u);
    }
}

/**
 * @brief Bind a list of vectors (k,u) and write them to a dictionary state_delta.
 */
template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static void associate(
    rt<D, _value, _key, ducks::rt_layout::row> &state_delta,
    /*const*/ rt<D, _time, _value, ducks::rt_layout::row> &v,
    /*const*/ rt<D, _time, _key, ducks::rt_layout::row> &k
) {
    rt<ACCUM, _value, _key> mma_state;

    zero(mma_state);
    auto &k_col = swap_layout_inplace(k);
    auto &v_col = swap_layout_inplace(v);
    mma_AtB(mma_state, v_col, k_col, mma_state);
    swap_layout_inplace(k_col);
    swap_layout_inplace(v_col);
    copy(state_delta, mma_state);
}

template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static void query(
    rt<D, _time, _value, ducks::rt_layout::row> &output_values,
    const rt<D, _value, _key, ducks::rt_layout::row> &state,
    const rt<D, _time, _key, ducks::rt_layout::row> &query
) {
    rt<ACCUM, _time, _value> mma;

    zero(mma);
    mma_ABt(mma, query, state, mma); // einsum('tk,vk->tv', query, state)
    copy(output_values, mma);
}

template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static void reverse_query(
    rt<D, _time, _key, ducks::rt_layout::row> &output_keys,
    const rt<D, _time, _value, ducks::rt_layout::row> &value_query,
    /*const*/ rt<D, _value, _key, ducks::rt_layout::row> &state
) {
    rt<ACCUM, _time, _key> mma;

    zero(mma);
    auto &state_col = swap_layout_inplace(state);
    mma_AB(mma, value_query, state_col, mma); // einsum('tv,vk->tk', value_query, state)
    copy(output_keys, mma);
    swap_layout_inplace(state_col);
}


template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _key>
__device__ static void kernel(
    rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    const rt<D, _time_source, _key, ducks::rt_layout::row> &query,
    const rt<D, _time_target, _key, ducks::rt_layout::row> &key
) {
    rt<ACCUM, _time_source, _time_target> qk;

    zero(qk);
    mma_ABt(qk, query, key, qk); // mma_TT = einsum('nsk,ntk->nst', q, k)
    copy(attention, qk);
}

template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _value>
__device__ static void attend(
    rt<D, _time_source, _value, ducks::rt_layout::row> &mixtures,
    const rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_target, _value, ducks::rt_layout::row> &values
) {
    rt<ACCUM, _time_source, _value> mma;

    zero(mma);
    auto &values_col = swap_layout_inplace(values);
    mma_AB(mma, attention, values_col, mma);
    copy(mixtures, mma);
    swap_layout_inplace(values_col);
}

template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _value>
__device__ static void reverse_attend(
    rt<D, _time_target, _value, ducks::rt_layout::row> &mixtures,
    /*const*/ rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_source, _value, ducks::rt_layout::row> &source_values
) {
    rt<ACCUM, _time_target, _value> mma;

    zero(mma);
    auto &attention_col = swap_layout_inplace(attention);
    auto &source_values_col = swap_layout_inplace(source_values);
    mma_AtB(mma, attention_col, source_values_col, mma);
    copy(mixtures, mma);
    swap_layout_inplace(attention_col);
    swap_layout_inplace(source_values_col);
}

struct op_negate {
    template<typename T> static __device__ inline T op(const T &x) { return -x; }
};

template<ducks::rt::all T>
__device__ static inline void negate(T &dst) {
    unary_map<op_negate, T>(dst, dst);
}

template <typename T, typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static void loop_backward(
    const T *_d_y,
    const T *_q,
    const T *_k,
    T *_w, // actually const
    T *_u, // will be updated
    T *_d_q,
    T *_d_k,
    T *_d_w,
    T *_d_u,
    const int num_chunks
) {
    rt<D, _value, _key> state, d_state, state_delta;
    rt<D, _time, _time> qk;
    rt<D, _time, _key> tk, q, k, w;
    rt<D, _time, _value> d_y, u, d_u, d_state_decays;
    rt<D, _time, _key> d_q, d_k;

    int chunk = 0;

    /*
     * recompute the loop forward
     */

    load(k, _k + chunk*k.num_elements, k.cols);
    load(u, _u + chunk*u.num_elements, u.cols);
    
    associate(state_delta, u, k);

    for (chunk = 1; chunk < num_chunks; chunk++) {
        load(w, _w + chunk*w.num_elements, w.cols);
        load(u, _u + chunk*u.num_elements, u.cols);

        query(tk, state, w);
        sub(u, u, tk);
        store(_u + chunk*u.num_elements, u, u.cols);

        load(k, _k + chunk*k.num_elements, k.cols);
        associate(state_delta, u, k);

        if (chunk < num_chunks - 1) {
            add(state, state, state_delta);
        }
    }

    /*
     * from now on, u's in global memory are decayed
     *
     * loop backward
     */

    zero(d_state);

    for (chunk = num_chunks - 1; chunk > 0; chunk--) {
        load(u, _u + chunk*u.num_elements, u.cols);
        load(k, _k + chunk*k.num_elements, k.cols);
        load(w, _w + chunk*w.num_elements, w.cols);

        // we already have state_delta set up correctly in the first iteration
        if (chunk < num_chunks - 1) {
            associate(state_delta, u, k);
            /*
             * uncompute the state backwards -- take that nonlinear models!
             */
            sub(state, state, state_delta);
            query(tk, state, w);
        }

        load(d_y, _d_y + chunk*d_y.num_elements, d_y.cols);
        negate(d_y);

        // d_q, d_k
        kernel(qk, d_y, tk);
        make_causal(qk, qk, 0);

        // d_q
        attend(d_q, qk, k);
        reverse_query(tk, d_y, state);
        sub(d_q, d_q, tk);
        store(_d_q + chunk*d_q.num_elements, d_q, d_q.cols);

        // d_k
        attend(d_k, qk, q);
        if (chunk < num_chunks - 1) {
            reverse_query(tk, u, d_state); // otherwise we know d_state is zero
            add(d_k, d_k, tk);
        }
        store(_d_k + chunk*d_k.num_elements, d_k, d_k.cols);

        // d_u
        if (chunk < num_chunks - 1) {
            query(d_u, d_state, k); // otherwise we know d_state is zero
            store(_d_u + chunk*d_u.num_elements, d_u, d_u.cols);
        }

        // d_state_decays
        load(q, _q + chunk*q.num_elements, q.cols);
        kernel(qk, q, k);
        make_causal(qk, qk, 0);

        auto &d_state_decays_buf = d_u; // alias
        reverse_attend(d_state_decays, qk, d_y);
        if (chunk < num_chunks - 1) {
            query(d_state_decays_buf, d_state, k);
            sub(d_state_decays, d_state_decays, d_state_decays_buf);
        }

        // d_w
        reverse_query(tk, d_state_decays, state);
        store(_d_w + chunk*tk.num_elements, tk, tk.cols);

        // backpropagate through time
        auto &state_buf = state_delta; // alias
        associate(state_buf, d_y, q);
        sub(d_state, d_state, state_buf);
        associate(state_buf, d_state_decays, w);
        add(d_state, d_state, state_buf);
    }

    chunk = 0;
    load(u, _u + chunk*u.num_elements, u.cols);
    load(k, _k + chunk*k.num_elements, k.cols);

    query(tk, d_state, k);
    store(_d_u + chunk*tk.num_elements, tk, tk.cols);

    reverse_query(tk, u, d_state);
    store(_d_k + chunk*tk.num_elements, tk, tk.cols);
}


template <typename H, typename T, typename D, typename ACCUM, int _time, int _key, int _value, int kNumWarps = 8, int kChunkSize = 16>
__global__ void backward_kernel(
    int num_chunks,
    H* __restrict__ __d_out_w__,
    H* __restrict__ __d_out_u__,
    const H* __restrict__ __d_out_y__,
    const H* __restrict__ __q__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const H* __restrict__ __beta__,
    H* __restrict__ __d_q__,
    H* __restrict__ __d_k__,
    H* __restrict__ __d_v__,
    H* __restrict__ __d_beta__,
    H* __restrict__ __w__,
    H* __restrict__ __u__,
    H* __restrict__ __y__
) {
    auto warpid           = kittens::warpid();
    auto block_start      = blockIdx.x*(num_chunks*kChunkSize*(_key*TILE_DIM));
    auto beta_block_start = blockIdx.x*(num_chunks*kChunkSize*1); // width is 1 for beta
    const T *_d_out_y = reinterpret_cast<const T *>(__d_out_y__) + block_start,
            *_q = reinterpret_cast<const T *>(__q__) + block_start,
            *_k = reinterpret_cast<const T *>(__k__) + block_start,
            *_v = reinterpret_cast<const T *>(__v__) + block_start,
            *_beta = reinterpret_cast<const T *>(__beta__) + beta_block_start;
          T *_d_out_w = reinterpret_cast<T *>(__d_out_w__) + block_start,
            *_d_out_u = reinterpret_cast<T *>(__d_out_u__) + block_start,
            *_d_q = reinterpret_cast<T *>(__d_q__) + block_start,
            *_d_k = reinterpret_cast<T *>(__d_k__) + block_start,
            *_d_v = reinterpret_cast<T *>(__d_v__) + block_start,
            *_d_beta = reinterpret_cast<T *>(__d_beta__) + beta_block_start,
            *_w = reinterpret_cast<T *>(__w__) + block_start,
            *_u = reinterpret_cast<T *>(__u__) + block_start,
            *_y = reinterpret_cast<T *>(__y__) + block_start;
    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K, V and beta live in shared memory -- this is about all that will fit.
    // st_bf<_time, _key, ducks::st_layout::swizzle> (&k_smem)[kNumWarps] = al.allocate<st_bf<_time, _key, ducks::st_layout::swizzle>, kNumWarps>();
    // st_bf<_time, _key, ducks::st_layout::swizzle> (&v_smem)[kNumWarps] = al.allocate<st_bf<_time, _value, ducks::st_layout::swizzle>, kNumWarps>();
    // st_bf<_time, 1, ducks::st_layout::swizzle> (&beta_smem)[kNumWarps] = al.allocate<st_bf<_time, ducks::st_layout::swizzle>, kNumWarps>();

    /*
     * register allocations
     */
    rt<D, _time, _key, ducks::rt_layout::col> q_reg;
    rt<D, _time, _key> k_reg, d_out_w_reg;
    rt<D, _time, _value> v_reg, d_out_y_reg;
    union {
        typename rt<D, _time, _key>::col_vec beta_reg;
        rv<D, _time, 2> d_beta_reg;
    };
    rt<D, _time, _key> w_reg, w_bases_reg, bk_reg, d_k_reg, tk_reg;
    rt<D, _time, _value> u_reg, u_bases_reg;
    rt<D, _time, _time> tt_reg, bKl_reg;

    rt<ACCUM, _time, _time> mma_TT;
    rt<ACCUM, _time, _key> mma_TD;

    const int time_blocks = (num_chunks*kChunkSize) / (w_reg.rows*kNumWarps);

    // int time_warp_index = time_blocks*kNumWarps + warpid;
    int time_warp_index = warpid; // TODO

    /*
     * load k, v, beta, d_out_w, d_out_u, d_out_y
     */

    load(k_reg, _k + time_warp_index*k_reg.num_elements, k_reg.cols); // k = k.clone()
    load(v_reg, _v + time_warp_index*v_reg.num_elements, v_reg.cols); // v = v.clone()
    load(beta_reg, _beta + time_warp_index); // beta = beta.clone()

    /*
     * decay_values_forward: compute w and u
     */

    __syncthreads();

    decay_values_forward(mma_TT, mma_TD, tt_reg, bKl_reg, k_reg, beta_reg, w_reg, u_reg, v_reg, u_bases_reg, bk_reg, tk_reg);

    store(_w, w_reg, w_reg.cols);
    store(_u, u_reg, u_reg.cols);

    // 
    // loop backwards
    //
    if (1) {
        __syncthreads();
        if (warpid == 0) {
            loop_backward<T, D, ACCUM, _time, _key, _value>(
                _d_out_y,
                _q, _k, _w, _u,
                _d_q, _d_k, _d_out_w, _d_out_u,
                num_chunks
            );
        }
        __syncthreads();
    }

    {
        zero(mma_TD);
        auto &w_reg_col = swap_layout_inplace(w_reg);
        mma_AB(mma_TD, tt_reg, w_reg_col, mma_TD); // mma_TD = einsum('nts,nsk->ntk', tt, w)
        w_reg = swap_layout_inplace(w_reg_col);
        copy(w_bases_reg, mma_TD);
        sub(w_bases_reg, k_reg, w_bases_reg); // w_bases_reg = k - mma_TD
    }

    {
        zero(mma_TD);
        auto &u_reg_col = swap_layout_inplace(u_reg);
        mma_AB(mma_TD, tt_reg, u_reg_col, mma_TD); // mma_TD = einsum('nts,nsw->ntw', tt, u)
        u_reg = swap_layout_inplace(u_reg_col);
        copy(v_reg, mma_TD);
        sub(u_bases_reg, u_bases_reg, v_reg); // u_bases = u_bases_reg - mma_TD
    }

    /*
     * causal_attend_backward for d_q, d_k_2, d_out_u
     */

    load(d_out_y_reg, _d_out_y + time_warp_index*d_out_y_reg.num_elements, d_out_y_reg.cols); // d_out_y = d_out_y.clone()

    zero(mma_TT);
    mma_ABt(mma_TT, d_out_y_reg, u_reg, mma_TT); // einsum('nsv,ntv->nst', d_out_y_reg, u)
    copy(tt_reg, mma_TT);
    make_causal(tt_reg, tt_reg, 0); // tt.tril_()

    zero(mma_TD);
    {
        auto &k_reg_col = swap_layout_inplace(k_reg);
        mma_AB(mma_TD, tt_reg, k_reg_col, mma_TD); // mma_TD = einsum('nst,ntk->nsk', tt, k)
        k_reg = swap_layout_inplace(k_reg_col);
    }
    copy(v_reg, mma_TD); // reuse v_reg for d_q

    if (1) {
        load(tk_reg, _d_q + time_warp_index*tk_reg.num_elements, tk_reg.cols);
        add(v_reg, v_reg, tk_reg);
    }
    store(_d_q + time_warp_index*q_reg.num_elements, v_reg, v_reg.cols);

    load(q_reg, _q + time_warp_index*q_reg.num_elements, q_reg.cols); // q = q.clone()

    zero(mma_TD);
    {
        auto &tt_reg_col = swap_layout_inplace(tt_reg);
        mma_AtB(mma_TD, tt_reg_col, q_reg, mma_TD); // mma_TD = einsum('nst,nsk->ntk', tt, q)
        copy(d_k_reg, mma_TD);
        tt_reg = swap_layout_inplace(tt_reg_col);
    }

    if (1) {
        load(tk_reg, _d_k + time_warp_index*tk_reg.num_elements, tk_reg.cols);
        add(d_k_reg, d_k_reg, tk_reg);
    }
    store(_d_k, d_k_reg, d_k_reg.cols); // first part of d_k

    zero(mma_TT);
    {
        auto &q_reg_row = swap_layout_inplace(q_reg);
        mma_ABt(mma_TT, q_reg_row, k_reg, mma_TT); // mma_TT = einsum('nsk,ntk->nst', q, k)
        //q_reg = swap_layout_inplace(q_reg_row);
    }
    copy(tt_reg, mma_TT);
    make_causal(tt_reg, tt_reg, 0); // tt.tril_()

    auto &d_out_u_reg = v_reg;
    zero(d_out_u_reg);
    load(d_out_u_reg, _d_out_u + time_warp_index*d_out_u_reg.num_elements, d_out_u_reg.cols); // d_out_u = d_out_u.clone()

    zero(mma_TD);
    {
        auto &tt_reg_col = swap_layout_inplace(tt_reg);
        auto &d_out_y_col = swap_layout_inplace(d_out_y_reg);
        mma_AtB(mma_TD, tt_reg_col, d_out_y_col, mma_TD);
        copy(tk_reg, mma_TD);
        add(d_out_u_reg, d_out_u_reg, tk_reg);

        tt_reg = swap_layout_inplace(tt_reg_col);
        //d_out_y = swap_layout_inplace(d_out_y_col);
    }

    /*
     * backward for d_k, d_v, d_beta
     */

    load(d_out_w_reg, _d_out_w + time_warp_index*d_out_w_reg.num_elements, d_out_w_reg.cols); // d_out_w = d_out_w.clone()
    zero(d_k_reg); // note that we have a part of d_k in global memory now

    for (auto t = _time * TILE_DIM - 1; t >= 0; t--) {
        __syncthreads();

        auto &k_reg_col = swap_layout_inplace(k_reg);

        // d_k
        zero(mma_TD);
        {
            zero(mma_TT);
            mma_ABt(mma_TT, w_reg, d_out_w_reg, mma_TT);
            copy(tt_reg, mma_TT);
            reset_trailing_rows(tt_reg, t);

            {
                auto &tt_reg_col = swap_layout_inplace(tt_reg);
                mma_AtB(mma_TD, tt_reg_col, k_reg_col, mma_TD);
                tt_reg = swap_layout_inplace(tt_reg_col);
            }

            zero(mma_TT);
            mma_ABt(mma_TT, u_reg, d_out_u_reg, mma_TT);
            copy(tt_reg, mma_TT);
            reset_trailing_rows(tt_reg, t);

            {
                auto &tt_reg_col = swap_layout_inplace(tt_reg);
                mma_AtB(mma_TD, tt_reg_col, k_reg_col, mma_TD);
                tt_reg = swap_layout_inplace(tt_reg_col);
                copy(tk_reg, mma_TD);
            }
   
            op_singlerow<base_ops::sum>(d_k_reg, d_k_reg, tk_reg, t);
        }

        k_reg = swap_layout_inplace(k_reg_col);

        // backpropagate through time, updating only remaining timestamps
        {
            zero(tt_reg);
            op_singlerow<base_ops::sum>(tt_reg, tt_reg, bKl_reg, t);
            auto &tt_reg_col = swap_layout_inplace(tt_reg);

            {
                zero(mma_TD);
                {
                    auto &d_out_w_reg_col = swap_layout_inplace(d_out_w_reg);
                    mma_AtB(mma_TD, tt_reg_col, d_out_w_reg_col, mma_TD);
                    d_out_w_reg = swap_layout_inplace(d_out_w_reg_col);
                    copy(tk_reg, mma_TD);
                }

                sub(d_out_w_reg, d_out_w_reg, tk_reg);

                zero(mma_TD);
                {
                    auto &d_out_u_reg_col = swap_layout_inplace(d_out_u_reg);
                    mma_AtB(mma_TD, tt_reg_col, d_out_u_reg_col, mma_TD);
                    d_out_u_reg = swap_layout_inplace(d_out_u_reg_col);
                    copy(tk_reg, mma_TD);
                }

                sub(d_out_u_reg, d_out_u_reg, tk_reg);
            }

            tt_reg = swap_layout_inplace(tt_reg_col);
        }

    }

    __syncthreads();

    sub(d_k_reg, d_out_w_reg, d_k_reg); // d_k = d_out_w - d_k
    mul_row(d_k_reg, d_k_reg, beta_reg); // d_k = einsum('ntk,nt->ntk', d_k, beta)

    // decay w and u
    zero(mma_TT);
    mma_ABt(mma_TT, d_out_w_reg, w_reg, mma_TT);
    mma_ABt(mma_TT, d_out_u_reg, u_reg, mma_TT);
    copy(tt_reg, mma_TT);
    make_causal(tt_reg, tt_reg, 0);
    set_diagonal(tt_reg, tt_reg, 0);

    zero(mma_TD);
    {
        auto &tt_reg_col = swap_layout_inplace(tt_reg);
        auto &bk_reg_col = swap_layout_inplace(bk_reg);
        mma_AtB(mma_TD, tt_reg_col, bk_reg_col, mma_TD);
        copy(tk_reg, mma_TD);
    }
    sub(d_k_reg, d_k_reg, tk_reg);

    load(tk_reg, _d_k + time_warp_index*d_k_reg.num_elements, d_k_reg.cols); // d_k = d_k.clone()
    __syncthreads();
    add(d_k_reg, d_k_reg, tk_reg); // d_k += tk, reload from global memory
    store(_d_k, d_k_reg, d_k_reg.cols);

    // d_beta
    mul(w_bases_reg, w_bases_reg, d_out_w_reg); // w_bases = einsum('ntk,ntk->ntk', w_bases, d_out_w)
    mul(u_bases_reg, u_bases_reg, d_out_u_reg); // u_bases = einsum('ntw,ntw->ntw', u_bases, d_out_u)

    // d_v using available d_out_u_reg register
    mul_row(d_out_u_reg, d_out_u_reg, beta_reg);
    store(_d_v, d_out_u_reg, d_out_u_reg.cols);

    // continue d_beta
    auto &w_bases_col = swap_layout_inplace(w_bases_reg);
    auto &u_bases_col = swap_layout_inplace(u_bases_reg);
    zero(d_beta_reg);
    row_sum(d_beta_reg, w_bases_col); // d_beta = einsum('tk->t', w_bases);
    row_sum(d_beta_reg, u_bases_col, d_beta_reg); // d_beta += einsum('tw->t', u_bases);
    store(_d_beta, d_beta_reg);
}

// see also: DISPATCH
#define TYPE_DISPATCH(scalar_type, FUNC)\
    switch (scalar_type) {\
        case c10::ScalarType::BFloat16: {\
            using H = c10::BFloat16;\
            using T = bf16;\
            using D = bf16_2;\
            using ACCUM = float2;\
            FUNC;\
        }\
            break;\
        default:\
            TORCH_CHECK(false, "Unsupported type! Try bfloat16");\
    }

#define DISPATCH_ME(d, seqlen) \
    if (d == 16) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 1, 1)); \
    } else if (d == 32) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 1)); \
    } else if (d == 64) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 4, 1)); \
    } else if (d == 128) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 8, 1)); \
    } else { \
        TORCH_CHECK(false, "[qkv].size(2) should be 16, 32, 64 or 128"); \
    }

void
forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
    torch::Tensor w, torch::Tensor u, torch::Tensor y
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(y);

    auto scalar_type = k.scalar_type();
    TORCH_CHECK(q.scalar_type() == scalar_type, "q type mismatch");
    TORCH_CHECK(v.scalar_type() == scalar_type, "v type mismatch");
    TORCH_CHECK(beta.scalar_type() == scalar_type, "beta type mismatch");
    TORCH_CHECK(w.scalar_type() == scalar_type, "w type mismatch");
    TORCH_CHECK(u.scalar_type() == scalar_type, "u type mismatch");
    TORCH_CHECK(y.scalar_type() == scalar_type, "y type mismatch");

    auto batch  = k.size(0);
    auto seqlen = k.size(1);
    auto d      = k.size(2);
    bool same = true;
    for(auto i = 0; i < 3; i++) { 
        same &= k.size(i) == v.size(i);
        same &= q.size(i) == v.size(i);
    }
    TORCH_CHECK(same, "Q, K and V should be same size");
    constexpr int kChunkSize = 16;
    auto num_chunks = seqlen / kChunkSize;

    // kHeight: tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    // kWidth: tiles per vector, 2 means head dimension is 2*16 = 32
#define DELTA_DISPATCH(_kHeight, _kWidth, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth; \
        constexpr int kNumWarps = _kNumWarps; \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        unsigned long mem_size = kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>); \
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(forward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size)); \
        forward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps, kChunkSize><<<batch,threads,mem_size>>>( \
            (int)num_chunks, \
            q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(), \
            w.data_ptr<H>(), u.data_ptr<H>(), y.data_ptr<H>())

    DISPATCH_ME(d, seqlen);
#undef DELTA_DISPATCH
}


void
backward(
    torch::Tensor d_out_w, torch::Tensor d_out_u, torch::Tensor d_out_y,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
    torch::Tensor d_q, torch::Tensor d_k, torch::Tensor d_v, torch::Tensor d_beta,
    torch::Tensor w, torch::Tensor u, torch::Tensor y
) {
    CHECK_INPUT(d_out_w);
    CHECK_INPUT(d_out_u);
    CHECK_INPUT(d_out_y);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);
    CHECK_INPUT(d_k);
    CHECK_INPUT(d_v);
    CHECK_INPUT(d_beta);
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(y);

    auto scalar_type = d_out_w.scalar_type();
    TORCH_CHECK(d_out_u.scalar_type() == scalar_type, "d_out_u type mismatch");
    TORCH_CHECK(d_out_y.scalar_type() == scalar_type, "d_out_y type mismatch");
    TORCH_CHECK(q.scalar_type() == scalar_type, "q type mismatch");
    TORCH_CHECK(k.scalar_type() == scalar_type, "k type mismatch");
    TORCH_CHECK(v.scalar_type() == scalar_type, "v type mismatch");
    TORCH_CHECK(beta.scalar_type() == scalar_type, "beta type mismatch");
    TORCH_CHECK(d_k.scalar_type() == scalar_type, "d_k type mismatch");
    TORCH_CHECK(d_v.scalar_type() == scalar_type, "d_v type mismatch");
    TORCH_CHECK(d_beta.scalar_type() == scalar_type, "d_beta type mismatch");
    TORCH_CHECK(w.scalar_type() == scalar_type, "w type mismatch");
    TORCH_CHECK(u.scalar_type() == scalar_type, "u type mismatch");
    TORCH_CHECK(y.scalar_type() == scalar_type, "y type mismatch");

    auto batch_head = k.size(0);
    auto seqlen = k.size(1);
    auto d      = k.size(2);
    bool same = true;
    for(auto i = 0; i < 3; i++) { 
        same &= k.size(i) == v.size(i);
        same &= q.size(i) == v.size(i);
    }
    TORCH_CHECK(same, "Q, K and V should be same size");
    constexpr int kChunkSize = 16;
    auto num_chunks = seqlen / kChunkSize;

    // kHeight: tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    // kWidth: tiles per vector, 2 means head dimension is 2*16 = 32
#define DELTA_DISPATCH(_kHeight, _kWidth, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth; \
        constexpr int kNumWarps = _kNumWarps; \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        unsigned long mem_size = kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>); \
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(backward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size)); \
        backward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps, kChunkSize><<<batch_head,threads,mem_size>>>( \
            (int)num_chunks, \
            d_out_w.data_ptr<H>(), d_out_u.data_ptr<H>(), d_out_y.data_ptr<H>(), \
            q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(), \
            d_q.data_ptr<H>(), d_k.data_ptr<H>(), d_v.data_ptr<H>(), d_beta.data_ptr<H>(), \
            w.data_ptr<H>(), u.data_ptr<H>(), y.data_ptr<H>())

    DISPATCH_ME(d, seqlen);
#undef DELTA_DISPATCH
}
