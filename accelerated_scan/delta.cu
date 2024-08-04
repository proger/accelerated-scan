#include <cooperative_groups.h>
#include <cuda/barrier>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "algebra.cu"

using namespace kittens; // this kernel only handles headdim=q_reg.cols for simplicity. Also n should be a multiple of 256 here.
using barrier = cuda::barrier<cuda::thread_scope_block>;

template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
static __device__ inline void decay_values_forward(
    rt<D, _time, _time> &tt_reg,
    rt<D, _time, _time> &bKl_reg,
    rt<D, _time, _key> &k_reg,
    typename rt<D, _time, _key>::col_vec &beta_reg,
    rt<D, _time, _key> &w_reg,
    rt<D, _time, _value> &u_reg,
    rt<D, _time, _value> &v_reg,
    rt<D, _time, _value> &u_bases_reg, // can be aliased to v_reg
    rt<D, _time, _key> &bk_reg
) {
    rt<D, _time, _key> tk_reg;
    rt<D, _time, _value> tv_reg;

    rt<D, _time, _time> tt;
    rv<D, _time, 2> decay;
    rv<D, _time, 1> decay_1;
    rv<D, _key, 2> decay_w;
    rv<D, _value, 2> decay_u;

    mul_row(bk_reg, k_reg, beta_reg); // k = einsum('ntk,nt->ntk', k, beta)

    kernel(tt_reg, k_reg, k_reg);
    make_causal(tt_reg, tt_reg, 0);
    set_diagonal(tt_reg, tt_reg, 0); // tt = tt.tril(diagnal=-1)

    mul_row(bKl_reg, tt_reg, beta_reg); // tt = einsum('nts,nt->nts', tt, beta)

    copy(u_bases_reg, v_reg);
    mul_row(v_reg, v_reg, beta_reg); // v = einsum('ntw,nt->ntw', v, beta)

    copy(w_reg, bk_reg);
    copy(u_reg, v_reg);

    #pragma unroll
    for (auto t = 1; t < w_reg.rows; t++) {
        // select row t from bKl
        zeroexcept(tt, bKl_reg, t);
        col_sum(decay, tt);
        copy(decay_1, decay);

        // ALT: reverse_query(tk_reg, bKl_reg, w_reg);
        broadcast_row(tk_reg, decay_1);
        mul(tk_reg, w_reg, tk_reg);
        col_sum(decay_w, tk_reg); // decay_w is 1xK vector
        broadcast_col(tk_reg, decay_w);

        op_singlerow<base_ops::sub>(w_reg, bk_reg, tk_reg, t); // w[t] = bk[t] - tk[t]

        // ALT: reverse_query(tv_reg, bKl_reg, u_reg);
        broadcast_row(tv_reg, decay_1);
        mul(tv_reg, u_reg, tv_reg);
        col_sum(decay_u, tv_reg); // decay_u is 1xV vector
        broadcast_col(tv_reg, decay_u);

        op_singlerow<base_ops::sub>(u_reg, v_reg, tv_reg, t); // u[t] = bv[t] - tv[t]
    }
}


template <typename H, typename T, typename D, typename ACCUM, int _time, int _key, int _value, int _value_groups, int kNumWarps = 1, int kChunkSize = 16>
__global__ void delta_forward_kernel(
    const int num_chunks,
    const H* __restrict__ __q__,
    const H* __restrict__ __k__,
    const H* __restrict__ __v__,
    const H* __restrict__ __beta__,
    H* __restrict__ __y__
) {
    auto warpid           = kittens::warpid();
    auto vg               = blockIdx.x % _value_groups;
    auto block_start      = blockIdx.z*(num_chunks*kChunkSize*(_key*TILE_DIM));
    auto beta_block_start = blockIdx.z*(num_chunks*kChunkSize*1); // width is 1 for beta
    const T *_q = reinterpret_cast<const T *>(__q__) + block_start,
            *_k = reinterpret_cast<const T *>(__k__) + block_start,
            *_v = reinterpret_cast<const T *>(__v__) + block_start,
            *_beta = reinterpret_cast<const T *>(__beta__) + beta_block_start;
          T *_y = reinterpret_cast<T *>(__y__) + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    st<T, _value, _key, ducks::st_layout::swizzle> (&shared_states)[1] = al.allocate<st<T, _value, _key, ducks::st_layout::swizzle>, 1>();
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state = shared_states[0];

    using col = rt<D, _time, _key, ducks::rt_layout::col>;
    /*
     * register allocations
     */
    rt<ACCUM, _value, _key> mma_state;
    rt<D, _time, _key> k, w, bk;
    rt<D, _time, _value> v, u;
    typename rt<D, _time, _key>::col_vec beta_reg;
    rt<D, _time, _time> qk;
    constexpr int v_num_elements = v.num_elements * _value_groups;
    constexpr int v_row_stride = v.cols * _value_groups;

    __shared__ barrier batons[kNumWarps];
    auto block = cooperative_groups::this_thread_block();

    init(&batons[warpid], 2);
    if (block.thread_rank() == 0) {
        auto token = batons[0].arrive();
    }

    for (int time_block = 0; time_block < (num_chunks + kNumWarps - 1) / kNumWarps; time_block++) {
        int chunk = time_block * kNumWarps + warpid;
        if (chunk >= num_chunks) {
            break;
        }

        /*
        * load k, v, beta
        */

        load(k, _k + chunk*k.num_elements, k.cols);
        load(beta_reg, _beta + chunk*beta_reg.outer_dim*TILE_DIM);
        load(v, _v + chunk*v_num_elements + v.cols * vg, v_row_stride);

        /*
        * decay_values_forward: compute w and u
        */

        decay_values_forward(qk, qk, k, beta_reg, w, u, v, v, bk);

        /*
        * attend to decayed values
        */

        auto &q = bk;
        load(q, _q + chunk*q.num_elements, q.cols);

        kernel(qk, q, k);
        make_causal(qk, qk, 0);

        auto &y = v;
        attend(y, qk, u);

        chunk_forward(shared_state, mma_state, q, k, w, u, qk, y, chunk, batons[warpid], batons[(warpid + 1) % kNumWarps]);

        store(_y + chunk*v_num_elements + y.cols * vg, y, v_row_stride);
    }
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value>
__device__ static inline void chunk_forward_nooutput(
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state,
    rt<ACCUM, _value, _key> &mma_state,
    rt<D, _time, _key> &k,
    rt<D, _time, _key> &w,
    rt<D, _time, _value> &u, // become decayed
    const int chunk,
    barrier &self,
    barrier &next
) {
    return chunk_forward_impl<T, D, ACCUM, _time, _key, _value, false>(
        shared_state, mma_state, k, w, u, nullptr, nullptr, nullptr, chunk, self, next
    );
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value>
__device__ static inline void chunk_forward(
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state,
    rt<ACCUM, _value, _key> &mma_state,
    rt<D, _time, _key> &q,
    rt<D, _time, _key> &k,
    rt<D, _time, _key> &w,
    rt<D, _time, _value> &u, // become decayed
    rt<D, _time, _time> &qk,
    rt<D, _time, _value> &y,
    const int chunk,
    barrier &self,
    barrier &next
) {
    return chunk_forward_impl<T, D, ACCUM, _time, _key, _value, true>(
        shared_state, mma_state, k, w, u, q, qk, y, chunk, self, next
    );
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value, bool NeedsY>
__device__ static inline void chunk_forward_impl(
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state,
    rt<ACCUM, _value, _key> &mma_state,
    rt<D, _time, _key> &k,
    rt<D, _time, _key> &w, // not necessary for chunk 0
    rt<D, _time, _value> &u, // become decayed
    std::conditional_t<NeedsY, rt<D, _time, _key> &, std::nullptr_t> q,
    std::conditional_t<NeedsY, rt<D, _time, _time> &, std::nullptr_t> qk,
    std::conditional_t<NeedsY, rt<D, _time, _value> &, std::nullptr_t> y,
    const int chunk,
    barrier &self,
    barrier &next
) {
    auto laneid = kittens::laneid();

    rt<D, _value, _key> state;
    rt<D, _time, _value> u_old;
    rt<D, _time, _value> y_buf;

    // all warps execute sequentially, passing state through the shared memory
    if (laneid == 0) {
        // wait until our state is ready
        auto token = self.arrive();
        self.wait(std::move(token));
    }
    __syncwarp();

    if (chunk > 0) {
        load(state, shared_state);
        copy(mma_state, state);

        query(u_old, state, w);
        sub(u, u, u_old);

        if constexpr (NeedsY) {
            query(y_buf, state, q);
        }
    } else {
        zero(mma_state);
    }

    if constexpr (NeedsY) {
        if (chunk > 0) {
            add(y, y, y_buf);

            attend(y_buf, qk, u_old);
            sub(y, y, y_buf);
        }
    }

    associate(state, u, k, mma_state, true);
    store(shared_state, state);

    if (laneid == 0) {
        // data ƒor the next chunk has arrived
        auto token = next.arrive();
    }
    __syncwarp();
}

/**
 * @brief Stitch the chunks backwards
 */
template <typename T, typename D, int _time, int _key, int _value>
__device__ static inline void chunk_backward(
    rt<D, _time, _key> &q, // not needed for chunk 0
    rt<D, _time, _key> &k,
    rt<D, _time, _key> &w, // not needed for chunk 0
    rt<D, _time, _value> &u,
    rt<D, _time, _value> &d_y,
    rt<D, _time, _key> &d_q,
    rt<D, _time, _key> &d_k,
    rt<D, _time, _key> &d_w,
    rt<D, _time, _value> &d_u,
    const int chunk,
    const int num_chunks,
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state,
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_d_state,
    barrier &self,
    barrier &prev
) {
    auto laneid = kittens::laneid();

    rt<D, _time, _time> qk;
    rt<D, _time, _value> d_state_decays, d_state_decays_buf;
    rt<D, _time, _key> tk;
    rt<D, _time, _value> tv;
    rt<D, _value, _key> state, d_state, state_delta;

    // all warps execute sequentially, passing state through the shared memory
    if (laneid == 0) {
        // wait until our state is ready
        auto token = self.arrive();
        self.wait(std::move(token));
    }
    __syncwarp();


    load(state, shared_state);
    load(d_state, shared_d_state);

    if (chunk == 0) {
        zero(d_q);
        reverse_query(d_k, u, d_state);
        zero(d_w);
        query(d_u, d_state, k);
    } else {
        associate(state_delta, u, k);
        /*
         * uncompute the state backwards
         */
        sub(state, state, state_delta);
        query(tv, state, w);

        negate(d_y);

        // d_q, d_k
        kernel(qk, d_y, tv);
        make_causal(qk, qk, 0);

        // d_q
        attend(d_q, qk, k);
        reverse_query(tk, d_y, state);
        sub(d_q, d_q, tk);

        // d_k
        reverse_attend(d_k, qk, q);
        if (chunk < num_chunks - 1) {
            reverse_query(tk, u, d_state); // otherwise we know d_state is zero
            add(d_k, d_k, tk);
        }

        // d_u
        if (chunk < num_chunks - 1) {
            query(d_u, d_state, k);
        } else {
            // otherwise we know d_state is zero
            zero(d_u);
        }

        // d_state_decays
        kernel(qk, q, k);
        make_causal(qk, qk, 0);

        reverse_attend(d_state_decays, qk, d_y);
        if (chunk < num_chunks - 1) {
            query(d_state_decays_buf, d_state, k);
            sub(d_state_decays, d_state_decays, d_state_decays_buf);
        }

        // d_w
        reverse_query(d_w, d_state_decays, state);

        // backpropagate through time
        auto &state_buf = state_delta; // alias
        associate(state_buf, d_y, q);
        sub(d_state, d_state, state_buf);
        associate(state_buf, d_state_decays, w);
        add(d_state, d_state, state_buf);

        negate(d_y); // undo
    }

    store(shared_state, state);
    store(shared_d_state, d_state);

    if (laneid == 0) {
        // gradients ƒor the previous chunk have arrived
        auto token = prev.arrive();
    }
    __syncwarp();
}


template <typename H>
struct DeltaBackwardArgs {
    unsigned long long num_chunks;
    const H* __restrict__ __d_out_y__;
    const H* __restrict__ __q__;
    const H* __restrict__ __k__;
    const H* __restrict__ __v__;
    const H* __restrict__ __beta__;
    H* __restrict__ __d_q__;
    H* __restrict__ __d_k__;
    H* __restrict__ __d_v__;
    H* __restrict__ __d_beta__;
    H* __restrict__ __u__;
    unsigned long long* __locks__;
};

template <typename H, typename T, typename D, typename ACCUM, int _time, int _key, int _value, int _value_groups, int kNumWarps = 8, int kChunkSize = 16>
__global__ void delta_backward_kernel(DeltaBackwardArgs<H> args) {
    const int num_chunks = args.num_chunks;
    auto warpid           = kittens::warpid();
    auto laneid           = kittens::laneid();
    auto vg               = blockIdx.x % _value_groups;
    auto block_start      = blockIdx.z*(num_chunks*kChunkSize*(_key*TILE_DIM));
    auto beta_block_start = blockIdx.z*(num_chunks*kChunkSize*1); // width is 1 for beta
    const T *_d_out_y = reinterpret_cast<const T *>(args.__d_out_y__) + block_start,
            *_q = reinterpret_cast<const T *>(args.__q__) + block_start,
            *_k = reinterpret_cast<const T *>(args.__k__) + block_start,
            *_v = reinterpret_cast<const T *>(args.__v__) + block_start,
            *_beta = reinterpret_cast<const T *>(args.__beta__) + beta_block_start;
          T *_d_q = reinterpret_cast<T *>(args.__d_q__) + block_start,
            *_d_k = reinterpret_cast<T *>(args.__d_k__) + block_start,
            *_d_v = reinterpret_cast<T *>(args.__d_v__) + block_start,
            *_d_beta = reinterpret_cast<T *>(args.__d_beta__) + beta_block_start,
            *_u = reinterpret_cast<T *>(args.__u__) + block_start;
    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st<T, _value, _key, ducks::st_layout::swizzle> (&shared_states)[2] = al.allocate<st<T, _value, _key, ducks::st_layout::swizzle>, 2>();
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state = shared_states[0];
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_d_state = shared_states[1];

    /*
     * register allocations
     */
    rt<D, _time, _key> q, d_q;
    rt<D, _time, _key> k_reg, d_w, d_k;
    rt<D, _time, _value> v_reg;
    rt<D, _time, _value> d_y, d_u;
    typename rt<D, _time, _key>::col_vec beta_reg;
    rt<D, _time, _key> w_reg, bk_reg;
    rt<D, _time, _value> u_reg, u_bases_reg, u_predecay_reg;
    rt<D, _time, _time> tt_reg, bKl_reg;

    rt<ACCUM, _value, _key> mma_state;
    rt<ACCUM, _time, _key> mma_TD;
    rt<ACCUM, _time, _time> mma_TT;
    rt<ACCUM, _time, _value> mma_TV;
    constexpr int v_num_elements = v_reg.num_elements * _value_groups;
    constexpr int v_row_stride = v_reg.cols * _value_groups;

    __shared__ barrier batons[kNumWarps];
    __shared__ barrier backward_batons[kNumWarps];
    auto block = cooperative_groups::this_thread_block();

    init(&batons[warpid], 2);
    init(&backward_batons[warpid], 2);
    if (block.thread_rank() == 0) {
        auto token = batons[0].arrive();
    }

    zero(shared_d_state);
    const int time_blocks = (num_chunks + kNumWarps - 1) / kNumWarps;

    for (int time_block = 0; time_block < time_blocks; time_block++) {
        const int chunk = time_block * kNumWarps + warpid;
        if (chunk >= num_chunks) {
            break;
        }

        /*
         * load k, v, beta
         */

        load(k_reg, _k + chunk*k_reg.num_elements, k_reg.cols);
        load(beta_reg, _beta + chunk*beta_reg.outer_dim*TILE_DIM);
        load(v_reg, _v + chunk*v_num_elements + v_reg.cols * vg, v_row_stride);

        /*
         * decay_values_forward: compute w and u
         */

        decay_values_forward(tt_reg, bKl_reg, k_reg, beta_reg, w_reg, u_reg, v_reg, u_bases_reg, bk_reg);

        barrier &forward_self = batons[warpid];
        barrier &forward_next = chunk == num_chunks - 1 ? backward_batons[warpid] : batons[(warpid + 1) % kNumWarps];

        copy(u_predecay_reg, u_reg); // chunk_forward_nooutput will mutate u_reg
        chunk_forward_nooutput(shared_state, mma_state, k_reg, w_reg, u_reg, chunk, forward_self, forward_next);
        store(_u + chunk*v_num_elements + u_reg.cols * vg, u_reg, v_row_stride); // store decayed u
    }

    for (int time_block = time_blocks - 1; time_block >= 0; time_block--) {
        const int chunk = time_block * kNumWarps + warpid;
        if (chunk >= num_chunks) {
            continue;
        }

        if (time_block < time_blocks - 1) {
            // reload
            load(k_reg, _k + chunk*k_reg.num_elements, k_reg.cols);
            load(beta_reg, _beta + chunk*beta_reg.outer_dim*TILE_DIM);
            load(v_reg, _v + chunk*v_num_elements + v_reg.cols * vg, v_row_stride);

            // recompute w_reg and u_reg
            decay_values_forward(tt_reg, bKl_reg, k_reg, beta_reg, w_reg, u_reg, v_reg, u_bases_reg, bk_reg);
            copy(u_predecay_reg, u_reg); // load will mutate u_reg

            load(u_reg, _u + chunk*v_num_elements + u_reg.cols * vg, v_row_stride);
        }

        load(q, _q + chunk*q.num_elements, q.cols);
        load(d_y, _d_out_y + chunk*v_num_elements + d_y.cols * vg, v_row_stride);

        barrier &backward_self = backward_batons[warpid];
        const int previous = ((warpid - 1) % kNumWarps + kNumWarps) % kNumWarps;
        barrier &backward_prev = backward_batons[previous];

        chunk_backward(
            q, k_reg, w_reg, u_reg,
            d_y,
            d_q, d_k, d_w, d_u,
            chunk, num_chunks,
            shared_state, shared_d_state,
            backward_self, backward_prev
        );

        copy(u_reg, u_predecay_reg); // restore non-decayed u

        rt<D, _time, _key, ducks::rt_layout::col> &q_reg = swap_layout_inplace(q);
        decay_values_backward<T, D, ACCUM, _time, _key, _value, _value_groups, kNumWarps>(
            q_reg, k_reg, w_reg, tt_reg, bk_reg, u_reg, u_bases_reg, bKl_reg, beta_reg,
            d_y,
            d_q, d_k, d_w, d_u,
            mma_TD, mma_TT, mma_TV,
            _d_out_y,
            _d_q, _d_k, _d_v, _d_beta,
            chunk
        );

        swap_layout_inplace(q_reg);
    }
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value, int _value_groups, int kNumWarps>
__device__ static inline void decay_values_backward(
    rt<D, _time, _key, ducks::rt_layout::col> &q_reg,
    rt<D, _time, _key> &k_reg,
    rt<D, _time, _key> &w_reg,
    rt<D, _time, _time> &tt_reg,
    rt<D, _time, _key> &bk_reg,
    rt<D, _time, _value> &u_reg,
    rt<D, _time, _value> &u_bases_reg,
    rt<D, _time, _time> &bKl_reg,
    rv<D, _time, 1> &beta_reg,
    rt<D, _time, _value> &d_out_y_reg,
    rt<D, _time, _key> &d_q,
    rt<D, _time, _key> &d_k,
    rt<D, _time, _key> &d_w_reg,
    rt<D, _time, _value> &d_u_reg,
    rt<ACCUM, _time, _key> &mma_TD,
    rt<ACCUM, _time, _time> &mma_TT,
    rt<ACCUM, _time, _value> &mma_TV,
    const T *_d_out_y,
    T *_d_q,
    T *_d_k,
    T *_d_v,
    T *_d_beta,
    const int chunk
) {
    auto vg = blockIdx.x % _value_groups;
    constexpr int v_num_elements = u_reg.num_elements * _value_groups;
    constexpr int v_row_stride = u_reg.cols * _value_groups;

    rt<D, _time, _key> w_bases_reg, tk_reg, d_k_reg, d_out_w_reg;
    rt<D, _time, _value> v_reg, tv_reg;
    rv<D, _time, 2> d_beta_reg;
    rv<D, _time, 2> d_beta_buf_reg;

    attend(w_bases_reg, tt_reg, w_reg);
    sub(w_bases_reg, k_reg, w_bases_reg);

    attend(v_reg, tt_reg, u_reg);
    sub(u_bases_reg, u_bases_reg, v_reg);

    /*
     * causal_attend_backward for d_q, d_k_2, d_u
     */


    // d_q
    kernel(tt_reg, d_out_y_reg, u_reg);
    make_causal(tt_reg, tt_reg, 0);
    attend(tk_reg, tt_reg, k_reg);
    add(d_q, d_q, tk_reg);

    auto g = cooperative_groups::this_grid();

    {
        #pragma unroll
        for (int vchunk = 0; vchunk < _value_groups; vchunk++) {
            if (vg == vchunk) {
                load(tk_reg, _d_q + chunk*tk_reg.num_elements, tk_reg.cols);
                add(d_q, d_q, tk_reg);
                store(_d_q + chunk*d_q.num_elements, d_q, d_q.cols);
            }
            g.sync();
        }
    }

    // d_k

    reverse_attend(d_k_reg, tt_reg, q_reg);
    add(d_k, d_k, d_k_reg);

    auto &q_reg_row = swap_layout_inplace(q_reg);
    kernel(tt_reg, q_reg_row, k_reg);
    //q_reg = swap_layout_inplace(q_reg_row); // won't need it later
    make_causal(tt_reg, tt_reg, 0); // tt.tril_()

    reverse_attend(tv_reg, tt_reg, d_out_y_reg); // don't need last swap_layout_inplace of d_out_y_reg
    add(d_u_reg, d_u_reg, tv_reg);

    /*
    * backward for d_k, d_v, d_beta
    */

    zero(d_k_reg);

    for (auto t = _time * TILE_DIM - 1; t >= 0; t--) {
        __syncthreads();

        auto &k_reg_col = swap_layout_inplace(k_reg);

        // d_k
        zero(mma_TD);
        {
            kernel(tt_reg, w_reg, d_w_reg);
            reset_trailing_rows(tt_reg, t);

            reverse_attend(tt_reg, k_reg_col, mma_TD, true);

            kernel(tt_reg, u_reg, d_u_reg);
            reset_trailing_rows(tt_reg, t);

            reverse_attend(tt_reg, k_reg_col, mma_TD, true);
            copy(tk_reg, mma_TD);

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
                    auto &d_w_reg_col = swap_layout_inplace(d_w_reg);
                    mma_AtB(mma_TD, tt_reg_col, d_w_reg_col, mma_TD);
                    d_w_reg = swap_layout_inplace(d_w_reg_col);
                    copy(tk_reg, mma_TD);
                }

                sub(d_w_reg, d_w_reg, tk_reg);

                zero(mma_TV);
                {
                    auto &d_u_reg_col = swap_layout_inplace(d_u_reg);
                    mma_AtB(mma_TV, tt_reg_col, d_u_reg_col, mma_TV);
                    d_u_reg = swap_layout_inplace(d_u_reg_col);
                    copy(tv_reg, mma_TV);
                }

                sub(d_u_reg, d_u_reg, tv_reg);
            }

            tt_reg = swap_layout_inplace(tt_reg_col);
        }

    }

    __syncthreads();

    sub(d_k_reg, d_w_reg, d_k_reg); // d_k = d_w - d_k
    mul_row(d_k_reg, d_k_reg, beta_reg); // d_k = einsum('ntk,nt->ntk', d_k, beta)

    // decay w and u
    zero(mma_TT);
    mma_ABt(mma_TT, d_w_reg, w_reg, mma_TT);
    mma_ABt(mma_TT, d_u_reg, u_reg, mma_TT);
    copy(tt_reg, mma_TT);
    make_causal(tt_reg, tt_reg, 0);
    set_diagonal(tt_reg, tt_reg, 0);

    zero(mma_TD);
    {
        auto &tt_reg_col = swap_layout_inplace(tt_reg);
        auto &bk_reg_col = swap_layout_inplace(bk_reg); // don't need the swap later
        mma_AtB(mma_TD, tt_reg_col, bk_reg_col, mma_TD);
        copy(tk_reg, mma_TD);
    }
    sub(d_k_reg, d_k_reg, tk_reg);
    add(d_k, d_k, d_k_reg);

    {
        #pragma unroll
        for (int vchunk = 0; vchunk < _value_groups; vchunk++) {
            if (vg == vchunk) {
                load(tk_reg, _d_k + chunk*d_k.num_elements, d_k.cols);
                add(d_k, d_k, tk_reg);
                store(_d_k + chunk*d_k.num_elements, d_k, d_k.cols);
            }
            g.sync();
        }
    }

    // d_beta
    mul(w_bases_reg, w_bases_reg, d_w_reg); // w_bases = einsum('ntk,ntk->ntk', w_bases, d_w)
    mul(u_bases_reg, u_bases_reg, d_u_reg); // u_bases = einsum('ntw,ntw->ntw', u_bases, d_u)

    // d_v using available d_u_reg register
    mul_row(d_u_reg, d_u_reg, beta_reg);
    store(_d_v + chunk*v_num_elements + d_u_reg.cols * vg, d_u_reg, v_row_stride);

    // continue d_beta
    auto &w_bases_col = swap_layout_inplace(w_bases_reg);
    auto &u_bases_col = swap_layout_inplace(u_bases_reg);
    zero(d_beta_reg);
    row_sum(d_beta_reg, w_bases_col); // d_beta = einsum('tk->t', w_bases);
    row_sum(d_beta_reg, u_bases_col, d_beta_reg); // d_beta += einsum('tw->t', u_bases);

    {
        #pragma unroll
        for (int vchunk = 0; vchunk < _value_groups; vchunk++) {
            if (vg == vchunk) {
                load(d_beta_buf_reg, _d_beta + chunk*beta_reg.outer_dim*TILE_DIM);
                add(d_beta_reg, d_beta_reg, d_beta_buf_reg);
                store(_d_beta + chunk*beta_reg.outer_dim*TILE_DIM, d_beta_reg);
            }
            g.sync();
        }
    }
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
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 1, 1, 8)); \
    } else if (d == 32) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 1, 8)); \
    } else if (d == 64) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 2, 4)); \
    } else if (d == 128) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 4, 2)); \
    } else if (d == 256) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 8, 2)); \
    } else if (d == 512) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 16, 2)); \
    } else if (d == 1024) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 32, 2)); \
    } else { \
        TORCH_CHECK(false, "[qkv].size(2) should be 16, 32, 64, 128, 256, 512 or 1024"); \
    }

#define DISPATCH_ME_FLAT(d, seqlen) \
    if (d == 16) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 1, 1, 8)); \
    } else if (d == 32) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 1, 8)); \
    } else if (d == 64) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 2, 8)); \
    } else if (d == 128) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 4, 2)); \
    } else { \
        TORCH_CHECK(false, "[qkv].size(2) should be 16, 32, 64, 128"); \
    }

void
forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
    torch::Tensor y
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);
    CHECK_INPUT(y);

    auto scalar_type = k.scalar_type();
    TORCH_CHECK(q.scalar_type() == scalar_type, "q type mismatch");
    TORCH_CHECK(v.scalar_type() == scalar_type, "v type mismatch");
    TORCH_CHECK(beta.scalar_type() == scalar_type, "beta type mismatch");
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
#define DELTA_DISPATCH(_kHeight, _kWidth, _kWidthGroups, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth; \
        constexpr int kWidthGroups = _kWidthGroups; \
        constexpr int kNumWarps = _kNumWarps; \
        constexpr int kKey = kWidth * kWidthGroups; \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        dim3 blocks(kWidthGroups, 1, batch); \
        unsigned long mem_size = sizeof(st<T, kWidth, kKey, ducks::st_layout::swizzle>) + kNumWarps*sizeof(barrier); \
        delta_forward_kernel<H, T, D, ACCUM, kHeight, kKey, kWidth, kWidthGroups, kNumWarps, kChunkSize><<<blocks,threads,mem_size>>>( \
            (int)num_chunks, \
            q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(), \
            y.data_ptr<H>())

    DISPATCH_ME(d, seqlen);
#undef DELTA_DISPATCH
}


void
backward(
    torch::Tensor d_out_y,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
    torch::Tensor d_q, torch::Tensor d_k, torch::Tensor d_v, torch::Tensor d_beta,
    torch::Tensor u
) {
    CHECK_INPUT(d_out_y);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);
    CHECK_INPUT(d_k);
    CHECK_INPUT(d_v);
    CHECK_INPUT(d_beta);
    CHECK_INPUT(u);

    auto scalar_type = d_out_y.scalar_type();
    TORCH_CHECK(q.scalar_type() == scalar_type, "q type mismatch");
    TORCH_CHECK(k.scalar_type() == scalar_type, "k type mismatch");
    TORCH_CHECK(v.scalar_type() == scalar_type, "v type mismatch");
    TORCH_CHECK(beta.scalar_type() == scalar_type, "beta type mismatch");
    TORCH_CHECK(d_k.scalar_type() == scalar_type, "d_k type mismatch");
    TORCH_CHECK(d_v.scalar_type() == scalar_type, "d_v type mismatch");
    TORCH_CHECK(d_beta.scalar_type() == scalar_type, "d_beta type mismatch");
    TORCH_CHECK(u.scalar_type() == scalar_type, "u type mismatch");

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
    unsigned long long num_chunks = seqlen / kChunkSize;
    // TORCH_CHECK(num_chunks <= 16, "num_chunks should be <= 32 (chunk size is 16)");

    // kHeight: tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    // kWidth: tiles per vector, 2 means head dimension is 2*16 = 32
#define DELTA_DISPATCH(_kHeight, _kWidth, _kWidthGroups, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth; \
        constexpr int kWidthGroups = _kWidthGroups; \
        constexpr int kNumWarps = _kNumWarps; \
        constexpr int kKey = kWidth * kWidthGroups; \
        unsigned long long *locks; \
        cudaMalloc(&locks, batch * kNumWarps * sizeof(unsigned long long)); \
        cudaMemset(locks, 0, batch * kNumWarps * sizeof(unsigned long long)); \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        dim3 gridDim(kWidthGroups, 1, batch); \
        dim3 blockDim(threads, 1, 1); \
        size_t mem_size = 2*sizeof(st<T, kWidth, kWidth*kWidthGroups, ducks::st_layout::swizzle>) + 2*kNumWarps*sizeof(barrier); \
        auto kernel = delta_backward_kernel<H, T, D, ACCUM, kHeight, kKey, kWidth, kWidthGroups, kNumWarps>; \
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(delta_backward_kernel<H, T, D, ACCUM, kHeight, kKey, kWidth, kWidthGroups, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size)); \
        DeltaBackwardArgs<H> args = {num_chunks, \
            d_out_y.data_ptr<H>(), \
            q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(), \
            d_q.data_ptr<H>(), d_k.data_ptr<H>(), d_v.data_ptr<H>(), d_beta.data_ptr<H>(), \
            u.data_ptr<H>(), locks}; \
        void *args_ptr[] = {&args}; \
        cudaLaunchCooperativeKernel((const void *)&delta_backward_kernel<H, T, D, ACCUM, kHeight, kKey, kWidth, kWidthGroups, kNumWarps>, gridDim, blockDim, args_ptr, mem_size, at::cuda::getCurrentCUDAStream().stream()); \
        cudaFree(locks)

    DISPATCH_ME_FLAT(d, seqlen);
#undef DELTA_DISPATCH
}
