#include <cooperative_groups.h>
#include <cuda/barrier>
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
    set_diagonal(tt_reg, tt_reg, 0); // tt = tt.tril(diagonal=-1)

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
    H* __restrict__ __w__,
    H* __restrict__ __u__,
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
          T *_w = reinterpret_cast<T *>(__w__) + block_start,
            *_u = reinterpret_cast<T *>(__u__) + block_start,
            *_y = reinterpret_cast<T *>(__y__) + block_start;

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

        load(k, _k + chunk*k.num_elements, k.cols); // k = k.clone()
        load(beta_reg, _beta + chunk*beta_reg.outer_dim*TILE_DIM); // beta = beta.clone()
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

        loop(shared_state, mma_state, q, k, qk, w, u, y, chunk, batons);

        store(_y + chunk*v_num_elements + v.cols * vg, y, v_row_stride);
    }
}

template <typename T, typename D, typename ACCUM, int _time, int _key, int _value, int kNumWarps>
__device__ static inline void loop(
    st<T, _value, _key, ducks::st_layout::swizzle> &shared_state,
    rt<ACCUM, _value, _key> &mma_state,
    rt<D, _time, _key> &q,
    rt<D, _time, _key> &k,
    rt<D, _time, _time> &qk,
    rt<D, _time, _key> &w, // will be repurposed
    rt<D, _time, _value> &u,
    rt<D, _time, _value> &y,
    const int chunk,
    barrier (&batons)[kNumWarps]
) {
    auto warpid = kittens::warpid();
    auto laneid = kittens::laneid();

    rt<D, _value, _key> state;
    rt<D, _time, _value> u_old;
    rt<D, _time, _value> y_buf;

    // all warps execute sequentially, passing state through the shared memory
    if (laneid == 0) {
        // wait until our state is ready
        barrier::arrival_token token = batons[warpid].arrive();
        batons[warpid].wait(std::move(token));
    }
    __syncwarp();

    if (chunk > 0) {
        load(state, shared_state);
        copy(mma_state, state);

        query(u_old, state, w);
        sub(u, u, u_old);

        query(y_buf, state, q);
    } else {
        zero(mma_state);
    }

    if (chunk > 0) {
        add(y, y, y_buf);

        attend(y_buf, qk, u_old);
        sub(y, y, y_buf);
    }

    associate(state, u, k, mma_state, true);
    store(shared_state, state);

    if (laneid == 0) {
        // data ƒor the next chunk has arrived
        auto token = batons[(warpid + 1) % kNumWarps].arrive();
    }
    __syncwarp();
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
    
    associate(state, u, k);

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
        load(q, _q + chunk*q.num_elements, q.cols);

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
        reverse_attend(d_k, qk, q);
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
        kernel(qk, q, k);
        make_causal(qk, qk, 0);

        reverse_attend(d_state_decays, qk, d_y);
        if (chunk < num_chunks - 1) {
            auto &d_state_decays_buf = d_u; // alias
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
__global__ void delta_backward_kernel(
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

    for (int time_block = 0; time_block < (num_chunks + kNumWarps - 1) / kNumWarps; time_block++) {
        int time_index = time_block * kNumWarps + warpid;
        if (time_index >= num_chunks) {
            break;
        }

        /*
        * load k, v, beta, d_out_w, d_out_u, d_out_y
        */

        load(k_reg, _k + time_index*k_reg.num_elements, k_reg.cols); // k = k.clone()
        load(v_reg, _v + time_index*v_reg.num_elements, v_reg.cols); // v = v.clone()
        load(beta_reg, _beta + time_index*beta_reg.outer_dim*TILE_DIM); // beta = beta.clone()

        /*
        * decay_values_forward: compute w and u
        */

        __syncthreads();

        decay_values_forward(tt_reg, bKl_reg, k_reg, beta_reg, w_reg, u_reg, v_reg, u_bases_reg, bk_reg);

        store(_w + time_index*w_reg.num_elements, w_reg, w_reg.cols);
        store(_u + time_index*w_reg.num_elements, u_reg, u_reg.cols);
    }

    /*
     * stitch chunks backwards using BPTT
     */
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

    for (int time_block = 0; time_block < (num_chunks + kNumWarps - 1) / kNumWarps; time_block++) {
        int time_index = time_block * kNumWarps + warpid;
        if (time_index >= num_chunks) {
            break;
        }
    
        /*
         * TODO: reload k, v, beta, d_out_w, d_out_u, d_out_y
         * when looping above actually happened
         */

        attend(w_bases_reg, tt_reg, w_reg);
        sub(w_bases_reg, k_reg, w_bases_reg);

        attend(v_reg, tt_reg, u_reg);
        sub(u_bases_reg, u_bases_reg, v_reg);

        /*
        * causal_attend_backward for d_q, d_k_2, d_out_u
        */

        load(d_out_y_reg, _d_out_y + time_index*d_out_y_reg.num_elements, d_out_y_reg.cols); // d_out_y = d_out_y.clone()

        kernel(tt_reg, d_out_y_reg, u_reg);
        make_causal(tt_reg, tt_reg, 0); // tt.tril_()

        auto &d_q = v_reg;
        attend(d_q, tt_reg, k_reg);

        if (1) {
            load(tk_reg, _d_q + time_index*tk_reg.num_elements, tk_reg.cols);
            add(d_q, d_q, tk_reg);
        }
        store(_d_q + time_index*q_reg.num_elements, d_q, d_q.cols);

        load(q_reg, _q + time_index*q_reg.num_elements, q_reg.cols); // q = q.clone()
        reverse_attend(d_k_reg, tt_reg, q_reg);

        if (1) {
            load(tk_reg, _d_k + time_index*tk_reg.num_elements, tk_reg.cols);
            add(d_k_reg, d_k_reg, tk_reg);
        }
        store(_d_k + time_index*d_k_reg.num_elements, d_k_reg, d_k_reg.cols); // first part of d_k

        auto &q_reg_row = swap_layout_inplace(q_reg);
        kernel(tt_reg, q_reg_row, k_reg);
        //q_reg = swap_layout_inplace(q_reg_row); // won't need it later
        make_causal(tt_reg, tt_reg, 0); // tt.tril_()

        auto &d_out_u_reg = v_reg;
        zero(d_out_u_reg);
        load(d_out_u_reg, _d_out_u + time_index*d_out_u_reg.num_elements, d_out_u_reg.cols); // d_out_u = d_out_u.clone()

        reverse_attend(tk_reg, tt_reg, d_out_y_reg); // don't need last swap_layout_inplace of d_out_y_reg
        add(d_out_u_reg, d_out_u_reg, tk_reg);

        /*
        * backward for d_k, d_v, d_beta
        */

        load(d_out_w_reg, _d_out_w + time_index*d_out_w_reg.num_elements, d_out_w_reg.cols); // d_out_w = d_out_w.clone()
        zero(d_k_reg); // note that we have a part of d_k in global memory now

        for (auto t = _time * TILE_DIM - 1; t >= 0; t--) {
            __syncthreads();

            auto &k_reg_col = swap_layout_inplace(k_reg);

            // d_k
            zero(mma_TD);
            {
                kernel(tt_reg, w_reg, d_out_w_reg);
                reset_trailing_rows(tt_reg, t);

                reverse_attend(tt_reg, k_reg_col, mma_TD, true);

                kernel(tt_reg, u_reg, d_out_u_reg);
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
            auto &bk_reg_col = swap_layout_inplace(bk_reg); // don't need the swap later
            mma_AtB(mma_TD, tt_reg_col, bk_reg_col, mma_TD);
            copy(tk_reg, mma_TD);
        }
        sub(d_k_reg, d_k_reg, tk_reg);

        load(tk_reg, _d_k + time_index*d_k_reg.num_elements, d_k_reg.cols); // d_k = d_k.clone()
        __syncthreads();
        add(d_k_reg, d_k_reg, tk_reg); // d_k += tk, reload from global memory
        store(_d_k + time_index*d_k_reg.num_elements, d_k_reg, d_k_reg.cols);

        // d_beta
        mul(w_bases_reg, w_bases_reg, d_out_w_reg); // w_bases = einsum('ntk,ntk->ntk', w_bases, d_out_w)
        mul(u_bases_reg, u_bases_reg, d_out_u_reg); // u_bases = einsum('ntw,ntw->ntw', u_bases, d_out_u)

        // d_v using available d_out_u_reg register
        mul_row(d_out_u_reg, d_out_u_reg, beta_reg);
        store(_d_v + time_index*d_k_reg.num_elements, d_out_u_reg, d_out_u_reg.cols);

        // continue d_beta
        auto &w_bases_col = swap_layout_inplace(w_bases_reg);
        auto &u_bases_col = swap_layout_inplace(u_bases_reg);
        zero(d_beta_reg);
        row_sum(d_beta_reg, w_bases_col); // d_beta = einsum('tk->t', w_bases);
        row_sum(d_beta_reg, u_bases_col, d_beta_reg); // d_beta += einsum('tw->t', u_bases);
        store(_d_beta + time_index*beta_reg.outer_dim*TILE_DIM, d_beta_reg);

        __syncthreads();
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
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 1, 1, 16)); \
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
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 1, 1, 16)); \
    } else if (d == 32) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 1, 8)); \
    } else if (d == 64) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 2, 4)); \
    } else if (d == 128) { \
        TYPE_DISPATCH(scalar_type, DELTA_DISPATCH(1, 2, 4, 4)); \
    } else { \
        TORCH_CHECK(false, "[qkv].size(2) should be 16, 32, 64, 128"); \
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

/**/
/*CHECK_CUDA_ERROR(cudaFuncSetAttribute(delta_forward_kernel<H, T, D, ACCUM, kHeight, kWidth*kWidthGroups, kWidth, kWidthGroups, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));*/

    // kHeight: tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    // kWidth: tiles per vector, 2 means head dimension is 2*16 = 32
#define DELTA_DISPATCH(_kHeight, _kWidth, _kWidthGroups, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth; \
        constexpr int kWidthGroups = _kWidthGroups; \
        constexpr int kNumWarps = _kNumWarps; \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        dim3 blocks(kWidthGroups, 1, batch); \
        unsigned long mem_size = sizeof(st<T, kWidth*kWidthGroups, kWidth, ducks::st_layout::swizzle>) + kNumWarps*sizeof(barrier); \
        delta_forward_kernel<H, T, D, ACCUM, kHeight, kWidth*kWidthGroups, kWidth, kWidthGroups, kNumWarps, kChunkSize><<<blocks,threads,mem_size>>>( \
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
    TORCH_CHECK(num_chunks <= 16, "num_chunks should be <= 32 (chunk size is 16)");

    // kHeight: tiles per sequence block, 2 means 2*16 = 32 sequence elements per warp
    // kWidth: tiles per vector, 2 means head dimension is 2*16 = 32
#define DELTA_DISPATCH(_kHeight, _kWidth, _kWidthGroups, _kNumWarps) \
        constexpr int kHeight = _kHeight;  \
        constexpr int kWidth = _kWidth*_kWidthGroups; \
        constexpr int kNumWarps = _kNumWarps; \
        auto threads = kNumWarps * kittens::WARP_THREADS; \
        unsigned long mem_size = kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>) \
                               + kNumWarps*sizeof(st_bf<kHeight, kWidth, ducks::st_layout::swizzle>); \
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(delta_backward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size)); \
        delta_backward_kernel<H, T, D, ACCUM, kHeight, kWidth, kWidth, kNumWarps, kChunkSize><<<batch_head,threads,mem_size>>>( \
            (int)num_chunks, \
            d_out_w.data_ptr<H>(), d_out_u.data_ptr<H>(), d_out_y.data_ptr<H>(), \
            q.data_ptr<H>(), k.data_ptr<H>(), v.data_ptr<H>(), beta.data_ptr<H>(), \
            d_q.data_ptr<H>(), d_k.data_ptr<H>(), d_v.data_ptr<H>(), d_beta.data_ptr<H>(), \
            w.data_ptr<H>(), u.data_ptr<H>(), y.data_ptr<H>())

    DISPATCH_ME_FLAT(d, seqlen);
#undef DELTA_DISPATCH
}
