
#include "tk/src/kittens.cuh"
#include "tk/src/common/pyutils/torch_helpers.cuh"

using namespace kittens;

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

template<typename T, int _time, int _key>
__device__ static inline void zeroexcept(
    rt<T, _time, _key> &dst,
    const rt<T, _time, _key> &src,
    const int row_index
) {
    const int row_top = laneid() / 4;
    const int row_bottom = row_top + 8;

    static_assert(dst.packed_per_tile == 4, "packed_per_tile must be 4");
    //using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {            
            if (row_top != row_index) {
                dst.tiles[i][j].data[0] = {0.0, 0.0};
                dst.tiles[i][j].data[2] = {0.0, 0.0};
            } else {
                dst.tiles[i][j].data[0] = src.tiles[i][j].data[0];
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2];
            }
            if (row_bottom != row_index) {
                dst.tiles[i][j].data[1] = {0.0, 0.0};
                dst.tiles[i][j].data[3] = {0.0, 0.0};
            } else {
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1];
                dst.tiles[i][j].data[3] = src.tiles[i][j].data[3];
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

/**
 * @brief Bind a list of vectors (k,u) and write them to a dictionary state_delta.
 */
template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static inline void associate(
    rt<D, _value, _key, ducks::rt_layout::row> &state_delta,
    /*const*/ rt<D, _time, _value, ducks::rt_layout::row> &v,
    /*const*/ rt<D, _time, _key, ducks::rt_layout::row> &k
) {
    rt<ACCUM, _value, _key> mma_state;
    associate(state_delta, v, k, mma_state, false);
}

/**
 * @brief Bind a list of vectors (k,u) and write them to a dictionary state_delta.
 */
template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static inline void associate(
    rt<D, _value, _key, ducks::rt_layout::row> &state_delta,
    /*const*/ rt<D, _time, _value, ducks::rt_layout::row> &v,
    /*const*/ rt<D, _time, _key, ducks::rt_layout::row> &k,
    rt<ACCUM, _value, _key> &mma_state,
    const bool accum = false
) {
    if (accum == false) {
        zero(mma_state);
    }
    auto &k_col = swap_layout_inplace(k);
    auto &v_col = swap_layout_inplace(v);
    mma_AtB(mma_state, v_col, k_col, mma_state);
    swap_layout_inplace(k_col);
    swap_layout_inplace(v_col);
    copy(state_delta, mma_state);
}

/**
 * @brief Bind a list of vectors (k,u) and write them to a dictionary state_delta.
 */
template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static inline void associate(
    rt<D, _value, _key, ducks::rt_layout::row> &state_delta,
    /*const*/ rt<D, _time, _value, ducks::rt_layout::col> &v_col,
    /*const*/ rt<D, _time, _key, ducks::rt_layout::row> &k,
    rt<ACCUM, _value, _key> &mma_state,
    const bool accum = false
) {
    if (accum == false) {
        zero(mma_state);
    }
    auto &k_col = swap_layout_inplace(k);
    mma_AtB(mma_state, v_col, k_col, mma_state);
    swap_layout_inplace(k_col);
    copy(state_delta, mma_state);
}


template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static inline void query(
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
__device__ static inline void reverse_query(
    rt<D, _time, _key, ducks::rt_layout::row> &output_keys,
    const rt<D, _time, _value, ducks::rt_layout::row> &value_query,
    /*const*/ rt<D, _value, _key, ducks::rt_layout::row> &state
) {
    auto &state_col = swap_layout_inplace(state);
    reverse_query(output_keys, value_query, state_col);
    swap_layout_inplace(state_col);
}

template <typename D, typename ACCUM = float2, int _time, int _key, int _value>
__device__ static inline void reverse_query(
    rt<D, _time, _key, ducks::rt_layout::row> &output_keys,
    const rt<D, _time, _value, ducks::rt_layout::row> &value_query,
    /*const*/ rt<D, _value, _key, ducks::rt_layout::col> &state_col
) {
    rt<ACCUM, _time, _key> mma;

    zero(mma);
    mma_AB(mma, value_query, state_col, mma); // einsum('tv,vk->tk', value_query, state)
    copy(output_keys, mma);
}



template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _key>
__device__ static inline void kernel(
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
__device__ static inline void attend(
    rt<D, _time_source, _value, ducks::rt_layout::row> &mixtures,
    const rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_target, _value, ducks::rt_layout::row> &values
) {
    auto &values_col = swap_layout_inplace(values);
    attend(mixtures, attention, values_col);
    swap_layout_inplace(values_col);
}

template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _value>
__device__ static inline void attend(
    rt<D, _time_source, _value, ducks::rt_layout::row> &mixtures,
    const rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    const rt<D, _time_target, _value, ducks::rt_layout::col> &values_col
) {
    rt<ACCUM, _time_source, _value> mma;

    zero(mma);
    mma_AB(mma, attention, values_col, mma);
    copy(mixtures, mma);
}

template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _value>
__device__ static inline void reverse_attend(
    rt<D, _time_target, _value, ducks::rt_layout::row> &mixtures,
    /*const*/ rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_source, _value, ducks::rt_layout::row> &source_values
) {
    auto &source_values_col = swap_layout_inplace(source_values);
    reverse_attend(mixtures, attention, source_values_col);
    swap_layout_inplace(source_values_col);
}

template <typename D, typename ACCUM = float2, int _time_source, int _time_target, int _value>
__device__ static inline void reverse_attend(
    rt<D, _time_target, _value, ducks::rt_layout::row> &mixtures,
    /*const*/ rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_source, _value, ducks::rt_layout::col> &source_values_col
) {
    rt<ACCUM, _time_target, _value> mma;
    reverse_attend(attention, source_values_col, mma);
    copy(mixtures, mma);
}

template <typename D, typename ACCUM, int _time_source, int _time_target, int _value>
__device__ static inline void reverse_attend(
    /*const*/ rt<D, _time_source, _time_target, ducks::rt_layout::row> &attention,
    /*const*/ rt<D, _time_source, _value, ducks::rt_layout::col> &source_values_col,
    rt<ACCUM, _time_target, _value> &mma,
    const bool accum = false
) {
    if (!accum) {
        zero(mma);
    }
    auto &attention_col = swap_layout_inplace(attention);
    mma_AtB(mma, attention_col, source_values_col, mma);
    swap_layout_inplace(attention_col);
}


struct op_negate {
    template<typename T> static __device__ inline T op(const T &x) { return -x; }
};

template<ducks::rt::all T>
__device__ static inline void negate(T &dst) {
    unary_map<op_negate, T>(dst, dst);
}

