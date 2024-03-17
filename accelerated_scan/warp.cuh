#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_STRIDE(x) TORCH_CHECK(x.stride(-1) == 1 || x.size(-1) == 1);

template<typename weight_t, int N>
class UnalignedTuple {
public:
    static constexpr int Size = N;
    using Type = weight_t;

    weight_t data[N];

    __device__ void reverse() {
        #pragma unroll
        for (int i = 0; i < N/2; i++) {
            weight_t temp = data[i];
            data[i] = data[N - (i+1)];
            data[N - (i+1)] = temp;
        }
    }
};

template<typename T, int N>
class alignas(16) AlignedTuple : public UnalignedTuple<T, N> {
};

template <typename Tuple, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence>
__global__ void scan(
    const Tuple* gates,
    const Tuple* tokens,
    Tuple* result,
    const int batch_stride,
    const int dim_stride,
    const bool reverse
) {
    using weight_t = typename Tuple::Type;

    __shared__ weight_t warpLastGate[kNWarpsPerBlock];
    __shared__ weight_t warpLastToken[kNWarpsPerBlock];
    __shared__ weight_t chunkAccGate, chunkAccToken;

    const int seqoffset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
    const int warpId = threadIdx.x / kNThreadsPerWarp;
    const int laneId = threadIdx.x % kNThreadsPerWarp;
    const int chunklen = blockDim.x * Tuple::Size;
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr int kThreadLast = Tuple::Size - 1;
    const weight_t kEmptyGate = 1.0;
    const weight_t kEmptyToken = 0.0;

    //
    // Read from global memory.
    // Scan sequentially in thread registers (level 0).
    // 

    for (int chunk = 0; chunk < kNChunksPerSequence; chunk++) {
        const int offset = seqoffset + (reverse ? kNChunksPerSequence - 1 - chunk : chunk) * chunklen;
        const int tupleOffset = (offset + (reverse ? chunklen - ((threadIdx.x + 1) * Tuple::Size) : (threadIdx.x * Tuple::Size))) / Tuple::Size;

        if (chunk) {
            __syncthreads();
        }

        Tuple loadedGate = gates[tupleOffset];
        Tuple loadedToken = tokens[tupleOffset];
        if (reverse) {
            loadedGate.reverse();
            loadedToken.reverse();
        }

        Tuple accGate;
        Tuple accToken;

        #pragma unroll
        for (int i = 0; i < Tuple::Size; ++i) {
            weight_t gate = loadedGate.data[i];
            weight_t token = loadedToken.data[i];
            if (i == 0) {
                if (chunk == 0) {
                    accGate.data[0] = threadIdx.x == 0 ? kEmptyGate : gate;
                    accToken.data[0] = token;
                } else {
                    if (threadIdx.x == 0) {
                        // Add the last element of the previous chunk to the first element of the current chunk.
                        accGate.data[0] = chunkAccGate * gate;
                        accToken.data[0] = chunkAccToken * gate + token;
                    } else {
                        accGate.data[0] = gate;
                        accToken.data[0] = token;
                    }
                }
            } else {
                accGate.data[i] = accGate.data[i - 1] * gate;
                accToken.data[i] = accToken.data[i - 1] * gate + token;
            }
        }

        //
        // Scan threads in a warp using shuffling (level 1).
        //

        #pragma unroll
        for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
            weight_t prev_gate = __shfl_up_sync(0xffffffff, accGate.data[kThreadLast], delta);
            weight_t prev_token = __shfl_up_sync(0xffffffff, accToken.data[kThreadLast], delta);

            if (laneId >= delta) {
                #pragma unroll
                for (int i = 0; i < Tuple::Size; ++i) {
                    accToken.data[i] = prev_token * accGate.data[i] + accToken.data[i];
                    accGate.data[i] = prev_gate * accGate.data[i];
                }
            }
        }

        __syncwarp();

        //
        // Store the last element of each warp in shared memory.
        //

        if (laneId == kWarpLast) {
            warpLastGate[warpId] = accGate.data[kThreadLast];
            warpLastToken[warpId] = accToken.data[kThreadLast];
        }

        __syncthreads();

        //
        // Leading warp scans every warp in a block (level 2).
        //

        if (warpId == 0) {
            weight_t warpAccGate, warpAccToken;
            warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
            warpAccToken = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : kEmptyToken;

            #pragma unroll
            for (int delta = 1; delta < warpSize; delta *= 2) {
                weight_t prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                weight_t prev_token = __shfl_up_sync(0xffffffff, warpAccToken, delta);

                if (laneId >= delta) {
                    warpAccToken = prev_token * warpAccGate + warpAccToken;
                    warpAccGate = prev_gate * warpAccGate;
                }
            }

            if (laneId < kNWarpsPerBlock) {
                warpLastGate[laneId] = warpAccGate;
                warpLastToken[laneId] = warpAccToken;
            }
        }

        __syncthreads();

        //
        // Add the last element of the previous warp to each element of the current warp (level 0).
        // Store to global memory.
        //

        #pragma unroll
        for (int i = 0; i < Tuple::Size; ++i) {
            if (warpId > 0) {
                accToken.data[i] = warpLastToken[warpId-1] * accGate.data[i] + accToken.data[i];
                accGate.data[i] = warpLastGate[warpId-1] * accGate.data[i];
            }
        }
        if (reverse) {
            accToken.reverse();
        }
        result[tupleOffset] = accToken;

        if (laneId == kWarpLast && warpId == kBlockLast) {
            chunkAccGate = accGate.data[kThreadLast];
            chunkAccToken = accToken.data[kThreadLast];
        }
    }
}

#define DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, batch_stride, dim_stride, reverse) \
    using AlignedT = AlignedTuple<weight_t, kNStepsPerThread>; \
    using UnalignedT = UnalignedTuple<weight_t, kNStepsPerThread>; \
    if (kNStepsPerThread == 4 && \
        ((long)gates.data_ptr()) % 16 == 0 && \
        ((long)tokens.data_ptr()) % 16 == 0 && \
        ((long)out.data_ptr()) % 16 == 0) { \
        scan<AlignedT, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( \
                reinterpret_cast<const AlignedT *>(gates.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const AlignedT *>(tokens.data_ptr<torch_weight_t>()), \
                reinterpret_cast<AlignedT *>(out.data_ptr<torch_weight_t>()), \
                batch_stride, dim_stride, reverse \
            ); \
    } else { \
        scan<UnalignedT, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( \
                reinterpret_cast<const UnalignedT*>(gates.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const UnalignedT*>(tokens.data_ptr<torch_weight_t>()), \
                reinterpret_cast<UnalignedT *>(out.data_ptr<torch_weight_t>()), \
                batch_stride, dim_stride, reverse \
            ); \
    }

template <typename weight_t, typename torch_weight_t>
void
warpscan(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
    const auto strides = tokens.strides();
    const int batch_stride = strides[0];
    const int dim_stride = strides[1];
    CHECK_STRIDE(tokens);
    CHECK_STRIDE(gates);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(batch_size, dim);
    constexpr int kNThreadsPerWarp = 32;

    if (seqlen == 32) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 64) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 1;
        constexpr int kNChunksPerSequence = 1;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 128) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 4;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 256) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 8;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 512) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 1024) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 2048) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 4096) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 8192) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 2;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 16384) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 4;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 32768) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 8;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 65536) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 16;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out,
            batch_stride, dim_stride, reverse);
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
    }
}

at::Tensor
warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.is_contiguous());
    TORCH_CHECK(gates.is_contiguous());

    if (tokens.scalar_type() == at::ScalarType::BFloat16) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::BFloat16);
        warpscan<__nv_bfloat16, at::BFloat16>(gates, tokens, out, reverse);
    } else if (tokens.scalar_type() == at::ScalarType::Half) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::Half);
        warpscan<__half, at::Half>(gates, tokens, out, reverse);
    } else if (tokens.scalar_type() == at::ScalarType::Float) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::Float);
        warpscan<float, float>(gates, tokens, out, reverse);
    } else {
        TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
    }
    return out;
}

template <typename Tuple, int offset>
Tuple load_shifted_tuple(const Tuple* ptr, int index, int limit) {
    using weight_t = typename Tuple::Type;

    const weight_t* rawPtr = reinterpret_cast<const weight_t *>(ptr);
    Tuple x;
    for (int i = 0; i < Tuple::Size; i++) {
        const int offset = index*4 + i + offset;
        if (offset >= 0 && offset < limit*4) {
            x.data[i] = rawPtr[offset];
        }
    }

    return x;
}

template <typename Tuple, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence>
__global__ void scan_grad(
    const Tuple* gates,
    const Tuple* output,
    const Tuple* outGrad,
    Tuple* gateGradOut,
    Tuple* valueGradOut,
    const int batch_stride,
    const int dim_stride,
    const bool reverse
) {
    using weight_t = typename Tuple::Type;

    __shared__ weight_t warpLastGate[kNWarpsPerBlock];
    __shared__ weight_t warpLastToken[kNWarpsPerBlock];
    __shared__ weight_t chunkAccGate, chunkAccToken;

    const int seqoffset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
    const int warpId = threadIdx.x / kNThreadsPerWarp;
    const int laneId = threadIdx.x % kNThreadsPerWarp;
    const int chunklen = blockDim.x * Tuple::Size;
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr int kThreadLast = Tuple::Size - 1;
    const weight_t kEmptyGate = 1.0;
    const weight_t kEmptyToken = 0.0;
    const int limit = blockDim.x * kNChunksPerSequence;

    for (int chunk = 0; chunk < kNChunksPerSequence; chunk++) {
        const int offset = seqoffset + (kNChunksPerSequence - 1 - chunk) * chunklen;
        const int tupleOffset = (offset + (chunklen - ((threadIdx.x + 1) * Tuple::Size))) / Tuple::Size;

        if (chunk) {
            __syncthreads();
        }

        // Load from global memory.
        Tuple loadedGate = load_shifted_tuple<Tuple, 1>(gates, tupleOffset, limit);
        Tuple loadedToken = outGrad[tupleOffset];
        loadedGate.reverse();
        loadedToken.reverse();

        Tuple accGate;
        Tuple accToken;

        // Scan within the current thread.
        #pragma unroll
        for (int i = 0; i < Tuple::Size; ++i) {
            weight_t gate = loadedGate.data[i];
            weight_t token = loadedToken.data[i];
            if (i == 0) {
                if (chunk == 0) {
                    accGate.data[0] = threadIdx.x == 0 ? kEmptyGate : gate;
                    accToken.data[0] = token;
                } else {
                    if (threadIdx.x == 0) {
                        // Add the last element of the previous chunk to the first element of the current chunk.
                        accGate.data[0] = chunkAccGate * gate;
                        accToken.data[0] = chunkAccToken * gate + token;
                    } else {
                        accGate.data[0] = gate;
                        accToken.data[0] = token;
                    }
                }
            } else {
                accGate.data[i] = accGate.data[i - 1] * gate;
                accToken.data[i] = accToken.data[i - 1] * gate + token;
            }
        }

        //
        // Scan threads in a warp using shuffling (level 1).
        //

        #pragma unroll
        for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
            weight_t prev_gate = __shfl_up_sync(0xffffffff, accGate.data[kThreadLast], delta);
            weight_t prev_token = __shfl_up_sync(0xffffffff, accToken.data[kThreadLast], delta);

            if (laneId >= delta) {
                #pragma unroll
                for (int i = 0; i < Tuple::Size; ++i) {
                    accToken.data[i] = prev_token * accGate.data[i] + accToken.data[i];
                    accGate.data[i] = prev_gate * accGate.data[i];
                }
            }
        }

        __syncwarp();

        //
        // Store the last element of each warp in shared memory.
        //

        if (laneId == kWarpLast) {
            warpLastGate[warpId] = accGate.data[kThreadLast];
            warpLastToken[warpId] = accToken.data[kThreadLast];
        }

        __syncthreads();

        //
        // Leading warp scans every warp in a block (level 2).
        //

        if (warpId == 0) {
            weight_t warpAccGate, warpAccToken;
            warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
            warpAccToken = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : kEmptyToken;

            #pragma unroll
            for (int delta = 1; delta < warpSize; delta *= 2) {
                weight_t prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                weight_t prev_token = __shfl_up_sync(0xffffffff, warpAccToken, delta);

                if (laneId >= delta) {
                    warpAccToken = prev_token * warpAccGate + warpAccToken;
                    warpAccGate = prev_gate * warpAccGate;
                }
            }

            if (laneId < kNWarpsPerBlock) {
                warpLastGate[laneId] = warpAccGate;
                warpLastToken[laneId] = warpAccToken;
            }
        }

        __syncthreads();

        //
        // Add the last element of the previous warp to each element of the current warp (level 0).
        // Store to global memory.
        //

        #pragma unroll
        for (int i = 0; i < Tuple::Size; ++i) {
            if (warpId > 0) {
                accToken.data[i] = warpLastToken[warpId-1] * accGate.data[i] + accToken.data[i];
                accGate.data[i] = warpLastGate[warpId-1] * accGate.data[i];
            }
        }
        if (reverse) {
            accToken.reverse();
        }
        valueGradOut[tupleOffset] = accToken;

        Tuple gateGrad = load_shifted_tuple<Tuple, -1>(output, tupleOffset, limit);
        for (int i = 0; i < Tuple::Size; i++) {
            gateGrad.data[i] *= accToken.data[i];
        }
        gateGradOut[tupleOffset] = gateGrad;

        if (laneId == kWarpLast && warpId == kBlockLast) {
            chunkAccGate = accGate.data[kThreadLast];
            chunkAccToken = accToken.data[kThreadLast];
        }
    }
}

#define DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, output, outGrad, gateGradOut, valueGradOut, batch_stride, dim_stride) \
    using AlignedT = AlignedTuple<weight_t, kNStepsPerThread>; \
    using UnalignedT = UnalignedTuple<weight_t, kNStepsPerThread>; \
    if (kNStepsPerThread == 4 && \
        ((long)gates.data_ptr()) % 16 == 0 && \
        ((long)output.data_ptr()) % 16 == 0 && \
        ((long)outGrad.data_ptr()) % 16 == 0 && \
        ((long)gateGradOut.data_ptr()) % 16 == 0 && \
        ((long)valueGradOut.data_ptr()) % 16 == 0) { \
        scan_grad<AlignedT, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( \
                reinterpret_cast<const AlignedT *>(gates.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const AlignedT *>(output.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const AlignedT *>(outGrad.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const AlignedT *>(gateGradOut.data_ptr<torch_weight_t>()), \
                reinterpret_cast<AlignedT *>(valueGradOut.data_ptr<torch_weight_t>()), \
                batch_stride, dim_stride \
            ); \
    } else { \
        scan_grad<UnalignedT, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( \
                reinterpret_cast<const UnalignedT*>(gates.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const UnalignedT*>(output.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const UnalignedT*>(outGrad.data_ptr<torch_weight_t>()), \
                reinterpret_cast<const UnalignedT*>(gateGradOut.data_ptr<torch_weight_t>()), \
                reinterpret_cast<UnalignedT *>(valueGradOut.data_ptr<torch_weight_t>()), \
                batch_stride, dim_stride \
            ); \
    }

template <typename weight_t, typename torch_weight_t>
void
warpscan_grad(const at::Tensor &gates, const at::Tensor &output, const at::Tensor &outGrad,
              const at::Tensor& gateGradOut, const at::Tensor& valueGradOut) {
    const auto strides = tokens.strides();
    const int batch_stride = strides[0];
    const int dim_stride = strides[1];
    CHECK_STRIDE(gates);
    CHECK_STRIDE(output);
    CHECK_STRIDE(outGrad);
    CHECK_STRIDE(gateGradOut);
    CHECK_STRIDE(valueGradOut);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(batch_size, dim);
    constexpr int kNThreadsPerWarp = 32;

    if (seqlen == 32) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 64) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 1;
        constexpr int kNChunksPerSequence = 1;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 128) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 4;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 256) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 8;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 512) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 1024) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 2048) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 4096) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 8192) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 2;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 16384) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 4;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 32768) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 8;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else if (seqlen == 65536) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 16;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN_GRAD(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream,
            gates, output, outGrad, gateGradOut, valueGradOut,
            batch_stride, dim_stride);
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
    }
}

at::Tensor
warpscan_backward(const at::Tensor &gates, const at::Tensor &output, const at::Tensor &outGrad, const at::Tensor& gateGradOut, const at::Tensor& valueGradOut) {
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(outGrad.is_cuda());
    TORCH_CHECK(gateGradOut.is_contiguous());
    TORCH_CHECK(valueGradOut.is_contiguous());
    TORCH_CHECK(gates.scalar_type() == output.scalar_type());
    TORCH_CHECK(gates.scalar_type() == outGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == gateGradOut.scalar_type());
    TORCH_CHECK(gates.scalar_type() == valueGradOut.scalar_type());
    TORCH_CHECK(gates.sizes() == output.sizes());
    TORCH_CHECK(gates.sizes() == outGrad.sizes());
    TORCH_CHECK(gates.sizes() == gateGradOut.sizes());
    TORCH_CHECK(gates.sizes() == valueGradOut.sizes());

    if (gates.scalar_type() == at::ScalarType::BFloat16) {
        warpscan_grad<__nv_bfloat16, at::BFloat16>(gates, outputs, outGrad, gateGradOut, valueGradOut);
    } else if (gates.scalar_type() == at::ScalarType::Half) {
        warpscan_grad<__half, at::Half>(gates, outputs, outGrad, gateGradOut, valueGradOut);
    } else if (gates.scalar_type() == at::ScalarType::Float) {
        warpscan_grad<float, float>(gates, outputs, outGrad, gateGradOut, valueGradOut);
    } else {
        TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
    }
    return out;
}
