#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <int kNStepsPerThread, int kNThreadsPerWarp, int kNWarpsPerBlock>
__global__ void scan(float* gates, float* tokens, float* result) {
    __shared__ float warpLastGate[kNWarpsPerBlock];
    __shared__ float warpLastToken[kNWarpsPerBlock];

    const int threadId = threadIdx.x;
    const int warpId = threadId / kNThreadsPerWarp;
    const int laneId = threadId % kNThreadsPerWarp;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr int kThreadLast = kNStepsPerThread - 1;
    constexpr float kEmptyGate = 1.0;
    constexpr float kEmptyToken = 0.0;

    //
    // Read from global memory.
    // Scan sequentially in thread registers (level 0).
    // 

    float2 acc[kNStepsPerThread];

    #pragma unroll
    for (int i = 0; i < kNStepsPerThread; ++i) {
        float gate = gates[threadId * kNStepsPerThread + i];
        float token = tokens[threadId * kNStepsPerThread + i];
        if (i == 0) {
            acc[i] = {threadId == 0 ? kEmptyGate : gate, token};
        } else {
            acc[i] = {acc[i - 1].x * gate, acc[i - 1].y * gate + token};
        }
    }

    //
    // Scan threads in a warp using shuffling (level 1).
    //

    #pragma unroll
    for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
        float prev_gate = __shfl_up_sync(0xffffffff, acc[kThreadLast].x, delta);
        float prev_token = __shfl_up_sync(0xffffffff, acc[kThreadLast].y, delta);

        if (laneId >= delta) {
            #pragma unroll
            for (int i = 0; i < kNStepsPerThread; ++i) {
                acc[i] = {prev_gate * acc[i].x, prev_token * acc[i].x + acc[i].y};
            }
        }
    }

    __syncwarp();

    //
    // Store the last element of each warp in shared memory.
    //

    if (laneId == kWarpLast) {
        warpLastGate[warpId] = acc[kThreadLast].x;
        warpLastToken[warpId] = acc[kThreadLast].y;
    }

    __syncthreads();

    //
    // Leading warp scans every warp in a block (level 2).
    //

    if (warpId == 0) {
        float2 warpAcc;
        warpAcc.x = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
        warpAcc.y = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : kEmptyToken;

        #pragma unroll
        for (int delta = 1; delta < warpSize; delta *= 2) {
            float prev_gate = __shfl_up_sync(0xffffffff, warpAcc.x, delta);
            float prev_token = __shfl_up_sync(0xffffffff, warpAcc.y, delta);

            if (laneId >= delta) {
                warpAcc = {prev_gate * warpAcc.x, prev_token * warpAcc.x + warpAcc.y};
            }
        }

        if (laneId < kNWarpsPerBlock) {
            warpLastGate[laneId] = warpAcc.x;
            warpLastToken[laneId] = warpAcc.y;
        }
    }

    __syncthreads();

    //
    // Add the last element of the previous warp to each element of the current warp (level 0).
    // Store to global memory.
    //

    #pragma unroll
    for (int i = 0; i < kNStepsPerThread; ++i) {
        if (warpId > 0) {
            result[threadId * kNStepsPerThread + i] = warpLastToken[warpId-1] * acc[i].x + acc[i].y;
        } else {
            result[threadId * kNStepsPerThread + i] = acc[i].y;
        }
    }
}

at::Tensor
warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out) {
    TORCH_CHECK(tokens.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(gates.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
    TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 grid(1,1);
    constexpr int kNThreadsPerWarp = 32;

    if (seqlen == 32) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 64) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 128) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 4;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 256) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 8;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 512) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 1024) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 2048) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 4096) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        scan<kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 4096");
    }

    return out;
}