from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name='kitten',
    cpp_sources=["""
extern void  attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f, torch::Tensor o);
"""],
    cuda_sources=[(Path(__file__).parent / 'kitten.cu').read_text()],
    functions=['attend'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--ptxas-options=-v",
        "-lineinfo",
        #"--fmad", "false",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DKITTENS_4090",
        f"-I{str(Path(__file__).parent)}"
    ]
)

delta_module = load_inline(
    name='delta',
    cpp_sources=["""\
extern void forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor y);
extern void backward(torch::Tensor d_out_w, torch::Tensor d_out_u, torch::Tensor d_out_y,
                     torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                     torch::Tensor d_q, torch::Tensor d_k, torch::Tensor d_v, torch::Tensor d_beta,
                     torch::Tensor u);
    """],
    cuda_sources=[(Path(__file__).parent / 'delta.cu').read_text()],
    functions=['forward', 'backward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--ptxas-options=-v",
        "-lineinfo",
        #"--fmad", "false",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DKITTENS_4090",
        f"-I{str(Path(__file__).parent)}"
    ]
)

attend = module.attend
delta_forward = delta_module.forward
delta_backward = delta_module.backward


class Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta):
        N, H, T, D = q.shape
        NH = N * H
        assert k.shape == v.shape == (N, H, T, D)
        assert beta.shape == (N, H, T)
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert beta.is_contiguous()

        ctx.save_for_backward(q, k, v, beta)
        q = q.view(NH, T, D)
        k = k.view(NH, T, D)
        v = v.view(NH, T, D)
        beta = beta.view(NH, T)
        y = torch.empty_like(q)
        delta_forward(q, k, v, beta, y)
        return y.view(N, H, T, D)

    @staticmethod
    def backward(ctx, d_y):
        q, k, v, beta = ctx.saved_tensors
        N, H, T, D = q.shape
        NH = N * H

        d_y = d_y.view(NH, T, D)
        q = q.view(NH, T, D)
        k = k.view(NH, T, D)
        v = v.view(NH, T, D)
        beta = beta.view(NH, T)
        d_q = q.new_zeros(NH, T, D)
        d_k = k.new_zeros(NH, T, D)
        d_v = v.new_zeros(NH, T, D)
        d_beta = beta.new_zeros(NH, T)
        u = v.new_zeros(NH, T, D) # buffer
        d_out_w = k.new_zeros(NH, T, D) # placeholder
        d_out_u = u.new_zeros(NH, T, D) # placeholder
        delta_backward(
            d_out_w, d_out_u, d_y,
            q, k, v, beta,
            d_q, d_k, d_v, d_beta,
            u
        )
        return d_q.view(N, H, T, D), d_k.view(N, H, T, D), d_v.view(N, H, T, D), d_beta.view(N, H, T)


def delta(query, key, value, beta):
    """Delta rule compressive attention.

    Maintains the state matrix for a linear model using online stochastic gradient descent
    on the mean squared error objective. Beta is the learning rate, key is the input and value is the target.

    At every step, the difference between current prediction given a key and the value is added to the state.
    When beta is constant ones this is equivalent to causal linear attention,
    which always stores the complete value.

    Arguments:
        query (torch.Tensor): shape (N, H, T, D), regression monitoring inputs
        key (torch.Tensor): shape (N, H, T, D), regression inputs
        value (torch.Tensor): shape (N, H, T, D), regression outputs
        beta (torch.Tensor): shape (N, H, T), learning rate

    Returns:
        torch.Tensor: shape (N, H, T, D), query outputs
    """
    return Delta.apply(query, key, value, beta)