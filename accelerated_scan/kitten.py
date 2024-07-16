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
                    torch::Tensor w, torch::Tensor u, torch::Tensor y);
extern void decay_values_backward(torch::Tensor d_out_w, torch::Tensor d_out_u, torch::Tensor d_out_y,
                                  torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                                  torch::Tensor d_q, torch::Tensor d_k, torch::Tensor d_v, torch::Tensor d_beta,
                                  torch::Tensor w, torch::Tensor u, torch::Tensor y);
    """],
    cuda_sources=[(Path(__file__).parent / 'delta.cu').read_text()],
    functions=['forward', 'decay_values_backward'],
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
decay_values_backward = delta_module.decay_values_backward