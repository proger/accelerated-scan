from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = (Path(__file__).parent / 'kitten.cu').read_text()

cpp_source = """
extern void  attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f, torch::Tensor o);
"""

module = load_inline(
    name='kitten',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
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
        "-DKITTENS_4090", "-arch=sm_80",
        f"-I{str(Path(__file__).parent)}"
    ]
)

attend = module.attend