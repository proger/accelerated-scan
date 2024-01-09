from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = (Path(__file__).parent / 'warp.cuh').read_text()

cpp_source = """
at::Tensor warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out);
"""

module = load_inline(
    name='warpscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['warpscan_forward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--ptxas-options=-v",
        "-lineinfo",
        "--fmad", "false"
    ]
)

def scan_forward(gates, tokens):
    output = torch.zeros_like(tokens)
    module.warpscan_forward(gates, tokens, output)
    return output