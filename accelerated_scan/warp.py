from pathlib import Path

import torch
from torch import Tensor
from torch.library import custom_op
from torch.utils.cpp_extension import load_inline

cuda_source = (Path(__file__).parent / 'warp.cuh').read_text()

cpp_source = """
at::Tensor warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse);
void warpscan_backward(const at::Tensor &gates, const at::Tensor &output, const at::Tensor &outGrad, const at::Tensor& gateGradOut, const at::Tensor& valueGradOut);
"""

module = load_inline(
    name='warpscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['warpscan_forward', 'warpscan_backward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "--ptxas-options=-v",
        "-lineinfo",
        "--fmad", "false",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
)
warpscan_forward = module.warpscan_forward
warpscan_backward = module.warpscan_backward

@custom_op("accelerated_scan::warp_forward", mutates_args=())
def warp_forward(gates: Tensor, tokens: Tensor) -> Tensor:
    torch._check(gates.shape == tokens.shape)
    torch._check(gates.device == tokens.device)
    torch._check(gates.is_cuda); torch._check(tokens.is_cuda)
    torch._check(gates.dtype == tokens.dtype)

    gates = gates.contiguous()
    tokens = tokens.contiguous()

    output = torch.empty_like(tokens)
    warpscan_forward(gates, tokens, output, False)

    return output

@warp_forward.register_fake
def _(gates: Tensor, tokens: Tensor) -> Tensor:
    torch._check(gates.shape == tokens.shape)
    torch._check(gates.device == tokens.device)
    torch._check(gates.is_cuda); torch._check(tokens.is_cuda)
    torch._check(gates.dtype == tokens.dtype)

    return torch.empty_like(tokens, memory_format=torch.contiguous_format)

# backward scan is a padded reverse scan
# See https://arxiv.org/abs/1709.04057 Section 2.2
@custom_op("accelerated_scan::warp_backward", mutates_args=())
def warp_backward(states: Tensor, gates: Tensor, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    torch._check(states.shape == gates.shape)
    torch._check(states.device == gates.device)
    torch._check(states.is_cuda); torch._check(gates.is_cuda)
    torch._check(states.dtype == gates.dtype)

    grad_output = grad_output.contiguous()
    states = states.contiguous()
    gates = gates.contiguous()

    d_gates = torch.empty_like(gates)
    d_tokens = torch.empty_like(states)
    warpscan_backward(gates, states, grad_output, d_gates, d_tokens)

    return d_gates, d_tokens

@warp_backward.register_fake
def _(states: Tensor, gates: Tensor, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    torch._check(states.shape == gates.shape)
    torch._check(states.device == gates.device)
    torch._check(states.is_cuda); torch._check(gates.is_cuda)
    torch._check(states.dtype == gates.dtype)

    return (
        torch.empty_like(states, memory_format=torch.contiguous_format),
        torch.empty_like(gates, memory_format=torch.contiguous_format),
    )

def _setup_context(ctx, inputs, output):
    gates, _tokens = inputs
    ctx.save_for_backward(output, gates)

def _backward(ctx, grad_output):
    states, gates = ctx.saved_tensors
    return warp_backward(states, gates, grad_output)

warp_forward.register_autograd(_backward, setup_context=_setup_context)


def scan(gates: Tensor, tokens: Tensor) -> Tensor:
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.
        tokens (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.

    Returns:
        (torch.Tensor): shape (B, C, T)
    """
    return warp_forward(gates, tokens)
