import torch
import triton
import triton.language as tl


@triton.jit
def combine1(xl, fl, xr, fr):
    """
    First order associative op. See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1
    """
    x = xl * fr + xr
    f = fl * fr
    return x, f


@triton.jit
def forward_scan(
    gates,
    tokens,
    outputs,
    T: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    strides = tl.arange(0, T) + sequence_id * T

    tokens_ = tl.load(tokens + strides)
    gates_ = tl.load(gates + strides)

    output_tokens_, output_gates_ = tl.associative_scan((tokens_, gates_), axis=0, combine_fn=combine1)
    tl.store(outputs + strides, output_tokens_)


@triton.jit
def backward_scan(
    gates,
    tokens,
    outputs,
    T: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    forward_strides = tl.arange(0, T) + sequence_id * T
    reverse_strides = (tl.num_programs(axis=0) * tl.num_programs(axis=1) * T - 1) - forward_strides

    tokens_ = tl.load(tokens + reverse_strides)
    gates_ = tl.load(gates + reverse_strides)

    output_tokens_, output_gates_ = tl.associative_scan((tokens_, gates_), axis=0, combine_fn=combine1)
    tl.store(outputs + reverse_strides, output_tokens_)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T)
        assert gates.is_contiguous()
        assert tokens.is_contiguous()

        states = torch.zeros_like(tokens)
        forward_scan[(B,C)](gates, tokens, states, T=T, enable_fp_fusion=False)

        ctx.save_for_backward(states, gates)
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, grad_output):
        states, gates = ctx.saved_tensors
        B, C, T = gates.shape

        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()

        d_states = torch.empty_like(states)
        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
        backward_scan[(B,C)](padded_shifted_gates, grad_output, d_states, T=T, enable_fp_fusion=False)

        padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
        d_gates = padded_outputs * d_states

        d_tokens = d_states
        return d_gates, d_tokens


def scan(gates, tokens):
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
    return Scan.apply(gates, tokens)
