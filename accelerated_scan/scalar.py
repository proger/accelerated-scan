import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

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
    forget,
    inputs,
    states,
    seqlen,
    BLOCK: tl.constexpr = 2048,
):
    nblocks = tl.cdiv(seqlen, BLOCK)
    bsz, chan = tl.program_id(axis=0), tl.program_id(axis=1)
    sequence_id = tl.num_programs(axis=1) * bsz + chan
    idx = tl.arange(0, BLOCK)
    h0 = tl.zeros((), dtype=tl.float32)

    for block_id in tl.range(0, nblocks):
        t = block_id * BLOCK + idx
        offset = sequence_id * seqlen + t

        i = tl.load(inputs + offset, mask=t < seqlen, other=0.0)
        f = tl.load(forget + offset, mask=t < seqlen, other=1.0)

        h, f = tl.associative_scan((i, f), axis=0, combine_fn=combine1)

        block_h = tl.sum(tl.where(idx == BLOCK - 1, h, 0.0), axis=0)
        block_f = tl.sum(tl.where(idx == BLOCK - 1, f, 0.0), axis=0)

        h = h + h0 * f
        tl.store(states + offset, h, mask=t < seqlen)

        h0 = block_h + h0 * block_f


@triton.jit
def backward_scan(
    forget,
    states,
    d_output,
    d_inputs,
    d_forget,
    seqlen,
    BLOCK: tl.constexpr = 2048,
):
    nblocks = tl.cdiv(seqlen, BLOCK)
    bsz, chan = tl.program_id(axis=0), tl.program_id(axis=1)
    sequence_id = tl.num_programs(axis=1) * bsz + chan
    idx = tl.arange(0, BLOCK)
    h0 = tl.zeros((), dtype=tl.float32)

    for block_id in tl.range(0, nblocks):
        reverse_block = nblocks - 1 - block_id
        t = reverse_block * BLOCK + idx
        offset = sequence_id * seqlen + t

        do = tl.load(d_output + offset, mask=t < seqlen, other=0.0)

        shifted_f = tl.load(forget + offset + 1, mask=t < seqlen - 1, other=1.0)

        di, f = tl.associative_scan((do, shifted_f), axis=0, combine_fn=combine1, reverse=True)

        block_di = tl.sum(tl.where(idx == 0, di, 0.0), axis=0)
        block_f = tl.sum(tl.where(idx == 0, f, 0.0), axis=0)

        di = di + h0 * f
        tl.store(d_inputs + offset, di, mask=t < seqlen)

        shifted_states = tl.load(states + offset - 1, mask=t > 0, other=0.0)

        df = shifted_states * di
        tl.store(d_forget + offset, df, mask=t < seqlen)

        h0 = block_di + h0 * block_f


@triton_op("accelerated_scan::scalar_forward", mutates_args=())
def scalar_forward(forget: Tensor, inputs: Tensor) -> Tensor:
    B, C, T = forget.shape
    torch._check(forget.shape == inputs.shape)
    torch._check(forget.device == inputs.device)
    torch._check(forget.is_cuda); torch._check(inputs.is_cuda)
    torch._check(forget.dtype == inputs.dtype)

    forget = forget.contiguous()
    inputs = inputs.contiguous()

    states = torch.empty_like(inputs)
    wrap_triton(forward_scan)[(B, C)](
        forget,
        inputs,
        states,
        seqlen=T,
        BLOCK=2048, # MUST add this or somehow the default value of BLOCK=2048 is not being passed to the kernel (when torch.compile)
        enable_fp_fusion=False,
    )

    return states

# backward scan is a padded reverse scan
# See https://arxiv.org/abs/1709.04057 Section 2.2
@triton_op("accelerated_scan::scalar_backward", mutates_args=())
def scalar_backward(states: Tensor, forget: Tensor, d_output: Tensor) -> tuple[Tensor, Tensor]:
    B, C, T = states.shape
    torch._check(states.shape == forget.shape)
    torch._check(states.device == forget.device)
    torch._check(states.is_cuda); torch._check(forget.is_cuda)
    torch._check(states.dtype == forget.dtype)

    d_output = d_output.contiguous()
    states = states.contiguous()
    forget = forget.contiguous()

    d_forget = torch.empty_like(forget)
    d_inputs = torch.empty_like(states)
    wrap_triton(backward_scan)[(B, C)](
        forget,
        states,
        d_output,
        d_inputs,
        d_forget,
        seqlen=T,
        BLOCK=2048, # MUST add this or somehow the default value of BLOCK=2048 is not being passed to the kernel (when torch.compile)
        enable_fp_fusion=False,
    )

    return d_forget, d_inputs

def _setup_context(ctx, inputs, output):
    forget, _inputs = inputs
    ctx.save_for_backward(output, forget)

def _backward(ctx, d_output):
    states, forget = ctx.saved_tensors
    return scalar_backward(states, forget, d_output)

scalar_forward.register_autograd(_backward, setup_context=_setup_context)


def scan(forget, inputs):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("forget") and :math:`b_t` ("inputs") are sequences of vectors.

    Arguments:
        forget (torch.Tensor): shape (B, C, T), must be contiguous.  T can be any length.
        inputs (torch.Tensor): shape (B, C, T), must be contiguous.

    Returns:
        (torch.Tensor): same shape as inputs
    """
    return scalar_forward(forget, inputs)
