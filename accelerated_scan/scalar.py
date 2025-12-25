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


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forget, inputs):
        B, C, T = forget.shape
        assert inputs.shape == (B, C, T)
        assert forget.is_contiguous()
        assert inputs.is_contiguous()

        states = torch.empty_like(inputs)
        forward_scan[(B, C)](forget, inputs, states, seqlen=T, enable_fp_fusion=False)

        ctx.save_for_backward(forget, states)
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, d_output):
        forget, states = ctx.saved_tensors
        B, C, T = forget.shape

        d_output = d_output.contiguous()
        assert forget.is_contiguous()
        assert states.is_contiguous()

        d_forget = torch.empty_like(forget)
        d_inputs = torch.empty_like(states)
        backward_scan[(B, C)](
            forget,
            states,
            d_output,
            d_inputs,
            d_forget,
            seqlen=T,
            enable_fp_fusion=False,
        )

        return d_forget, d_inputs


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
    return Scan.apply(forget, inputs)
