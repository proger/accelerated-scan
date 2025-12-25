import torch
import triton
import triton.language as tl


@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def combine1_complex(xl_r, xl_i, fl_r, fl_i, xr_r, xr_i, fr_r, fr_i):
    x_r, x_i = complex_mul(xl_r, xl_i, fr_r, fr_i)
    x_r += xr_r
    x_i += xr_i
    f_r, f_i = complex_mul(fl_r, fl_i, fr_r, fr_i)
    return x_r, x_i, f_r, f_i


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
    state_x_r = tl.zeros((), dtype=tl.float32)
    state_x_i = tl.zeros((), dtype=tl.float32)

    for block_id in tl.range(0, nblocks):
        t = block_id * BLOCK + idx
        offset = (sequence_id * seqlen + t) * 2

        inputs_r = tl.load(inputs + offset, mask=t < seqlen, other=0.0)
        inputs_i = tl.load(inputs + offset + 1, mask=t < seqlen, other=0.0)
        forget_r = tl.load(forget + offset, mask=t < seqlen, other=1.0)
        forget_i = tl.load(forget + offset + 1, mask=t < seqlen, other=0.0)

        states_r, states_i, f_r, f_i = tl.associative_scan(
            (inputs_r, inputs_i, forget_r, forget_i),
            axis=0,
            combine_fn=combine1_complex,
            reverse=False,
        )

        block_x_r = tl.sum(tl.where(idx == BLOCK - 1, states_r, 0.0), axis=0)
        block_x_i = tl.sum(tl.where(idx == BLOCK - 1, states_i, 0.0), axis=0)
        block_f_r = tl.sum(tl.where(idx == BLOCK - 1, f_r, 0.0), axis=0)
        block_f_i = tl.sum(tl.where(idx == BLOCK - 1, f_i, 0.0), axis=0)

        state_mul_r, state_mul_i = complex_mul(state_x_r, state_x_i, f_r, f_i)
        states_r = states_r + state_mul_r
        states_i = states_i + state_mul_i

        tl.store(states + offset, states_r, mask=t < seqlen)
        tl.store(states + offset + 1, states_i, mask=t < seqlen)

        state_mul_r, state_mul_i = complex_mul(state_x_r, state_x_i, block_f_r, block_f_i)
        state_x_r = state_mul_r + block_x_r
        state_x_i = state_mul_i + block_x_i


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
    state_x_r = tl.zeros((), dtype=tl.float32)
    state_x_i = tl.zeros((), dtype=tl.float32)
    state_f_r = tl.full((), 1.0, dtype=tl.float32)
    state_f_i = tl.zeros((), dtype=tl.float32)

    for block_id in tl.range(0, nblocks):
        reverse_block = nblocks - 1 - block_id
        t = reverse_block * BLOCK + idx
        offset = (sequence_id * seqlen + t) * 2

        d_output_r = tl.load(d_output + offset, mask=t < seqlen, other=0.0)
        d_output_i = tl.load(d_output + offset + 1, mask=t < seqlen, other=0.0)

        shifted_forget_r = tl.load(forget + offset + 2, mask=t < seqlen - 1, other=1.0)
        shifted_forget_i = -tl.load(forget + offset + 3, mask=t < seqlen - 1, other=0.0)

        d_inputs_r, d_inputs_i, f_r, f_i = tl.associative_scan(
            (d_output_r, d_output_i, shifted_forget_r, shifted_forget_i),
            axis=0,
            combine_fn=combine1_complex,
            reverse=True,
        )

        block_x_r = tl.sum(tl.where(idx == 0, d_inputs_r, 0.0), axis=0)
        block_x_i = tl.sum(tl.where(idx == 0, d_inputs_i, 0.0), axis=0)
        block_f_r = tl.sum(tl.where(idx == 0, f_r, 0.0), axis=0)
        block_f_i = tl.sum(tl.where(idx == 0, f_i, 0.0), axis=0)

        state_mul_r, state_mul_i = complex_mul(d_inputs_r, d_inputs_i, state_f_r, state_f_i)
        d_inputs_r, d_inputs_i = state_mul_r + state_x_r, state_mul_i + state_x_i
        tl.store(d_inputs + offset, d_inputs_r, mask=t < seqlen)
        tl.store(d_inputs + offset + 1, d_inputs_i, mask=t < seqlen)

        shifted_states_r = tl.load(states + offset - 2, mask=t > 0, other=0.0)
        shifted_states_i = -tl.load(states + offset - 1, mask=t > 0, other=0.0)

        d_forget_r, d_forget_i = complex_mul(shifted_states_r, shifted_states_i, d_inputs_r, d_inputs_i)
        tl.store(d_forget + offset, d_forget_r, mask=t < seqlen)
        tl.store(d_forget + offset + 1, d_forget_i, mask=t < seqlen)

        state_mul_r, state_mul_i = complex_mul(block_x_r, block_x_i, state_f_r, state_f_i)
        state_x_r, state_x_i = state_mul_r + state_x_r, state_mul_i + state_x_i
        state_f_r, state_f_i = complex_mul(block_f_r, block_f_i, state_f_r, state_f_i)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forget, inputs):
        if not torch.is_complex(forget):
            raise TypeError("forget must be a complex tensor")
        real_inputs = not torch.is_complex(inputs)
        B, C, T = forget.shape
        assert inputs.shape == (B, C, T)
        assert forget.is_contiguous()
        assert inputs.is_contiguous()

        if real_inputs:
            inputs = torch.complex(inputs, torch.zeros_like(inputs))

        forget_real = torch.view_as_real(forget)
        inputs_real = torch.view_as_real(inputs)
        states_real = torch.empty_like(inputs_real)
        forward_scan[(B, C)](
            forget_real,
            inputs_real,
            states_real,
            seqlen=T,
            enable_fp_fusion=False,
        )

        states = torch.view_as_complex(states_real)
        ctx.save_for_backward(forget, states)
        ctx.real_inputs = real_inputs
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, d_output):
        forget, states = ctx.saved_tensors
        B, C, T = forget.shape
        real_inputs = ctx.real_inputs

        d_output = d_output.contiguous()
        assert forget.is_contiguous()
        assert states.is_contiguous()

        forget_real = torch.view_as_real(forget)
        states_real = torch.view_as_real(states)
        d_output_real = torch.view_as_real(d_output)

        d_forget_real = torch.empty_like(forget_real)
        d_inputs_real = torch.empty_like(states_real)
        backward_scan[(B, C)](
            forget_real,
            states_real,
            d_output_real,
            d_inputs_real,
            d_forget_real,
            seqlen=T,
            enable_fp_fusion=False,
        )

        d_forget = torch.view_as_complex(d_forget_real)
        d_inputs = torch.view_as_complex(d_inputs_real)
        if real_inputs:
            d_inputs = d_inputs.real
        return d_forget, d_inputs


def scan(forget, inputs):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("forget") and :math:`b_t` ("inputs") are sequences of vectors.

    Arguments:
        forget (torch.Tensor): shape (B, C, T), complex dtype, must be contiguous.
            T can be any length.
        inputs (torch.Tensor): shape (B, C, T), complex or real dtype, must be contiguous.
            If real, the imaginary part is treated as zero.

    Returns:
        (torch.Tensor): same shape as inputs
    """
    return Scan.apply(forget, inputs)
