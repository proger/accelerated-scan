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
    T: tl.constexpr,
    REVERSE: tl.constexpr = False,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    stride = sequence_id * T + tl.arange(0, T)

    inputs_ = tl.load(inputs + stride)
    forget_ = tl.load(forget + stride)

    states_, forget_states_ = tl.associative_scan((inputs_, forget_), axis=0, combine_fn=combine1, reverse=REVERSE)
    tl.store(states + stride, states_)


@triton.jit
def backward_scan(
    forget,
    states,
    d_output,
    d_forget,
    d_inputs,
    T: tl.constexpr,
):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    stride = sequence_id * T + tl.arange(0, T)

    d_output_ = tl.load(d_output + stride)
    shifted_forget_ = tl.load(forget + stride + 1, mask=tl.arange(0, T) < T-1, other=1.0)

    d_inputs_, _ = tl.associative_scan((d_output_, shifted_forget_), axis=0, combine_fn=combine1, reverse=True)
    tl.store(d_inputs + stride, d_inputs_)

    shifted_states_ = tl.load(states + stride - 1, mask=tl.arange(0, T) > 0, other=0.0)
    d_forget_ = shifted_states_ * d_inputs_
    tl.store(d_forget + stride, d_forget_)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forget, inputs):
        B, C, T = forget.shape
        assert inputs.shape == (B, C, T)
        assert forget.is_contiguous()
        assert inputs.is_contiguous()

        states = torch.zeros_like(inputs)
        forward_scan[(B,C)](forget, inputs, states, T=T, enable_fp_fusion=False)

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

        if False:
            shifted_forget = torch.cat([forget, torch.ones_like(forget[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
            forward_scan[(B,C)](shifted_forget, d_output, d_inputs, T=T, REVERSE=True, enable_fp_fusion=False)
            shifted_states = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
            d_forget = shifted_states * d_inputs
        else:
            backward_scan[(B,C)](forget, states, d_output, d_forget, d_inputs, T=T, enable_fp_fusion=False)

        return d_forget, d_inputs


def scan(forget, inputs):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("forget") and :math:`b_t` ("inputs") are sequences of vectors.

    Arguments:
        forget (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.
        inputs (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.

    Returns:
        (torch.Tensor): shape (B, C, T)
    """
    return Scan.apply(forget, inputs)
