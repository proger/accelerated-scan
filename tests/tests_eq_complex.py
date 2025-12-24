import pytest
import torch

from accelerated_scan.complex import scan as scan_complex
from accelerated_scan.ref import scan as scan_ref

seeds = [1, 11, 22]
seqlens = [2**i for i in range(12, 18)] + [16323, 8000, 33753, 127323]
atol, rtol = 1e-5, 1e-4


def init(seed, batch_size=1, dim=512, seqlen=32, requires_grad=False):
    torch.manual_seed(seed)
    angle = 2 * torch.pi * torch.rand(
        batch_size,
        dim,
        seqlen,
        dtype=torch.float32,
        device="cuda",
    )
    radius = 0.4 + 0.6 * torch.rand_like(angle)
    forget = torch.polar(radius, angle).to(torch.complex64).requires_grad_(requires_grad)
    inputs = torch.randn(
        batch_size,
        dim,
        seqlen,
        requires_grad=requires_grad,
        device="cuda",
    ).tanh()
    assert not torch.isnan(inputs).any()
    if requires_grad:
        forget.retain_grad()
        inputs.retain_grad()
    return forget, inputs


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
@torch.inference_mode()
def test_complex_forward(seed, seqlen):
    forget, inputs = init(seed, seqlen=seqlen)
    out = scan_complex(forget, inputs)
    out_ref = scan_ref(forget, inputs)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
def test_complex_backward(seed, seqlen):
    forget, inputs = init(seed, seqlen=seqlen, requires_grad=True)
    scan_complex(forget, inputs).real.mean().backward()
    forget_grad = forget.grad
    inputs_grad = inputs.grad
    del forget
    del inputs

    forget_ref, inputs_ref = init(seed, seqlen=seqlen, requires_grad=True)
    scan_ref(forget_ref, inputs_ref).real.mean().backward()

    torch.testing.assert_close(forget_grad, forget_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(inputs_grad, inputs_ref.grad, atol=atol, rtol=rtol)
