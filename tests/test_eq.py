import pytest

import torch
from accelerated_scan.ref import scan as scan_ref
from accelerated_scan.warp import scan_forward as scan
#from accelerated_scan.triton import scan

# https://arxiv.org/abs/2109.08203
seeds = [3407,4,42,57]
seqlens = [2**i for i in range(5, 13)]


def init(seed, batch_size=3, dim=1536, seqlen=32, requires_grad=False):
    torch.manual_seed(seed)
    gates = 0.999 + 0.001 * torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, device="cuda")
    tokens = torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, device="cuda")
    if requires_grad:
        gates.retain_grad()
        tokens.retain_grad()
    return gates, tokens


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
@torch.inference_mode()
def test_eq_forward(seed, seqlen):
    gates, tokens = init(seed, seqlen=seqlen)
    out = scan(gates, tokens)
    out_ref = scan_ref(gates, tokens)

    print('max error', (out - out_ref).abs().max())
    print(out, 'out')
    print(out_ref, 'ref')

    assert torch.allclose(out, out_ref) 


@pytest.mark.parametrize("seed", seeds)
def test_eq_backward(seed):
    gates, tokens = init(seed, requires_grad=True)
    scan(gates, tokens).sum().backward()

    gates_ref, tokens_ref = init(seed, requires_grad=True)
    scan_ref(gates_ref, tokens_ref).sum().backward()

    assert torch.allclose(gates.grad, gates_ref.grad)
    assert torch.allclose(tokens.grad, tokens_ref.grad)
    