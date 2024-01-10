import pytest

import torch
from accelerated_scan.ref import scan as scan_ref
from accelerated_scan.warp import scan as scan_warp
from accelerated_scan.triton import scan as scan_triton

# https://arxiv.org/abs/2109.08203
seeds = [3407,4,42,57]
seqlens = [2**i for i in range(5, 17)]
scans = [scan_warp, scan_triton]

def init(seed, batch_size=3, dim=1536, seqlen=32, requires_grad=False):
    torch.manual_seed(seed)
    gates = 0.999 + 0.001 * torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, device="cuda")
    tokens = torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, device="cuda")
    if requires_grad:
        gates.retain_grad()
        tokens.retain_grad()
    return gates, tokens


@pytest.mark.parametrize("scan", scans)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
@torch.inference_mode()
def test_eq_forward(scan, seed, seqlen):
    gates, tokens = init(seed, seqlen=seqlen)
    out = scan(gates, tokens)
    out_ref = scan_ref(gates, tokens)

    print('max error', (out - out_ref).abs().max())
    # print(out, 'out')
    # print(out_ref, 'ref')

    assert torch.allclose(out, out_ref) 


@pytest.mark.parametrize("scan", scans)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
def test_eq_backward(scan, seed, seqlen):
    gates, tokens = init(seed, seqlen=seqlen, requires_grad=True)
    scan(gates, tokens).sum().backward()
    gates_grad = gates.grad
    tokens_grad = tokens.grad
    del gates
    del tokens

    gates_ref, tokens_ref = init(seed, seqlen=seqlen, requires_grad=True)
    scan_ref(gates_ref, tokens_ref).sum().backward()

    assert torch.allclose(gates_grad, gates_ref.grad)
    assert torch.allclose(tokens_grad, tokens_ref.grad)
    