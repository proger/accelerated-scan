#%%

import os
import torch
from torch import einsum, randn, allclose, stack, eye, manual_seed, no_grad, set_float32_matmul_precision, compile, arange

import pytest

from tests.deltanet import tileprint
from tests.deltanet import shape, make_example
from tests.deltanet import backward as backward_ref
from tests.deltanet import forward as forward_ref



@pytest.mark.parametrize('T', [16, 32, 64, 128, 256])
@pytest.mark.parametrize('D', [16, 32, 64, 128])
def test_backward(T, D):
    NH, T, D = 1, T, D

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)
    d_out_w = torch.zeros_like(k) # placeholder
    d_out_u = torch.zeros_like(v) # placeholder
    d_out_y = torch.randn_like(v) / D**0.5 # actual gradient

    d_q0, d_k0, d_v0, d_beta0 = backward_ref(
        d_out_w.clone(), d_out_u.clone(), d_out_y.clone(), q, k, v, beta,
        chunk_size=16
    )

    d_q1 = q.new_zeros(NH, T, D)
    d_k1 = k.new_zeros(NH, T, D)
    d_v1 = v.new_zeros(NH, T, D)
    d_beta1 = beta.new_zeros(NH, T)
    u1 = v.new_zeros(NH, T, D) # placeholder
    from accelerated_scan import kitten
    kitten.delta_backward(
        d_out_w.clone(), d_out_u.clone(), d_out_y.clone(),
        q, k, v, beta,
        d_q1, d_k1, d_v1, d_beta1,
        u1
    )

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=300)
    #torch.set_printoptions(linewidth=300)

    #print(d_q0 - d_q1, 'd_q diff')
    assert allclose(d_q0, d_q1, atol=1e-2), 'd_q is wrong'

    #print(d_k0, 'd_k ref')
    #print(d_k1, 'd_k hyp')
    #print((d_k1 - d_k0).abs().topk(10), 'd_k diff')
    assert allclose(d_k0, d_k1, atol=1e-2), 'd_k is wrong' # ???
    assert allclose(d_v0, d_v1, atol=1e-2), 'd_v is wrong'

    #print(d_beta0, 'ref')
    #print(d_beta1, 'hyp')
    # XXX: atol=1e-2 might be too low. cast to float32?
    assert allclose(d_beta0, d_beta1, atol=1e-2), 'd_beta is wrong'


@pytest.mark.parametrize('T', [16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize('D', [16, 32, 64, 128])
def test_forward(T, D):
    NH, T, D = 1, T, D

    chunk_size = 16
    C = T // chunk_size

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)

    # q, k, v, beta = (
    #     q.view(NH, C, chunk_size, D).view(NH*C,chunk_size,D),
    #     k.view(NH, C, chunk_size, D).view(NH*C,chunk_size,D),
    #     v.view(NH, C, chunk_size, D).view(NH*C,chunk_size,D),
    #     beta.view(NH, C, chunk_size).view(NH*C,chunk_size)
    # )

    torch.set_printoptions(linewidth=300, sci_mode=False, precision=3)
    w, u, y = forward_ref(q, k, v, beta, chunk_size=chunk_size)

    # w, u, y = (
    #     w.view(NH, C, chunk_size, D).view(NH,T,D),
    #     u.view(NH, C, chunk_size, D).view(NH,T,D),
    #     y.view(NH, C, chunk_size, D).view(NH,T,D)
    # )

    y2 = y.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.delta_forward(q, k, v, beta, y2)

    assert allclose(y, y2, atol=1e-2), 'y2 is wrong'


@pytest.mark.parametrize('T', [2048, 4096, 8192, 16384])
@pytest.mark.parametrize('D', [16, 32, 64, 128])
def test_longf(T, D):
    return test_forward(T, D)


if __name__ == '__main__':
    #pytest.main([__file__, "--disable-warnings", "-v"])
    #test_forward(T=128, D=16)
    test_backward(T=128, D=16)
    #test_forward(T=32, D=16)
