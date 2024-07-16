#%%

import os
import torch
from torch import einsum, randn, allclose, stack, eye, manual_seed, no_grad, set_float32_matmul_precision, compile, arange

import pytest

from tests.deltanet import tileprint
from tests.deltanet import shape, make_example
from tests.deltanet import decay_values_backward as decay_values_backward_ref
from tests.deltanet import decay_values as decay_values_ref
from tests.deltanet import forward as forward_ref



def test_decay_values_backward_cu():
    NH, T, D = 1, 16, 64

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)
    d_out_w = torch.randn_like(k) / D**0.5
    d_out_u = torch.randn_like(v) / D**0.5
    d_out_y = torch.randn_like(v) / D**0.5

    w, u, y = decay_values_ref(q, k, v, beta)

    d_q0, d_k0, d_v0, d_beta0 = decay_values_backward_ref(
        d_out_w.clone(), d_out_u.clone(), d_out_y.clone(), q, k, v, beta
    )

    d_q1 = q.new_zeros(NH, T, D)
    d_k1 = k.new_zeros(NH, T, D)
    d_v1 = v.new_zeros(NH, T, D)
    d_beta1 = beta.new_zeros(NH, T)
    w1 = k.new_zeros(NH, T, D)
    u1 = v.new_zeros(NH, T, D)
    y1 = v.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.decay_values_backward(
        d_out_w.clone(), d_out_u.clone(), d_out_y.clone(), q, k, v, beta,
        d_q1, d_k1, d_v1, d_beta1, w1, u1, y1
    )

    #torch.set_printoptions(precision=4, sci_mode=False, linewidth=300)
    torch.set_printoptions(linewidth=300)

    assert allclose(u, u1, atol=1e-2), 'u is wrong'
    assert allclose(w, w1, atol=1e-3), 'w is wrong'

    print(d_q0 - d_q1, 'd_q diff')
    assert allclose(d_q0, d_q1, atol=1e-5), 'd_q is wrong'

    print(d_k0, 'd_k ref')
    print(d_k1, 'd_k hyp')
    #print((d_k1 - d_k0).abs().topk(10), 'd_k diff')
    assert allclose(d_k0, d_k1, atol=1e-2), 'd_k is wrong' # ???
    assert allclose(d_v0, d_v1, atol=1e-3), 'd_v is wrong'

    print(d_beta0, 'ref')
    print(d_beta1, 'hyp')
    # XXX: atol=1e-2 might be too low. cast to float32?
    assert allclose(d_beta0, d_beta1, atol=1e-2), 'd_beta is wrong'



def test_forward_cu():
    NH, T, D = 1, 64, 16

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

    w2 = w.new_zeros(NH, T, D)
    u2 = u.new_zeros(NH, T, D)
    y2 = y.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.delta_forward(q, k, v, beta, w2, u2, y2)

    #print((w2 - w).abs().max())

    assert allclose(u, u2, atol=1e-2), 'u2 is wrong'
    assert allclose(w, w2, atol=1e-2), 'w2 is wrong'
    # print(y, 'y')
    # print(y2, 'y2')
    #print(y - y2, 'y diff')
    assert allclose(y, y2, atol=1e-2), 'y2 is wrong'


if __name__ == '__main__':
    pytest.main([__file__])