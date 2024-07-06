#%%

import os
import torch
from torch import einsum, randn, allclose, stack, eye, manual_seed, no_grad, set_float32_matmul_precision, compile, arange

set_float32_matmul_precision('high')

def shape(q, k, v, beta=None):    
    NH, T, D = (q if q is not None else k).shape
    if q is not None:
        assert q.shape == (NH, T, D)
    if v is not None:
        assert k.shape == v.shape
    if beta is not None:
        assert beta.shape == (NH, T)
    return NH, T, D


def make_example(NH, T, D, device='cpu', dtype=torch.float32):
    manual_seed(0)
    q = randn(NH, T, D, device=device, dtype=dtype) / D**0.5
    q.requires_grad_()
    k = torch.randn_like(q) / D**0.5
    k.requires_grad_()
    v = torch.randn_like(k) / D**0.5
    v.requires_grad_()
    beta = randn(NH, T, device=device, dtype=dtype).sigmoid()
    beta.requires_grad_()
    return q, k, v, beta


def decay_values_forward_thr(k, v, beta):
    NH, T, D = shape(None, k, v, beta)
    DV = D

    """
    w_t = b_t k_t - b_t \sum_{s=0}^{t-1} k_s^T k_t w_s
    u_t = b_t v_t - b_t \sum_{s=0}^{t-1} k_s^T k_t u_s
    """

    w = k.new_zeros(NH, T, D)
    u = v.new_zeros(NH, T, D)

    K = einsum('ntd,nsd->nts', k, k) # (T,T) matrix
    K = einsum('nt,nts->nts', beta, K) # multiply each row of K by beta

    k = einsum('nt,ntd->ntd', beta, k)
    v = einsum('nt,ntd->ntd', beta, v)

    for t in range(T):
        c_w = einsum('nts,nsd->ntd', K, w)
        w[:, t] = k[:, t, :] - c_w[:, t, :]
        c_u = einsum('nts,nsd->ntd', K, u)
        u[:, t] = v[:, t, :] - c_u[:, t, :]

    return w, u


def decay_values_forward_thr2(k, v, beta):
    NH, T, D = shape(None, k, v, beta)
    DV = D
    K = einsum('nsd,ntd->nst', k, k) # (T,T) matrix

    beta_ = beta.unsqueeze(-1)

    """
    w_0 = b_0 k_0
    w_t = b_t k_t - b_t \sum_{s=0}^{t-1} k_s^T k_t w_s
    u_t = b_t v_t - b_t \sum_{s=0}^{t-1} k_s^T k_t u_s
    """

    w = beta_ * k.clone()
    u = beta_ * v.clone()

    for t in range(1, T):
        c_w = 0 # row
        c_u = 0 # row
        for s in range(t):
            c = K[:, s, t]
            c_w += c * w[:, s]
            c_u += c * u[:, s]
        b = beta_[:, t]

        w[:, t] -= b * c_w
        u[:, t] -= b * c_u

    return w, u

torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)


def test_decay_values_forward_thr2():
    NH, T, D = 1, 16, 16

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)

    #v = torch.arange(1, T*D+1).to(torch.bfloat16).reshape(T, D).unsqueeze(0).expand(NH, T, D).to('cuda') / T*D
    #beta = torch.arange(1, T+1).to(torch.bfloat16).unsqueeze(0).expand(NH, T).to('cuda') / T
    w, u = decay_values_forward_thr(k, v, beta)

    w1, u1 = decay_values_forward_thr2(k, v, beta)

    assert allclose(u, u1, atol=1e-2), 'u is wrong'

def test_decay_values_forward_thr():
    NH, T, D = 1, 16, 16

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)

    #v = torch.arange(1, T*D+1).to(torch.bfloat16).reshape(T, D).unsqueeze(0).expand(NH, T, D).to('cuda')
    #beta = torch.arange(1, T+1).to(torch.bfloat16).unsqueeze(0).expand(NH, T).to('cuda') / T
    w, u = decay_values_forward_thr(k, v, beta)

    print(beta[0], 'beta')
    print(v[0], 'v')

    d_out_w = q.new_zeros(NH, T, D)
    d_out_u = q.new_zeros(NH, T, D)
    d_k = q.new_zeros(NH, T, D)
    d_v = q.new_zeros(NH, T, D)
    d_beta = q.new_zeros(NH, T)
    d_k = k.new_zeros(NH, T, D)
    d_v = v.new_zeros(NH, T, D)
    d_beta = beta.new_zeros(NH, T)
    w1 = w.new_zeros(NH, T, D)
    u1 = u.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.decay_values_backward(d_out_w, d_out_u, k, v, beta, d_k, d_v, d_beta, w1, u1)

    print(u[0], 'u')
    print(u1[0], 'u1')
    print(u1[0] - u[0], 'u1-u0')

    assert allclose(u, u1, atol=1e-5), 'u is wrong'
    assert allclose(w, w1, atol=1e-5), 'w is wrong'

test_decay_values_forward_thr()

