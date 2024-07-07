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
    d_beta = beta.new_zeros(NH, T)
    w1 = w.new_zeros(NH, T, D)
    u1 = u.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.decay_values_backward(d_out_w, d_out_u, k, v, beta, d_k, d_v, d_beta, w1, u1)

    # print(u[0], 'u')
    # print(u1[0], 'u1')
    # print(u1[0] - u[0], 'u1-u0')

    assert allclose(u, u1, atol=1e-5), 'u is wrong'
    assert allclose(w, w1, atol=1e-5), 'w is wrong'

#test_decay_values_forward_thr()

#%%


def tileprint(K, name='K'):
    "format matches tileprint in tk code so you can diff it"
    assert K.shape == (16, 16)
    for laneid in range(32):
        row_top = laneid // 4
        row_bottom = row_top + 8
        col_left = laneid % 4 * 2
        col_right = col_left + 8

        def fmt(r,c,tag):
            odd = "y" in tag
            if odd: # do not print r for odd rows because cuda printf silently runs out of function arguments
                return f"{name}[,{c:02}] {tag}={K[r,c]: .3f}"
            else:
                return f"{name}[{r:02},{c:02}] {tag}={K[r,c]: .3f}"

        print(f"lane={laneid:02}", "    ".join([
            " ".join([fmt(row_top, col_left, "0x"), fmt(row_top, col_left+1, "0y")]),
            " ".join([fmt(row_bottom, col_left, "1x"), fmt(row_bottom, col_left+1, "1y")]),
            " ".join([fmt(row_top, col_right, "2x"), fmt(row_top, col_right+1, "2y")]),
            " ".join([fmt(row_bottom, col_right, "3x"), fmt(row_bottom, col_right+1, "3y")])
        ]))



def decay_values_backward_wb1(d_out_w, d_out_u, k, v, w, u, beta):
    NH, T, D = shape(None, k, None, beta)
    S = T

    K = einsum('nsd,ntd->nst', k, k) # (T,T) matrix

    WB = w.new_zeros(NH, T, S, D) # d w_t / d beta_s
    KWB = w.new_zeros(NH, S, D)

    for t in range(T):
        KWB[:, :t] = einsum('nt,ntsk->nsk', K[:, :t, t], WB[:, :t, :t])
        WB[:, t, :t] = einsum('n,nsk->nsk', -beta[:, t], KWB[:, :t])
        WB[:, t, t] = k[:, t] - einsum('nt,ntk->nk', K[:, :t, t], w[:, :t])


    UB = u.new_zeros(NH, T, S, D) # d u_t / d beta_s

    for t in range(T):
        for s in range(t):
            for l in range(t):
                c = (k[:, t] * k[:, l]).sum(-1)
                UB[:, t, s] -= einsum('n,n,nj->nj', beta[:, t], c, UB[:, l, s])
                    
        UB[:, t, t] = v[:, t] - einsum('ns,nsd->nd', K[:, :t, t], u[:, :t])


    d_beta = einsum('ntk,ntsk->ns', d_out_w, WB) + einsum('ntv,ntsv->ns', d_out_u, UB)

    return d_beta

def decay_values_backward_wb2(d_out_w, d_out_u, k, v, w, u, beta):
    "uses less memory than wb1, but spends more time"
    NH, T, D = shape(None, k, None, beta)
    assert w.shape == d_out_w.shape
    S = T

    K = einsum('nsd,ntd->nts', k, k) # (T,T) matrix

    """
    w_t = b_t k_t - b_t \sum_{s=0}^{t-1} k_s^T k_t w_s
    """

    d_beta = beta.new_zeros(NH, T)

    for s in range(S):
        WBT = w.new_zeros(NH, T, D)

        WBT[:, s] += k[:, s] - einsum('ns,nsk->nk', K[:, s, :s], w[:, :s])
        for t in range(s+1,T):
            #WBT[:, t] += -beta[:, t] * einsum('nj,njk->nk', K[:, :t, t], WBT[:, :t])
            WBT[:, t] = -beta[:, t] * einsum('njk,njt->nkt', WBT, K)[:, :, t]

        d_beta[:, s] = einsum('ntk,ntk->n', d_out_w, WBT)


    UB = u.new_zeros(NH, T, S, D) # d u_t / d beta_s
    K = einsum('nsd,ntd->nts', k, k) # (T,T) matrix
    for t in range(T):
        for s in range(t):
            for l in range(t):
                UB[:, t, s] -= einsum('n,n,nj->nj', beta[:, t], K[:, t, l], UB[:, l, s])
        UB[:, t, t] = v[:, t] - einsum('ns,nsv->nv', K[:, :t, t], u[:, :t])
    d_beta += einsum('ntv,ntsv->ns', d_out_u, UB)

    return d_beta


def decay_values_backward_wb3(d_out_w, d_out_u, k, v, w, u, beta):
    NH, T, D = shape(None, k, None, beta)
    assert w.shape == d_out_w.shape
    S = T

    K = einsum('nsd,ntd->nts', k, k) # (T,T) matrix
    K = K.tril(diagonal=-1) # make_causal(0); set_diagonal(0)

    w_bases = k - einsum('nts,nsk->ntk', K, w)
    u_bases = v - einsum('nts,nsv->ntv', K, u)
    tileprint(w_bases[0], "bases")

    K = einsum('nt,nts->nts', beta, K) # multiply each row of K by beta
    #tileprint(K[0], "Kbeta")

    decays = k.new_zeros(NH, T, S) # (T, T) matrix

    for t in range(T):
        decays[:, t, t] = 1 # add_diagonal(1)
        pre = einsum('ntj,njs->nts', K, decays)[:, t, :]
        decays[:, t, :] -= pre
    #d_beta = beta.new_zeros(NH, T)
    #d_beta = einsum('nts,nts->ns', decays, d_out_w_bases)

    d_out_w_bases = einsum('ntk,nsk->nts', d_out_w, w_bases) # (T, T) matrix
    d_out_w_bases *= decays

    #tileprint(d_out_w_bases[0], "pre_reg")

    d_beta = d_out_w_bases.sum(dim=1)

    d_out_u_bases = einsum('ntv,nsv->nts', d_out_u, u_bases) * decays # (T, T) matrix
    d_beta += d_out_u_bases.sum(dim=1)

    return d_beta


def test_decay_values_backward():
    NH, T, D = 1, 16, 16
    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.float32)
    w, u = decay_values_forward_thr(k, v, beta)
    d_out_w = torch.randn_like(w) / D**0.5
    d_out_u = torch.randn_like(u) / D**0.5
    d_beta = decay_values_backward_wb3(d_out_w, d_out_u, k, v, w, u, beta)

    d_out_w1 = d_out_w.clone()
    d_out_u1 = d_out_u.clone()
    d_beta1 = decay_values_backward_wb2(d_out_w1, d_out_u1, k, v, w, u, beta)

    print(d_beta, 'wb3')
    print(d_beta1, 'wb2')
    assert allclose(d_beta, d_beta1, atol=1e-5), 'd_beta is wrong'

test_decay_values_backward()

def test_decay_values_backward_cu():
    NH, T, D = 1, 16, 16

    q, k, v, beta = make_example(NH, T, D, device='cuda', dtype=torch.bfloat16)
    w, u = decay_values_forward_thr(k, v, beta)
    d_out_w = torch.randn_like(w) / D**0.5
    d_out_u = torch.randn_like(u) / D**0.5
    d_beta = decay_values_backward_wb3(d_out_w, d_out_u, k, v, w, u, beta)

    d_k = k.new_zeros(NH, T, D)
    d_v = v.new_zeros(NH, T, D)
    d_beta1 = beta.new_zeros(NH, T)
    w1 = w.new_zeros(NH, T, D)
    u1 = u.new_zeros(NH, T, D)
    from accelerated_scan import kitten
    kitten.decay_values_backward(d_out_w, d_out_u, k, v, beta, d_k, d_v, d_beta1, w1, u1)

    assert allclose(u, u1, atol=1e-5), 'u is wrong'
    assert allclose(w, w1, atol=1e-5), 'w is wrong'
    print(d_beta, 'ref')
    print(d_beta1, 'hyp')
    # XXX: atol=1e-2 might be too low. cast to float32?
    assert allclose(d_beta, d_beta1, atol=1e-2), 'd_beta is wrong'

test_decay_values_backward_cu()