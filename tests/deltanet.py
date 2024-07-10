"""
DeltaNet implementation reference for Accelerated Scan.  DeltaNet performs efficient management of a large fixed-sized memory.

For a simple single chunk version see `forward_simple`.
It computes decayed values by a little bit of recurrence (`decay_values`)
and then applies linear attention (`causal_attend`).

`forward_chunkwise` is inspired by Yang 2024. It applies single chunk version pointwise and
then performs chunk-level stitching.

forward_ogloop and forward_scanloop are reference implementations of straightforward recurrences.

References:
[1] The WY Representation for Products of Householder Matrices (Bischof and Van Loan 1985)

Method 1, section 3 guides `decay_values`.
https://ecommons.cornell.edu/items/92a11030-dca1-45d4-a0ba-732cf962b2b2

[2] Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Yang et al 2024)

- equation 5 is a specialization of method 1 of [1] is in `decay_values`
- equation 6 is application of decayed keys to values is also in `decay_values`
- `forward_chunkwise` uses the distributed form of equation 7 and 8
  (actually look the two equations before it instead, they are easier to read)

https://arxiv.org/abs/2406.06484 

[3] Linear Transformers Are Secretly Fast Weight Programmers (Schlag et al 2021)

Introduction to Transformers as RNNs. Ignore all of the kernel stuff.
https://arxiv.org/abs/2102.11174
"""
#%%

import os
os.environ['TORCH_LOGS'] = 'output_code'
import torch
from torch import einsum, randn, allclose, stack, eye, manual_seed, no_grad, set_float32_matmul_precision, compile, arange

set_float32_matmul_precision('high')


def decay_values(k, v, beta):
    "decay values applying deltanet forgetting rules"
    NH, T, D = shape(None, k, v, beta)

    beta_ = beta.unsqueeze(-1)
    w = beta_ * k.clone()
    u = beta_ * v.clone()
    K = einsum('nsd,ntd->nst', k, k) # (T,T) matrix

    for t in range(1,T):
        w[:, t] -= beta_[:, t] * einsum('nt,ntd->nd', K[:, :t, t], w[:, :t].clone())
        u[:, t] -= beta_[:, t] * einsum('nt,ntd->nd', K[:, :t, t], u[:, :t].clone())

    return w, u


def causal_attend(q, k, v, diagonal=0):
    "apply linear attention with a causal mask"
    NH, T, D = shape(q, k, v)
    mask = q.new_ones(T, T).tril(diagonal=diagonal)
    y = einsum("nsi,nti,st,ntj->nsj", q, k, mask, v)
    return y


def forward_simple(q, k, v, beta):
    "simple deltanet: linear attention to decayed values"
    w, u = decay_values(k, v, beta)
    return causal_attend(q, k, u)


def forward_chunkwise(q, k, v, beta, chunk_size=2):
    NH, T, D = shape(q, k, v, beta)
    C = T // chunk_size
    q_, k_, v_, beta_ = (
        q.view(NH*C, chunk_size, D), k.view(NH*C, chunk_size, D),
        v.view(NH*C, chunk_size, D), beta.view(NH*C, chunk_size)
    )

    # evaluate all chunks in parallel
    w, u = decay_values(k_, v_, beta_)
    y = causal_attend(q_, k_, u)

    # stitch chunks sequentially
    y_delta, _ = stitch_forward(q, k, w, u, C=C, chunk_size=chunk_size)
    return y.view(NH, T, D) + y_delta.view(NH, T, D)


def stitch_forward(q, k, w, u, C, chunk_size):
    "stitch chunks sequentially"
    NH, T, D = shape(q, k, None, None)

    q_ = q.view(NH, C, chunk_size, D)
    k_ = k.view(NH, C, chunk_size, D)
    u = u.view(NH, C, chunk_size, D)
    w = w.view(NH, C, chunk_size, D)

    # materialize the state for the leading chunk    
    state = einsum('ntv,ntk->nvk', u[:, 0], k_[:, 0])

    deltas = [u.new_zeros(NH, chunk_size, D)]
    for c in range(1, C):
        y_delta1, state = stitch1_forward(state, q_[:, c], k_[:, c], w[:, c], u[:, c])
        deltas.append(y_delta1)

    y_delta = torch.stack(deltas, dim=1)

    return y_delta, state


def stitch1_forward(state, q, k, w, u):
    state_decays = einsum('nvk,ntk->ntv', state, w)
    state_add = einsum('ntv,ntk->nvk', u - state_decays, k)

    delta = causal_attend(q, k, state_decays)
    prev_output = einsum('nvk,nsk->nsv', state, q)
    y = prev_output - delta

    return y, state + state_add


def forward_ogloop(q, k, v, beta):
    "reference: w_t = w_{t-1} + beta_t (v_t - w_t k_t) k_t"
    NH, T, D = shape(q, k, v, beta)

    w = k.new_zeros(NH, D, D)
    y = []

    for t in range(T):
        q_ = q[:, t]
        k_ = k[:, t]
        v_ = v[:, t]
        beta_ = beta[:, t].unsqueeze(-1)

        v_old = einsum("nij,nj->ni", w, k_)
        delta = beta_ * (v_ - v_old)
        w = w + einsum("ni,nj->nij", delta, k_)

        y.append(einsum("nij,nj->ni", w, q_))

    return stack(y, dim=1)


def forward_scanloop(q, k, v, beta):
    "reference via linear-time scan: w_t = w_{t-1} (I - beta_t k_t k_t.T) + beta v_t k_t.T"
    NH, T, D = shape(q, k, v, beta)

    w = k.new_zeros(NH, D, D)
    id = eye(D, device=w.device).expand(NH, D, D)
    y = []

    for t in range(T):
        q_ = q[:, t]
        k_ = k[:, t]
        v_ = v[:, t]
        beta_ = beta[:, t].unsqueeze(-1).unsqueeze(-1)
        beta_sqrt_ = beta_.squeeze(-1).sqrt()

        forget = id - einsum("ni,nj->nij", beta_sqrt_ * k_, beta_sqrt_ * k_)
        update = beta_ * einsum("ni,nj->nij", v_, k_)
        w = einsum("nik,nkj->nij", w, forget) + update

        y.append(einsum("nij,nj->ni", w, q_))

    return stack(y, dim=1)


def shape(q, k, v, beta=None):    
    NH, T, D = (q if q is not None else k).shape
    if q is not None:
        assert q.shape == (NH, T, D)
    if v is not None:
        assert k.shape == v.shape
    if beta is not None:
        assert beta.shape == (NH, T)
    return NH, T, D


def make_example(NH, T, D):
    manual_seed(0)
    q = randn(NH, T, D) / D**0.5
    q.requires_grad_()
    k = randn(NH, T, D) / D**0.5
    k.requires_grad_()
    v = randn(NH, T, D) / D**0.5
    v.requires_grad_()
    beta = randn(NH, T).sigmoid()
    beta.requires_grad_()
    return q, k, v, beta


def test_equal(atol=1e-6):
    NH, T, D = 2*3, 128, 16
    #NH, T, D = 1, 8, 3
    q, k, v, beta = make_example(NH, T, D)

    y1 = forward_ogloop(q, k, v, beta)
    y2 = forward_scanloop(q, k, v, beta)
    y3 = forward_simple(q, k, v, beta)

    assert allclose(y1, y2, atol=atol), (y1 - y2).abs().max()
    assert allclose(y1, y3, atol=atol), (y1 - y3).abs().max()

    for chunk_size in (1,2,4,8):
        y = forward_chunkwise(q, k, v, beta, chunk_size)
        assert allclose(y1, y, atol=atol), (y1 - y).abs().max()


test_equal()

#%%


@no_grad()
def attend_backward(d_out, q, k, v, g):
    d_q = einsum('nsv,ntk,ntv,nst->nsk', d_out, k, v, g)
    d_k = einsum('nsv,nsk,ntv,nst->ntk', d_out, q, v, g)
    d_v = einsum('nsv,nsk,ntk,nst->ntv', d_out, q, k, g)
    d_g = einsum('nsv,nsk,ntk,ntv,nst->nst', d_out, q, k, v, g)
    return d_q, d_k, d_v, d_g


@no_grad()
def causal_attend_backward(d_out, q, k, v, diagonal=0):
    NH, T, D = shape(q, k, v)
    mask = q.new_ones(T, T).tril(diagonal=diagonal).unsqueeze(0)
    d_q, q_k, d_v, _d_mask = attend_backward(d_out, q, k, v, mask)
    return d_q, q_k, d_v


class CausalAttend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        return causal_attend(q, k, v)

    @staticmethod
    def backward(ctx, d_out):
        q, k, v = ctx.saved_tensors
        return causal_attend_backward(d_out, q, k, v)


def test_equal_attend_backward(atol=1e-5):
    NH, T, D = 1*1, 512, 64
    q, k, v, beta = make_example(NH, T, D)

    y = causal_attend(q, k, v)
    d_q, d_k, d_v = causal_attend_backward(torch.ones_like(y), q, k, v)
    y.sum().backward()

    assert allclose(q.grad, d_q, atol=atol), 'q.grad is wrong'
    assert allclose(k.grad, d_k, atol=atol), 'k.grad is wrong'
    assert allclose(v.grad, d_v, atol=atol), 'v.grad is wrong'

    ## TODO: test gates (g)
    # print((g_hook.grad - d_g).pow(2).mean(), 'error')
    # print((g_hook.grad - d_g).abs().max(), 'max abs error')
    # assert torch.allclose(g_hook.grad, d_g, atol=1e-1), 'g.grad is wrong'


def test_equal_attend_backward2(atol=1e-5):
    NH, T, D = 1, 2, 2
    q1, k1, v1, beta1 = make_example(NH, T, D)
    y1 = causal_attend(q1, k1, v1)
    (y1 - torch.ones_like(y1)).pow(2).mean().backward()

    q, k, v, beta = make_example(NH, T, D)
    y = CausalAttend.apply(q, k, v)
    (y - torch.ones_like(y)).pow(2).mean().backward()

    # print(q1.grad - q.grad, 'q.grad diff')
    # print(k1.grad - k.grad, 'k.grad diff')
    # print(v1.grad - v.grad, 'v.grad diff')
    # print(k.grad, 'k.grad')
    # print(k1.grad, 'k1.grad')
    # print(v.grad, 'v.grad')
    # print(v1.grad, 'v1.grad')

    assert (q1.grad - q.grad).abs().max() < atol, 'q.grad is wrong'
    assert (k1.grad - k.grad).abs().max() < atol, 'k.grad is wrong'
    assert (v1.grad - v.grad).abs().max() < atol, 'v.grad is wrong'


test_equal_attend_backward()
test_equal_attend_backward2()


#%%

@no_grad()
def decay_values_backward(d_out_w, d_out_u, k, v, beta):
    NH, T, D = shape(None, k, v, beta)

    # recompute w and u TK-style
    w = k.new_zeros(NH, T, D) # ntk
    u = v.new_zeros(NH, T, D) # ntw

    bk = einsum('nt,ntk->ntk', beta, k)
    bv = einsum('nt,ntw->ntw', beta, v)

    K = einsum('ntd,nsd->nts', k, k) # (T,T) matrix
    K = K.tril(diagonal=-1) # make_causal(0); set_diagonal(0)
    bKl = einsum('nt,nts->nts', -beta, K) # multiply each row of K by beta

    for t in range(T):
        c_w = einsum('nts,nsk->ntk', bKl, w)
        w[:, t] = bk[:, t, :] + c_w[:, t, :]
        c_u = einsum('nts,nsw->ntw', bKl, u)
        u[:, t] = bv[:, t, :] + c_u[:, t, :]

    # compute gradients for d_k, d_v, d_beta
    d_k = k.new_zeros(NH, T, D) # nsk
    d_beta = beta.new_zeros(NH, T) # ns
    d_v = v.new_zeros(NH, T, D) # nsv

    eye = torch.eye(D, device=k.device, dtype=k.dtype)
    eye = eye.unsqueeze(0).expand(NH, D, D)

    w_bases = k - einsum('nts,nsk->ntk', K, w)
    u_bases = v - einsum('nts,nsw->ntw', K, u)

    w0 = w.clone() # we will be mutating these, but the kernel also returns the original w and u
    u0 = u.clone()

    d_out_w_backward = d_out_w.clone() # ntk
    d_out_u_backward = d_out_u.clone() # ntw

    for t in range(T-1,-1,-1):
        w[:, t, :] = 0
        k[:, t, :] = 0
        u[:, t, :] = 0
        wk = einsum('njw,njk->nwk', w, k)
        wk = eye - wk
        wk = einsum('n,nwk->nwk', beta[:, t], wk)
        uk = einsum('njw,njk->nwk', u, k)
        uk = einsum('n,nwk->nwk', beta[:, t], uk)

        # d_k
        d_k[:,  t] += einsum('nw,nwk->nk', d_out_w_backward[:, t], wk)
        d_k[:,  t] -= einsum('nw,nwk->nk', d_out_u_backward[:, t], uk)

        decay_w = einsum('nw,nsw->ns', d_out_w_backward[:, t], w[:, :t])
        decay_u = einsum('nw,nsw->ns', d_out_u_backward[:, t], u[:, :t])

        d_k[:, :t] -= einsum('nk,ns->nsk', bk[:, t], decay_w)
        d_k[:, :t] -= einsum('nk,ns->nsk', bk[:, t], decay_u)

        # backpropagate through time
        d_out_w_backward[:, :t] += einsum('nj,nk->njk', bKl[:, t, :t], d_out_w_backward[:, t])
        d_out_u_backward[:, :t] += einsum('nj,nk->njk', bKl[:, t, :t], d_out_u_backward[:, t])

    # d_beta
    d_beta += einsum('ntk,ntk->nt', w_bases, d_out_w_backward)
    d_beta += einsum('ntk,ntk->nt', u_bases, d_out_u_backward)

    # d_v
    d_v = einsum('nt,ntv->ntv', beta, d_out_u_backward)

    return d_k, d_v, d_beta


class DecayValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, beta):
        w, u = decay_values(k, v, beta)
        ctx.save_for_backward(k, v, beta)
        return w, u

    @staticmethod
    def backward(ctx, d_out_w, d_out_u):
        k, v, beta = ctx.saved_tensors
        return decay_values_backward(d_out_w, d_out_u, k, v, beta)


def test_equal_decay_values_backward():
    NH, T, D = 1, 16, 3

    q, k, v, beta = make_example(NH, T, D)
    w, u = decay_values(k, v, beta)
    (w + u - torch.ones_like(w)).pow(2).mean().backward()
    #(w - torch.ones_like(w)).pow(2).mean().backward()

    q1, k1, v1, beta1 = make_example(NH, T, D)
    w1, u1 = DecayValues.apply(k1, v1, beta1)
    (w1 + u1 - torch.ones_like(w1)).pow(2).mean().backward()
    #(w1 - torch.ones_like(w1)).pow(2).mean().backward()

    # print(v.grad, 'v.grad', v.grad.shape)
    # print(v1.grad, 'v1.grad')
    # print(v.grad - v1.grad, 'v diff')
    assert allclose(v.grad, v1.grad, atol=1e-5), 'v1_grad is wrong'

    # print(beta.grad, 'beta.grad du')
    # print(beta1.grad, 'beta1.grad du')
    assert allclose(beta.grad, beta1.grad, atol=1e-5), 'beta1.grad is wrong'

    # print(k.grad, 'k.grad du')
    # print(k1.grad, 'k1.grad du')
    # print(k.grad - k1.grad, 'diff du')
    assert allclose(k.grad, k1.grad, atol=1e-5), 'k1_grad is wrong'

test_equal_decay_values_backward()

# %%


def stitch1_backward(d_y, d_state, state, q, k, w, u):
    state_decays = einsum('nvk,ntk->ntv', state, w)

    d_q1 = einsum('nsv,nvk->nsk', d_y, state) # prev_output
    d_state1 = einsum('nsv,nsk->nvk', d_y, q) # prev_output

    d_q2, d_k1, d_state_decays1 = causal_attend_backward(-d_y, q, k, state_decays) # delta

    d_k2 = einsum('nvk,ntv->ntk', d_state, u - state_decays) # state_add
    d_u = einsum('nvk,ntk->ntv', d_state, k) # state_add
    d_state_decays2 = einsum('nvk,ntk->ntv', -d_state, k) # state_add

    d_state_decays = d_state_decays1 + d_state_decays2
    d_w = einsum('ntv,nvk->ntk', d_state_decays, state) # state_decays
    d_state2 = einsum('ntv,ntk->nvk', d_state_decays, w) # state_decays

    return d_state + d_state1 + d_state2, d_q1 + d_q2, d_k1 + d_k2, d_w, d_u


class Stitch1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, q, k, w, u):
        y, new_state = stitch1_forward(state, q, k, w, u)
        ctx.save_for_backward(state, q, k, w, u)
        return y, new_state

    @staticmethod
    def backward(ctx, d_y, d_state):
        state, q, k, w, u = ctx.saved_tensors
        return stitch1_backward(d_y, d_state, state, q, k, w, u)


def stitch_backward(d_y_delta, q, k, w, u, C, chunk_size):
    NH, T, D = shape(q, k, None, None)

    d_y_delta = d_y_delta.view(NH, C, chunk_size, D)
    q_ = q.view(NH, C, chunk_size, D)
    k_ = k.view(NH, C, chunk_size, D)
    u = u.view(NH, C, chunk_size, D)
    w = w.view(NH, C, chunk_size, D)

    d_q_ = q.new_zeros(NH, C, chunk_size, D)
    d_k_ = k.new_zeros(NH, C, chunk_size, D)
    d_w = w.new_zeros(NH, C, chunk_size, D)
    d_u = u.new_zeros(NH, C, chunk_size, D)
    d_state = w.new_zeros(NH, D, D) # NHVK
    y_delta = u.new_zeros(NH, C, chunk_size, D) # leading chunk has zero delta

    # storing all states for BPTT
    states = k.new_zeros(NH, C, D, D) # NHCVK
    # materialize the state for the leading chunk
    states[:, 0] = einsum('ntv,ntk->nvk', u[:, 0], k_[:, 0])

    for c in range(1, C):
        y_delta[:, c], states[:, c] = stitch1_forward(states[:, c-1], q_[:, c], k_[:, c], w[:, c], u[:, c])

    for c in range(C-1, 0, -1):
        (
            d_state, d_q_[:, c], d_k_[:, c], d_w[:, c], d_u[:, c]
        ) = stitch1_backward(d_y_delta[:, c], d_state, states[:, c-1], q_[:, c], k_[:, c], w[:, c], u[:, c])

    (
        d_state, d_q_[:, 0], d_k_[:, 0], d_w[:, 0], d_u[:, 0]
    ) = stitch1_backward(d_y_delta[:, 0], d_state, torch.zeros_like(d_state), q_[:, 0], k_[:, 0], w[:, 0], u[:, 0])
 
    return d_q_.view(NH, T, D), d_k_.view(NH, T, D), d_w.view(NH, T, D), d_u.view(NH, T, D)


class Stitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, w, u, C, chunk_size):
        y_delta, state = stitch_forward(q, k, w, u, C, chunk_size)
        ctx.save_for_backward(q, k, w, u)
        ctx.C = C
        ctx.chunk_size = chunk_size
        return y_delta

    @staticmethod
    def backward(ctx, d_y_delta):
        q, k, w, u = ctx.saved_tensors
        return *stitch_backward(d_y_delta, q, k, w, u, ctx.C, ctx.chunk_size), None, None


def test_stitch_all(atol=1e-5):
    NH, T, D = 1, 4, 2
    C, chunk_size = 2, 2
    q, k, v, beta = make_example(NH, T, D)
    w, u = decay_values(k, v, beta)

    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    beta.requires_grad_()
    w.retain_grad()
    u.retain_grad()

    y, new_state = stitch_forward(q, k, w, u, C=C, chunk_size=chunk_size)
    loss = (y - torch.ones_like(y)).pow(2).mean()
    loss.backward()

    # print(q.grad, 'q.grad')
    # print(k.grad, 'k.grad')
    # print(v.grad, 'v.grad')
    # print(beta.grad, 'beta.grad')
    # print(w.grad, 'w.grad')
    # print(u.grad, 'u.grad')

    q1, k1, v1, beta1 = make_example(NH, T, D)
    w1, u1 = decay_values(k1, v1, beta1)

    q1.requires_grad_()
    k1.requires_grad_()
    v1.requires_grad_()
    beta1.requires_grad_()
    w1.retain_grad()
    u1.retain_grad()

    y1 = Stitch.apply(q1, k1, w1, u1, C, chunk_size)
    loss = (y1 - torch.ones_like(y1)).pow(2).mean()
    loss.backward()

    assert allclose(y, y1, atol=atol), 'y is wrong'

    assert allclose(u.grad, u1.grad, atol=atol), 'u.grad is wrong'
    assert allclose(v.grad, v1.grad, atol=atol), 'v.grad is wrong'
    # print(k.grad, 'k.grad')
    # print(k1.grad, 'k1.grad')
    assert allclose(k.grad, k1.grad, atol=atol), 'k.grad is wrong'
    assert allclose(q.grad, q1.grad, atol=atol), 'q.grad is wrong'
    assert allclose(beta.grad, beta1.grad, atol=atol), 'beta.grad is wrong'
    assert allclose(w.grad, w1.grad, atol=atol), 'w.grad is wrong'


test_stitch_all()


#%%


class DeltaChunkwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, chunk_size):
        y = forward_chunkwise(q, k, v, beta, chunk_size)
        ctx.save_for_backward(q, k, v, beta)
        ctx.chunk_size = chunk_size
        return y

    @staticmethod
    def backward(ctx, d_y):
        q, k, v, beta = ctx.saved_tensors
        NH, T, D = shape(q, k, v, beta)
        chunk_size = ctx.chunk_size
        C = T // chunk_size

        q_, k_, v_, beta_ = (
            q.view(NH*C, chunk_size, D), k.view(NH*C, chunk_size, D),
            v.view(NH*C, chunk_size, D), beta.view(NH*C, chunk_size)
        )

        w, u = decay_values(k_, v_, beta_)

        d_q_1, d_k_1, d_w, d_u1 = stitch_backward(d_y, q, k, w, u, C=C, chunk_size=chunk_size)

        d_w = d_w.view(NH*C, chunk_size, D)
        d_u1 = d_u1.view(NH*C, chunk_size, D)
        d_y = d_y.view(NH*C, chunk_size, D)
        u = u.view(NH*C, chunk_size, D)

        d_q_2, d_k_2, d_u2 = causal_attend_backward(d_y, q_, k_, u)
        d_u = d_u1 + d_u2
        d_k_3, d_v_, d_beta_ = decay_values_backward(d_w, d_u, k_, v_, beta_)

        d_q_1 = d_q_1.view(NH, T, D)
        d_q_2 = d_q_2.view(NH, C, chunk_size, D)
        d_q_2 = d_q_2.reshape(NH, T, D)
        d_q = d_q_1 + d_q_2

        d_k_2 = d_k_2.reshape(NH, T, D)
        d_k_3 = d_k_3.reshape(NH, T, D)
        d_k = d_k_1 + d_k_2 + d_k_3
        d_v = d_v_.reshape(NH, T, D)
        d_beta = d_beta_.view(NH, T)

        return d_q, d_k, d_v, d_beta, None


def test_delta_chunkwise_backward():
    NH, T, D = 2, 16, 2
    q1, k1, v1, beta1 = make_example(NH, T, D)
    
    y1 = forward_chunkwise(q1, k1, v1, beta1, chunk_size=2)
    (y1 - torch.ones_like(y1).detach()).pow(2).mean().backward()

    q, k, v, beta = make_example(NH, T, D)
    y = DeltaChunkwise.apply(q, k, v, beta, 2)
    (y - torch.ones_like(y).detach()).pow(2).mean().backward()

    assert allclose(y1, y, atol=1e-5), 'y is wrong'

    # print(beta1.grad - beta.grad, 'beta.grad diff')
    # print(q1.grad - q.grad, 'q.grad diff')
    # print(k1.grad - k.grad, 'k.grad diff')
    # print(v1.grad - v.grad, 'v.grad diff')

    assert allclose(q1.grad, q.grad, atol=1e-5), 'q.grad is wrong'
    assert allclose(beta1.grad, beta.grad, atol=1e-5), 'beta.grad is wrong'
    assert allclose(k1.grad, k.grad, atol=1e-5), 'k.grad is wrong'
    assert allclose(v1.grad, v.grad, atol=1e-5), 'v.grad is wrong'


test_delta_chunkwise_backward()