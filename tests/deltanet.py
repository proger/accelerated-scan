"""
DeltaNet implementation reference for Accelerated Scan.  DeltaNet performs efficient management of a large fixed-sized memory.

`forward` is inspired by Yang 2024. It applies single chunk version pointwise and then performs chunk-level stitching.
`forward_loop` is the reference implementation of the original recurrence.

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

#set_float32_matmul_precision('high')


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


def decay_values_forward(q_, k_, v_, beta_):
    chunk_size = k_.size(1)

    # evaluate all chunks in parallel
    beta__ = beta_.unsqueeze(-1)
    w = beta__ * k_.clone()
    u = beta__ * v_.clone()
    K = einsum('nsd,ntd->nst', k_, k_) # (chunk_size,chunk_size) matrix

    for t in range(1,chunk_size):
        w[:, t] -= beta__[:, t] * einsum('nt,ntd->nd', K[:, :t, t], w[:, :t].clone())
        u[:, t] -= beta__[:, t] * einsum('nt,ntd->nd', K[:, :t, t], u[:, :t].clone())

    # attend to decayed values
    qk = einsum("nsk,ntk->nst", q_, k_)
    qk.tril_()
    y = einsum("nst,ntj->nsj", qk, u)

    return w, u, y


def forward(q, k, v, beta, chunk_size=2):
    "decay values applying deltanet forgetting rules, then stitch chunks"
    NH, T, D = shape(q, k, v, beta)
    C = T // chunk_size
    q_, k_, v_, beta_ = (
        q.view(NH*C, chunk_size, D), k.view(NH*C, chunk_size, D),
        v.view(NH*C, chunk_size, D), beta.view(NH*C, chunk_size)
    )

    w, u, y = decay_values_forward(q_, k_, v_, beta_)

    # stitch chunks sequentially
    q_ = q.view(NH, C, chunk_size, D)
    k_ = k.view(NH, C, chunk_size, D)
    u = u.view(NH, C, chunk_size, D)
    w = w.view(NH, C, chunk_size, D)

    # materialize the state for the leading chunk
    state = einsum('ntv,ntk->nvk', u[:, 0], k_[:, 0])

    y_deltas = [u.new_zeros(NH, chunk_size, D)]
    for c in range(1, C):
        if c == 1:
            # early on this could be cheaper than a state query
            kw = einsum('nsk,ntk->nst', k_[:, c-1], w[:, c]) # TDT
            u_old = einsum('nsv,nst->ntv', u[:, c-1], kw) # TDT

            kq = einsum('nsk,ntk->nst', k_[:, c-1], q_[:, c]) # TDT
            y_cur = einsum('nsv,nst->ntv', u[:, c-1], kq) # TTD            
        else:
            u_old = einsum('nvk,ntk->ntv', state, w[:, c]) # DDT
            y_cur = einsum('nvk,nsk->nsv', state, q_[:, c]) # DDT

        # attend to old values
        qk = einsum("nsi,nti->nst", q_[:, c], k_[: ,c]) # TDT
        qk = qk.tril()

        y_prev = einsum("nst,ntv->nsv", qk, u_old) # TTD

        y_deltas.append(y_cur - y_prev)

        state_delta = einsum('ntv,ntk->nvk', u[:, c] - u_old, k_[:, c])
        state = state + state_delta

    y_delta = torch.stack(y_deltas, dim=1)

    w = w.view(NH, T, D)
    u = u.view(NH, T, D)
    y = y.view(NH, T, D) + y_delta.view(NH, T, D)

    return w, u, y


def forward_loop(q, k, v, beta):
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
    k = randn(NH, T, D, device=device, dtype=dtype) / D**0.5
    k.requires_grad_()
    v = randn(NH, T, D, device=device, dtype=dtype) / D**0.5
    v.requires_grad_()
    beta = randn(NH, T, device=device, dtype=dtype).sigmoid()
    beta.requires_grad_()
    return q, k, v, beta


@no_grad()
def decay_values_backward(d_out_w, d_out_u, d_out_y, q, k, v, beta):
    NH, T, D = shape(q, k, v, beta)

    #
    # allocations
    #

    # this group is loaded from global memory
    q = q.clone() # load q
    k = k.clone() # load k
    v = v.clone() # load v
    beta = beta.clone() # load beta
    d_out_w = d_out_w.clone() # ntk
    d_out_y = d_out_y.clone() # ntv

    w = k.new_zeros(NH, T, D) # ntk
    u = v.new_zeros(NH, T, D) # ntw
    w_bases = w.clone() # ntk
    u_bases = u.clone() # ntw

    bk = einsum('nt,ntk->ntk', beta, k)

    bKl = k.new_zeros(NH, T, T)
    tt = k.new_zeros(NH, T, T)

    d_k = k.new_zeros(NH, T, D) # nsk
    tk = k.new_zeros(NH, T, D) # ntk

    #
    # forward
    #

    tt = einsum('ntk,nsk->nts', k, k)
    tt = tt.tril(diagonal=-1) # make_causal(0); set_diagonal(0)
    bKl = einsum('nt,nts->nts', beta, tt) # multiply each row of K by beta

    u_bases = v
    v = einsum('nt,ntw->ntw', beta, v)

    for t in range(T):
        tk = einsum('nts,nsk->ntk', bKl, w) # matmul for the sake of one row
        w[:, t] = bk[:, t, :] - tk[:, t, :]
        tk = einsum('nts,nsw->ntw', bKl, u) # matmul for the sake of one row
        u[:, t] = v[:, t, :] - tk[:, t, :]

    w.clone() # store w
    u.clone() # store u

    w_bases = einsum('nts,nsk->ntk', tt, w)
    w_bases = k - w_bases
    v = einsum('nts,nsw->ntw', tt, u)
    u_bases = u_bases - v

    #
    # causal_attend_backward for d_q, d_k_2, d_out_u
    #

    tt = einsum('nsv,ntv->nst', d_out_y, u)
    tt = tt.tril()
    d_q = einsum('nst,ntk->nsk', tt, k)
    d_q.clone() # store

    d_k_2 = einsum('nst,nsk->ntk', tt, q)
    d_k_2.clone() # store to shared memory?

    tt = einsum('nsk,ntk->nst', q, k)
    tt = tt.tril()

    v.zero_() # reuse register space of v for d_out_u
    d_out_u = d_out_u.clone() # load ntw
    d_out_u += einsum('nst,nsv->ntv', tt, d_out_y)

    #
    # backward for d_k, d_v, d_beta
    #

    d_k.zero_()

    for t in range(T-1,-1,-1):
        # d_k
        tt = einsum('njw,ntw->njt', w, d_out_w) # matmul for the sake of one column t
        tt[:, t:, :] = 0
        tk = einsum('njt,njk->ntk', tt, k)

        tt = einsum('njv,ntv->njt', u, d_out_u) # matmul for the sake of one column t
        tt[:, t:, :] = 0
        tk += einsum('njt,njk->ntk', tt, k)

        d_k[:, t] += tk[:, t]

        # backpropagate through time, updating only remaining timestamps
        tt.zero_()
        tt[:, t] += bKl[:, t]
        tk = einsum('ntj,ntk->njk', tt, d_out_w)
        d_out_w = d_out_w - tk
        tk = einsum('ntj,ntk->njk', tt, d_out_u)
        d_out_u = d_out_u - tk

    d_k = d_out_w - d_k
    d_k = einsum('ntk,nt->ntk', d_k, beta)

    # decay w and u
    tt = einsum('ntw,njw->ntj', d_out_w, w)
    tt += einsum('ntw,njw->ntj', d_out_u, u)
    tt.tril_(diagonal=-1)

    tk = einsum('ntj,ntk->njk', tt, bk)
    d_k = d_k - tk
    d_k_2 = d_k_2.clone() # load from shared memory
    d_k = d_k_2 + d_k
    d_k = d_k.clone() # store

    # d_beta
    w_bases = einsum('ntk,ntk->ntk', w_bases, d_out_w)
    u_bases = einsum('ntw,ntw->ntw', u_bases, d_out_u)

    # d_v using d_out_u register
    d_out_u = einsum('nt,ntv->ntv', beta, d_out_u)
    d_v = d_out_u.clone() # store

    # continue d_beta reusing the beta register
    beta = einsum('ntk->nt', w_bases)
    beta += einsum('ntv->nt', u_bases)
    d_beta = beta.clone() # store

    return d_q, d_k, d_v, d_beta


def stitch_backward(d_y_delta, q, k, w, u, C, chunk_size):
    NH, T, D = shape(q, k, None, None)

    # outputs
    d_q_ = q.new_zeros(NH, C, chunk_size, D)
    d_k_ = k.new_zeros(NH, C, chunk_size, D)
    d_w = w.new_zeros(NH, C, chunk_size, D)
    d_u = u.new_zeros(NH, C, chunk_size, D)

    # chunked inputs
    d_y_delta = d_y_delta.view(NH, C, chunk_size, D)
    q_ = q.view(NH, C, chunk_size, D)
    k_ = k.view(NH, C, chunk_size, D)
    w = w.view(NH, C, chunk_size, D)

    # shared memory copy
    u = u.view(NH, C, chunk_size, D).clone()

    state = w.new_zeros(NH, D, D)
    d_state = w.new_zeros(NH, D, D) # NHVK
    state_delta = w.new_zeros(NH, D, D) # NHVK # can this be float32?

    tt = k.new_zeros(NH, chunk_size, C)
    tk = k.new_zeros(NH, chunk_size, D)

    # materialize the state for the leading chunk
    state = einsum('ntv,ntk->nvk', u[:, 0], k_[:, 0])

    for c in range(1, C):
        tk = einsum('nvk,ntk->ntv', state, w[:, c])
        u[:, c] = u[:, c] - tk
        state_delta = einsum('ntv,ntk->nvk', u[:, c], k_[:, c])
        if c < C-1:
            state = state + state_delta

    # from now on, u's are decayed

    # stitch backward
    for c in range(C-1, 0, -1):
        d_y_delta_c = d_y_delta[:, c]
        d_y_delta_c = -d_y_delta_c # neg

        if c < C-1:
            state_delta = einsum('ntv,ntk->nvk', u[:, c], k_[:, c])
            state = state - state_delta # uncompute the state
            tk = einsum('nvk,ntk->ntv', state, w[:, c]) # state_decay

        # d_q, d_k
        tt = einsum('nsv,ntv->nst', d_y_delta_c, tk)
        tt.tril_()

        # d_q
        tk = einsum('nst,ntk->nsk', tt, k_[:, c]) # causal_attend_backward for delta
        tk.sub_(einsum('nsv,nvk->nsk', d_y_delta_c, state)) # prev_output
        d_q_[:, c] = tk

        # d_k
        tk = einsum('nst,nsk->ntk', tt, q_[:, c])
        if c < C-1:
            tk.add_(einsum('nvk,ntv->ntk', d_state, u[:, c])) # state_add
        else:
            # d_state is zero
            pass
        d_k_[:, c] = tk

        # d_u
        if c < C-1:
            d_u[:, c] = einsum('nvk,ntk->ntv', d_state, k_[:, c]) # state_add
        else:
            # d_state is zero
            pass

        # d_state_decays
        tt = einsum('nsk,ntk->nst', q_[:, c], k_[:, c])
        tt.tril_()
        d_state_decays = einsum('nsv,nst->ntv', d_y_delta_c, tt)
        if c < C-1:
            d_state_decays.sub_(einsum('nvk,ntk->ntv', d_state, k_[:, c])) # state_add

        # d_w
        tk = einsum('ntv,nvk->ntk', d_state_decays, state)
        d_w[:, c] = tk # state_decays

        # backpropagate through time
        d_state.sub_(einsum('nsv,nsk->nvk', d_y_delta_c, q_[:, c])) # prev_output
        d_state.add_(einsum('ntv,ntk->nvk', d_state_decays, w[:, c])) # state_decays

    tk = einsum('nvk,ntk->ntv', d_state, k_[:, 0])
    d_u[:, 0] = tk # state_add
    tk = einsum('nvk,ntv->ntk', d_state, u[:, 0])
    d_k_[:, 0] = tk # state_add

    return d_q_.view(NH, T, D), d_k_.view(NH, T, D), d_w.view(NH, T, D), d_u.view(NH, T, D)


class DeltaChunkwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, chunk_size):
        w, u, y = forward(q, k, v, beta, chunk_size)
        ctx.save_for_backward(q, k, v, beta, w, u)
        ctx.chunk_size = chunk_size
        return y

    @staticmethod
    def backward(ctx, d_y):
        q, k, v, beta, w, u = ctx.saved_tensors
        NH, T, D = shape(q, k, v, beta)
        chunk_size = ctx.chunk_size
        C = T // chunk_size

        q_, k_, v_, beta_ = (
            q.view(NH*C, chunk_size, D), k.view(NH*C, chunk_size, D),
            v.view(NH*C, chunk_size, D), beta.view(NH*C, chunk_size)
        )

        d_q_1, d_k_1, d_w, d_u = stitch_backward(d_y, q, k, w, u, C=C, chunk_size=chunk_size)

        d_w = d_w.view(NH*C, chunk_size, D)
        d_u = d_u.view(NH*C, chunk_size, D)
        d_y = d_y.view(NH*C, chunk_size, D)
        u = u.view(NH*C, chunk_size, D)

        d_q_2, d_k_2, d_v_, d_beta_ = decay_values_backward(d_w, d_u, d_y, q_, k_, v_, beta_)

        d_q_1 = d_q_1.view(NH, T, D)
        d_q_2 = d_q_2.view(NH, C, chunk_size, D)
        d_q_2 = d_q_2.reshape(NH, T, D)
        d_q = d_q_1 + d_q_2

        d_k_2 = d_k_2.reshape(NH, T, D)
        d_k = d_k_1 + d_k_2
        d_v = d_v_.reshape(NH, T, D)
        d_beta = d_beta_.view(NH, T)

        return d_q, d_k, d_v, d_beta, None


def test_delta_chunkwise_backward():
    NH, T, D = 2, 16, 3
    q1, k1, v1, beta1 = make_example(NH, T, D)

    y0 = forward_loop(q1, k1, v1, beta1)
    
    w1, u1, y1 = forward(q1, k1, v1, beta1, chunk_size=2)
    (y1 - torch.ones_like(y1).detach()).pow(2).mean().backward()

    assert allclose(y0, y1, atol=1e-5), 'y1 is wrong'

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