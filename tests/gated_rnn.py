"""
Gated Linear RNNs with state expansion are Linear Transformers with data-dependent cumulative masking
"""
#%%
import torch
from torch import tensor


def pascal1(g):
    "compute the mask: prescan gates in log space (explicit)"
    N, T = g.shape
    l = g.new_zeros(N, T, T) + float('-inf')

    for t in range(T):
        l[:, t, t] = 0
        for s in range(t):
            l[:, s, t] = sum(g[:, k] for k in range(s+1, t)) + g[:, t]
    return l


def pascal(g):
    "compute the mask: prescan gates in log space (dynamic programming)"
    N, T = g.shape
    l = g.new_zeros(N, T, T) + float('-inf')

    for t in range(T):
        l[:, t, t] = 0
        for s in range(t-1, -1, -1):
            l[:, s, t] = l[:, s+1, t] + g[:, s+1]
    return l


@torch.no_grad()
def attend_backward(q, k, v, g):
    d_q = torch.einsum('ntk,ntv,nts->nsk', k, v, g.exp())
    d_k = torch.einsum('nsk,ntv,nts->ntk', q, v, g.exp())
    d_v = torch.einsum('nsk,ntk,nts->nt', q, k, g.exp()).unsqueeze(-1).expand_as(v)
    g_mask = g.exp().bool().float()
    d_g = torch.einsum('nsk,ntk,ntv,nts->nts', q, k, v, g_mask)
    return d_q, d_k, d_v, d_g


def attend(q, k, v, g):
    "masked linear attention: mask is data dependent, no softmax -- can be an RNN"
    y = torch.einsum('nsk,ntk,ntv,nts->nsv', q, k, v, g.exp())
    return y


def lscan(q, k, v, f):
    "linear time scan: fast weight programmer-style loop"
    N, D, T = k.shape
    N, T = f.shape
    h = k.new_zeros(N, D, D, T)
    y = k.new_zeros(N, D, T)
    h[..., 0] = torch.einsum('nk,nv->nkv', k[..., 0], v[..., 0])
    y[..., 0] = torch.einsum('nk,nkv->nv', q[..., 0], h[..., 0])
    for i in range(1, T):
        h[..., i] = f[:, None, None, i] * h[..., i-1] + torch.einsum('nk,nv->nkv', k[..., i], v[..., i])
        y[..., i] = torch.einsum('nk,nkv->nv', q[..., i], h[..., i])
    return y


if __name__ == '__main__':
    primes = tensor([1,  2,  3,  5,  7, 11, 13, 17, 19])[None, :]
    a = pascal(primes.log()).exp()
    b = primes.cumprod(dim=-1).float()
    assert torch.allclose(a[:, 0, :], b), f'{a[:,0,:]} != {b}'

    torch.manual_seed(0)

    N, T, D = 1, 512, 64
    q = torch.randn(N, T, D, requires_grad=True) / D**0.5 / D**0.25
    q.retain_grad()
    k = (torch.randn(N, T, D, requires_grad=True) / D**0.5 / D**0.25).sigmoid()
    k.retain_grad()
    v = torch.randn(N, T, D, requires_grad=True) / D**0.5
    v.retain_grad()

    #f = torch.rand(N, T)            # token-level forget gates:        "Gated RNN" with outer product state expansion
    f = torch.ones(N, T, requires_grad=True)*0.999         # same sequence-level forget gate: FWP with Decay
    f.retain_grad()

    k = (1-f).unsqueeze(-1).expand_as(k).clone().detach().requires_grad_(True)

    g = pascal(f.log())              # Prescan of all gates
    g = g.clone().detach().requires_grad_(True)
    ## add gradient hook to g
    #def g_hook(grad):
    #    g_hook.grad = grad
    #g.register_hook(g_hook)
    g_hook = g

    y1 = attend(q, k, v, g)
    print(y1, 'gated_attend') # N,T,D
    y2 = lscan(q.mT, k.mT, v.mT, f).mT
    print(y2, 'lscan')
    assert torch.allclose(y1, y2, atol=1e-5), 'gate_attend and lscan should be the same'

    def y_hook(grad):
        print(grad, 'y.grad')
        y_hook.grad = grad
    y1.register_hook(y_hook)
    y1.sum().backward()
    print(g_hook.grad, 'g.grad')

    d_q, d_k, d_v, d_g = attend_backward(q, k, v, g)
    print(d_g, 'd_g')

    assert torch.allclose(q.grad, d_q, atol=1e-5), 'q.grad is wrong'
    assert torch.allclose(k.grad, d_k, atol=1e-5), 'k.grad is wrong'
    assert torch.allclose(v.grad, d_v, atol=1e-5), 'v.grad is wrong'
    # print((g_hook.grad - d_g).pow(2).mean(), 'error')
    # print((g_hook.grad - d_g).abs().max(), 'max abs error')
    # assert torch.allclose(g_hook.grad, d_g, atol=1e-1), 'g.grad is wrong'
