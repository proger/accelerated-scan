import torch

torch.set_grad_enabled(False)

def init(B, C, T, *, device, requires_grad=False):
    torch.manual_seed(12312323)
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    gates = gates.half().float()
    #tokens = torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    tokens = torch.ones(B, C, T, device=device, requires_grad=requires_grad)

    return gates, tokens

device = 'cuda'

#for SEQUENCE_LENGTH in [512,1024,2048,4096]:
for SEQUENCE_LENGTH in [256]:
    direction = "forward"
    provider = "kittenexp"
    #B, H, D, T = 32, 64, 16, SEQUENCE_LENGTH
    B, H, D, T = 1, 1, 32, SEQUENCE_LENGTH
    #B, H, D, T = 1, 1, 64, SEQUENCE_LENGTH
    print("running", T)

    # from triton:
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    gates, tokens = init(B, H, T, device=device, requires_grad=direction=="train")

    k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous()
    q = torch.ones_like(k).bfloat16().contiguous()
    v = torch.ones_like(q).bfloat16().contiguous()
    f = gates.float().contiguous()
    o = torch.empty_like(v).bfloat16().contiguous()

    from accelerated_scan.kitten import attend

    if B <= 32:
        qk = torch.einsum('bhsd,bhtd->bhst', q, k)
        causal_mask = torch.tril(torch.ones(T, T, device=device))
        causal_mask = causal_mask[None, None].expand(B, H, T, T)
        causal_mask = causal_mask.to(dtype=qk.dtype)
        y = torch.einsum('bhst,bhte->bhse', qk * causal_mask, v)

    torch.cuda.synchronize()
    for _ in range(1):
        cache.zero_()
        attend(q, k, v, f, o)
        from torch.nn.functional import scaled_dot_product_attention
        scaled_dot_product_attention(q, k, v, is_causal=True)
        print('flash')

    if B <= 32:
        try:
            assert torch.allclose(y, o, atol=1e-3, rtol=1e-3)
        except:
            print(y[:,:,:,0], 'ref')
            print(o[:,:,:,0], 'ker')
            # print(y[:,:,0], 'ref, t=0')
            # print(o[:,:,0], 'ker, t=0')
            # for t in range(SEQUENCE_LENGTH):
            #     if (y[:,:,t,:]- o[:,:,t,:]).pow(2).mean().item() > 0:
            #         print(t, y[:,:,t,:]- o[:,:,t,:])
            raise
    torch.cuda.synchronize()
