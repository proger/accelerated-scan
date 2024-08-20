import torch
import triton
from typing import Literal


def init(B, C, T, *, device, requires_grad=False):
    torch.manual_seed(12312323)
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    gates = gates.half().float()
    tokens = torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    return gates, tokens


def make_benchmark(plot_name, *, direction, batch_size=82, dim=64, max_exponent=15):
    return triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(8, max_exponent)],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_names=["triton", "ref", "warp"],
        #line_vals=["triton", "ref", "warp"],
        #line_names=["flash", "kittenexp", "warp"],
        #line_vals=["flash", "kittenexp", "warp"],
        #line_names=["linear", "flash2", "delta", "scan"],
        #line_vals=["kitten", "flash", "delta", "warp"],
        #line_names=["delta", "fla"],
        #line_vals=["delta", "fla"],
        
        #line_names=["linear", "flash2", "delta", "fla-delta", "warpscan"],
        #line_vals=["kitten", "flash", "delta", "fla", "warp"],

        line_names=["flash2", "delta", "fla-delta", "warpscan"],
        line_vals=["flash", "delta", "fla", "warp"],

        # line_names=["flash2", "delta", "fla", "scan"],
        # line_vals=["flash", "delta", "fla", "warp"],
        plot_name=plot_name,
        args={
            "direction": direction,
            "dim": dim,
            "batch_size": batch_size,
        }
    )


def grad2(f, x, y, grad_out):
    grad = torch.autograd.grad(f(x, y), (x, y), grad_out)
    sum(x.sum().item() for x in grad)


from collections import defaultdict
c = defaultdict(int)

def bench(provider, SEQUENCE_LENGTH, device="cuda", batch_size: int = 82, dim: int = 32, direction: Literal["forward", "backward", "train"] = "forward"):
    B, H, D, T = 1, batch_size, dim, SEQUENCE_LENGTH
    gates, tokens = init(B, H*D, T, device=device, requires_grad=direction=="train")
    outputs = torch.empty_like(tokens)
    grad_outputs = torch.empty_like(tokens)

    match provider:
        case "triton":
            print(f"Running {direction} {provider} with sequence length {SEQUENCE_LENGTH}")
            match direction:
                case "forward":
                    from accelerated_scan.triton import forward_scan
                    scan = lambda: forward_scan[(B,H*D)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
                case "backward":
                    from accelerated_scan.triton import backward_scan
                    scan = lambda: backward_scan[(B,H*D)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
                case "train":
                    # note that these measurements include time for memory allocation for forward output tensors
                    from accelerated_scan.triton import scan as train_scan
                    scan = lambda: grad2(train_scan, gates, tokens, grad_outputs)
        case "ref":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.ref import scan as scan_ref
            match direction:
                case "forward":
                    scan = lambda: scan_ref(gates, tokens)
                case "backward":
                    scan = lambda: scan_ref(gates, tokens, reverse=True)
                case "train":
                    scan = lambda: grad2(scan_ref, gates, tokens, grad_outputs)
        case "warp":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            match direction:
                case "forward":
                    from accelerated_scan.warp import warpscan_forward
                    o = torch.empty_like(tokens)
                    def scan():
                        warpscan_forward(gates, tokens, outputs, False)
                        return o * outputs
                    scan# = lambda: warpscan_forward(gates, tokens, outputs, False)
                case "backward":
                    from accelerated_scan.warp import warpscan_forward
                    scan = lambda: warpscan_forward(gates, tokens, outputs, True)
                case "train":
                    # note that these measurements include time for memory allocation for forward output tensors
                    from accelerated_scan.warp import scan as train_scan
                    scan = lambda: grad2(train_scan, gates, tokens, grad_outputs)

        case "kitten":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.kitten import attend

            gates, tokens = init(B, H, T, device=device, requires_grad=direction=="train")

            k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous()
            q = torch.ones_like(k).bfloat16().contiguous()
            v = torch.ones_like(q).bfloat16().contiguous()
            f = gates.float().contiguous()
            
            match direction:
                case "forward":
                    def scan():
                        o = torch.empty_like(v).bfloat16().contiguous()
                        attend(q, k, v, f, o)

        case "flash":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from torch.nn.functional import scaled_dot_product_attention

            gates, tokens = init(B, H, T, device=device, requires_grad=direction=="train")

            k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous().requires_grad_()
            q = torch.ones_like(k).bfloat16().contiguous().requires_grad_()
            v = torch.ones_like(q).bfloat16().contiguous().requires_grad_()
            f = gates.float().contiguous()
            o = torch.empty_like(v).bfloat16().contiguous()
            do = torch.randn_like(o)
            
            match direction:
                case "forward":
                    scan = lambda: scaled_dot_product_attention(q, k, v, is_causal=True)
                case "train":
                    def scan():
                        grad = torch.autograd.grad(scaled_dot_product_attention(q, k, v), (q, k, v), do)
                        sum(x.sum().item() for x in grad)
        case "delta":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.kitten import delta_forward, delta_backward, delta

            gates, tokens = init(B, H, T, device=device, requires_grad=direction=="train")

            k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous().requires_grad_()
            q = torch.ones_like(k).bfloat16().contiguous().requires_grad_()
            v = torch.ones_like(q).bfloat16().contiguous().requires_grad_()
            f = gates.bfloat16().contiguous()
            o = torch.empty_like(v).bfloat16().contiguous()
            do = torch.randn_like(o)

            match direction:
                case "forward":
                    def scan():
                        delta(q, k, v, f)
                case "train":
                    def scan():
                        grad = torch.autograd.grad(delta(q, k, v, f), (q, k, v, f), do)
                        sum(x.sum().item() for x in grad)

        case "fla":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from fla.ops.delta_rule.chunk_fuse import fused_chunk_delta_rule
            gates, tokens = init(B, H, T, device=device, requires_grad=direction=="train")

            k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous()
            q = torch.ones_like(k).bfloat16().contiguous()
            v = torch.ones_like(q).bfloat16().contiguous()
            f = gates.bfloat16().contiguous()
            o = torch.empty_like(v).bfloat16().contiguous()

            k = k.view(B, H, T, D).requires_grad_()
            q = q.view(B, H, T, D).requires_grad_()
            v = v.view(B, H, T, D).requires_grad_()
            f = f.view(B, H, T).requires_grad_()
            o = o.view(B, H, T, D)
            do = torch.randn_like(o)
            
            match direction:
                case "forward":
                    def scan():
                        fused_chunk_delta_rule(q, k, v, f, BT=16)

                case "train":
                    def scan():
                        y, _ = fused_chunk_delta_rule(q, k, v, f, BT=16)
                        grad = torch.autograd.grad(y, (q, k, v, f), do)
                        sum(x.sum().item() for x in grad)
        case _:
            raise ValueError(f"Unknown provider {provider}")

    # large warmup for benefit of torch.compile
    if direction == "train":
        ms = triton.testing.do_bench(scan, warmup=5000, rep=100)
    else:
        with torch.inference_mode():
            ms = triton.testing.do_bench(scan, warmup=5000, rep=100)
    return ms


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=82)
    parser.add_argument("--max-exponent", type=int, default=15)
    parser.add_argument("--direction", choices=["forward", "backward", "train", "all"], default="all")
    args = parser.parse_args()

    directions = {
        'forward': make_benchmark("accelerated_scan: forward speed", dim=args.dim, direction="forward", max_exponent=args.max_exponent, batch_size=args.batch_size),
        'backward': make_benchmark(f"accelerated_scan: backward speed of ({args.batch_size},{args.dim},seqlen), inference mode", dim=args.dim, direction="backward", max_exponent=args.max_exponent, batch_size=args.batch_size),
        'train': make_benchmark(f"accelerated_scan: training speed of ({args.batch_size},{args.dim},seqlen)", direction="train", dim=args.dim, max_exponent=args.max_exponent, batch_size=args.batch_size),
    }

    benchmarks = []
    match args.direction:
        case "all":
            benchmarks.append(directions['forward'])
            benchmarks.append(directions['backward'])
            benchmarks.append(directions['train'])
        case dir:
            benchmarks.append(directions[dir])

    try:
        triton.testing.perf_report(benchmarks)(bench).run(save_path=".", print_data=True)
    finally:
        print(f"{dict(c)=}")
