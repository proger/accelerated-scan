import torch
import triton
from typing import Literal


def init(B, C, T, *, device, requires_grad=False):
    torch.manual_seed(12312323)
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    gates = gates.half().float()
    tokens = torch.rand(B, C, T, device=device, requires_grad=requires_grad)
    return gates, tokens


def make_benchmark(plot_name, *, direction, max_exponent=12):
    return triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        #x_vals=[2**i for i in range(7, max_exponent)],
        #x_vals=[512,1024,2048,4096,8192,16384],
        x_vals=[512,1024,2048],
        #x_vals=[512,1024,2048,4096,8192,16384],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_names=["triton", "ref", "warp"],
        #line_vals=["triton", "ref", "warp"],
        #line_names=["flash", "kittenexp", "warp"],
        #line_vals=["flash", "kittenexp", "warp"],
        line_names=["kittenexp", "flash"],
        line_vals=["kittenexp", "flash"],
        plot_name=plot_name,
        args={
            "direction": direction,
        }
    )


def grad2(f, x, y, grad_out):
    grad = torch.autograd.grad(f(x, y), (x, y), grad_out)
    sum(x.sum().item() for x in grad)


from collections import defaultdict
c = defaultdict(int)

def bench(provider, SEQUENCE_LENGTH, device="cuda", direction: Literal["forward", "backward", "train"] = "forward"):
    B, H, D, T = 32, 64, 64, SEQUENCE_LENGTH
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

        case "kittenexp":
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

            k = tokens.unsqueeze(-1).expand(B, H, T, D).bfloat16().contiguous()
            q = torch.ones_like(k).bfloat16().contiguous()
            v = torch.ones_like(q).bfloat16().contiguous()
            f = gates.float().contiguous()
            o = torch.empty_like(v).bfloat16().contiguous()
            
            match direction:
                case "forward":
                    scan = lambda: scaled_dot_product_attention(q, k, v, is_causal=True)
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
    parser.add_argument("--direction", choices=["forward", "backward", "train", "all"], default="all")
    args = parser.parse_args()

    directions = {
        'forward': make_benchmark("accelerated_scan: forward speed", direction="forward"),
        'backward': make_benchmark("accelerated_scan: backward speed of (8,1536,seqlen), inference mode", direction="backward"),
        'train': make_benchmark("accelerated_scan: training speed of (8,1536,seqlen)", direction="train", max_exponent=15),
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
