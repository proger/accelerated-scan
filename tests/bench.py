import torch
import triton


def init(B, C, T, device):
    torch.manual_seed(12312323)
    gates = 0.999 + 0.001 * torch.rand(B, C, T, device=device)
    gates = gates.half().float()
    tokens = torch.rand(B, C, T, device=device)
    return gates, tokens


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7,17)],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_names=["triton", "ref", "warp"],
        #line_vals=["triton", "ref", "warp"],
        line_names=["warp"],
        line_vals=["warp"],
        plot_name="accelerated_scan: forward speed of (8,1536,seqlen), inference mode",  # name of the plot
        args={}
    ),
    triton.testing.Benchmark(
        x_names=["SEQUENCE_LENGTH"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7,17)],
        xlabel='sequence length',
        ylabel='ms',
        x_log=True,
        y_log=True,
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        #line_names=["triton", "ref", "warp"],
        #line_vals=["triton", "ref", "warp"],
        line_names=["warp"],
        line_vals=["warp"],
        plot_name="accelerated_scan: reverse speed of (8,1536,seqlen), inference mode",  # name of the plot
        args={
            "reverse": True,
        }
    ),
])
@torch.inference_mode()
def bench(provider, SEQUENCE_LENGTH, CHUNK_LENGTH=64, device="cuda", reverse=False):
    B, C, T = 8, 1536, SEQUENCE_LENGTH
    gates, tokens = init(B, C, T, device)
    outputs = torch.empty_like(tokens)

    direction = "reversed" if reverse else "forward"
    match provider:
        case "triton":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            output_gates = torch.zeros_like(gates).contiguous()
            from accelerated_scan.triton import forward_scan, backward_scan
            if reverse:
                scan = lambda: backward_scan[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
            else:
                scan = lambda: forward_scan[(B,C)](gates, tokens, outputs, SEQUENCE_LENGTH, enable_fp_fusion=False)
        case "ref":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.ref import scan as scan_ref
            scan = lambda: scan_ref(gates, tokens, reverse=reverse)
        case "warp":
            print(f"Running {provider} with sequence length {SEQUENCE_LENGTH} {direction}")
            from accelerated_scan.warp import warpscan_forward
            scan = lambda: warpscan_forward(gates, tokens, outputs, reverse)
        case _:
            raise ValueError(f"Unknown provider {provider}")

    # large warmup for benefit of torch.compile
    ms = triton.testing.do_bench(scan, warmup=5000, rep=100)
    return ms

if __name__ == '__main__':
    bench.run(save_path=".", print_data=True)
