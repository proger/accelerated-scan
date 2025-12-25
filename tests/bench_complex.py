import torch
import triton

from accelerated_scan.complex import backward_scan, forward_scan


def init(B, C, T, *, device):
    torch.manual_seed(12312323)
    angle = 2 * torch.pi * torch.rand(B, C, T, device=device, dtype=torch.float32)
    radius = 0.4 + 0.6 * torch.rand_like(angle)
    forget = torch.polar(radius, angle).to(torch.complex64)
    inputs = torch.randn(B, C, T, device=device, dtype=torch.float32).tanh()
    return forget, inputs


def bench(provider, block, *, B, C, T, device):
    forget, inputs = init(B, C, T, device=device)
    inputs_complex = torch.complex(inputs, torch.zeros_like(inputs))

    forget_real = torch.view_as_real(forget).contiguous()
    inputs_real = torch.view_as_real(inputs_complex).contiguous()
    states_real = torch.empty_like(inputs_real)

    d_output = torch.randn_like(inputs_complex)
    d_output_real = torch.view_as_real(d_output).contiguous()
    d_forget_real = torch.empty_like(forget_real)
    d_inputs_real = torch.empty_like(states_real)

    def scan():
        forward_scan[(B, C)](
            forget_real,
            inputs_real,
            states_real,
            seqlen=T,
            BLOCK=block,
            enable_fp_fusion=False,
        )
        backward_scan[(B, C)](
            forget_real,
            states_real,
            d_output_real,
            d_inputs_real,
            d_forget_real,
            seqlen=T,
            BLOCK=block,
            enable_fp_fusion=False,
        )

    with torch.inference_mode():
        return triton.testing.do_bench(scan, warmup=2000, rep=100)


def make_benchmark(plot_name, *, block_sizes):
    return triton.testing.Benchmark(
        x_names=["BLOCK"],
        x_vals=block_sizes,
        xlabel="block size",
        ylabel="ms",
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_names=["triton"],
        line_vals=["triton"],
        plot_name=plot_name,
        args={},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--min-exp", type=int, default=5)
    parser.add_argument("--max-exp", type=int, default=14)
    args = parser.parse_args()

    B, C, T = 1, 512, args.sequence_length
    block_sizes = [2**i for i in range(args.min_exp, args.max_exp + 1) if 2**i <= T]

    bench_run = triton.testing.perf_report(
        [
            make_benchmark(
                f"complex forward scan: ({B},{C},{T}) block size sweep",
                block_sizes=block_sizes,
            )
        ]
    )(lambda provider, BLOCK: bench("complex", BLOCK, B=B, C=C, T=T, device=args.device))
    bench_run.run(save_path=".", print_data=True)
