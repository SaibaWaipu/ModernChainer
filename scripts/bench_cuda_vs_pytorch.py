"""Compare Modern Chainer's CUDA backend with PyTorch CUDA.

Runs the Rust `cuda_bench` example (JSON line), then benchmarks the same shapes
with PyTorch, including CUDA Graph replay.  Use a Python environment with the
CUDA build of PyTorch:

    /tmp/opencode/torch-venv/bin/python scripts/bench_cuda_vs_pytorch.py

Set CUDA_VISIBLE_DEVICES to the target GPU (UUID is safest).
"""

import json
import os
import subprocess
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE_UUID = os.environ.get(
    "CUDA_VISIBLE_DEVICES",
    "GPU-2eb6e70d-6292-dd98-da37-a52b5c2ba679",
)
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_UUID

NVIDIA_LIBS = [
    "cuda_nvrtc/lib",
    "cublas/lib",
    "cuda_runtime/lib",
]


def rust_bench():
    env = os.environ.copy()
    venv = os.path.join(ROOT, "python", ".venv")
    libs = [
        os.path.join(venv, "lib", "python3.13", "site-packages", "nvidia", lib)
        for lib in NVIDIA_LIBS
    ]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = DEVICE_UUID
    output = subprocess.check_output(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--example",
            "cuda_bench",
            "-p",
            "modern-chainer",
        ],
        cwd=ROOT,
        env=env,
        text=True,
    )
    for line in output.splitlines():
        if line.startswith("JSON "):
            return json.loads(line[5:])
    raise RuntimeError(f"no JSON line in cuda_bench output:\n{output}")


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def main():
    ours = rust_bench()
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}")

    n = 512
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    matmul_eager = bench(lambda: torch.matmul(a, b))
    out = torch.empty_like(a)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            torch.matmul(a, b, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.matmul(a, b, out=out)
    matmul_graph = bench(graph.replay)

    batch, in_features, out_features = 128, 784, 256
    x = torch.randn(batch, in_features, device="cuda")
    w = torch.randn(out_features, in_features, device="cuda")
    bias = torch.randn(out_features, device="cuda")
    linear = bench(lambda: torch.nn.functional.linear(x, w, bias))

    params = 1 << 20
    p = torch.randn(params, device="cuda", requires_grad=True)
    grad = torch.randn_like(p)
    opt = torch.optim.Adam([p], lr=1e-3)
    p.grad = grad
    opt.step()
    adam_eager = bench(opt.step, warmup=2, iters=10)

    adam_graph = None
    try:
        p2 = torch.randn(params, device="cuda", requires_grad=True)
        opt2 = torch.optim.Adam([p2], lr=1e-3, capturable=True)
        p2.grad = torch.randn_like(p2)
        opt2.step()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                opt2.step()
        torch.cuda.current_stream().wait_stream(side)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            opt2.step()
        adam_graph = bench(g2.replay, warmup=2, iters=10)
    except Exception as error:  # noqa: BLE001
        print(f"torch Adam graph capture unavailable: {error}")

    rows = [
        ("matmul eager 512^3", ours["matmul_cuda"], matmul_eager),
        ("matmul graph 512^3", ours["matmul_graph"], matmul_graph),
        ("linear eager 128x784x256", ours["linear_cuda"], linear),
        ("adam eager 1M", ours["adam_cuda"], adam_eager),
    ]
    if adam_graph is not None:
        rows.append(("adam graph 1M", ours["adam_graph"], adam_graph))

    print(f"{'workload':<28} {'ours ms':>9} {'torch ms':>9} {'torch/ours':>11}")
    for name, own, their in rows:
        print(f"{name:<28} {own:9.4f} {their:9.4f} {their / own:11.2f}")


if __name__ == "__main__":
    main()
