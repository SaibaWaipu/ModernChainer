"""Compare Modern Chainer and PyTorch from Python.

Run from the repository root with the Python environment that has the maturin
build installed:

    python/.venv/bin/python scripts/bench_vs_pytorch.py

Choose devices with BENCH_DEVICES, for example:

    BENCH_DEVICES=cpu,cuda,vulkan:1 \
      python/.venv/bin/python scripts/bench_vs_pytorch.py

Vulkan adapter indices come from chainer.vulkan_devices(). For a fair GPU
comparison, choose the same physical GPU as the torch CUDA device (set
CUDA_VISIBLE_DEVICES and the Vulkan adapter index accordingly).
"""

import os
import statistics
import time

THREADS = int(os.environ.get("BENCH_THREADS", "12"))
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))

import numpy as np
import torch

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O


def requested_devices():
    raw = os.environ.get("BENCH_DEVICES", "cpu,cuda,vulkan:0")
    out = []
    for item in raw.split(","):
        device = item.strip()
        if not device:
            continue
        if device == "cuda" and not chainer.cuda_available():
            print("skipping cuda: chainer.cuda_available() is false")
            continue
        if device.startswith("vulkan") and not chainer.vulkan_available():
            print("skipping vulkan: chainer.vulkan_available() is false")
            continue
        out.append(device)
    return out


def torch_device_for(device):
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return os.environ.get("BENCH_TORCH_DEVICE", "cuda")


def sync(device, torch_device):
    chainer.synchronize(device)
    if torch_device != "cpu":
        torch.cuda.synchronize()


def bench(fn, device, torch_device, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    sync(device, torch_device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device, torch_device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def report(name, ours, theirs):
    print(
        f"{name:<34} ours={ours * 1e3:9.3f} ms  "
        f"torch={theirs * 1e3:9.3f} ms  speedup={theirs / ours:5.2f}x"
    )
    return {"name": name, "ours": ours, "torch": theirs, "speedup": theirs / ours}


def bench_matmul(device, torch_device, size=512):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((size, size), dtype=np.float32)
    b_np = rng.standard_normal((size, size), dtype=np.float32)
    a = chainer.Variable(chainer.Tensor.from_numpy(a_np, device=device))
    b = chainer.Variable(chainer.Tensor.from_numpy(b_np, device=device))
    ta = torch.from_numpy(a_np).to(torch_device)
    tb = torch.from_numpy(b_np).to(torch_device)
    return (
        bench(lambda: F.matmul(a, b), device, torch_device),
        bench(lambda: torch.matmul(ta, tb), device, torch_device),
    )


def bench_add(device, torch_device, n=1 << 22):
    rng = np.random.default_rng(1)
    a_np = rng.standard_normal(n, dtype=np.float32)
    b_np = rng.standard_normal(n, dtype=np.float32)
    a = chainer.Variable(chainer.Tensor.from_numpy(a_np, device=device))
    b = chainer.Variable(chainer.Tensor.from_numpy(b_np, device=device))
    ta = torch.from_numpy(a_np).to(torch_device)
    tb = torch.from_numpy(b_np).to(torch_device)
    return (
        bench(lambda: F.add(a, b), device, torch_device, iters=15),
        bench(lambda: torch.add(ta, tb), device, torch_device, iters=15),
    )


def bench_linear(device, torch_device, batch=128, in_features=784, out_features=256):
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((batch, in_features), dtype=np.float32)
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np, device=device))
    tx = torch.from_numpy(x_np).to(torch_device)
    layer = L.Linear(in_features, out_features, seed=0, device=device)
    torch_layer = torch.nn.Linear(in_features, out_features).to(torch_device)

    def ours():
        layer.cleargrads()
        loss = F.sum(layer(x))
        loss.backward()

    def theirs():
        torch_layer.zero_grad()
        loss = torch_layer(tx).sum()
        loss.backward()

    return (
        bench(ours, device, torch_device),
        bench(theirs, device, torch_device),
    )


def bench_adam(device, torch_device, rows=1024, cols=1024):
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((64, rows), dtype=np.float32)
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np, device=device))
    tx = torch.from_numpy(x_np).to(torch_device)
    layer = L.Linear(rows, cols, seed=0, device=device)
    torch_layer = torch.nn.Linear(rows, cols).to(torch_device)
    optimizer = O.Adam(0.001)
    optimizer.setup(layer.parameters())
    torch_optimizer = torch.optim.Adam(torch_layer.parameters(), lr=0.001)

    def ours():
        optimizer.zero_grad()
        F.sum(layer(x)).backward()
        optimizer.update()

    def theirs():
        torch_optimizer.zero_grad()
        torch_layer(tx).sum().backward()
        torch_optimizer.step()

    return (
        bench(ours, device, torch_device, iters=10),
        bench(theirs, device, torch_device, iters=10),
    )


def run_device(device):
    torch_device = torch_device_for(device)
    if device.startswith("vulkan"):
        adapters = chainer.vulkan_devices()
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        name = adapters[index] if index < len(adapters) else "unknown"
        print(f"vulkan adapter: {index} {name}")
    print(f"device: {device}   torch: {torch_device}")
    results = []
    results.append(report("matmul 512x512x512", *bench_matmul(device, torch_device)))
    results.append(report("elementwise add (4M)", *bench_add(device, torch_device)))
    results.append(report("linear fwd+bwd 128x784x256", *bench_linear(device, torch_device)))
    results.append(report("adam step 1024x1024", *bench_adam(device, torch_device)))

    ours = statistics.geometric_mean(r["ours"] for r in results)
    theirs = statistics.geometric_mean(r["torch"] for r in results)
    print(
        f"{'geometric mean':<34} ours={ours * 1e3:9.3f} ms  "
        f"torch={theirs * 1e3:9.3f} ms  speedup={theirs / ours:5.2f}x"
    )


def main():
    torch.set_num_threads(THREADS)
    print(f"threads: {THREADS}")
    print(f"chainer: {chainer.__version__}   torch: {torch.__version__}")
    if chainer.vulkan_available():
        print("vulkan adapters:")
        for index, adapter in enumerate(chainer.vulkan_devices()):
            print(f"  vulkan:{index} {adapter}")
    for device in requested_devices():
        print()
        run_device(device)


if __name__ == "__main__":
    main()
