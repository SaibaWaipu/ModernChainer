"""Micro-benchmarks comparing Modern Chainer's Python API with PyTorch (CPU).

Run from the repository root with the Python virtualenv that has the maturin
build installed:

    python/.venv/bin/python scripts/bench_vs_pytorch.py
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


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def ratio(ours, theirs):
    return theirs / ours


def report(name, ours, theirs):
    print(
        f"{name:<34} ours={ours * 1e3:9.3f} ms  "
        f"torch={theirs * 1e3:9.3f} ms  speedup={ratio(ours, theirs):5.2f}x"
    )
    return {"name": name, "ours": ours, "torch": theirs, "speedup": ratio(ours, theirs)}


def bench_matmul(size=512):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((size, size), dtype=np.float32)
    b_np = rng.standard_normal((size, size), dtype=np.float32)
    a = chainer.Variable(chainer.Tensor.from_numpy(a_np))
    b = chainer.Variable(chainer.Tensor.from_numpy(b_np))
    ta = torch.from_numpy(a_np)
    tb = torch.from_numpy(b_np)
    return (
        bench(lambda: F.matmul(a, b)),
        bench(lambda: torch.matmul(ta, tb)),
    )


def bench_add(n=1 << 22):
    rng = np.random.default_rng(1)
    a_np = rng.standard_normal(n, dtype=np.float32)
    b_np = rng.standard_normal(n, dtype=np.float32)
    a = chainer.Variable(chainer.Tensor.from_numpy(a_np))
    b = chainer.Variable(chainer.Tensor.from_numpy(b_np))
    ta = torch.from_numpy(a_np)
    tb = torch.from_numpy(b_np)
    return (
        bench(lambda: F.add(a, b), iters=15),
        bench(lambda: torch.add(ta, tb), iters=15),
    )


def bench_linear(batch=128, in_features=784, out_features=256):
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((batch, in_features), dtype=np.float32)
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np))
    tx = torch.from_numpy(x_np)
    layer = L.Linear(in_features, out_features, seed=0)
    torch_layer = torch.nn.Linear(in_features, out_features)

    def ours():
        layer.cleargrads()
        loss = F.sum(layer(x))
        loss.backward()

    def theirs():
        torch_layer.zero_grad()
        loss = torch_layer(tx).sum()
        loss.backward()

    return bench(ours), bench(theirs)


def bench_adam(rows=1024, cols=1024):
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((64, rows), dtype=np.float32)
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np))
    tx = torch.from_numpy(x_np)
    layer = L.Linear(rows, cols, seed=0)
    torch_layer = torch.nn.Linear(rows, cols)
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

    return bench(ours, iters=10), bench(theirs, iters=10)


def main():
    torch.set_num_threads(THREADS)
    print(f"threads: {THREADS}")
    results = []
    results.append(report("matmul 512x512x512", *bench_matmul()))
    results.append(report("elementwise add (4M)", *bench_add()))
    results.append(report("linear fwd+bwd 128x784x256", *bench_linear()))
    results.append(report("adam step 1024x1024", *bench_adam()))

    ours = statistics.geometric_mean(r["ours"] for r in results)
    theirs = statistics.geometric_mean(r["torch"] for r in results)
    print(
        f"{'geometric mean':<34} ours={ours * 1e3:9.3f} ms  "
        f"torch={theirs * 1e3:9.3f} ms  speedup={theirs / ours:5.2f}x"
    )


if __name__ == "__main__":
    main()
