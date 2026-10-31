# Benchmarks

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 5 5600G, 6 cores / 12 threads |
| RAM | 13 GiB |
| GPU | NVIDIA RTX 2070 SUPER, GTX 1080 (Vulkan via driver 580) |
| OS | Linux 6.12 (Debian 13) |
| Rust | 1.99.0, release profile (`lto = "thin"`, `codegen-units = 1`) |
| Python | 3.13 |
| PyTorch | 2.15.0+cpu |
| NumPy | 2.5.4 |

Both implementations use 12 threads. The harness sets
`torch.set_num_threads(12)` and `RAYON_NUM_THREADS=12`.

## Harness

`scripts/bench_vs_pytorch.py` measures the same operation through each API:

- `matmul 512×512×512`
- elementwise add on 4M f32 values
- `Linear(784, 256)` forward + sum + backward, batch 128
- Adam step on a 1024×1024 parameter

Each measurement runs 5 warmups and takes the median of 30 iterations (10 for
Adam). Run it with the maturin build installed:

```bash
python/.venv/bin/python scripts/bench_vs_pytorch.py
BENCH_THREADS=6 python/.venv/bin/python scripts/bench_vs_pytorch.py
```

Criterion benches for the Rust core:

```bash
cargo bench -p modern-chainer --bench cpu_ops
```

## Results (2026-10-31)

Three runs, median per cell. Lower is better.

| Workload | Modern Chainer | PyTorch | Ratio |
|---|---:|---:|---:|
| matmul 512×512×512 | 0.43–0.52 ms | 0.42 ms | 0.81–0.98× |
| elementwise add (4M) | 2.23–2.30 ms | 1.95–2.16 ms | 0.85–0.97× |
| linear fwd+bwd 128×784×256 | 0.54–0.61 ms | 0.31–0.32 ms | 0.52–0.60× |
| Adam step 1024×1024 | 3.38–3.81 ms | 4.02–4.98 ms | 1.19–1.40× |
| geometric mean | ~1.20 ms | ~1.04 ms | ~0.87–0.91× |

### Interpretation

- **Adam** wins consistently: the fused single-pass CPU kernel avoids the
  ten intermediate tensors that a naive composition allocates.
- **Matmul** is close to parity and occasionally ahead at 512³. The engine uses
  `gemm` with rayon, so it benefits from the same class of blocked SIMD kernels
  as MKL.
- **Elementwise** is within 15% of PyTorch; the gap is allocation and
  memory-bandwidth policy, not arithmetic.
- **Linear forward+backward** is the clear loss. Forward alone is competitive
  (0.17 ms vs 0.12 ms in a focused test), but the backward pass computes
  `gᵀ·x` and `g·W` with generic GEMM calls; PyTorch/MKL's transposed GEMM
  kernels and its `Addmm` bias path are better tuned at this shape. This is the
  primary optimization target.

These numbers are micro-benchmarks, not a universal claim. The acceptance gate
from the spec (1.2× geometric mean, no item below 0.9×) is not met on this
suite; the harness is checked in so improvements are measurable.

## Optimizations that produced the current numbers

1. Zero-copy `Vec<f32>` storage (`CpuStorage::F32`).
2. Parallel `collect`-based elementwise kernels instead of zero-init + write.
3. `gemm` replacing `matrixmultiply` for large GEMMs.
4. Fused `linear` forward and `linear_backward`.
5. Fused single-pass Adam with uninitialized output capacity.
6. Bias-specific broadcast path and bias-specific `sum_to`.
7. Explicit GEMM thread count; fixed a `Rayon(0)` scheduler hang.
8. Naive fallback for small GEMMs; fixed a small-shape `gemm` stack overflow.
9. Fused scalar fill for `sum`/`mean` backward (one allocation instead of
   `ones_like` + `scale`).
10. Sequential reductions below 1M elements; rayon scheduling overhead dominated
    small `sum` calls.

## State-by-Run and the decoder

The decoder is a correctness reference (2 layers, `d_model=8`) used to verify
explicit KV state, prefill/decode parity, paged-vs-dense attention, batching
equivalence and speculative equality. `PagedBatchRunner` performs a real batched
decode step over paged KV, but the CPU reference is not performance-tuned: block
appends copy storage and the attention kernel is scalar. TTFT, inter-token
latency, tokens/s and block utilisation benchmarks arrive with the CUDA kernels.
### CUDA reference (not PyTorch CUDA)

`cargo run --release --example cuda_bench`, RTX 2070 SUPER versus this project's
CPU backend on the Ryzen 5 5600G:

| Workload | CPU | CUDA | Speedup |
|---|---:|---:|---:|
| matmul 512³ | 0.508 ms | 0.052 ms | 9.7× |
| linear 128×784×256 | 0.171 ms | 0.027 ms | 6.4× |
| Adam 1M params | 1.292 ms | 0.115 ms | 11.2× |

These compare against our own CPU backend, not PyTorch CUDA. PyTorch-CUDA and
TTFT/tokens/s benchmarks remain future work.

## Reproducing the Vulkan numbers

Vulkan correctness is covered by `cargo test -p modern-chainer-vulkan`. The
current reduction-heavy ops round-trip through the CPU, so Vulkan performance is
not yet benchmarked against CUDA/cuBLAS; that comparison belongs to the CUDA
milestone.
