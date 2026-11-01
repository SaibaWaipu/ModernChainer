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
appends copy storage and the attention kernel is scalar. CUDA static prefill and
decode are benchmarked below; CPU paged block utilisation and continuous
batching latency are still open.
### CUDA vs PyTorch CUDA

RTX 2070 SUPER, PyTorch 2.11.0+cu128, 20 iterations after 3 warmups (Adam:
10/2). `torch/ours` above 1 favors Modern Chainer.

| Workload | Modern Chainer | PyTorch | torch/ours |
|---|---:|---:|---:|
| matmul eager 512³ | 0.048 ms | 0.046 ms | 0.95× |
| matmul graph 512³ | 0.043 ms | 0.045 ms | 1.04× |
| linear 128×784×256 | 0.023 ms | 0.038 ms | 1.61× |
| linear+gelu 128×784×256 | 0.023 ms | 0.040 ms | 1.70× |
| Adam eager 1M | 0.156 ms | 0.212 ms | 1.36× |
| Adam graph 1M | 0.077 ms | 0.231 ms | 3.01× |

Method:

- Modern Chainer: `cargo run --release --example cuda_bench`; direct `Ops`
  calls with a non-default stream and explicit synchronization.
- PyTorch: `scripts/bench_cuda_vs_pytorch.py`; `torch.matmul`,
  `F.linear`, `torch.optim.Adam(capturable=True)`, and `torch.cuda.CUDAGraph`.
- Our linear uses the cuBLASLt bias/GELU epilogue; the decoder MLP uses
  bias+GELU. The CUDA memory pool is active for all measurements. Our Adam
  graph replays the fused single-kernel update into preallocated buffers.
- CUDA 12.8 wheels; the GTX 1080 (sm_61) is not supported by this PyTorch
  build, so both sides run on the RTX 2070 SUPER (sm_75).

### Batched static prefill / decode

Fixed-shape batched decoder, F32, B=8, 32 decoded tokens, random weights,
4 layers / d_model 64 / 4 heads, RTX 2070 SUPER. PyTorch is 2.11.0+cu128 eager
with explicit matmul/softmax attention. Both sides copy `[batch, vocab]` logits
to the host and run argmax each decode step, then copy tokens back; decode
throughput includes that host loop. GPU clocks vary, so ranges are shown.

| Metric | Modern Chainer | PyTorch eager | Ratio |
|---|---:|---:|---:|
| TTFT, prefill 16 | 0.26 ms | 1.45–1.58 ms | 5.5× |
| prefill throughput, 16 | 488k tok/s | 81k–88k tok/s | 5.5× |
| decode, context 16→48 | 35k–41k tok/s | 4.0k–6.7k tok/s | 5–10× |
| TTFT, prefill 512 | 7.6 ms | 5.20 ms | 0.69× |
| prefill throughput, 512 | 540k tok/s | 787k tok/s | 0.69× |
| decode, context 512→544 | 35.5k tok/s | 6.9k tok/s | 5.1× |

Variable-length prompt batching (`VARIABLE=1`, lengths descend from the
capacity) keeps TTFT around 7.6 ms at capacity 512 and reports real-token
prefill throughput around 420k tok/s.

KV cache element type (B=8, max_seq=560, 4 layers):

| KV dtype | Cache | TTFT, prefill 512 | Decode, context 512→544 |
|---|---:|---:|---:|
| F32 | 8 MiB | 7.6 ms | 35.5k tok/s |
| F16 | 4 MiB | 7.6–8.4 ms | ~19k tok/s |
| BF16 | 4 MiB | 7.6 ms | ~19k tok/s |

Method:

- Modern Chainer: `cargo run --release --example static_batched_decoder`.
  Env: `BATCH`, `PREFILL_LEN`, `DECODE_TOKENS`, `MAX_SEQ`, `KV_DTYPE`,
  `VARIABLE`, `FLASH_PREFILL`, `D_MODEL`, `N_HEADS`, `D_FF`.
  Prefill is one captured graph replay: row-batched linear/LayerNorm/MLP,
  fused flash attention (head_dim ≤ 128) or cuBLAS strided-batched attention,
  then LM head only on each real last token. Decode is one captured graph
  replay per step: per-sequence positions, batched GEMM attention for F32, or
  native low-precision split-K attention for F16/BF16.
- PyTorch: `scripts/bench_static_batched_vs_pytorch.py`, same shape, eager
  `F.linear`/`F.layer_norm`/`F.gelu`, explicit KV cache and Python per-step
  dispatch.
- The PyTorch side is eager, not `torch.compile` or a captured graph. The
  comparison measures graph replay plus the kernel layout for this shape, not a
  like-for-like compiler comparison.

At prefill 512 the workload is dominated by the linear/MLP GEMMs; flash
attention keeps low-precision KV native but does not make the 64-wide model
faster than the GEMM path. At larger head dimensions the fused kernel is the
intended path. F32 decode remains GEMM-based and scans `max_seq`; native
low-precision decode avoids the upcast but is currently slower at long context
than the F32 upcast path.

Matmul is near parity; linear and Adam show the benefit of fused epilogues and
fewer launches. Graph replay helps both, but our Adam graph has one kernel
versus PyTorch's optimizer step machinery.

## Unified Python device benchmark

`scripts/bench_vs_pytorch.py` now runs the core microbenchmarks through the
Python API on CPU, CUDA and Vulkan, with PyTorch on the matching device. The
same RTX 2070 SUPER was used for CUDA and `vulkan:1`.

```bash
PYTHONPATH="$PWD/python" BENCH_DEVICES=cpu,cuda,vulkan:1 \
  /tmp/opencode/torch-venv/bin/python scripts/bench_vs_pytorch.py
```

Milliseconds, `ours / torch`; PyTorch 2.11.0+cu128, 5 warmups, median of 30
(10 for Adam). CPU uses the PyTorch CPU path.

| Device | matmul 512³ | add 4M | linear fwd+bwd | Adam 1024² | geomean |
|---|---:|---:|---:|---:|---:|
| CPU | 0.49 / 0.43 | 2.19 / 8.00 | 0.58 / 0.32 | 3.56 / 4.94 | 1.22 / 1.53 (1.25×) |
| CUDA | 0.074 / 0.070 | 0.185 / 0.142 | 0.207 / 0.198 | 0.457 / 0.526 | 0.190 / 0.179 (0.94×) |
| Vulkan:1 | 0.85 / 0.071 | 0.44 / 0.141 | 3.13 / 0.200 | 3.85 / 0.52 | 1.46 / 0.179 (0.12×) |

Python/CUDA is close to PyTorch for this suite; Adam is slightly ahead.
Python/Vulkan improved from 2.11 to 1.46 ms geomean: fused Adam (10.7 → 3.85 ms),
fused softmax CE and GELU/SiLU, native `sum_all` / `sum_to`, 32×32 and
small-shape matmul kernels, transpose-free linear and linear backward,
vec4 elementwise kernels, storage/uniform buffer pools, and command batching
for linear forward/backward and reductions. It is still about 0.12× PyTorch;
the scalar WGSL matmul dominates.

## Reproducing the Vulkan numbers

Vulkan correctness is covered by `cargo test -p modern-chainer-vulkan`. The
current reduction-heavy ops round-trip through the CPU, so Vulkan performance is
not yet benchmarked against CUDA/cuBLAS; that comparison belongs to the CUDA
milestone.
