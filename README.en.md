# Modern Chainer

[日本語](README.md) | **English**

Define-by-Run and State-by-Run deep learning with a Rust core — a modern revival
of [Chainer](https://github.com/chainer/chainer)'s ideas.

Chainer pioneered the *define-by-run* execution model: the computational graph
is built while `forward` runs, so ordinary Python control flow is the model
definition. Modern Chainer keeps that mental model and the vocabulary
(`Variable`, `F.*`, `L.*`, `Chain`, `Optimizer`) while moving the engine into
Rust so the same core can serve Rust and Python, with CPU and GPU backends.

The second principle is **State by Run**. Persistent execution state — KV
caches, recurrent hidden state, sampler RNG, scheduler-visible buffers — is
declared while a step executes but owned by an explicit `Run`. Nothing that
survives a step is hidden inside a module.

```python
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

class MLP(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, 100)
            self.l2 = L.Linear(100, 10)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))
```

## Status (v0.12.0)

| Area | Status |
|---|---|
| Define-by-Run autograd (Rust) | Stable |
| State by Run (`Run`, `Step`, `StateStore`) | Stable: fork, snapshot/rollback, transactions |
| Dense KV decoder | Working: 2-layer reference, prefill/decode, sampling |
| Paged state + CPU paged attention | Working: block allocator, block table, append/truncate, COW |
| Batched decode | Working: `PagedBatchRunner` + batched paged attention |
| Continuous batching | Working: `RequestTable` / `BatchPlan` (logical batching) |
| Speculative decoding | Working: greedy fork/verify/commit reference |
| CUDA backend | Working: NVRTC + cuBLAS/cuBLASLt, F32/TF32/F16/BF16 compute types, AMP, memory pool, CUDA Graph, wide paged attention |
| Static Step API / Run-level CUDA Graph | Working: `CudaStaticStep` + `StaticRun`; includes `LinearMseAdamStep`, `LinearForwardStep` and `StaticDecoderStep` |
| Batched static prefill / decode | Working: `BatchedStaticDecoder`; variable-length prompts, per-sequence positions, F16/BF16 KV, native low-precision decode attention, flash prefill, CUDA Graph replay |
| Python bindings | CPU / CUDA / Vulkan: `Tensor`, `Variable`, `F.*`, `L.Linear(device=...)`, `SGD`/`Adam`, `Chain`; `Tensor.to(device)`, `chainer.synchronize()` |
| Serialization | Native `.mcn`; Chainer `.npz` reader planned |
| Trainer / Updater / Extensions | Planned |
| Documentation | README (Japanese default + English), `SPEC.md` / `docs/` (English) |

The Python package is intentionally not a byte-for-byte reimplementation of
Chainer 7. It keeps the shape of the API and the define-by-run feel while
cleaning up the parts that no longer make sense on modern Python. State by Run
is Rust-first in v0.4.

## Quick start — Rust

```bash
cargo run --release --example mlp_regression
cargo run --release --example state_decoder
cargo run --release --example continuous_batching
cargo run --release --example paged_batching
cargo run --release --example speculative
cargo run --release --example cuda_matmul     # requires a CUDA driver and NVRTC/cuBLAS
cargo run --release --example cuda_bench
cargo run --release --example graph_trainer
cargo run --release --example static_step
cargo run --release --example static_decoder
cargo run --release --example static_batched_decoder
cargo run --release --example vulkan_matmul   # requires a Vulkan loader/ICD
```

Select a device from Python:

```python
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

print(chainer.vulkan_devices())          # adapter names in index order
layer = L.Linear(4, 2, device="vulkan:1")
x = chainer.Variable(chainer.Tensor.from_numpy(x_np, device="vulkan:1"))
loss = F.sum(layer(x))
loss.backward()
optimizer = O.Adam(0.001)
optimizer.setup(layer.parameters())
optimizer.update()
```

```rust
use modern_chainer::prelude::*;

let model = Linear::new(4, 2, 0)?;
let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &[1.0; 8], [2, 4])?);
let y = model.forward(&x)?;
let loss = F::sum(&y)?;
loss.backward()?;
```

### CUDA

When using the pip CUDA libraries, expose NVRTC/cuBLAS before running:

```bash
export LD_LIBRARY_PATH="$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
cargo run --release --example cuda_matmul
```

`cudarc` uses dynamic loading, so no CUDA toolkit or `nvcc` is required at
build time. NVRTC compiles kernels at runtime for the device's compute
capability. A memory pool reuses device allocations, and `ComputeType` selects
the GEMM precision.

```rust
let gpu = CudaBackend::new(0)?;
gpu.set_compute_type(ComputeType::F16); // F32 / Tf32 / F16 / Bf16

// AMP: GEMMs run in f16/bf16; LayerNorm/Softmax/loss stay f32.
let loss = autocast(DType::F16, || {
    let y = model.forward(&x)?;
    F::mean(&F::mul(&F::sub(&y, &target)?, &F::sub(&y, &target)?)?)
})??;
let scaled = scaler.scale_loss(&loss)?;
scaled.backward()?;
scaler.step(&mut optimizer, &params)?;
```

`GradScaler` handles loss scaling, gradient unscaling, inf/nan detection and
scale growth/backoff. Autocast lowers GEMM-heavy ops only; LayerNorm, Softmax
and reductions remain f32.

`CudaStaticStep` is a reusable trait: a model declares fixed buffers and
enqueues one in-place step. `StaticRun` owns the buffers, captures the CUDA
Graph, and replays it with a new input. Included models: `LinearMseAdamStep`
(Linear + MSE + Adam), `LinearForwardStep`, and `StaticDecoderStep` (one-token
decode with device-resident KV cache and position). Counters live on the
device, so the same graph replays without host arguments.

`BatchedStaticDecoder` captures fixed-shape batched prefill and decode as two
CUDA Graphs over one shared KV cache. `prefill_batch(&[&[u32]])` accepts
variable-length prompts, pads tokens, sets per-sequence positions, and applies
the LM head only to each real last token. Prefill attention uses a fused flash
kernel for head_dim ≤ 128 (reading F32/F16/BF16 directly) or cuBLAS
strided-batched GEMM otherwise. Decode processes one token per batch element
with per-sequence positions and advances each position inside the graph.
`KV_DTYPE=f16` / `bf16` decode uses native low-precision attention. TTFT and
tok/s are measured by `examples/static_batched_decoder.rs`.

## Quick start — Python

```bash
cd python
uv venv
uv pip install maturin numpy pytest
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python examples/train_mlp.py
```

```python
import chainer
import chainer.functions as F
import chainer.links as L

model = L.Linear(4, 2)
x = chainer.Variable(chainer.Tensor.from_list([1.0, 2.0, 3.0, 4.0], [1, 4]))
loss = F.sum(model(x))
loss.backward()
print(model.W.grad.array)
```

## State by Run

```rust
use modern_chainer::prelude::*;

let decoder = Decoder::new(CpuBackend::new(), config, 42)?;
let mut run = Run::new(decoder, CpuBackend::new(), RunMode::Inference, 7)?;

let mut logits = run.step(prompt.clone())?;   // prefill
for (key, id) in run.plan().iter() {          // state is visible
    println!("{key} -> {id}");
}
let mut draft = run.fork();                   // branch the KV cache
draft.step(vec![next_token])?;                // parent is untouched
```

- `StateStore` owns tensors, variables, scalars and paged state under
  namespaced `StateKey`s.
- `Run` owns the store, RNG and step counter.
- `Step` declares state once and receives typed `StateId` handles, so the hot
  path has no string lookup.
- Dense state uses immutable `Arc`-backed tensors, which makes `fork` cheap and
  rollback trivial.
- `snapshot` / `restore` / `transaction` (rollback on drop) are provided.

### Paged KV and batched decode

```rust
let mut runner = PagedBatchRunner::new(decoder, CpuBackend::new(), 2, 16, 7)?;
runner.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
let output = runner.step()?;   // one paged-attention call for the whole batch
```

- `BlockAllocator` manages free blocks and reference counts.
- `PagedKvCache` supports `append` / `truncate` / `fork` with
  tensor-granularity COW, so forks are isolated.
- `Ops::paged_attention_batch` takes per-request block tables and sequence
  lengths and processes the whole batch in one kernel.
- Prefill runs through the dense `Run` path; its KV is converted into paged
  caches before batched decode.
- The v0.4 CPU implementation is a reference: block appends copy storage and
  attention is scalar.

### Continuous batching and speculative decoding

```rust
// Logical batching: per-request Run, with visible prefill/decode plans.
let mut batcher = ContinuousBatcher::new(decoder, CpuBackend::new(), 7);
batcher.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
let output = batcher.step()?;
```

```rust
// Greedy speculative decoding; output matches target-only greedy decoding.
let tokens = speculative_generate(&mut target, &mut draft, prompt, 8, 3)?;
```

- `ContinuousBatcher` maintains a `RequestTable` and a
  `BatchPlan { prefill, decode }`; requests can be added between steps.
- `PagedBatchRunner` is the real batched decode path using paged KV.
- Speculative decoding drafts tokens, verifies them with `target.fork()`,
  commits only accepted tokens, and resynchronises the draft with
  `snapshot`/`restore`.

## Architecture

```
                 ┌────────────────────────────┐
  Python API ────▶  modern-chainer-python      │  PyO3 / maturin
                 └────────────┬───────────────┘
                              │
  Rust API  ─────────────────▶│
                 ┌────────────▼───────────────┐
                 │  modern-chainer (facade)   │  prelude, F::, L::
                 └──┬────────┬────────┬───────┘
                    │        │        │
        ┌───────────▼──┐ ┌───▼─────┐ ┌▼─────────────┐
        │ autograd      │ │ nn      │ │ optim        │
        │ Variable/Op   │ │ modules │ │ SGD/Adam     │
        └───────┬───────┘ └────┬────┘ └──────┬───────┘
                │              │             │
        ┌───────▼──────────────▼─────────────▼───────┐
        │  modern-chainer-run: Run, Step, StateStore  │
        │  modern-chainer-llm: decoder, paged KV,     │
        │                      batching, speculation  │
        └───────┬─────────────────────────────────────┘
                │
        ┌───────▼─────────────────────────────────────┐
        │  core: Tensor, Shape, DType, Device,        │
        │        Storage, Backend, Ops                │
        └───────┬──────────────────────┬──────────────┘
                │                      │
        ┌───────▼──────┐       ┌───────▼──────┐
        │ CPU (gemm,   │       │ Vulkan       │
        │ rayon)       │       │ (wgpu, WGSL) │
        └──────────────┘       └──────────────┘
```

`Backend` / `Ops` are the kernel ABI. `Tensor<B>` is generic over the backend,
so CPU code monomorphises while the Python layer wraps a `DynTensor` enum. See
[`docs/architecture.md`](docs/architecture.md).

## Performance

Measured on 2026-10-31, AMD Ryzen 5 5600G (6 cores / 12 threads), 12 threads for
both implementations, PyTorch 2.15.0+cpu. Median of repeated runs; lower is
better. These are micro-benchmarks, not a claim about every workload.

| Workload | Modern Chainer | PyTorch | Ratio |
|---|---:|---:|---:|
| matmul 512×512×512 | 0.43–0.52 ms | 0.42 ms | 0.81–0.98× |
| elementwise add (4M f32) | 2.23–2.30 ms | 1.95–2.16 ms | 0.85–0.97× |
| linear fwd+bwd 128×784×256 | 0.54–0.61 ms | 0.32 ms | 0.52–0.60× |
| Adam step 1024×1024 | 3.38–3.81 ms | 4.02–4.98 ms | 1.19–1.40× |
| **geometric mean** | **~1.20 ms** | **~1.04 ms** | **~0.87–0.91×** |

Adam wins, matmul/elementwise are close, and Linear backward is the weak spot.
The original 1.2× geometric-mean target is not met, but the benchmark harness is
checked in so improvements are measurable. CUDA (RTX 2070 SUPER) versus PyTorch 2.11.0+cu128 on the same GPU.
`torch/ours` above 1 means Modern Chainer is faster.

| Workload | Modern Chainer | PyTorch | torch/ours |
|---|---:|---:|---:|
| matmul eager 512³ | 0.046 ms | 0.045 ms | 0.99× |
| matmul graph 512³ | 0.043 ms | 0.044 ms | 1.03× |
| linear 128×784×256 | 0.021 ms | 0.038 ms | 1.82× |
| linear+gelu 128×784×256 | 0.021 ms | 0.041 ms | 1.91× |
| Adam eager 1M | 0.112 ms | 0.216 ms | 1.93× |
| Adam graph 1M | 0.077 ms | 0.229 ms | 2.99× |

Batched static decoder (RTX 2070 SUPER, F32, B=8, decode=32, random weights).
TTFT is the median batched prefill; decode includes a host loop that copies
logits to CPU and runs argmax each step. GPU clocks vary, so treat these as
representative.

| Metric | Modern Chainer | PyTorch 2.11 eager | Ratio |
|---|---:|---:|---:|
| TTFT (prefill=16) | 0.26 ms | 1.45–1.58 ms | 5.5× |
| prefill (prefill=16) | 488k tok/s | 81k–88k tok/s | 5.5× |
| decode (context 16→48) | 35k–41k tok/s | 4.0k–6.7k tok/s | 5–10× |
| TTFT (prefill=512) | 7.6 ms | 5.20 ms | 0.69× |
| prefill (prefill=512) | 540k tok/s | 787k tok/s | 0.69× |
| decode (context 512→544) | 35.5k tok/s | 6.9k tok/s | 5.1× |

Variable-length prompts (`VARIABLE=1`, prefill capacity 512) keep TTFT around
7.6 ms and real-token prefill throughput around 420k tok/s. Prefill uses flash
attention for head_dim ≤ 128 (reading F32/F16/BF16 directly) or cuBLAS
strided-batched GEMM; decode uses per-sequence-position GEMM or native
low-precision attention. KV cache storage is F32 / F16 / BF16; B=8,
max_seq=560, 4 layers uses 8 / 4 / 4 MiB. F16/BF16 decode attention reads the
cache directly without upcasting.

```bash
cargo run --release --example cuda_bench
cargo run --release --example graph_trainer
cargo run --release --example static_step
cargo run --release --example static_decoder
cargo run --release --example static_batched_decoder
PREFILL_LEN=512 DECODE_TOKENS=32 KV_DTYPE=f16 \
  cargo run --release --example static_batched_decoder
VARIABLE=1 PREFILL_LEN=512 DECODE_TOKENS=32 \
  cargo run --release --example static_batched_decoder
# PyTorch comparison (CUDA-enabled torch venv)
CUDA_VISIBLE_DEVICES=<RTX 2070S UUID> \
  /tmp/opencode/torch-venv/bin/python scripts/bench_cuda_vs_pytorch.py
CUDA_VISIBLE_DEVICES=<RTX 2070S UUID> \
  /tmp/opencode/torch-venv/bin/python scripts/bench_static_batched_vs_pytorch.py
```

cuBLASLt fuses bias and GELU into the GEMM epilogue; the decoder MLP uses
bias+GELU in one kernel. `ComputeType` selects F32 / TF32 / F16 / BF16. A CUDA
memory pool reuses device allocations. CUDA Graph replays fixed-shape
matmul/Adam updates; the PyTorch Adam graph is measured with
`capturable=True`.

- [`scripts/bench_vs_pytorch.py`](scripts/bench_vs_pytorch.py)
- [`docs/benchmarks.md`](docs/benchmarks.md)

### Unified Python device benchmark

Python can compare CPU, CUDA and Vulkan against PyTorch. Vulkan accepts an
adapter index; `vulkan:1` is the RTX 2070 SUPER on this machine.

```bash
PYTHONPATH="$PWD/python" BENCH_DEVICES=cpu,cuda,vulkan:1 \
  /tmp/opencode/torch-venv/bin/python scripts/bench_vs_pytorch.py
```

RTX 2070 SUPER, PyTorch 2.11.0+cu128, `matmul 512³` / `add 4M` /
`linear fwd+bwd 128×784×256` / `Adam 1024²`. Milliseconds, `ours / torch`.

| Device | matmul | add 4M | linear fwd+bwd | Adam | geomean |
|---|---:|---:|---:|---:|---:|
| CPU | 0.49 / 0.43 | 2.19 / 8.00 | 0.58 / 0.32 | 3.56 / 4.94 | 1.22 / 1.53 (1.25×) |
| CUDA | 0.074 / 0.070 | 0.185 / 0.142 | 0.207 / 0.198 | 0.457 / 0.526 | 0.190 / 0.179 (0.94×) |
| Vulkan:1 | 1.00 / 0.071 | 0.543 / 0.142 | 3.48 / 0.207 | 11.0 / 0.573 | 2.14 / 0.186 (0.09×) |

Python/CUDA is close to parity. Python/Vulkan is correct but still far slower;
the tiled matmul and CPU-fallback reductions dominate. Native reductions and
matmul optimization are next.

## Build and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude modern-chainer-python
cargo bench -p modern-chainer --bench cpu_ops

cd python
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest
```

Vulkan tests use the first available adapter (NVIDIA/AMD/Intel or Mesa
lavapipe) and skip cleanly when no Vulkan loader is present.

## Documentation

- [`SPEC.md`](SPEC.md) — specification
- [`docs/state-by-run.md`](docs/state-by-run.md) — State by Run, Run/Step, paged KV, batching, speculation
- [`docs/architecture.md`](docs/architecture.md) — crate graph, autograd, kernel ABI
- [`docs/backends.md`](docs/backends.md) — CPU, Vulkan and the CUDA plan
- [`docs/python-api.md`](docs/python-api.md) — Python surface and differences from Chainer 7
- [`docs/benchmarks.md`](docs/benchmarks.md) — methodology and results
- [`docs/roadmap.md`](docs/roadmap.md) — roadmap
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow

## Known limitations

- CUDA native coverage is elementwise, reductions, matmul, linear, Adam,
  softmax cross entropy, embedding, layer_norm, attention and paged attention.
  Only generic `sum_to` (non-bias) falls back to CPU.
- Batch size and `prefill_len` capacity are fixed. Variable-length prompts fit
  within the capacity, but continuous admission, mid-sequence request joins, and
  prompts larger than the capacity are not supported.
- F32 decode attention runs over the full `max_seq` GEMM, including zero regions
  beyond each sequence position, so large `max_seq` with short sequences is
  inefficient.
- Native low-precision decode attention supports head_dim ≤ 256; flash prefill
  supports head_dim ≤ 128. Larger heads fall back to an F32 upcast path.
- `CudaStaticStep` is CUDA-specific; a generic static step for arbitrary `Run`
  is still future work.
- The static decoder is bounded by its fixed `max_seq` buffer.
- F16/BF16 are supported as compute types and via autocast AMP. Autocast lowers
  GEMM-heavy ops; LayerNorm/Softmax/reductions stay f32. A full-model
  f16/bf16 dtype path is not implemented.
- Wide paged attention uses a block-per-query CUDA kernel.
- Python does not expose CUDA tensors, `Run`, `PagedBatchRunner` or `Trainer`
  yet (Rust-first).
- Paged COW is tensor-granularity; block-level in-place COW and CUDA
  paged-attention kernels are future work.
- No release has been made.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
