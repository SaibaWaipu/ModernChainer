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

## Status (v0.5.0)

| Area | Status |
|---|---|
| Define-by-Run autograd (Rust) | Stable |
| State by Run (`Run`, `Step`, `StateStore`) | Stable: fork, snapshot/rollback, transactions |
| Dense KV decoder | Working: 2-layer reference, prefill/decode, sampling |
| Paged state + CPU paged attention | Working: block allocator, block table, append/truncate, COW |
| Batched decode | Working: `PagedBatchRunner` + batched paged attention |
| Continuous batching | Working: `RequestTable` / `BatchPlan` (logical batching) |
| Speculative decoding | Working: greedy fork/verify/commit reference |
| CUDA backend | Working: NVRTC + cuBLAS core ops; reductions/attention fall back to CPU |
| Python bindings | Core subset: `Tensor`, `Variable`, `F.*`, `L.Linear`, `SGD`/`Adam`, `Chain` |
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
cargo run --release --example vulkan_matmul   # requires a Vulkan loader/ICD
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
capability.

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
checked in so improvements are measurable. Paged/batched decode is a CPU
reference and not yet part of the performance comparison. CUDA core ops are
verified against CPU, but CUDA performance has not been benchmarked yet.

- [`scripts/bench_vs_pytorch.py`](scripts/bench_vs_pytorch.py)
- [`docs/benchmarks.md`](docs/benchmarks.md)

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

- CUDA native coverage is elementwise / matmul / linear / Adam. Reductions,
  softmax cross entropy and the LLM attention/paged ops fall back to CPU.
- Python does not expose CUDA tensors, `Run`, `PagedBatchRunner` or `Trainer`
  yet (Rust-first).
- Paged COW is tensor-granularity; block-level in-place COW and CUDA
  paged-attention kernels are future work.
- No release has been made.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
