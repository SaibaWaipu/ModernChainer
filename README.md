# Modern Chainer

Define-by-run deep learning with explicit run state and a Rust core — a modern
revival of [Chainer](https://github.com/chainer/chainer)'s ideas.

Chainer pioneered the *define-by-run* execution model: the computational graph
is built while `forward` runs, so ordinary Python control flow is the model
definition. Modern Chainer keeps that mental model and the vocabulary
(`Variable`, `F.*`, `L.*`, `Chain`, `Optimizer`) while moving the engine into
Rust so the same core can serve Rust and Python, with CPU and GPU backends.

The second principle is **State by Run**: persistent execution state (KV caches,
recurrent state, sampler RNG, scheduler-visible buffers) is declared while a
step executes but owned by an explicit `Run`. Nothing that survives a step is
hidden inside a module.

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

## Status (v0.2.0)

| Area | Status |
|---|---|
| Define-by-run autograd (Rust) | Stable |
| State by Run (`Run`, `Step`, `StateStore`) | Stable: fork, snapshot/rollback, transactions |
| Dense KV decoder | Working: 2-layer reference decoder, prefill/decode, sampling |
| Paged state / continuous batching / speculative decoding | Planned (state types reserved) |
| Training-By-Run | Planned |
| CPU backend | Stable, fused Linear + fused Adam, `gemm` + rayon |
| Vulkan backend | Working (wgpu/Vulkan), core kernels + CPU fallbacks |
| CUDA backend | Scaffold: capability probe only (see roadmap) |
| Python bindings | Core subset: `Tensor`, `Variable`, `F.*`, `L.Linear`, `SGD`/`Adam`, `Chain` |
| Serialization | Native `.mcn`; Chainer `.npz` reader planned |
| Trainer / Updater / Extensions | Planned |
| Docs | This README, `SPEC.md`, `docs/` |

The Python package is intentionally not a byte-for-byte reimplementation of
Chainer 7. It keeps the shape of the API and the define-by-run feel while
cleaning up the parts that no longer make sense on modern Python.

## Quick start — Rust

```bash
cargo run --release --example mlp_regression
cargo run --release --example state_decoder
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
import chainer.optimizers as O

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

`StateStore` owns tensors, variables and scalars under namespaced `StateKey`s.
`Run` owns the store, RNG and step counter; `Step` declares state once and gets
typed `StateId` handles, so the hot path has no string lookup. Dense state uses
immutable `Arc`-backed tensors, which makes `fork` cheap and rollback trivial.

See [`docs/state-by-run.md`](docs/state-by-run.md).

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
        │  modern-chainer-llm: dense KV decoder       │
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

The `Backend`/`Ops` traits are the kernel ABI. `Tensor<B>` is generic over the
backend, so CPU code monomorphises while the Python layer wraps a `DynTensor`
enum. See [`docs/architecture.md`](docs/architecture.md).

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

The original acceptance target was a 1.2× geometric mean. The current engine
wins on fused Adam and is close on matmul/elementwise, but loses on
`Linear` backward, where cuBLAS-class GEMM transposes are still faster. The
benchmark harness is checked in so the number can be defended and improved:
[`scripts/bench_vs_pytorch.py`](scripts/bench_vs_pytorch.py),
[`docs/benchmarks.md`](docs/benchmarks.md).

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

- [`SPEC.md`](SPEC.md) — product and technical specification
- [`docs/state-by-run.md`](docs/state-by-run.md) — explicit state, Run/Step, KV cache, transactions
- [`docs/architecture.md`](docs/architecture.md) — crate graph, autograd, kernel ABI
- [`docs/backends.md`](docs/backends.md) — CPU, Vulkan and the CUDA plan
- [`docs/python-api.md`](docs/python-api.md) — Python surface and differences from Chainer 7
- [`docs/benchmarks.md`](docs/benchmarks.md) — methodology and results
- [`docs/roadmap.md`](docs/roadmap.md) — next milestones

## License

Apache-2.0. See [`LICENSE`](LICENSE).
