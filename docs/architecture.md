# Architecture

## Crate graph

```
modern-chainer (facade)
 ├── modern-chainer-core
 ├── modern-chainer-autograd ── modern-chainer-core
 ├── modern-chainer-nn ──────── modern-chainer-autograd
 ├── modern-chainer-optim ───── modern-chainer-nn
 ├── modern-chainer-data ────── modern-chainer-core
 ├── modern-chainer-serial ──── modern-chainer-core
 ├── modern-chainer-run ─────── modern-chainer-core + autograd
 ├── modern-chainer-llm ─────── modern-chainer-run
 ├── modern-chainer-cpu ─────── modern-chainer-core
 ├── modern-chainer-vulkan ──── modern-chainer-core + cpu (fallbacks)
 └── modern-chainer-cuda ────── (probe only)
modern-chainer-python ──────── modern-chainer (PyO3)
```

The dependency direction is always towards `core`. Backends never depend on
autograd or nn; they implement the `Ops` trait and are selected through the
`Backend` associated type.

## Tensor and storage

`Tensor<B: Backend>` is a cheap handle:

- `Arc<B::Storage>` — reference-counted backend memory
- `DType`, `Shape`, `Strides`, element `offset`
- a small backend handle `B` (e.g. `CpuBackend` is a zero-sized type)

Tensors are immutable. Kernels allocate a new storage and return a new tensor.
The CPU storage has two variants:

- `Bytes(Vec<u8>)` for uploads and non-f32 dtypes
- `F32(Vec<f32>)` for kernel output, allowing `Vec<f32>` → tensor with no byte
  copy

`f32` is the implemented dtype for kernels; `i64`/`i32`/`u8` are used for
labels. `Tensor::to_bytes` copies through the backend, which lets generic code
(gradcheck, serialization, GPU readback) inspect values without knowing the
device.

## Backend and Ops

```rust
pub trait Backend: Clone + Send + Sync + 'static {
    type Storage: Storage;
    type Ops: Ops<Self>;
    fn device(&self) -> Device;
    fn ops(&self) -> &Self::Ops;
    fn alloc(&self, bytes: usize) -> Result<Self::Storage>;
    fn alloc_zeroed(&self, bytes: usize) -> Result<Self::Storage>;
    fn copy_from_host(&self, dst: &mut Self::Storage, src: &[u8]) -> Result<()>;
    fn copy_to_host(&self, src: &Self::Storage, dst: &mut [u8]) -> Result<()>;
}
```

`Ops` is the kernel ABI. Every method has an `Unsupported` default so a backend
can land incrementally. Besides the primitive ops it contains two fused hooks
used by the training path:

- `linear(x, w, bias)` and `linear_backward(g, x, w)`
- `adam_update(param, grad, m, v, cfg) -> (param, m, v)`

Defaults compose primitives, so a new backend works before it has fused
kernels. CPU overrides them; Vulkan inherits the default `linear` path (bias
broadcast is a native Vulkan kernel).

## Autograd

A `Variable<B>` holds `Arc<VarInner<B>>`:

- `data: RwLock<Tensor<B>>` (parameters can be replaced by optimizers)
- `grad: Mutex<Option<Tensor<B>>>`
- `node: RwLock<Option<Arc<dyn Node<B>>>>`
- `requires_grad: AtomicBool`
- an optional name

`apply(op, inputs)` runs `op.forward`, then attaches an `OpNode` that saves the
inputs and outputs. `Variable::backward`:

1. Collects reachable variables with an explicit stack (no recursion).
2. Seeds the output with `ones_like`.
3. Processes the reverse topological order.
4. Calls `Node::backward` and accumulates each input gradient with `Ops::add`
   (broadcast cases are reduced by the op itself with `sum_to`).

Gradients are stored on intermediates as well as leaves. `unchain_backward`
drops the node, which truncates the graph without touching data. The gradient
enable flag is thread-local; `NoGradGuard` restores it on drop, including
during unwinding.

## Explicit run state

`modern-chainer-run` adds a second execution model alongside the autograd graph:

- `StateStore` is an arena of typed slots (`Tensor`, `Variable`, `Scalar`,
  `Paged`). Slots are declared with `StateSpec` (layout, dtype, gradient policy,
  fork policy, persistence).
- `StateKey` is the namespaced identity; `StateId` is a generational handle.
- `StatePlan` is the sorted introspection view. Model-specific `Handles` hold
  `StateId`s, so the hot path never hashes strings.
- `Step` is the model contract: `declare_state` once, then `step(&mut RunContext,
  &Handles, input)`.
- `Run` owns the model (`Arc`), handles, store, plan, mode, RNG and step counter.
- Dense values are immutable `Arc`-backed tensors, so `fork` is a shallow store
  clone and a child cannot mutate its parent.
- `snapshot`/`restore` and `transaction` (with rollback on drop) make speculative
  and failure-recovery flows explicit.

`modern-chainer-llm` builds a decoder on this layer. Each layer owns a `k`/`v`
slot and a shared `position` scalar. Prefill and decode use the same
`embedding -> layer_norm -> q/k/v -> attention -> MLP` path; the KV cache grows
by `concat_seq`. `StateLayout::Paged` and `Ops::paged_attention` provide the
block-table path, `ContinuousBatcher` adds a request table / `BatchPlan`, and
`speculative_generate` uses fork/verify/commit. See `docs/state-by-run.md`.

## Modules and optimizers

`Module<B>` exposes `forward` and `parameters`. `Linear` stores `[out, in]`
weights (Chainer layout) and an optional bias; its `forward` calls the fused
`linear` op. `Sequential` is a simple container.

Optimizers are stateful objects that receive a slice of `Parameter`. State
(`m`, `v`, velocity) is keyed by `Variable::id()` (the `Arc` pointer), which is
stable while the parameter handle lives. `zero_grad` clears gradients on every
parameter.

## Concurrency

- `Tensor` handles are `Send + Sync` when the backend is.
- Gradients are behind a `Mutex`; parameter data behind an `RwLock`. Kernel
  hot paths read data once per op and do not lock per element.
- Run state is single-writer: a `Step` receives `&mut RunContext`, so state
  mutation is explicit and does not need per-slot locks.
- CPU kernels use rayon; GEMM uses the `gemm` crate with an explicit thread
  count (see `docs/backends.md` for the `Rayon(0)` caveat).
- Vulkan uses a single `wgpu::Device`/`Queue` per backend; commands are
  submitted synchronously. A future async path can batch submissions.

## Serialization

`modern-chainer-serial` implements a versioned `.mcn` container: magic, count,
then per-entry name, dtype, shape and host bytes. It is intentionally small and
portable. A Chainer `.npz` reader is on the roadmap; the native format is not
meant to replace it long-term.

## Python binding

`modern-chainer-python` is a `cdylib` built by maturin. It wraps
`Tensor<CpuBackend>` and `Variable<CpuBackend>` in PyO3 classes and exposes the
functional API as module functions. NumPy interop goes through host bytes:

- `Tensor.from_numpy` preserves `int64`/`int32`/`uint8`, otherwise casts to
  `float32`
- `Tensor.numpy` / `Variable.array` use `numpy.frombuffer` + `reshape`

The binding is thin on purpose: the engine, graph and optimizer state stay in
Rust. Python-only convenience (`Chain`, `init_scope`) lives in
`python/chainer/__init__.py`. The State-by-Run layer (`Run`, `StateStore`,
decoder) is Rust-first in v0.2; the Python wrapper is planned.
