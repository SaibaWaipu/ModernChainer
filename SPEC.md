# Modern Chainer — Specification

Status: v0.10.0 implementation. This document records the product intent, the
implemented surface, and the deliberately deferred work.

## 1. Vision

Chainer's contribution was not a particular layer catalogue; it was the
*define-by-run* execution model and a vocabulary that made neural networks feel
like ordinary programs. Modern Chainer revives that model on a Rust core:

- The graph is constructed during `forward`.
- `Variable::backward` walks the graph in reverse topological order.
- `no_grad` / `unchain_backward` give explicit control over graph lifetime.
- The same core is usable from Rust and Python.

Non-goals for v1: byte-for-byte Chainer 7 compatibility, a native C++ backend,
distributed training, and a complete model zoo.

## 2. Design principles

1. **Define-by-run first.** No static graph, no tracing requirement.
2. **Explicit over implicit.** Device, dtype, gradient enablement and graph
   truncation are explicit. Global state is limited to a thread-local gradient
   flag.
3. **Small core, pluggable backends.** `Backend` + `Ops` are the kernel ABI.
4. **Rust is the reference implementation.** Python is a binding with a
   Chainer-shaped façade.
5. **Numerical correctness is testable.** Every differentiable op is covered by
   central-difference `gradcheck`; GPU backends are compared against CPU.
6. **State is explicit.** Anything that survives a step is owned by a `Run` and
   visible through the state plan; modules do not hide caches.

## 3. Public API

### Rust

```rust
use modern_chainer::prelude::*;

let model = Linear::new(784, 10, 0)?;
let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &data, [batch, 784])?);
let y = model.forward(&x)?;
let loss = F::softmax_cross_entropy(&y, &target)?;
loss.backward()?;
optimizer.update(&model.parameters())?;
```

Implemented:

- `Tensor<B>`, `Shape`, `Strides`, `DType`, `Device`
- `Variable<B>`: `data`, `grad`, `backward`, `cleargrad`, `detach`,
  `unchain_backward`, `requires_grad`, `name`
- `Op<B>` / `Node<B>` / `apply`
- `no_grad`, `NoGradGuard`, `is_grad_enabled`
- `gradcheck`
- `F`: add, sub, mul, div, sqrt, relu, matmul, transpose, sum, mean, scale, neg,
  linear, softmax_cross_entropy
- `L`: `Linear`, `Sequential`, `Module`, `Parameter`
- Optimizers: `SGD`, `MomentumSGD`, `Adam` (fused CPU kernel), `zero_grad`
- Data: `Dataset`, `VecDataset`, `BatchIterator`
- Serialization: native `.mcn` save/load
- Backends: `CpuBackend`, `VulkanBackend`, `CudaBackend` (NVRTC + cuBLAS),
  `cuda::probe`, `vulkan::probe`
- State by Run: `Run`, `Step`, `RunContext`, `RunMode`, `StateStore`,
  `StateSpec`, `StateId`, `StateKey`, `StatePlan`, `Transaction`
- Dense LLM state: `Decoder`, `DecoderConfig`, `KvCacheHandles`,
  `Sampler`, `generate`
- Paged state: `StateLayout::Paged`, `PagedState`, `BlockAllocator`,
  `PagedKvCache`, `Ops::paged_attention`, `Ops::paged_append`,
  `Ops::paged_attention_batch`
- Batching: `RequestId`, `BatchPlan`, `ContinuousBatcher`, `BatchOutput`,
  `PagedBatchRunner` (real batched decode)
- Speculation: `speculative_generate`, `argmax_token`
- LLM kernels: embedding, layer_norm, softmax, SiLU, split/merge heads,
  dense causal attention, paged attention, sequence concat, last-row extraction

### Python

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

Implemented: `Tensor`, `Variable`, `Chain`, `L.Linear`, `F` functions listed
above, `O.SGD`, `O.Adam`, NumPy interop (`Tensor.from_numpy`, `Tensor.numpy`,
`Variable.array`).

Not yet implemented in Python: `Run`/state APIs, `Trainer`, `Updater`,
`Iterator` wrappers, `Extension`, `chainer.config`, `FunctionNode` subclassing,
`.npz` serialization, `chainer.cuda`/`chainer.backends`. The State-by-Run layer
is Rust-first in v0.5; `chainer.cuda_available()` reports the CUDA probe.

## 4. State by Run

Define-by-Run builds the graph. **State by Run** declares persistent state while
a step executes, but the `Run` owns it. The formal contract is a Mealy machine:

```
step: (params, input, state) -> (output, state')
```

### State taxonomy

- Graph / activations: transient, owned by the autograd graph.
- Model state: KV cache, recurrent hidden state, memory.
- Optimization state: parameters, grads, Adam m/v, EMA, scaler (v0.3).
- Execution state: request table, batch plan, block table, position, RNG.
- Environment state: data cursor, metrics.

Only the last four categories enter `StateStore` or typed run fields.
Activations never do.

### Core API

```rust
pub trait Step<B: Backend>: Send + Sync {
    type Input;
    type Output;
    type Handles: Clone + Send + Sync;
    fn declare_state(&self, builder: &mut StateBuilder<'_, B>) -> Result<Self::Handles>;
    fn step(
        &self,
        ctx: &mut RunContext<'_, B>,
        handles: &Self::Handles,
        input: Self::Input,
    ) -> Result<Self::Output>;
}
```

- `StateKey` is a namespaced identity; `StateId` is a generational arena handle.
- `StateSpec` declares layout, dtype, gradient policy, fork policy and
  persistence policy.
- `StateValue` is a tensor, variable, scalar, or paged state (paged reserved for
  v0.3).
- `StateStore` owns values; `StatePlan` is the sorted introspection view.
- Module-specific `Handles` carry `StateId`s so hot paths avoid string lookup.
- `Run` owns the model, handles, store, plan, mode, RNG and step counter.
- `Run::fork` clones the store cheaply (immutable `Arc`-backed dense tensors),
  `snapshot`/`restore` and `transaction` provide rollback. `Transaction` also
  rolls back on drop.

### Dense KV decoder (v0.2)

`modern-chainer-llm` implements a 2-layer reference decoder with explicit dense
KV state:

```
layers.{i}.kv.k  [heads, 0, head_dim]
layers.{i}.kv.v  [heads, 0, head_dim]
kv.position      i64
```

Prefill and decode use one path; only the sequence length differs. The
acceptance test is that prefill logits equal token-by-token decode logits.

Implemented: embedding, layer_norm, softmax, SiLU, split/merge heads, causal
attention, sequence concat, last-row extraction, greedy/temperature/top-k
sampling.

### v0.3 paged state, batching and speculation

- `StateLayout::Paged { block_size, num_blocks, trailing }` allocates
  `blocks [num_blocks, block_size, trailing...]` and an i64 block table.
- `Ops::paged_attention` reads K/V through a logical-to-physical block table;
  the CPU reference matches dense attention for prefill and decode.
- `ContinuousBatcher` maintains a request table and a `BatchPlan` with separate
  `prefill` and `decode` lists. Requests can be added between steps. v0.3
  executes a plan sequentially; vectorised paged kernels are the next step.
- `speculative_generate(target, draft, ...)` uses target-side greedy
  verification. The target model decides every accepted token; rejected draft
  state is discarded with `snapshot`/`restore`. The result is identical to
  target-only greedy decoding, which is the acceptance test.

### v0.4 paged allocator, COW and batched decode

- `BlockAllocator` with reference counts and free-list reuse.
- `PagedKvCache` wraps block tensors + a block table, supports `append`,
  `truncate`, `fork` and `from_dense_with_capacity`. Appends copy block storage
  (tensor-granularity COW), so a fork cannot observe child writes.
- `Ops::paged_append` writes new K/V through the block table.
- `Ops::paged_attention_batch` runs one kernel over a batch with per-request
  block tables and sequence lengths.
- `PagedBatchRunner` prefills each request through the dense `Run` path,
  converts KV into paged caches, then performs a single batched decode step over
  all active requests. Generated tokens match dense greedy decoding.

### v0.5 CUDA core backend

- `modern-chainer-cuda` uses `cudarc` with dynamic loading. No CUDA toolkit or
  `nvcc` is needed at build time.
- Device init via the driver API, typed `CudaStorage` (`f32`/`i64`/`u8`),
  host/device copies.
- NVRTC compiles elementwise, ReLU/SiLU, sqrt, scale, fill, bias-add,
  transpose, reductions, softmax CE, embedding, layer_norm, attention and
  paged-attention kernels at runtime for the device's compute capability.
- cuBLAS handles `matmul`, fused `linear`, and `linear_backward`.
- Native paged KV: `paged_append`, `paged_attention`, `paged_attention_batch`.
- cuBLASLt fuses bias and GELU into the linear GEMM epilogue.
- `ComputeType` F32/TF32/F16/BF16. F32 uses cuBLAS+bias; TF32 uses cuBLASLt;
  F16/BF16 cast f32 inputs, run cuBLASLt mixed GEMM and cast back.
- A per-runtime CUDA memory pool reuses f32/i64/u8/f16/bf16 buffers.
- AMP: `autocast(F16/BF16)` lowers linear/matmul/linear_activation while
  LayerNorm, Softmax, reductions and the loss stay f32. `GradScaler` provides
  loss scaling, gradient unscaling, inf/nan detection and dynamic scale
  growth/backoff.
- CUDA Graph capture/replay for preallocated matmul (`graph_matmul`), Adam
  (`graph_adam`), and the generalized Static Step API: `CudaStaticStep`,
  `StaticBuilder`, `CudaStaticContext`, `StaticRun`. `LinearMseAdamStep`,
  `LinearForwardStep`, `StaticDecoderStep` and `BatchedStaticDecoder` cover
  training, forward, one-token decode and fixed-shape batched prefill/decode
  over fixed buffers with device-resident state.
- Batched static prefill uses cuBLAS strided-batched QK/PV and a warp causal
  softmax; the LM head runs on last tokens only. Batched decode uses
  per-sequence positions and batched GEMM attention. KV storage is F32, F16 or
  BF16 (F16/BF16 upcast to F32 for attention).
- Wide paged attention: block-per-query kernels for single and batched decode.
- Only generic `sum_to` (non-bias) falls back to CPU.
- Verified against CPU for elementwise, matmul, linear, softmax CE, LLM
  kernels, dense decoder logits, paged attention and paged batched decode; a
  40-step Adam loop runs on an RTX 2070 SUPER.

Deferred: variable-length/continuous static batching, native low-precision
attention and flash-attention prefill, block-level in-place COW,
optimizer-state migration, Training-By-Run, Python bindings for Run/static
APIs.

## 5. Autograd semantics

- `Variable::new` is differentiable by default; `Variable::constant` is not.
- A node is created when at least one input requires grad and gradients are
  enabled.
- `backward()` seeds with ones, traverses inputs before outputs, accumulates
  into `grad`, and skips inputs whose `requires_grad` is false.
- Repeated `backward()` accumulates; `cleargrad()` resets.
- Gradients are stored on intermediate variables as well as leaves.
- `no_grad` is thread-local and panic-safe via a guard.
- Higher-order derivatives are not yet exposed, but `Op::backward` composes
  differentiable ops so `create_graph` can be added without changing the ABI.

Known limitation: the Python binding does not expose `FunctionNode`
subclassing; custom ops must currently be written in Rust.

## 6. Backends

### CPU (stable)

- Contiguous row-major f32 tensors; zero-copy `Vec<f32>` storage.
- `gemm` (rayon) for large GEMMs; naive loops below ~1M multiply-adds to avoid
  small-size scheduler overhead and a known `gemm` edge case.
- Rayon-parallel elementwise/reduction kernels; broadcasting with a bias fast
  path.
- Fused `linear` and fused `linear_backward`; fused `adam_update`.
- Numerically stable softmax cross entropy in f32 with f64 accumulation.

### Vulkan (beta)

- `wgpu` with the Vulkan backend on Linux; WGSL kernels compiled at build time
  by naga.
- Kernels: add/sub/mul/div, sqrt, relu, relu backward, scale, fill, bias add,
  transpose, tiled 16×16 matmul.
- Reduction-heavy ops (`sum_all`, `sum_to`, softmax cross entropy) currently
  round-trip through the CPU backend. This is correctness-first and is the
  first target for the next Vulkan milestone.
- Verified on RTX 2070 SUPER and Mesa lavapipe; tests skip when no adapter is
  available.

### CUDA (core ops)

- `cudarc` with dynamic loading; NVRTC runtime compilation; cuBLAS GEMM.
- Native kernels: add/sub/mul/div, sqrt, relu, relu backward, SiLU, scale,
  fill, bias add, transpose, fused Adam.
- Native cuBLAS path: matmul, linear, linear_backward.
- Native attention, paged append/attention/batch, reductions, softmax CE,
  embedding, layer_norm, split/merge heads, concat and last-row.
- Only generic `sum_to` (non-bias) falls back to CPU.
- cuBLASLt fused bias / GELU for `linear`; `ComputeType` F32/TF32/F16/BF16;
  memory pool; CUDA Graph for fixed-shape matmul/Adam and a static Linear
  training step; block-per-query wide paged attention.
- CUDA Graph capture requires preallocated buffers and a non-default stream;
  arbitrary `Run` capture is deferred.
- Tested on RTX 2070 SUPER (sm_75) and compiled for the detected compute
  capability; GTX 1080 (sm_61) is supported by the same mechanism.
- `LD_LIBRARY_PATH` must expose `libnvrtc` / `libcublas` when using pip CUDA
  wheels. See `docs/backends.md`.

## 7. Performance

Acceptance gates proposed in the design phase:

| Workload | Gate |
|---|---|
| tiny-op eager overhead | 2.0× vs PyTorch |
| MNIST MLP train step (CPU) | 1.2× |
| ResNet block fwd+bwd (CUDA) | 1.0× |
| Transformer block fwd+bwd (CUDA) | 1.0× |
| peak memory | ≤ PyTorch |
| geometric mean (CPU/CUDA) | 1.2×, no item below 0.9× |

Current CPU result on the checked-in micro-suite is a ~0.87–0.91× geometric mean
(Adam wins, Linear backward loses). On the RTX 2070 SUPER against PyTorch
2.11.0+cu128:

| Workload | Modern Chainer | PyTorch | torch/ours |
|---|---:|---:|---:|
| matmul eager 512³ | 0.046 ms | 0.045 ms | 0.99× |
| matmul graph 512³ | 0.043 ms | 0.044 ms | 1.03× |
| linear 128×784×256 | 0.021 ms | 0.038 ms | 1.82× |
| linear+gelu 128×784×256 | 0.021 ms | 0.041 ms | 1.91× |
| Adam eager 1M | 0.112 ms | 0.216 ms | 1.93× |
| Adam graph 1M | 0.077 ms | 0.229 ms | 2.99× |

See `docs/benchmarks.md` for methodology and caveats.

## 8. Repository layout

```
crates/
  modern-chainer/          facade, prelude, examples
  modern-chainer-core/     Tensor, Shape, DType, Device, Storage, Backend, Ops
  modern-chainer-autograd/ Variable, Op, Node, gradcheck, functional ops
  modern-chainer-nn/       Module, Linear, Sequential, Parameter
  modern-chainer-optim/    SGD, MomentumSGD, Adam
  modern-chainer-data/     Dataset, VecDataset, BatchIterator
  modern-chainer-serial/   native .mcn format
  modern-chainer-cpu/      CPU kernels
  modern-chainer-vulkan/   wgpu/Vulkan backend and WGSL shaders
  modern-chainer-cuda/     CUDA scaffold
  modern-chainer-run/      StateStore, Run, Step, transactions
  modern-chainer-llm/      dense KV decoder, attention helpers, sampling
  modern-chainer-python/   PyO3 extension
python/                    maturin project and Python package
scripts/                   benchmark harness
docs/                      architecture, backends, API, benchmarks, roadmap
```

## 9. Testing strategy

- Numerical `gradcheck` for every differentiable op, including fused Linear.
- Known-value tests for kernels (Linear, matmul, bias broadcast, Adam update).
- Backend parity: Vulkan vs CPU for elementwise, matmul, transpose, reductions.
- End-to-end training tests that assert loss decreases (SGD/Momentum/Adam,
  CPU and Vulkan).
- Dataset permutation tests, serializer round-trip tests.
- Python tests for import, forward/backward, optimizer updates, Chain and NumPy
  interop.
- State lifecycle: declaration, reuse, shape growth, fork isolation,
  snapshot/restore, transaction commit/rollback/drop.
- KV parity: prefill vs incremental decode logits, position and shape checks,
  max-sequence enforcement.
- Paged attention parity against dense attention for prefill and decode.
- Block allocator reuse and refcount invariants.
- Paged KV append/truncate/fork isolation.
- Batched paged attention parity against per-request dense attention.
- Paged batched decode parity against dense greedy generation.
- Continuous batching equivalence against independent per-request generation.
- Speculative decoding equality against target-only greedy generation, including
  target/draft position synchronisation.
- Generation determinism and sampler range checks.
- CUDA parity: elementwise, matmul, linear, softmax CE, reduction, LLM kernels,
  dense decoder logits, paged attention, paged batched decode and an Adam
  training loop against CPU. Graph replay is checked against eager matmul/Adam.
- F16/BF16 compute types compared against F32 linear/matmul within tolerance;
  a BF16 decoder logit comparison covers the mixed path.
- Autocast decoder parity and an AMP training loop with `GradScaler` on CUDA;
  scaler unscale/growth/backoff and inf/nan skipping are tested on CPU.
- `LinearMseAdamStep` via `StaticRun` drives loss down over 100 steps with the
  device step counter in sync; `LinearForwardStep` replays with changed inputs;
  `StaticDecoderStep` matches a dense CPU decoder token-by-token over a prompt
  and then generates from the replayed graph.
- `BatchedStaticDecoder` runs fixed-shape batched prefill and decode over a
  shared KV cache in separate captured graphs; prefill/decode logits match the
  dense CPU decoder per batch element, including per-sequence positions and
  F16/BF16 caches.
- Memory-pool reuse and zeroed reuse are tested.
- `cargo fmt`, `cargo clippy -D warnings`, `cargo test` in CI.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `gemm` `Rayon(0)` scheduler hang | Explicit thread count via `rayon::current_num_threads()`; stress-tested |
| `gemm` small-shape crash | Naive kernel fallback below 1M MACs |
| Vulkan reduction performance | CPU fallback now; native reduction kernels next |
| cuDNN-free convolution performance | im2col + cuBLASLt, Winograd; optional cuDNN later |
| Python compatibility expectations | Documented subset; no `Trainer` yet |
| GPU absence in CI | Tests skip; lavapipe job in CI |
| State becomes an untyped god-object | Five-category taxonomy, typed `StateSpec`, namespaced keys |
| String lookup in hot loops | `StateId` handles compiled at first step |
| Fork/rollback bugs | Immutable dense values in v0.2; COW property tests in v0.3 |
| Training/inference state semantics diverge | `GradPolicy` and `RunMode` on every state declaration |
| Logical batching hides a throughput gap | `PagedBatchRunner` provides real batched decode; vectorised kernels and throughput gates next |
| Paged block-table bugs | Dense-vs-paged parity tests and bounds checks on every table lookup |
| Batched decode drifts from dense decoding | Token-level parity test against dense greedy generation |
| Tensor-granularity COW costs bandwidth | Correctness first; block-level in-place COW and vectorised kernels in the next CUDA milestone |
| CUDA fallbacks hide a performance cliff | Native/fallback boundary documented per op; CUDA attention/reduction kernels next |
