# Modern Chainer — Specification

Status: v0.1.0 implementation. This document records the product intent, the
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
- Backend probes: `cuda::probe`, `vulkan::probe`

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

Not yet implemented in Python: `Trainer`, `Updater`, `Iterator` wrappers,
`Extension`, `chainer.config`, `FunctionNode` subclassing, `.npz` serialization,
`chainer.cuda`/`chainer.backends`.

## 4. Autograd semantics

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

## 5. Backends

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

### CUDA (scaffold)

- Capability probe only; the planned implementation uses `cudarc` with NVRTC
  (no `nvcc` at build time), cuBLAS/cuBLASLt for GEMM, and im2col/Winograd for
  convolution. cuDNN is not a requirement. See `docs/backends.md`.

## 6. Performance

Acceptance gates proposed in the design phase:

| Workload | Gate |
|---|---|
| tiny-op eager overhead | 2.0× vs PyTorch |
| MNIST MLP train step (CPU) | 1.2× |
| ResNet block fwd+bwd (CUDA) | 1.0× |
| Transformer block fwd+bwd (CUDA) | 1.0× |
| peak memory | ≤ PyTorch |
| geometric mean (CPU/CUDA) | 1.2×, no item below 0.9× |

Current CPU result on the checked-in micro-suite is a ~0.87× geometric mean
(Adam wins, Linear backward loses). CUDA and transformer/ResNet benchmarks are
blocked on the CUDA backend. See `docs/benchmarks.md` for the full table and
caveats.

## 7. Repository layout

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
  modern-chainer-python/   PyO3 extension
python/                    maturin project and Python package
scripts/                   benchmark harness
docs/                      architecture, backends, API, benchmarks, roadmap
```

## 8. Testing strategy

- Numerical `gradcheck` for every differentiable op, including fused Linear.
- Known-value tests for kernels (Linear, matmul, bias broadcast, Adam update).
- Backend parity: Vulkan vs CPU for elementwise, matmul, transpose, reductions.
- End-to-end training tests that assert loss decreases (SGD/Momentum/Adam,
  CPU and Vulkan).
- Dataset permutation tests, serializer round-trip tests.
- Python tests for import, forward/backward, optimizer updates, Chain and NumPy
  interop.
- `cargo fmt`, `cargo clippy -D warnings`, `cargo test` in CI.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `gemm` `Rayon(0)` scheduler hang | Explicit thread count via `rayon::current_num_threads()`; stress-tested |
| `gemm` small-shape crash | Naive kernel fallback below 1M MACs |
| Vulkan reduction performance | CPU fallback now; native reduction kernels next |
| cuDNN-free convolution performance | im2col + cuBLASLt, Winograd; optional cuDNN later |
| Python compatibility expectations | Documented subset; no `Trainer` yet |
| GPU absence in CI | Tests skip; lavapipe job in CI |
