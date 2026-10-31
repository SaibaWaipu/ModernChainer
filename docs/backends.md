# Backends

## Device selection

```rust
use modern_chainer::{Device, vulkan, cuda};

let cpu = Device::Cpu;
let gpu = Device::Vulkan(0);
let cuda = Device::Cuda(0);

println!("{:?}", vulkan::probe());
println!("{:?}", cuda::probe());
```

`Device` parsing accepts `cpu`, `cuda:0`, `vulkan:0` and the aliases `gpu`/`vk`.
The Python surface exposes `chainer.cuda_available()` and
`chainer.vulkan_available()`; tensor execution in Python is CPU-only in
v0.1.0.

## CPU

The CPU backend is the reference implementation.

- **Storage**: contiguous row-major buffers. `CpuStorage::F32` keeps kernel
  output as `Vec<f32>` so no byte copy is needed; `Bytes` handles uploads.
- **GEMM**: [`gemm`](https://crates.io/crates/gemm) with rayon. The thread count
  is passed explicitly (`rayon::current_num_threads()`); `Parallelism::Rayon(0)`
  was found to hang under load and is avoided. Below roughly 1M multiply-adds a
  naive kernel is used: it avoids scheduler overhead and a small-shape edge
  case in `gemm`.
- **Elementwise**: rayon parallel maps with a fast path for equal shapes and
  for a 2-D tensor plus trailing 1-D bias.
- **Fused kernels**: `linear` (x·Wᵀ+b), `linear_backward` (two GEMMs without
  materialising Wᵀ), `adam_update` (one parallel pass over p/g/m/v using
  uninitialized output capacity).
- **Softmax cross entropy**: stable max/log-sum-exp, f64 accumulation, fused
  backward `(softmax - onehot)/batch`.
- **Threads**: `RAYON_NUM_THREADS` is honoured. `MC_GEMM_THREADS=0` forces
  single-threaded GEMM for debugging; `MC_GEMM_THREADS=n` pins the GEMM pool.

`lto = "thin"` and `codegen-units = 1` are enabled for release and bench
profiles.

## Vulkan

The Vulkan backend is implemented with `wgpu` configured for the Vulkan backend.
This gives a portable, memory-safe Vulkan path (NVIDIA, AMD, Intel, Mesa
lavapipe) without hand-written `unsafe` Vulkan plumbing. WGSL shaders are
embedded and translated to SPIR-V by naga at pipeline creation.

### Implemented kernels

| Kernel | Notes |
|---|---|
| add / sub / mul / div | equal shapes; bias add for `[rows, cols] + [cols]` |
| sqrt, relu, relu backward, scale, fill | elementwise |
| transpose2d | linear index remap |
| matmul | 16×16 tiled, shared memory, 2-D dispatch beyond 65535 workgroups |

### Host fallbacks (correctness-first)

`sum_all`, `sum_to`, `softmax_cross_entropy` and its backward currently copy to
the CPU backend and upload the result. This keeps the full training path usable
(matrix multiplication, bias, ReLU, transpose and optimizer arithmetic all stay
on the GPU) while the native reduction kernels are developed. Expect PCIe
traffic and CPU-side reduction time in those ops.

### Dispatch and synchronization

- Workgroups are 64 threads for elementwise kernels; 2-D dispatch lifts the
  65535 limit of a single dimension.
- Each op creates a command encoder, records one compute pass (or a copy) and
  submits. There is no async pipelining yet.
- Readback uses a staging buffer, `map_async`, and `PollType::wait_indefinitely`.
- Buffers are `STORAGE | COPY_SRC | COPY_DST`; sizes are rounded to 4 bytes.

### Verification

```bash
cargo test -p modern-chainer-vulkan
```

Tests compare elementwise, matmul, transpose, bias add and reductions against
CPU, then run a 100-step Linear training loop and assert the loss decreases.
They skip with a message when `probe()` reports no adapter. A CI job installs
Mesa's lavapipe and runs the same suite headlessly.

## CUDA (core ops)

The CUDA backend is implemented with [`cudarc`](https://crates.io/crates/cudarc)
and dynamic library loading, so a build does not require the CUDA toolkit or
`nvcc`.

### Runtime

- Device selection through the CUDA driver API (`CudaContext::new(ordinal)`).
- One default stream and one cuBLAS handle per backend.
- NVRTC compiles a single kernel module at runtime using the detected compute
  capability (`compute_61` for GTX 1080, `compute_75` for RTX 2070 SUPER).
- Typed storage: `CudaStorage::{F32, I64, U8}`. `Backend::alloc_for` allocates
  the dtype the tensor needs.

### Native ops

| Op | Implementation |
|---|---|
| add / sub / mul / div | NVRTC elementwise kernels (equal shapes) |
| sqrt, relu, relu backward, SiLU, scale, fill | NVRTC kernels |
| one/zero/full like | fill kernel |
| bias add | row-bias kernel |
| transpose2d | NVRTC kernel |
| matmul | cuBLAS sgemm (row-major via the standard swap) |
| linear | ComputeType F32 (cuBLAS+bias), TF32 (cuBLASLt bias/GELU), F16/BF16 (mixed cuBLASLt) |
| linear_backward | cuBLAS + transpose |
| adam_update | fused NVRTC kernel |
| sum_all / 2-D bias sum_to | NVRTC reductions |
| softmax CE / backward | NVRTC row kernels |
| embedding / layer_norm / softmax | NVRTC row kernels |
| split/merge heads / concat / last-row | NVRTC index kernels |
| dense attention | NVRTC per-query kernel |
| wide paged attention | block-per-query kernels for single and batched decode |
| paged append / attention / batched attention | NVRTC block-table kernels |
| to_scalar | device-to-host readback |
| graph matmul / graph Adam | CUDA Graph capture/replay over preallocated buffers |

### CPU fallbacks

Only generic `sum_to` (shapes other than a 2-D tensor reduced to its trailing
1-D bias) copies to the CPU backend. Everything on the training and decoder
paths — including dense and paged attention — is native.

### Memory pool and compute types

`CudaPool` caches f32/i64/u8/f16/bf16 `CudaSlice` allocations and returns them
on drop. Exact-size matching keeps storage lengths consistent; `alloc_zeroed_for`
memsets reused buffers. The pool removes most `cudaMalloc`/`cudaFree` traffic
from the hot path and is the prerequisite for future graph capture of dynamic
steps.

`CudaBackend::set_compute_type` selects:

| Compute type | Path |
|---|---|
| F32 | cuBLAS GEMM + bias kernel (strict f32) |
| TF32 | cuBLASLt (FAST_TF32; hardware dependent) |
| F16 | cast x/w/bias to f16, cuBLASLt f16 GEMM, cast output to f32 |
| BF16 | same with bf16 |

### AMP (autocast + GradScaler)

`autocast(F16 | BF16, || ...)` is a thread-local policy. CUDA lowers
`linear`, `linear_activation` and `matmul` to mixed f16/bf16 cuBLASLt
(tensor-core friendly on supported hardware); layer_norm, softmax, attention,
reductions and the loss stay f32.

`GradScaler` scales the loss before backward, unscales gradients before the
optimizer step, skips the step on inf/nan, and grows/backs off the scale. The
current policy is GEMM-focused mixed precision; a full-model f16/bf16 dtype is
not implemented.

### CUDA Graph

`CudaBackend::capture` begins stream capture on the backend's non-default
stream, runs a closure that only enqueues work, and returns a
`CudaGraphRunner`.  `graph_matmul` and `graph_adam` preallocate their outputs
and capture a cuBLAS GEMM / fused Adam kernel respectively.

The **Static Step API** generalizes run-level graphs:

- `CudaStaticStep` declares fixed buffers and enqueues one in-place step.
- `StaticBuilder` allocates the buffers (`Tensor<CudaBackend>`, device scalars).
- `CudaStaticContext` provides allocation-free in-place primitives: linear,
  grad_w, MSE, grad reductions, Adam, step bump, host input/output.
- `StaticRun` owns the buffers, captures the step once and replays it with a
  new input.

`LinearMseAdamStep` (Linear + MSE + Adam), `LinearForwardStep`,
`StaticDecoderStep`, and `BatchedStaticDecoder` are included.  The single
static decoder keeps weights, per-layer K/V caches, activations and a position
counter in fixed buffers; each replay appends one token at the device position,
runs causal attention over the cache, and advances the position inside the
graph.  It matches a dense CPU decoder token-by-token.

`BatchedStaticDecoder` captures two graphs over one shared KV cache:

- **Prefill** processes `[batch, prefill_len]` tokens. Embedding, LayerNorm and
  Q/K/V projections run as row-batched GEMMs. K/V rows are written to the cache,
  Q is split head-major, and attention uses cuBLAS strided-batched GEMMs for
  QK and PV with a warp-per-row causal softmax. Only the last token per batch
  goes through the final LayerNorm and LM head. The graph sets the device
  positions to `prefill_len`.
- **Decode** processes one token per batch element at its own position. A
  batched append kernel writes each sequence's K/V, batched GEMM attention
  computes scores for `0..=position` and masks longer positions per sequence,
  and the graph increments every position. Positions can be changed with
  `set_positions`.

KV cache storage is selectable (`KvDType::F32`, `F16`, `Bf16`); weights and
activations stay F32. F16/BF16 caches are upcast into an F32 workspace for the
prefill/decode GEMMs. Both graphs are replay-only: buffers never reallocate.
Batched prefill and decode logits match the dense CPU decoder per batch element,
including low-precision caches. The current limitation is fixed batch/prompt
length and no variable-length/continuous admission; native low-precision
attention and flash-attention prefill are future work.

The step/position counters are device scalars advanced by kernels inside the
graph, so replay performs a complete step without host arguments.
`GraphLinearTrainer` remains as a convenience wrapper-style API for the
Linear+MSE case.

CUDA Graph requires preallocated buffers: allocation and synchronization inside
the closure are not capturable.  Event tracking is disabled on the backend
context because the backend uses a single stream; this also keeps capture clean.
A general `Run` / stateful decoder graph requires a static-step and in-place
state API; that is the next milestone.

### Running

Using pip CUDA wheels:

```bash
export LD_LIBRARY_PATH="$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
cargo test -p modern-chainer-cuda
cargo run --release --example cuda_matmul
cargo run --release --example cuda_bench
# PyTorch CUDA comparison
CUDA_VISIBLE_DEVICES=<uuid> /tmp/opencode/torch-venv/bin/python scripts/bench_cuda_vs_pytorch.py
```

Tests skip when no CUDA driver/runtime is available.

### Next

- Full-model f16/bf16 dtype (all ops) and master-weight optimizer state.
- Variable-length / continuous static batching and request admission.
- Native low-precision attention and fused flash-attention prefill.
- Backend-agnostic Static Step trait (currently CUDA-specific).
- Further paged-attention/flash-decode tuning and TTFT/tokens/s benchmarks.
- Run-level CUDA Graph capture with a memory pool.
- TTFT/tokens/s and larger transformer benchmarks.

## Adding a backend

1. Implement `Storage` for the allocation handle.
2. Implement `Backend` (`alloc`, copies, device, `Ops` type).
3. Implement the `Ops` methods that matter, leaving the rest to the defaults.
4. Add parity tests against the CPU backend using `Tensor::to_bytes`.
5. Add a fused `linear`/`linear_backward`/`adam_update` when the naive
   composition is a bottleneck.
