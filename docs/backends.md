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

## CUDA (planned)

The existing `modern-chainer-cuda` crate only probes availability. The intended
implementation:

1. **Runtime**: [`cudarc`](https://crates.io/crates/cudarc) with dynamic
   loading, so builds do not require a CUDA toolkit.
2. **Kernels**: NVRTC compiles CUDA C at runtime; no `nvcc` in the build.
3. **GEMM**: cuBLAS for general matmul, cuBLASLt for fused
   GEMM+bias+activation.
4. **Convolution**: im2col + GEMM first, Winograd for 3×3 later. cuDNN is
   optional, not required.
5. **Memory**: pooling allocator, multiple streams, CUDA Graph capture for
   static-shape training loops.
6. **Precision**: fp32 first, then fp16/bf16 autocast with loss scaling.

Driver/runtime notes: the development machine has an NVIDIA driver but no
CUDA runtime or headers. CUDA 12.x is the target because CUDA 13 drops Pascal
(sm_61, GTX 1080); the RTX 2070 SUPER (sm_75) works with either.

## Adding a backend

1. Implement `Storage` for the allocation handle.
2. Implement `Backend` (`alloc`, copies, device, `Ops` type).
3. Implement the `Ops` methods that matter, leaving the rest to the defaults.
4. Add parity tests against the CPU backend using `Tensor::to_bytes`.
5. Add a fused `linear`/`linear_backward`/`adam_update` when the naive
   composition is a bottleneck.
