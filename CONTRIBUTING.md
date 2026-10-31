# Contributing

## Setup

```bash
cargo build --workspace
cd python
uv venv
uv pip install maturin numpy pytest
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
```

## Checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude modern-chainer-python
cargo test -p modern-chainer-vulkan           # skips without an adapter
cargo test -p modern-chainer-cuda             # skips without CUDA; see below
cd python && .venv/bin/pytest
```

## Adding a differentiable op

1. Implement an `Op<B>` in `modern-chainer-autograd/src/ops.rs` with `forward`
   and `backward`. Backward must compose other `Ops` (not raw kernels) so
   higher-order gradients remain possible.
2. Add a public function that calls `apply` / `apply1`.
3. Add a case to `crates/modern-chainer-autograd/tests/gradcheck.rs`.
4. If the op is on the training hot path, add a fused kernel to `Ops` with a
   default composition and override it in the CPU (and GPU) backend.
5. Expose it through `modern-chainer/src/lib.rs` (`F` / `L`).

### CUDA tests

The CUDA backend loads NVRTC and cuBLAS dynamically. With pip CUDA wheels:

```bash
export LD_LIBRARY_PATH="$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
cargo test -p modern-chainer-cuda
```

Tests skip when the driver or runtime libraries are unavailable.

## Adding stateful models

- Implement `Step<B>` and declare every persistent buffer in `declare_state`.
  The model must not keep a cache in its own fields across steps.
- Return typed handles (`StateId`s); do not look state up by string in `step`.
- Use `ctx.state()` for read-only introspection and `ctx.set_*` for explicit
  updates.
- Add tests for fork isolation, transaction rollback, and prefill-vs-incremental
  parity where applicable.
- Mark differentiable state with `GradPolicy::Differentiable`; otherwise keep it
  detached.

## Adding a backend

See `docs/backends.md`. Keep the CPU backend as the reference and add parity
tests that compare values through `Tensor::to_bytes`.

## Performance work

- Use `cargo bench -p modern-chainer --bench cpu_ops` for the Rust core.
- Use `scripts/bench_vs_pytorch.py` for end-to-end comparisons.
- Record both the absolute time and the ratio; machine noise on shared hosts is
  significant, so quote a range from repeated runs.
- Never trade correctness for a benchmark: run gradcheck and backend parity
  tests first.

## Commit style

Small, focused commits with an imperative subject, for example:

```
core: add zero-copy f32 storage
cpu: fuse linear forward and backward
vulkan: add tiled matmul kernel
python: expose div/sqrt/transpose
docs: record CPU benchmark methodology
```
