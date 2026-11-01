# Roadmap

State by Run reorders the original roadmap: explicit state and the step contract
come before backends.

## v0.2 — State core and dense KV decoder (done)

- `modern-chainer-run`: `StateStore`, `StateSpec`, `StatePlan`, `Run`, `Step`,
  `RunContext`, `RunMode`, fork, snapshot/restore, transactions, metrics.
- `modern-chainer-llm`: 2-layer reference decoder, dense per-layer KV cache,
  prefill/decode, greedy/temperature/top-k sampling.
- CPU kernels: embedding, layer_norm, softmax, SiLU, split/merge heads, causal
  attention, sequence concat, last-row extraction.
- Tests: fork isolation, rollback, prefill vs incremental decode parity,
  generation determinism, state visibility.

## v0.3 — Paged primitives, batching, speculation (done, reference)

- `StateLayout::Paged` allocation with a block table.
- CPU paged attention matching dense attention for prefill and decode.
- `RequestTable` / `BatchPlan` / `ContinuousBatcher`; requests can be added
  between steps. Execution is per-request in v0.3.
- Greedy speculative decoding with `fork` / `snapshot` / verify / commit;
  output equals target-only greedy decoding.
- Tests: paged-vs-dense parity, batching equivalence, speculative equality,
  position synchronisation.

Remaining for v0.4: block allocator with refcounts and block-level COW,
vectorised paged attention, vectorised batched step.

## v0.4 — Paged allocator and batched decode (done, reference)

- `BlockAllocator` with free list and refcounts.
- `PagedKvCache` append/truncate/fork with tensor-granularity COW.
- `Ops::paged_append` and batched paged attention.
- `PagedBatchRunner`: dense prefill converted to paged KV, then one batched
  decode step over all active requests.
- Tests: allocator reuse, KV append/truncate, paged-vs-dense attention,
  batched decode vs dense greedy generation.

Remaining: block-level in-place COW, wider kernels, CUDA.

## v0.5 — CUDA backend (done)

- `cudarc` dynamic loading, NVRTC kernels, cuBLAS.
- Native elementwise, reductions, matmul, linear/linear_backward, transpose,
  fused Adam, softmax CE, embedding, layer_norm, dense attention and paged
  attention/append/batched attention.
- Only generic `sum_to` falls back to CPU.
- CPU-parity tests for decoder logits, LLM kernels and paged batched decode;
  CUDA Adam training loop and `cuda_bench` on RTX 2070 SUPER.

## v0.6 — CUDA throughput (in progress)

- Done: native reductions / softmax CE / layer_norm / embedding,
  dense and paged attention, cuBLASLt fused linear, CUDA Graph matmul/Adam,
  PyTorch CUDA comparison.
- Done: fused bias+GELU linear for the decoder MLP, wide block-per-query
  paged attention, CUDA memory pool, F32/TF32/F16/BF16 compute types,
  autocast AMP + GradScaler.
- Done: generalized Static Step API (`CudaStaticStep` / `StaticBuilder` /
  `CudaStaticContext` / `StaticRun`), Linear+MSE+Adam, Linear forward,
  `StaticDecoderStep`, and `BatchedStaticDecoder` with captured batched
  prefill/decode over a shared KV cache.
- Done: per-sequence positions, F16/BF16 KV caches, variable-length static
  prefill, native low-precision decode attention, fused flash-attention prefill,
  last-token-only LM head, and TTFT/tokens/s benchmarks against PyTorch eager.
- Done: Python device execution for CPU/CUDA/Vulkan, Vulkan adapter selection,
  pipeline cache, a unified Python benchmark against PyTorch, fused Adam,
  fused softmax CE / GELU / SiLU, native generic `sum_to` / `sum_all`,
  vectorized elementwise, 32×32 and small-shape matmul, transpose-free
  linear/backward, fused bias gradients, TN-small matmul, local command
  batching, GPU-side scalar broadcast, and storage/uniform buffer pools.
- Done: image / GAN / VAE / VQ-VAE / audio / Transformer examples with synthetic
  data, plus CPU autograd for embedding, LayerNorm, softmax, exp/log/sigmoid,
  and image/audio utilities.
- Next: CPU-side bind-group/autograd overhead, multi-tile cooperative-matrix
  scheduling (the 16×16 single-tile
  prototype was slower than scalar on Turing), faster scalar matmul,
  continuous static batching/request admission, faster long-context
  low-precision decode, backend-agnostic Static Step trait, full-model f16/bf16
  dtype.

## v0.7 — Training-By-Run

- Move optimizer state into `StateStore` namespaces (`optim.*`).
- `TrainingRun`: params, grads, optimizer, RNG, data cursor, metrics.
- Checkpoint/resume from `RunSnapshot`; `PersistPolicy` drives what is saved.
- Truncated BPTT with `GradPolicy::Differentiable` state.
- RL rollouts via `fork` + `no_grad`.

## v0.8 — Python and ecosystem

- Python `chainer.Run`, `run.state` (read-only views), `run.prefill/decode`,
  `run.fork`, and a `Trainer` built on the same `Step` contract.
- `Trainer` / `Updater` / `Iterator` / `Extension` compatibility layer.
- Chainer `.npz` reader/writer, safetensors, ONNX export.
- Vision/NLP examples and a docs site.

## Explicitly deferred

- ChainerX native C++ backend.
- Distributed training / ChainerMN equivalent.
- Full ChainerCV/ChainerRL ports.
- Windows/macOS GPU backends (wgpu keeps the door open).
