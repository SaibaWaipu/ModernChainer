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

## v0.3 — Paged state, batching, speculation

- `StateLayout::Paged` allocation: block allocator, block table, refcounts,
  copy-on-write.
- Native paged attention (CPU reference first), block gather/append.
- `RequestTable`, `BatchPlan`, scheduler trait, continuous batching.
- Speculative decoding driver: draft fork, verify, commit accepted prefix,
  rollback the rest.
- Property tests for block refcounts, batch equivalence (batched vs sequential),
  and speculative acceptance/rollback.

## v0.4 — CUDA

- `cudarc` dynamic loading, NVRTC compute kernels, cuBLAS/cuBLASLt.
- Paged-attention / flash-decode kernels.
- CUDA Graph capture for static-shape training loops.
- Benchmarks: TTFT, inter-token latency, tokens/s, block utilisation.

## v0.5 — Training-By-Run

- Move optimizer state into `StateStore` namespaces (`optim.*`).
- `TrainingRun`: params, grads, optimizer, RNG, data cursor, metrics.
- Checkpoint/resume from `RunSnapshot`; `PersistPolicy` drives what is saved.
- Truncated BPTT with `GradPolicy::Differentiable` state.
- RL rollouts via `fork` + `no_grad`.

## v0.6 — Python and ecosystem

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
