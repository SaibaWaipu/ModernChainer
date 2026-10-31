# State by Run

Status: v0.5 implementation (dense + paged state, block allocator, batched
decode, continuous batching, greedy speculative decoding; CUDA core backend).

## Motivation

Chainer's define-by-run model made graph construction explicit. The LLM era
adds a second problem: the *execution state* of a model — KV caches, request
tables, block tables, sampler state, recurrent hidden state — is where the
interesting engineering lives, yet most frameworks hide it inside an inference
engine. Training code and serving code then diverge.

State by Run is the corresponding principle for state:

- **Define-by-Run**: build the graph while `forward` executes.
- **State-by-Run**: declare persistent state while `step` executes, but let the
  `Run` own it.
- **Training-By-Run**: the same `step` carried by optimizer, RNG, data cursor
  and (optionally) differentiable state.
- **Inference-By-Run**: the same `step` carried by KV cache, request state and
  sampling state.

The model is a Mealy machine:

```
step: (params, input, state) -> (output, state')
```

The graph is transient. Persistent state is first-class, introspectable,
forkable and rollback-able. Nothing that survives a step is allowed to hide.

## State taxonomy

| Category | Examples | Where it lives |
|---|---|---|
| Graph / activations | intermediate tensors, backward graph | autograd graph (transient) |
| Model persistent state | KV cache, RNN hidden state, memory | `RunState` |
| Optimization state | parameters, grads, Adam m/v, EMA, scaler | `RunState` (v0.3) |
| Execution state | request table, batch plan, block table, position, sampler RNG | `RunState` / typed run fields |
| Environment state | data cursor, metrics | `Run` / `TrainingRun` |

Activations are not put in the state store. They belong to the define-by-run
graph.

## Core API

```rust
pub trait Step<B: Backend>: Send + Sync {
    type Input;
    type Output;
    type Handles: Clone + Send + Sync;

    fn declare_state(&self, builder: &mut StateBuilder<'_, B>)
        -> Result<Self::Handles>;

    fn step(
        &self,
        ctx: &mut RunContext<'_, B>,
        handles: &Self::Handles,
        input: Self::Input,
    ) -> Result<Self::Output>;
}
```

```rust
let mut run = Run::new(model, backend, RunMode::Inference, seed)?;
let logits = run.step(vec![1, 2, 3, 4])?;

for (key, id) in run.plan().iter() {
    println!("{key} -> {id}");
}
let k = run.state().tensor(run.handles().kv.layers[0].k)?;
println!("kv shape = {:?}", k.shape());
```

- `StateKey` is a namespaced identity (`layers.0.kv.k`).
- `StateId` is a generational arena handle.
- `StateSpec` declares layout, dtype, grad policy, fork policy, persistence.
- `StateStore` owns `StateValue`: tensor, variable, scalar, or paged (reserved).
- `StatePlan` is the sorted introspection view; hot paths use the IDs held in
  `Handles`, so there is no string lookup per op.

State is declared on the first step and reused afterwards. This keeps the
dynamic feel of define-by-run while making ownership explicit. Shapes may grow
(KV caches do); `set_tensor` replaces the value and revalidates dtype.

## Transactions

```rust
let mut draft = run.fork();          // O(#slots), immutable dense values share storage
draft.step(vec![next_token])?;       // parent is untouched
run.commit_from(&draft, &[kv_ids])?; // v0.3 helper for accepted prefixes

let mut tx = run.transaction();      // snapshot + auto-rollback on drop
tx.run_mut().step(vec![token])?;
tx.commit();                         // or tx.rollback()
```

Dense state uses `Arc`-backed immutable tensors, so `fork` is cheap and the
parent cannot observe child mutation. Paged state will use block refcounts and
copy-on-write without changing the contract.

## KV cache as explicit state

`modern-chainer-llm` declares one dense `k` and `v` slot per layer plus a
position scalar:

```
layers.{i}.kv.k  [heads, 0, head_dim]  (grows along seq)
layers.{i}.kv.v  [heads, 0, head_dim]
kv.position      i64
```

Prefill and decode share one path: `embedding -> layer_norm -> q/k/v ->
split_heads -> concat_seq(old_kv, new_kv) -> attention -> merge_heads ->
residual -> MLP`. The difference is only `seq` (prompt length vs 1).

```rust
let mut run = Run::new(decoder, CpuBackend::new(), RunMode::Inference, 7)?;
let logits = run.step(prompt.clone())?;     // prefill
let token = sample_token(&logits, &Sampler::Greedy, run.rng_mut())?;
let logits = run.step(vec![token])?;         // decode
```

The acceptance test is parity: prefill logits must equal token-by-token decode
logits for the same prompt. See `crates/modern-chainer-llm/tests/decoder.rs`.

## Paged state and paged attention

`StateLayout::Paged { block_size, num_blocks, trailing }` allocates:

```
blocks       [num_blocks, block_size, heads, head_dim]
block_table  [num_blocks]  i64  (logical block -> physical block)
len          logical token count
```

`Ops::paged_attention` takes `q [heads, q_seq, head_dim]`, K/V blocks, the block
table and `seq_len`, and performs the same causal attention as the dense path
without materialising a gathered KV tensor. The CPU reference matches dense
attention for both prefill and decode, including a shuffled block table.

v0.4 adds `BlockAllocator` (free list + refcounts) and `PagedKvCache`
(`append`, `truncate`, `fork`, `from_dense_with_capacity`). Block storage is
still immutable: appends copy the block tensors, so a fork is isolated at
tensor granularity. `Ops::paged_append` writes new K/V through the block table.
Block-level in-place COW is the next optimisation, not a semantic change.

`Ops::paged_attention_batch` takes per-request block tables and sequence lengths
and runs one kernel for the whole decode batch. `PagedBatchRunner` prefills each
request on the dense `Run` path, converts the KV into paged caches, and then
performs one batched decode step for all active requests. Its token output is
tested against dense greedy generation.

## Continuous batching

```rust
let mut batcher = ContinuousBatcher::new(decoder, backend, seed);
let id = batcher.add_request(prompt, max_new_tokens, Sampler::Greedy)?;
let output = batcher.step()?;    // BatchPlan { prefill, decode }
```

- `RequestTable` is the batcher's request vector; each request owns its own
  `Run` and therefore its own KV cache and RNG.
- `BatchPlan` separates newly added requests (`prefill`) from ongoing ones
  (`decode`). Requests can be added or removed between steps.
- `ContinuousBatcher` executes a plan sequentially; `PagedBatchRunner` is the
  real batched decode path. It batches every active request into one paged
  attention call after converting prefilled KV into blocks.
- Acceptance tests cover both logical batching equivalence and paged batched
  decode parity with dense greedy generation.

## Speculative decoding

```rust
let tokens = speculative_generate(&mut target, &mut draft, prompt, max_new, 3)?;
```

The greedy reference:

1. Draft proposes `k` tokens from a snapshot of the draft state.
2. A `target.fork()` verifies the proposals without touching the real target.
3. The longest matching prefix is accepted; on a mismatch the target's own
   choice is accepted for that position and the round ends.
4. Only accepted tokens are committed to the target.
5. The draft is restored from its snapshot and stepped with the accepted tokens,
   keeping the two KV states aligned.

Because the target decides every token, the output is exactly target-only greedy
decoding. The tests assert this equality and that target/draft positions match.

## Mapping LLM features onto state

| Feature | State-by-Run representation |
|---|---|
| KV cache | per-layer dense `k`/`v` slots + position |
| PagedAttention | `StateLayout::Paged`, block table, COW blocks (v0.3) |
| Continuous batching | `Scheduler` produces a `BatchPlan`; `Run` executes it |
| Speculative decoding | `run.fork()` → draft decode → verify → commit/rollback |
| Prefix caching | fork + shared immutable blocks (COW) |
| Beam search | multiple forked runs + score state |
| Truncated BPTT | `GradPolicy::Differentiable` state + detach at boundaries (v0.3) |

The scheduler stays outside the model. Continuous batching is an execution
policy, not a model property.

## Invariants

1. **No hidden state.** Anything that survives a step is reachable from the
   `Run` (or from a typed run field such as a request table).
2. **Stateless path is zero-cost.** `Module::forward` and the existing autograd
   path are unchanged.
3. **No string lookup in the hot path.** `StateSpec` is resolved once into
   `StateId`s held by `Handles`.
4. **Fork isolation.** A child run cannot mutate the parent's dense state.
5. **Rollback correctness.** A transaction restores state, RNG, step counter and
   mode.
6. **Determinism.** RNG is part of the run, not a global.
7. **Introspection.** Key, layout, dtype, shape, byte size and plan are public.

## v0.2 scope

Implemented:

- `modern-chainer-run`: `StateStore`, `StateSpec`, `StatePlan`, `Run`, `Step`,
  `RunContext`, `fork`, snapshot/restore, transactions, metrics.
- `modern-chainer-llm`: tiny 2-layer decoder, dense KV cache, prefill/decode,
  greedy/temperature/top-k sampling.
- New CPU kernels: embedding, layer_norm, softmax, SiLU, split/merge heads,
  dense causal attention, sequence concat, last-row extraction.
- Tests: state lifecycle, fork isolation, rollback, prefill vs incremental
  decode parity, generation determinism, max sequence enforcement.

Not in v0.2:

- PagedAttention and block allocators (types reserved).
- Continuous batching scheduler.
- Speculative decoding driver (fork/rollback primitives are tested, the
  verification algorithm is next).
- CUDA attention/reduction kernels (CUDA core ops are implemented; the rest
  falls back to CPU).
- Training-By-Run, optimizer-state migration, checkpoints.
- Python `Run` binding.

## Next

1. Paged state + block allocator + COW + native paged attention (CPU reference).
2. Continuous batching (`RequestTable`, `BatchPlan`, scheduler).
3. Speculative decoding on top of fork/verify/commit.
4. CUDA paged-attention kernel and throughput benchmarks.
5. Training-By-Run: move optimizer state into `StateStore`, truncated BPTT,
   checkpoint/resume, RL rollout forks.
6. Python API: `chainer.Run`, `run.state`, `run.prefill/decode`, and a
   `Trainer` built on the same step contract.
