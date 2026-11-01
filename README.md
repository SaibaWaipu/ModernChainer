# Modern Chainer

**日本語** | [English](README.en.md)

Define-by-Run と State-by-Run を備えた、Rust コアの深層学習フレームワーク。
[Chainer](https://github.com/chainer/chainer) の思想を現代に復興させるプロジェクトです。

Chainer が広めた **Define-by-Run** は「forward を実行しながら計算グラフを構築する」モデルです。Modern Chainer はその感覚と語彙（`Variable`、`F.*`、`L.*`、`Chain`、`Optimizer`）を保ちながら、エンジンを Rust に移しました。同じコアを Rust と Python の両方から使い、CPU と GPU バックエンドを選べます。

第二の原則が **State by Run** です。KV キャッシュ、RNN の隠れ状態、サンプラ RNG、スケジューラが見るバッファといった「ステップを跨いで残る状態」を、モジュールの中に隠さず `Run` が明示的に所有します。状態は step の実行中に宣言されますが、所有・可視化・fork・rollback は Run の責務です。

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

## ステータス（v0.18.0）

| 領域 | 状態 |
|---|---|
| Define-by-Run autograd（Rust） | 安定 |
| State by Run（`Run` / `Step` / `StateStore`） | 安定: fork、snapshot/rollback、transaction |
| Dense KV decoder | 動作: 2層リファレンス、prefill/decode、sampling |
| Paged state + CPU paged attention | 動作: block allocator、block table、append/truncate、COW |
| Batched decode | 動作: `PagedBatchRunner` + batched paged attention |
| Continuous batching | 動作: `RequestTable` / `BatchPlan`（論理バッチ） |
| Speculative decoding | 動作: greedy の fork/verify/commit 参照実装 |
| CUDA バックエンド | 動作: NVRTC + cuBLAS / cuBLASLt、F32/TF32/F16/BF16 compute type、AMP、メモリプール、CUDA Graph、wide paged attention |
| Static Step API / Run-level CUDA Graph | 動作: `CudaStaticStep` + `StaticRun`。`LinearMseAdamStep` / `LinearForwardStep` / `StaticDecoderStep` を同梱 |
| Batched static prefill / decode | 動作: `BatchedStaticDecoder`。可変長 prompt、per-sequence position、F16/BF16 KV、native low-precision decode attention、flash prefill、CUDA Graph replay |
| Python バインディング | CPU / CUDA / Vulkan: `Tensor`、`Variable`、`F.*`、`L.Linear(device=...)`、`SGD`/`Adam`、`Chain`。`Tensor.to(device)`、`chainer.synchronize()` |
| シリアライズ | ネイティブ `.mcn`、Chainer `.npz` は計画中 |
| Trainer / Updater / Extensions | 計画中 |
| ドキュメント | README（日本語）、SPEC / docs（英語） |

Python パッケージは Chainer 7 の完全互換ではありません。API の形と Define-by-Run の感触を残しつつ、現代の Python に合わない部分は整理しています。State by Run は v0.4 時点では Rust ファーストです。

## クイックスタート — Rust

```bash
cargo run --release --example mlp_regression
cargo run --release --example state_decoder
cargo run --release --example continuous_batching
cargo run --release --example paged_batching
cargo run --release --example speculative
cargo run --release --example cuda_matmul     # CUDA ドライバ + NVRTC/cuBLAS が必要
cargo run --release --example cuda_bench
cargo run --release --example graph_trainer
cargo run --release --example static_step
cargo run --release --example static_decoder
cargo run --release --example static_batched_decoder
cargo run --release --example vulkan_matmul   # Vulkan ローダ/ICD が必要
```

Python から device を選択:

```python
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

print(chainer.vulkan_devices())          # ['... GTX 1080 ...', '... RTX 2070 SUPER ...', ...]
layer = L.Linear(4, 2, device="vulkan:1")
x = chainer.Variable(chainer.Tensor.from_numpy(x_np, device="vulkan:1"))
loss = F.sum(layer(x))
loss.backward()
optimizer = O.Adam(0.001)
optimizer.setup(layer.parameters())
optimizer.update()
```

```rust
use modern_chainer::prelude::*;

let model = Linear::new(4, 2, 0)?;
let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &[1.0; 8], [2, 4])?);
let y = model.forward(&x)?;
let loss = F::sum(&y)?;
loss.backward()?;
```

### CUDA

pip 版の CUDA ライブラリを使う場合は、NVRTC / cuBLAS のパスを通してから実行します。

```bash
export LD_LIBRARY_PATH="$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/python/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
cargo run --release --example cuda_matmul
```

`cudarc` の動的ロードを使うため、ビルド時に CUDA toolkit / `nvcc` は不要です。NVRTC はデバイスの compute capability に合わせて実行時コンパイルします。メモリプールがデバイス割り当てを再利用し、`ComputeType` で GEMM の計算精度を選べます。

```rust
let gpu = CudaBackend::new(0)?;
gpu.set_compute_type(ComputeType::F16); // F32 / Tf32 / F16 / Bf16

// AMP: GEMM は f16/bf16、LayerNorm / Softmax / 損失は f32
let loss = autocast(DType::F16, || {
    let y = model.forward(&x)?;
    F::mean(&F::mul(&F::sub(&y, &target)?, &F::sub(&y, &target)?)?)
})??;
let scaled = scaler.scale_loss(&loss)?;
scaled.backward()?;
scaler.step(&mut optimizer, &params)?;
```

`GradScaler` が loss scaling、grad の unscale、inf/nan 検出、scale の増減を担当します。autocast は GEMM 系のみ低精度化し、LayerNorm / Softmax / reduction は f32 のままです。

## クイックスタート — Python

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
for (key, id) in run.plan().iter() {          // 状態が明示的に見える
    println!("{key} -> {id}");
}
let mut draft = run.fork();                   // KV キャッシュを分岐
draft.step(vec![next_token])?;                // 親は変更されない
```

- `StateStore` がテンソル / 変数 / スカラー / paged state を名前空間付き `StateKey` で所有します。
- `Run` がストア、RNG、step カウンタを所有します。
- `Step` は初回に状態を宣言し、型付き `StateId` ハンドルを受け取ります。ホットパスに文字列検索は入りません。
- Dense 状態は `Arc` ベースの不変テンソルなので、`fork` は軽量で rollback も単純です。
- `snapshot` / `restore` / `transaction`（drop 時自動 rollback）を提供します。

### Paged KV と batched decode

```rust
let mut runner = PagedBatchRunner::new(decoder, CpuBackend::new(), 2, 16, 7)?;
runner.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
let output = runner.step()?;   // 全リクエストを 1 回の paged attention で decode
```

- `BlockAllocator` がフリーブロックと参照カウントを管理します。
- `PagedKvCache` は `append` / `truncate` / `fork` を持ち、テンソル単位の COW で fork を分離します。
- `Ops::paged_attention_batch` はリクエストごとの block table と系列長を受け取り、バッチ全体を 1 カーネルで処理します。
- prefill は dense な `Run` で行い、その KV を paged cache に変換してから batched decode に移ります。
- v0.4 の CPU 実装は参照実装です。ブロック append はコピーを伴い、attention はスカラー実装です。

### Continuous batching と Speculative decoding

```rust
// 論理バッチ（リクエスト単位の Run、prefill/decode の計画が見える）
let mut batcher = ContinuousBatcher::new(decoder, CpuBackend::new(), 7);
batcher.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
let output = batcher.step()?;
```

```rust
// greedy な speculative decoding。target の greedy 出力と完全一致する
let tokens = speculative_generate(&mut target, &mut draft, prompt, 8, 3)?;
```

- `ContinuousBatcher` は `RequestTable` と `BatchPlan { prefill, decode }` を持ち、途中でリクエストを追加できます。
- `PagedBatchRunner` は paged KV を使った本物の batched decode を行います。
- speculative decoding は draft 提案 → `target.fork()` で検証 → 受理分だけ commit → draft を snapshot/restore で再同期、という流れです。

## アーキテクチャ

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
        │  modern-chainer-llm: decoder, paged KV,     │
        │                      batching, speculation  │
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

`Backend` / `Ops` がカーネル ABI です。`Tensor<B>` はバックエンドに対してジェネリックで、CPU はモノモルフィズされ、Python 層は `DynTensor` enum で包みます。詳細は [`docs/architecture.md`](docs/architecture.md)（英語）を参照してください。

## 性能

2026-10-31、AMD Ryzen 5 5600G（6 コア / 12 スレッド）、両実装とも 12 スレッド、PyTorch 2.15.0+cpu で測定。複数回の中央値で、小さいほど高速です。これはマイクロベンチマークであり、あらゆるワークロードに対する主張ではありません。

| ワークロード | Modern Chainer | PyTorch | 比 |
|---|---:|---:|---:|
| matmul 512×512×512 | 0.43–0.52 ms | 0.42 ms | 0.81–0.98× |
| elementwise add（4M f32） | 2.23–2.30 ms | 1.95–2.16 ms | 0.85–0.97× |
| Linear fwd+bwd 128×784×256 | 0.54–0.61 ms | 0.32 ms | 0.52–0.60× |
| Adam step 1024×1024 | 3.38–3.81 ms | 4.02–4.98 ms | 1.19–1.40× |
| **幾何平均** | **~1.20 ms** | **~1.04 ms** | **~0.87–0.91×** |

Adam は勝利、matmul / elementwise はほぼ互角、Linear backward が課題です。目標だった幾何平均 1.2× は未達ですが、ベンチマークハーネスをリポジトリに含めているので改善を測定できます。Paged / batched decode は CPU 参照実装のため、まだ PyTorch との比較対象外です。

CUDA（RTX 2070 SUPER）対 PyTorch 2.11.0+cu128（同じ GPU）。`torch/ours` は大きいほど Modern Chainer が高速です。

| 処理 | Modern Chainer | PyTorch | torch/ours |
|---|---:|---:|---:|
| matmul eager 512³ | 0.046 ms | 0.045 ms | 0.99× |
| matmul graph 512³ | 0.043 ms | 0.044 ms | 1.03× |
| linear 128×784×256 | 0.021 ms | 0.038 ms | 1.82× |
| linear+gelu 128×784×256 | 0.021 ms | 0.041 ms | 1.91× |
| Adam eager 1M | 0.112 ms | 0.216 ms | 1.93× |
| Adam graph 1M | 0.077 ms | 0.229 ms | 2.99× |

Batched static decoder（RTX 2070 SUPER、F32、B=8、decode=32、ランダム重み）。TTFT は batched prefill の中央値、decode は 1 step ごとに logits を CPU にコピーして argmax するホストループ込みです。GPU クロックの変動があるため範囲で見てください。

| 指標 | Modern Chainer | PyTorch 2.11 eager | 比 |
|---|---:|---:|---:|
| TTFT（prefill=16） | 0.26 ms | 1.45–1.58 ms | 5.5× |
| prefill（prefill=16） | 488k tok/s | 81k–88k tok/s | 5.5× |
| decode（context 16→48） | 35k–41k tok/s | 4.0k–6.7k tok/s | 5–10× |
| TTFT（prefill=512） | 7.6 ms | 5.20 ms | 0.69× |
| prefill（prefill=512） | 540k tok/s | 787k tok/s | 0.69× |
| decode（context 512→544） | 35.5k tok/s | 6.9k tok/s | 5.1× |

可変長 prompt（`VARIABLE=1`、prefill=512）でも TTFT は約 7.6 ms、実トークン throughput は約 420k tok/s です。prefill は flash attention（head_dim ≤ 128、F32/F16/BF16 を直接読む）または cuBLAS strided-batched GEMM、decode は per-sequence position の GEMM / native low-precision attention です。KV キャッシュは F32 / F16 / BF16 を選択でき、B=8・max_seq=560・4 層で 8 / 4 / 4 MiB。F16/BF16 の decode attention は upcast せず KV を直接読みます。

```bash
cargo run --release --example cuda_bench
cargo run --release --example graph_trainer
cargo run --release --example static_step
cargo run --release --example static_decoder
cargo run --release --example static_batched_decoder
PREFILL_LEN=512 DECODE_TOKENS=32 KV_DTYPE=f16 \
  cargo run --release --example static_batched_decoder
VARIABLE=1 PREFILL_LEN=512 DECODE_TOKENS=32 \
  cargo run --release --example static_batched_decoder
# PyTorch 比較（CUDA 版 PyTorch の venv を使用）
CUDA_VISIBLE_DEVICES=<RTX 2070S の UUID> \
  /tmp/opencode/torch-venv/bin/python scripts/bench_cuda_vs_pytorch.py
CUDA_VISIBLE_DEVICES=<RTX 2070S の UUID> \
  /tmp/opencode/torch-venv/bin/python scripts/bench_static_batched_vs_pytorch.py
```

cuBLASLt は bias、および GELU を GEMM エピローグで融合します。decoder の MLP は GELU に切り替え、bias+GELU を 1 カーネルで実行します。`ComputeType` で F32 / TF32 / F16 / BF16 を選べます。メモリプールがデバイス割り当てを再利用します。

`CudaStaticStep` は固定バッファを宣言し、in-place で 1 step を enqueue する汎用 trait です。`StaticRun` がバッファを所有して CUDA Graph をキャプチャし、入力だけ差し替えて replay します。`LinearMseAdamStep`（Linear + MSE + Adam）、`LinearForwardStep`（Linear forward）、`StaticDecoderStep`（KV キャッシュ + position をデバイス常駐にした 1 トークン decode）を同梱しています。step / position はデバイス上で進むため、同じ graph を繰り返し replay できます。

`BatchedStaticDecoder` は固定 shape の batched prefill と batched decode を別々の CUDA Graph としてキャプチャし、同じ KV キャッシュを共有します。`prefill_batch(&[&[u32]])` は可変長 prompt を受け取り、pad と実長を分離して最後の実トークンだけ lm_head に通します。prefill attention は head_dim ≤ 128 なら flash attention（F32/F16/BF16 を直接読む）、それ以外は cuBLAS strided-batched GEMM です。decode は batch ごとに 1 トークンずつ、per-sequence position で attention し、position を graph 内で進めます。`KV_DTYPE=f16` / `bf16` の decode は native low-precision attention です。TTFT と tok/s は `examples/static_batched_decoder.rs` で計測できます。

- [`scripts/bench_vs_pytorch.py`](scripts/bench_vs_pytorch.py)
- [`docs/benchmarks.md`](docs/benchmarks.md)（英語）

### Python unified device benchmark

CUDA / Vulkan を含めて Python から PyTorch と比較できます。Vulkan は adapter index を指定でき、`vulkan:1` は RTX 2070 SUPER です。

```bash
PYTHONPATH="$PWD/python" BENCH_DEVICES=cpu,cuda,vulkan:1 \
  /tmp/opencode/torch-venv/bin/python scripts/bench_vs_pytorch.py
```

RTX 2070 SUPER、PyTorch 2.11.0+cu128、`matmul 512³` / `add 4M` / `linear fwd+bwd 128×784×256` / `Adam 1024²`。単位 ms、`ours / torch`。

| Device | matmul | add 4M | linear fwd+bwd | Adam | geomean |
|---|---:|---:|---:|---:|---:|
| CPU | 0.49 / 0.43 | 2.19 / 8.00 | 0.58 / 0.32 | 3.56 / 4.94 | 1.22 / 1.53 (1.25×) |
| CUDA | 0.074 / 0.070 | 0.185 / 0.142 | 0.207 / 0.198 | 0.457 / 0.526 | 0.190 / 0.179 (0.94×) |
| Vulkan:1 | 0.77 / 0.071 | 0.44 / 0.141 | 2.46 / 0.204 | 3.08 / 0.519 | 1.27 / 0.180 (0.14×) |

Python/CUDA はほぼ互角、Python/Vulkan は正しく動くがまだ大幅に遅いです。カーネル改善で Adam は 10.7 → 約 3.1 ms、幾何平均は 2.11 → 1.27 ms になりました。linear は bias 融合・bias-grad 融合・TN small で 3.19 → 2.46 ms。残る主因は scalar WGSL matmul です。

## ビルドとテスト

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude modern-chainer-python
cargo bench -p modern-chainer --bench cpu_ops

cd python
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest
```

Vulkan テストは利用可能なアダプタ（NVIDIA / AMD / Intel / Mesa lavapipe）を使い、Vulkan ローダがない環境ではスキップします。

## ドキュメント

- [`SPEC.md`](SPEC.md) — 仕様（英語）
- [`docs/state-by-run.md`](docs/state-by-run.md) — State by Run、Run/Step、paged KV、batching、speculation（英語）
- [`docs/architecture.md`](docs/architecture.md) — クレート構成、autograd、カーネル ABI（英語）
- [`docs/backends.md`](docs/backends.md) — CPU / Vulkan / CUDA 計画（英語）
- [`docs/python-api.md`](docs/python-api.md) — Python API と Chainer 7 との差分（英語）
- [`docs/benchmarks.md`](docs/benchmarks.md) — 計測方法と結果（英語）
- [`docs/roadmap.md`](docs/roadmap.md) — ロードマップ（英語）
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — 開発手順（英語）

## 主な制約

- CUDA のネイティブ実装は elementwise / reduction / matmul / linear / Adam / softmax CE / embedding / layer_norm / attention / paged attention まで。汎用 `sum_to`（bias 以外）のみ CPU フォールバックです。
- static decoder の batch と `prefill_len`（capacity）は固定です。可変長 prompt は投入できますが、連続 batching・リクエストの途中参加・capacity を超える prompt は未対応です。
- F32 decode は `max_seq` 長の GEMM なので、現在位置より先のゼロ領域も走査します。長い `max_seq` で短い系列は非効率です。
- native low-precision decode attention は head_dim ≤ 256、flash prefill は head_dim ≤ 128。超える場合は F32 workspace へ upcast するフォールバックです。
- `CudaStaticStep` は CUDA 固有です。任意の `Run` をそのままキャプチャする汎用 static step はまだありません。
- static decoder の max_seq は固定バッファ長で制限されます。
- F16/BF16 は **compute type** と **autocast AMP** に対応しています。autocast は GEMM 系（linear / matmul）を低精度化し、LayerNorm / Softmax / reduction は f32 のままです。モデル全体を f16/bf16 dtype にする経路は未対応です。
- wide paged attention は block-per-query の CUDA カーネルです。
- Python から CUDA / Vulkan の Tensor / Variable / Linear / SGD / Adam / F.* を実行できます。`Run` / `PagedBatchRunner` / `BatchedStaticDecoder` / `Trainer` は未公開（Rust ファースト）。
- Vulkan の reduction / softmax CE はまだ CPU フォールバックで、Python でもそのコストが乗ります。CUDA と比べた性能は現状 0.09× 程度です。
- Paged COW はテンソル単位。ブロック単位の in-place COW と CUDA paged-attention カーネルは今後。
- リリースはまだ行っていません。

## ライセンス

Apache-2.0。詳細は [`LICENSE`](LICENSE) を参照してください。
