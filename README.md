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

## ステータス（v0.5.0）

| 領域 | 状態 |
|---|---|
| Define-by-Run autograd（Rust） | 安定 |
| State by Run（`Run` / `Step` / `StateStore`） | 安定: fork、snapshot/rollback、transaction |
| Dense KV decoder | 動作: 2層リファレンス、prefill/decode、sampling |
| Paged state + CPU paged attention | 動作: block allocator、block table、append/truncate、COW |
| Batched decode | 動作: `PagedBatchRunner` + batched paged attention |
| Continuous batching | 動作: `RequestTable` / `BatchPlan`（論理バッチ） |
| Speculative decoding | 動作: greedy の fork/verify/commit 参照実装 |
| CUDA バックエンド | 動作: NVRTC + cuBLAS / cuBLASLt 融合、CUDA Graph、LLM / paged attention までネイティブ |
| Python バインディング | コアサブセット: `Tensor`、`Variable`、`F.*`、`L.Linear`、`SGD`/`Adam`、`Chain` |
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
cargo run --release --example vulkan_matmul   # Vulkan ローダ/ICD が必要
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

`cudarc` の動的ロードを使うため、ビルド時に CUDA toolkit / `nvcc` は不要です。NVRTC はデバイスの compute capability に合わせて実行時コンパイルします。

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
| matmul eager 512³ | 0.048 ms | 0.046 ms | 0.95× |
| matmul graph 512³ | 0.043 ms | 0.045 ms | 1.04× |
| linear 128×784×256 | 0.023 ms | 0.038 ms | 1.62× |
| Adam eager 1M | 0.146 ms | 0.214 ms | 1.46× |
| Adam graph 1M | 0.078 ms | 0.231 ms | 2.97× |

```bash
cargo run --release --example cuda_bench
# PyTorch 比較（CUDA 版 PyTorch の venv を使用）
CUDA_VISIBLE_DEVICES=<RTX 2070S の UUID> \
  /tmp/opencode/torch-venv/bin/python scripts/bench_cuda_vs_pytorch.py
```

cuBLASLt は bias を GEMM エピローグで融合し、CUDA Graph は固定 shape の matmul / Adam 更新を再再生します。PyTorch 側の Adam Graph は `capturable=True` で測定しています。

- [`scripts/bench_vs_pytorch.py`](scripts/bench_vs_pytorch.py)
- [`docs/benchmarks.md`](docs/benchmarks.md)（英語）

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
- CUDA Graph は preallocated バッファ前提です（`graph_matmul` / `graph_adam`）。任意の `Run` をそのままキャプチャすることはまだできません。
- CUDA の reduction / attention / paged attention はネイティブですが、fp16/bf16 は未対応です。
- Python の CUDA テンソル API、`Run` / `PagedBatchRunner` / `Trainer` は未公開（Rust ファースト）。
- Paged COW はテンソル単位。ブロック単位の in-place COW と CUDA paged-attention カーネルは今後。
- リリースはまだ行っていません。

## ライセンス

Apache-2.0。詳細は [`LICENSE`](LICENSE) を参照してください。
