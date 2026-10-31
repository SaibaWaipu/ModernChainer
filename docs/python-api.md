# Python API

The Python package is `modern-chainer`; it is imported as `chainer` and built
from `crates/modern-chainer-python` with maturin.

## Build

```bash
cd python
uv venv
uv pip install maturin numpy pytest
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest
```

## Surface

### `chainer.Tensor`

| Member | Description |
|---|---|
| `Tensor.zeros(shape)` | f32 zeros |
| `Tensor.from_list(flat, shape)` | build from a flat float list |
| `Tensor.from_numpy(array)` | preserves int64/int32/uint8, otherwise casts f32 |
| `Tensor.numpy()` | NumPy copy (requires NumPy) |
| `.shape`, `.dtype`, `.ndim`, `.numel()` | metadata |
| `.tolist()` | flat Python floats |

### `chainer.Variable`

| Member | Description |
|---|---|
| `Variable(tensor)` | differentiable variable |
| `Variable.constant(tensor)` | no gradient |
| `.data` | underlying `Tensor` |
| `.array` | NumPy view of the data |
| `.grad` | `Tensor` or `None` |
| `.shape`, `.requires_grad`, `.grad_fn` | metadata |
| `.backward()` | reverse-mode traversal |
| `.cleargrad()` | drop the gradient |
| `.unchain_backward()` | truncate the graph |

### `chainer.functions` (`F`)

`add`, `sub`, `mul`, `div`, `sqrt`, `relu`, `neg`, `matmul`, `transpose`,
`sum`, `mean`, `softmax_cross_entropy`.

Broadcasting is supported for binary elementwise ops and for
`softmax_cross_entropy` logits of shape `[batch, classes]` with integer labels.

### `chainer.links` (`L`)

`Linear(in_features, out_features, seed=0)` with `W`, `b`, `parameters()`,
`cleargrads()` and `__call__`.

### `chainer.optimizers` (`O`)

- `SGD(lr)`
- `Adam(lr=0.001)`

Both implement `setup(parameters)`, `update()` and `zero_grad()`.

### `chainer.Chain`

Minimal Chainer-style container:

```python
class MLP(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 8)
            self.l2 = L.Linear(8, 2)
    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))
```

`Chain` records `Linear`/`Chain` attributes, aggregates `parameters()`, and
provides `cleargrads()`. `init_scope()` is a no-op context manager kept for
source compatibility.

## Training loop

```python
model = L.Linear(784, 10)
optimizer = O.Adam(0.001)
optimizer.setup(model.parameters())

for x_np, t_np in batches:
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np))
    t = chainer.Variable.constant(chainer.Tensor.from_numpy(t_np))
    loss = F.softmax_cross_entropy(model(x), t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.update()
```

## Differences from Chainer 7

- **No `Trainer`/`Updater`/`Iterator`/`Extension` yet.** Write the loop
  explicitly, as above.
- **No `chainer.config`.** Gradient control is `Variable.requires_grad` plus the
  Rust `no_grad` guard; inference graphs are not built in Python in v0.1.0.
- **No `FunctionNode` subclassing.** Custom differentiable ops are Rust `Op`
  implementations.
- **No `.npz` serialization yet.** Use the Rust `modern-chainer-serial` API.
- **CPU execution only in Python.** The Rust API exposes the Vulkan backend;
  a Python device selector is planned.
- **No State-by-Run API yet.** `Run`, `StateStore`, the dense KV decoder and
  fork/rollback are available from Rust (`modern_chainer::run` /
  `modern_chainer::llm`). A Python `Run` wrapper is the v0.6 milestone.
- **Partial dtype support.** Kernels are f32; integer dtypes are used for
  labels.

## Testing

`python/tests/test_basic.py` covers import/version, Linear forward/backward,
SGD updates, `Chain`, and NumPy round-trips. The example
`python/examples/train_mlp.py` trains a synthetic three-class classifier.
