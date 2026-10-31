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
| `Tensor.zeros(shape, device="cpu")` | f32 zeros |
| `Tensor.from_list(flat, shape, device="cpu")` | build from a flat float list |
| `Tensor.from_numpy(array, device="cpu")` | preserves int64/int32/uint8 on CPU; casts f32 on GPU |
| `Tensor.numpy()` | NumPy copy (requires NumPy) |
| `Tensor.to(device)` | copy to `cpu`, `cuda[:n]` or `vulkan[:n]` |
| `.shape`, `.dtype`, `.device`, `.ndim`, `.numel()` | metadata |
| `.tolist()` | flat Python floats |

### `chainer.Variable`

| Member | Description |
|---|---|
| `Variable(tensor)` | differentiable variable |
| `Variable.constant(tensor)` | no gradient |
| `.data` | underlying `Tensor` |
| `.array` | NumPy view of the data |
| `.grad` | `Tensor` or `None` |
| `.shape`, `.device`, `.requires_grad`, `.grad_fn` | metadata |
| `.to(device)` | copy data to another device (new variable) |
| `.backward()` | reverse-mode traversal |
| `.cleargrad()` | drop the gradient |
| `.unchain_backward()` | truncate the graph |

### `chainer.functions` (`F`)

`add`, `sub`, `mul`, `div`, `sqrt`, `relu`, `neg`, `matmul`, `transpose`,
`sum`, `mean`, `softmax_cross_entropy`.

Broadcasting is supported for binary elementwise ops and for
`softmax_cross_entropy` logits of shape `[batch, classes]` with integer labels.

### `chainer.links` (`L`)

`Linear(in_features, out_features, seed=0, device="cpu")` with `W`, `b`,
`parameters()`, `cleargrads()` and `__call__`. The layer and its input must be
on the same device.

### Devices

```python
chainer.cuda_available()      # CUDA driver + NVRTC/cuBLAS available
chainer.vulkan_available()    # at least one Vulkan adapter
chainer.vulkan_devices()      # adapter names, in vulkan:0, vulkan:1 ... order
chainer.synchronize(device)   # wait for queued CUDA/Vulkan work
```

Device strings are `cpu`, `cuda`, `cuda:N`, `vulkan`, `vulkan:N`. Vulkan adapter
selection uses the order returned by `vulkan_devices()`. All tensors on the
same device share one backend context, so operations can mix tensors created
independently.

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
    x = chainer.Variable(chainer.Tensor.from_numpy(x_np, device="cuda"))
    t = chainer.Variable.constant(chainer.Tensor.from_numpy(t_np, device="cuda"))
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
- **Vulkan performance is not yet competitive.** Device execution works from
  Python, but reductions and softmax cross entropy still round-trip through
  the CPU, and the tiled matmul is much slower than cuBLAS.
- **LLM/static APIs are Rust-only.** `Run`, `PagedBatchRunner` and
  `BatchedStaticDecoder` are not exposed to Python yet.
- **No State-by-Run API yet.** `Run`, `StateStore`, the dense KV decoder and
  fork/rollback are available from Rust (`modern_chainer::run` /
  `modern_chainer::llm`). A Python `Run` wrapper is the v0.6 milestone.
- **Partial dtype support.** Kernels are f32; integer dtypes are used for
  labels.

## Testing

`python/tests/test_basic.py` covers import/version, Linear forward/backward,
SGD updates, `Chain`, and NumPy round-trips. `python/tests/test_devices.py`
runs matmul, Linear backward and Adam on every available CPU/CUDA/Vulkan
device. The example `python/examples/train_mlp.py` trains a synthetic
three-class classifier.
