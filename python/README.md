# modern-chainer (Python)

Python bindings for Modern Chainer.  The extension is built with maturin from
`crates/modern-chainer-python`.

```bash
cd python
uv venv
uv pip install maturin numpy pytest
uv run maturin develop --release
uv run pytest
```

```python
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

print(chainer.cuda_available())
print(chainer.vulkan_available())
print(chainer.vulkan_devices())          # adapter names in index order

layer = L.Linear(4, 2, device="vulkan:1")
x = chainer.Variable(chainer.Tensor.from_list([1, 2, 3, 4], [1, 4], device="vulkan:1"))
y = layer(x)
loss = F.sum(y)
loss.backward()
print(layer.W.grad.array)

optimizer = O.Adam(0.001)
optimizer.setup(layer.parameters())
optimizer.update()
chainer.synchronize("vulkan:1")
```

Devices are `cpu`, `cuda`, `cuda:N`, `vulkan` and `vulkan:N`. Vulkan adapter
indices follow `chainer.vulkan_devices()`. `Tensor.to(device)` copies between
devices.

`Variable.array` and `Tensor.numpy()` require NumPy.

## Unified device benchmark

From the repository root:

```bash
PYTHONPATH="$PWD/python" BENCH_DEVICES=cpu,cuda,vulkan:1 \
  python/.venv/bin/python scripts/bench_vs_pytorch.py
```
