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

layer = L.Linear(4, 2)
x = chainer.Variable(chainer.Tensor.from_list([1, 2, 3, 4], [1, 4]))
y = layer(x)
loss = F.sum(y)
loss.backward()
print(layer.W.grad.array)
```

`Variable.array` and `Tensor.numpy()` require NumPy.
