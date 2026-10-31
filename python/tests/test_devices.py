import numpy as np
import pytest

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O


def available_devices():
    devices = ["cpu"]
    if chainer.cuda_available():
        devices.append("cuda")
    if chainer.vulkan_available():
        devices.append("vulkan:0")
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_tensor_move_and_matmul(device):
    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = chainer.Tensor.from_numpy(a_np).to(device)
    b = chainer.Tensor.from_numpy(b_np, device=device)
    assert a.device.split(":")[0] == device.split(":")[0]
    assert a.numpy().tolist() == a_np.tolist()
    moved = chainer.Variable(a)
    out = F.matmul(moved, chainer.Variable(b))
    assert out.device.split(":")[0] == device.split(":")[0]
    np.testing.assert_allclose(out.array, a_np @ b_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_linear_backward_and_adam(device):
    layer = L.Linear(3, 2, seed=1, device=device)
    x = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones((4, 3), dtype=np.float32), device=device)
    )
    y = layer(x)
    loss = F.sum(y)
    loss.backward()
    assert y.device.split(":")[0] == device.split(":")[0]
    assert layer.W.grad is not None
    assert layer.b.grad is not None

    before = layer.W.data.numpy().copy()
    optimizer = O.Adam(0.01)
    optimizer.setup(layer.parameters())
    optimizer.zero_grad()
    F.sum(layer(x)).backward()
    optimizer.update()
    after = layer.W.data.numpy()
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("device", available_devices())
def test_synchronize(device):
    chainer.synchronize(device)


def test_vulkan_devices_listing():
    if chainer.vulkan_available():
        assert chainer.vulkan_devices()
