import numpy as np
import pytest

import chainer
import chainer.functions as F


def available_devices():
    devices = ["cpu"]
    if chainer.cuda_available():
        devices.append("cuda")
    if chainer.vulkan_available():
        devices.append("vulkan:0")
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_softmax_layer_norm_backward(device):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((2, 3), dtype=np.float32)
    x = chainer.Variable(chainer.Tensor.from_numpy(values, device=device))
    probs = F.softmax(x)
    expected = np.exp(values - values.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs.array, expected, rtol=1e-4, atol=1e-5)
    F.sum(probs).backward()
    assert x.grad is not None

    x = chainer.Variable(chainer.Tensor.from_numpy(values, device=device))
    gamma = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones(3, dtype=np.float32), device=device)
    )
    beta = chainer.Variable(
        chainer.Tensor.from_numpy(np.zeros(3, dtype=np.float32), device=device)
    )
    norm = F.layer_norm(x, gamma, beta)
    np.testing.assert_allclose(
        norm.array,
        (values - values.mean(-1, keepdims=True))
        / np.sqrt(values.var(-1, keepdims=True) + 1e-5),
        rtol=1e-4,
        atol=1e-4,
    )
    F.sum(norm).backward()
    assert x.grad is not None and gamma.grad is not None and beta.grad is not None


@pytest.mark.parametrize("device", available_devices())
def test_conv_pool_batchnorm(device):
    image = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 10.0
    x = chainer.Variable(chainer.Tensor.from_numpy(image, device=device))
    weight = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones((1, 1, 2, 2), dtype=np.float32), device=device)
    )
    conv = F.conv2d(x, weight, None, 1, 0)
    expected = np.array(
        [
            [image[0, 0, 0:2, 0:2].sum(), image[0, 0, 0:2, 1:3].sum()],
            [image[0, 0, 1:3, 0:2].sum(), image[0, 0, 1:3, 1:3].sum()],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(conv.array.reshape(2, 2), expected, rtol=1e-5, atol=1e-5)
    pooled = F.max_pool2d(x, 2, 1, 0)
    assert pooled.shape == [1, 1, 2, 2]

    gamma = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones(1, dtype=np.float32), device=device)
    )
    beta = chainer.Variable(
        chainer.Tensor.from_numpy(np.zeros(1, dtype=np.float32), device=device)
    )
    norm = F.batch_norm(x, gamma, beta)
    np.testing.assert_allclose(norm.array.mean(), 0.0, atol=1e-5)


def test_embedding_attention_reshape():
    ids = chainer.Variable(
        chainer.Tensor.from_numpy(np.array([0, 2, 0], dtype=np.int64))
    )
    weight = chainer.Variable(
        chainer.Tensor.from_numpy(
            np.arange(6, dtype=np.float32).reshape(3, 2) / 10.0
        )
    )
    out = F.embedding(ids, weight)
    assert out.shape == [3, 2]
    F.sum(out).backward()
    assert weight.grad is not None

    q = chainer.Variable(
        chainer.Tensor.from_numpy(np.array([[[0.2, -0.1], [0.4, 0.3]]], dtype=np.float32))
    )
    k = chainer.Variable(
        chainer.Tensor.from_numpy(np.array([[[0.1, 0.2], [-0.3, 0.4]]], dtype=np.float32))
    )
    v = chainer.Variable(
        chainer.Tensor.from_numpy(np.array([[[0.5, -0.2], [0.3, 0.1]]], dtype=np.float32))
    )
    attn = F.multi_head_attention(q, k, v, 1, True)
    assert attn.shape == [1, 2, 2]
    F.sum(attn).backward()
    assert q.grad is not None

    x = chainer.Variable(
        chainer.Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    )
    assert F.reshape(x, [3, 2]).shape == [3, 2]
    assert F.permute(x, [1, 0]).shape == [3, 2]


def test_new_links_cpu():
    import chainer.links as L

    conv = L.Conv2d(1, 2, 3, stride=1, padding=1)
    x = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones((1, 1, 4, 4), dtype=np.float32))
    )
    y = conv(x)
    assert y.shape == [1, 2, 4, 4]
    bn = L.BatchNorm(2)
    assert bn(y).shape == y.shape
    ln = L.LayerNorm(4)
    flat = F.reshape(y, [8, 4])
    assert ln(flat).shape == [8, 4]
    emb = L.Embedding(8, 4)
    ids = chainer.Variable(
        chainer.Tensor.from_numpy(np.array([0, 2, 4], dtype=np.int64))
    )
    assert emb(ids).shape == [3, 4]
    attn = L.MultiHeadAttention(4, 2)
    seq = chainer.Variable(
        chainer.Tensor.from_numpy(np.ones((2, 3, 4), dtype=np.float32))
    )
    assert attn(seq).shape == [2, 3, 4]
