import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O


def test_version():
    assert chainer.__version__
    assert chainer.cuda_available() in (True, False)
    assert chainer.vulkan_available() in (True, False)


def test_linear_forward_and_backward():
    layer = L.Linear(2, 1, seed=1)
    x = chainer.Variable(chainer.Tensor.from_list([1.0, 2.0, 3.0, 4.0], [2, 2]))
    y = layer(x)
    assert y.shape == [2, 1]
    loss = F.sum(y)
    loss.backward()
    assert layer.W.grad is not None
    assert layer.b.grad is not None


def test_sgd_updates_parameters():
    layer = L.Linear(1, 1, seed=3)
    x = chainer.Variable(chainer.Tensor.from_list([1.0, 2.0], [2, 1]))
    target = chainer.Variable.constant(
        chainer.Tensor.from_list([2.0, 4.0], [2, 1])
    )
    optimizer = O.SGD(0.1)
    optimizer.setup(layer.parameters())
    before = layer.W.data.tolist()
    for _ in range(5):
        prediction = layer(x)
        diff = F.sub(prediction, target)
        loss = F.mean(F.mul(diff, diff))
        optimizer.zero_grad()
        loss.backward()
        optimizer.update()
    after = layer.W.data.tolist()
    assert before != after
    assert loss.data.tolist()[0] < 100.0


def test_chain_container():
    class MLP(chainer.Chain):
        def __init__(self):
            super().__init__()
            with self.init_scope():
                self.l1 = L.Linear(2, 3, seed=0)
                self.l2 = L.Linear(3, 1, seed=0)

        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    model = MLP()
    assert len(model.parameters()) == 4
    out = model(
        chainer.Variable(chainer.Tensor.from_list([0.1, 0.2], [1, 2]))
    )
    assert out.shape == [1, 1]


def test_numpy_interop():
    import numpy as np

    tensor = chainer.Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert tensor.shape == [2, 3]
    back = tensor.numpy()
    assert back.shape == (2, 3)
    assert back.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
