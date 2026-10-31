"""Train a small MLP classifier on a synthetic three-class problem."""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O


class MLP(chainer.Chain):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden, seed=0)
            self.l2 = L.Linear(n_hidden, n_out, seed=1)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


def make_data(samples=512, features=8, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((samples, features), dtype=np.float32)
    scores = x @ rng.standard_normal((features, classes), dtype=np.float32)
    labels = scores.argmax(axis=1).astype(np.int64)
    return x, labels


def main():
    x_np, t_np = make_data()
    model = MLP(8, 32, 3)
    optimizer = O.Adam(0.01)
    optimizer.setup(model.parameters())

    for epoch in range(30):
        order = np.random.default_rng(epoch).permutation(len(x_np))
        total = 0.0
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            x = chainer.Variable(chainer.Tensor.from_numpy(x_np[indices]))
            target = chainer.Variable.constant(
                chainer.Tensor.from_numpy(t_np[indices])
            )
            logits = model(x)
            loss = F.softmax_cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.update()
            total += float(loss.data.tolist()[0])
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d}  loss {total / (len(order) // 64):.4f}")


if __name__ == "__main__":
    main()
