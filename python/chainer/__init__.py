"""Modern Chainer: define-by-run deep learning with a Rust core.

The Python package is a thin layer over the Rust implementation.  The public
surface intentionally follows the original Chainer's vocabulary:

    import chainer
    import chainer.functions as F
    import chainer.links as L

    class MLP(chainer.Chain):
        def __init__(self):
            super().__init__()
            with self.init_scope():
                self.l1 = L.Linear(784, 100)
                self.l2 = L.Linear(100, 10)

        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))
"""

from contextlib import contextmanager

from ._chainer import (
    Adam,
    Linear,
    SGD,
    Tensor,
    Variable,
    __version__,
    cuda_available,
    synchronize,
    vulkan_available,
    vulkan_devices,
)
from . import functions, links, optimizers


class Chain:
    """Minimal Chainer-style container for links."""

    def __init__(self):
        object.__setattr__(self, "_children", {})

    @contextmanager
    def init_scope(self):
        yield

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, (Linear, Chain)):
            self._children[name] = value

    def add_link(self, name, link):
        setattr(self, name, link)
        return link

    def children(self):
        return dict(self._children)

    def parameters(self):
        params = []
        for link in self._children.values():
            params.extend(link.parameters())
        return params

    def cleargrads(self):
        for parameter in self.parameters():
            parameter.cleargrad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


__all__ = [
    "Adam",
    "Chain",
    "Linear",
    "SGD",
    "Tensor",
    "Variable",
    "__version__",
    "cuda_available",
    "synchronize",
    "vulkan_available",
    "vulkan_devices",
    "functions",
    "links",
    "optimizers",
]
