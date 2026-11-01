"""Compare the example workloads in Modern Chainer and PyTorch.

Run with the project Python venv (CUDA torch installed):

    CUDA_VISIBLE_DEVICES=<RTX UUID> PYTHONPATH="$PWD/python" \
      python/.venv/bin/python scripts/bench_examples_vs_pytorch.py

Devices:
  * CPU: Modern Chainer CPU vs PyTorch CPU
  * CUDA: Modern Chainer CUDA vs PyTorch CUDA (same GPU)
  * Vulkan: Modern Chainer Vulkan (RTX adapter) vs PyTorch CUDA reference
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time

os.environ.setdefault("RAYON_NUM_THREADS", "12")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from PIL import Image

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_iris():
    rows = []
    with open(os.path.join(DATA, "iris.data"), encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) != 5:
                continue
            label = 0 if "setosa" in fields[4] else 1 if "versicolor" in fields[4] else 2
            rows.append(([float(v) for v in fields[:4]], label))
    rng = np.random.default_rng(3)
    rng.shuffle(rows)
    train, test = rows[:120], rows[120:]
    x = np.asarray([r[0] for r in train], dtype=np.float32)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    def norm(values):
        return ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32)
    return (
        norm([r[0] for r in train]), np.asarray([r[1] for r in train], dtype=np.int64),
        norm([r[0] for r in test]), np.asarray([r[1] for r in test], dtype=np.int64),
    )


def read_idx_images(path):
    raw = np.fromfile(path, dtype=np.uint8)
    magic, count, rows, cols = np.frombuffer(raw[:16].tobytes(), dtype=">u4")
    assert magic == 2051
    return raw[16:16 + int(count) * int(rows) * int(cols)].reshape(count, rows * cols)


def read_idx_labels(path):
    raw = np.fromfile(path, dtype=np.uint8)
    magic, count = np.frombuffer(raw[:8].tobytes(), dtype=">u4")
    assert magic == 2049
    return raw[8:8 + int(count)]


def load_mnist():
    train_x = read_idx_images(os.path.join(DATA, "mnist/train-images-idx3-ubyte"))
    train_y = read_idx_labels(os.path.join(DATA, "mnist/train-labels-idx1-ubyte"))
    test_x = read_idx_images(os.path.join(DATA, "mnist/t10k-images-idx3-ubyte"))
    test_y = read_idx_labels(os.path.join(DATA, "mnist/t10k-labels-idx1-ubyte"))
    rng = np.random.default_rng(9)
    train_order = rng.permutation(len(train_x))[:6000]
    test_order = rng.permutation(len(test_x))[:1000]
    return (
        train_x[train_order].astype(np.float32) / 255.0,
        train_y[train_order].astype(np.int64),
        test_x[test_order].astype(np.float32) / 255.0,
        test_y[test_order].astype(np.int64),
    )


def load_cifar():
    root = os.path.join(DATA, "cifar10")
    paths = []
    for split in ("train", "test"):
        classes = sorted(
            name for name in os.listdir(os.path.join(root, split))
            if os.path.isdir(os.path.join(root, split, name))
        )
        for label, name in enumerate(classes):
            for path in glob.glob(os.path.join(root, split, name, "*.png")):
                paths.append((path, label))
    rng = np.random.default_rng(15)
    rng.shuffle(paths)
    train_paths = paths[:5000]
    test_paths = paths[5000:6000]

    def decode(items):
        images = np.stack([
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).reshape(-1) / 255.0
            for path, _ in items
        ])
        labels = np.asarray([label for _, label in items], dtype=np.int64)
        return images, labels

    train_x, train_y = decode(train_paths)
    test_x, test_y = decode(test_paths)
    return train_x[:4000], train_y[:4000], test_x[:1000], test_y[:1000]


def load_cnn():
    rng = np.random.default_rng(5)
    def make(label, count):
        values = np.zeros((count, 1, 16, 16), dtype=np.float32)
        for n in range(count):
            y = np.arange(16)[:, None]
            x = np.arange(16)[None, :]
            if label == 0:
                mask = y == 7
            elif label == 1:
                mask = x == 7
            else:
                mask = x == y
            values[n, 0] = mask.astype(np.float32) + rng.normal(0, 0.1, (16, 16)).astype(np.float32)
        return values
    train_parts, train_labels = [], []
    test_parts, test_labels = [], []
    for label in range(3):
        train_parts.append(make(label, 214))
        train_labels.extend([label] * 214)
        test_parts.append(make(label, 10))
        test_labels.extend([label] * 10)
    return (
        np.concatenate(train_parts),
        np.asarray(train_labels, dtype=np.int64),
        np.concatenate(test_parts),
        np.asarray(test_labels, dtype=np.int64),
    )


def load_wiki(chars=60_000):
    path = os.path.join(DATA, "wikipedia.txt")
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else (
        "Wikipedia is a free online encyclopedia written by volunteers. "
        "Machine learning learns patterns from data. "
    )
    text = text[:chars]
    alphabet = sorted(set(text))
    stoi = {ch: i + 4 for i, ch in enumerate(alphabet)}
    itos = {i + 4: ch for i, ch in enumerate(alphabet)}
    tokens = np.asarray([stoi[ch] for ch in text], dtype=np.int64)
    return tokens, len(alphabet) + 4, itos


# ---------------------------------------------------------------------------
# Modern Chainer models
# ---------------------------------------------------------------------------

def mc_tensor(array, device):
    return chainer.Tensor.from_numpy(np.asarray(array, dtype=np.float32), device=device)


def mc_ids(array, device):
    return chainer.Tensor.from_numpy(np.asarray(array, dtype=np.int64)).to(device)


def mc_sync(device):
    chainer.synchronize(device)


class MCMLP:
    def __init__(self, in_features, hidden, out_features, lr, device):
        self.l1 = L.Linear(in_features, hidden, seed=1, device=device)
        self.l2 = L.Linear(hidden, out_features, seed=2, device=device)
        self.params = self.l1.parameters() + self.l2.parameters()
        self.opt = O.Adam(lr)
        self.opt.setup(self.params)
        self.device = device

    def logits(self, x):
        return self.l2(F.relu(self.l1(x)))

    def train_step(self, x, y):
        xv = chainer.Variable(mc_tensor(x, self.device))
        tv = chainer.Variable.constant(mc_ids(y, self.device))
        loss = F.softmax_cross_entropy(self.logits(xv), tv)
        self.opt.zero_grad()
        loss.backward()
        self.opt.update()


class MCConvNet:
    def __init__(self, device):
        self.conv1 = L.Conv2d(1, 4, 3, stride=1, padding=1, seed=1, device=device)
        self.bn1 = L.BatchNorm(4, device=device)
        self.conv2 = L.Conv2d(4, 8, 3, stride=1, padding=1, seed=2, device=device)
        self.bn2 = L.BatchNorm(8, device=device)
        self.fc = L.Linear(8 * 4 * 4, 3, seed=3, device=device)
        self.params = (
            self.conv1.parameters() + self.bn1.parameters()
            + self.conv2.parameters() + self.bn2.parameters() + self.fc.parameters()
        )
        self.opt = O.Adam(0.002)
        self.opt.setup(self.params)
        self.device = device

    def logits(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2, 2, 0)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2, 2, 0)
        h = F.reshape(h, [h.shape[0], 8 * 4 * 4])
        return self.fc(h)

    def train_step(self, x, y):
        xv = chainer.Variable(mc_tensor(x, self.device))
        tv = chainer.Variable.constant(mc_ids(y, self.device))
        loss = F.softmax_cross_entropy(self.logits(xv), tv)
        self.opt.zero_grad()
        loss.backward()
        self.opt.update()


class MCTransformer:
    def __init__(self, vocab, device, seq=32, dim=32, heads=4):
        self.seq, self.dim = seq, dim
        self.emb = L.Embedding(vocab, dim, seed=1, device=device)
        pos = np.asarray([
            ((i * 13 % 97) / 97.0 - 0.5) * 0.2 for i in range(seq * dim)
        ], dtype=np.float32).reshape(seq, dim)
        self.pos = chainer.Variable.constant(chainer.Tensor.from_numpy(pos).to(device))
        self.attn = L.MultiHeadAttention(dim, heads, causal=True, device=device)
        self.ln1 = L.LayerNorm(dim, device=device)
        self.w1 = L.Linear(dim, dim * 4, seed=2, device=device)
        self.w2 = L.Linear(dim * 4, dim, seed=3, device=device)
        self.ln2 = L.LayerNorm(dim, device=device)
        self.head = L.Linear(dim, vocab, seed=4, device=device)
        self.params = (
            self.emb.parameters() + self.attn.parameters() + self.ln1.parameters()
            + self.w1.parameters() + self.w2.parameters() + self.ln2.parameters()
            + self.head.parameters()
        )
        self.opt = O.Adam(0.003)
        self.opt.setup(self.params)
        self.device = device

    def logits(self, tokens):
        ids = chainer.Variable.constant(mc_ids(tokens, self.device))
        h = F.add(self.emb(ids), self.pos)
        h3 = F.reshape(h, [1, self.seq, self.dim])
        a = F.reshape(self.attn(h3), [self.seq, self.dim])
        x = self.ln1(F.add(h, a))
        ff = self.w2(F.relu(self.w1(x)))
        x = self.ln2(F.add(x, ff))
        return self.head(x)

    def train_step(self, tokens, target):
        logits = self.logits(tokens)
        tv = chainer.Variable.constant(mc_ids(target, self.device))
        loss = F.softmax_cross_entropy(logits, tv)
        self.opt.zero_grad()
        loss.backward()
        self.opt.update()


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------

class TorchMLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.l2(TF.relu(self.l1(x)))


class TorchConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 4 * 4, 3)

    def forward(self, x):
        h = TF.relu(self.bn1(self.conv1(x)))
        h = TF.max_pool2d(h, 2, 2)
        h = TF.relu(self.bn2(self.conv2(h)))
        h = TF.max_pool2d(h, 2, 2)
        return self.fc(h.flatten(1))


class TorchTransformer(nn.Module):
    def __init__(self, vocab, seq=32, dim=32, heads=4):
        super().__init__()
        self.seq, self.dim = seq, dim
        self.emb = nn.Embedding(vocab, dim)
        pos = np.asarray([
            ((i * 13 % 97) / 97.0 - 0.5) * 0.2 for i in range(seq * dim)
        ], dtype=np.float32).reshape(1, seq, dim)
        self.register_buffer("pos", torch.from_numpy(pos))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim * 4, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, tokens):
        h = self.emb(tokens) + self.pos
        a, _ = self.attn(h, h, h, attn_mask=self.mask, need_weights=False)
        x = self.ln1(h + a)
        ff = self.w2(TF.relu(self.w1(x)))
        x = self.ln2(x + ff)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def sync_for(track, device, torch_device):
    if track == "mc":
        mc_sync(device)
    elif torch_device != "cpu":
        torch.cuda.synchronize()


def eval_mlp(track, device, model, x, y, torch_device):
    if track == "mc":
        logits = model.logits(chainer.Variable.constant(mc_tensor(x, device))).array
    else:
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(torch_device)).cpu().numpy()
        model.train()
    return float((logits.argmax(1) == y).mean())


def benchmark_mlp(track, device, torch_device, data, cfg, runs):
    train_x, train_y, test_x, test_y = data
    batch, steps, hidden, out, lr = cfg
    def one():
        if track == "mc":
            model = MCMLP(train_x.shape[1], hidden, out, lr, device)
        else:
            model = TorchMLP(train_x.shape[1], hidden, out).to(torch_device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        for step in range(steps):
            begin = (step * batch) % len(train_x)
            idx = (np.arange(batch) + begin) % len(train_x)
            xb, yb = train_x[idx], train_y[idx]
            if track == "mc":
                model.train_step(xb, yb)
            else:
                xv = torch.from_numpy(xb).to(torch_device)
                tv = torch.from_numpy(yb).to(torch_device)
                opt.zero_grad()
                loss = TF.cross_entropy(model(xv), tv)
                loss.backward()
                opt.step()
        sync_for(track, device, torch_device)
        train_s = time.perf_counter() - start
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        acc = eval_mlp(track, device, model, test_x, test_y, torch_device)
        sync_for(track, device, torch_device)
        return train_s, time.perf_counter() - start, acc
    one()
    results = [one() for _ in range(runs)]
    return (
        statistics.median(r[0] for r in results),
        statistics.median(r[1] for r in results),
        results[-1][2],
    )


def benchmark_conv(track, device, torch_device, data, runs):
    train_x, train_y, test_x, test_y = data
    batch, steps = 8, 80
    def one():
        if track == "mc":
            model = MCConvNet(device)
        else:
            model = TorchConvNet().to(torch_device)
            opt = torch.optim.Adam(model.parameters(), lr=0.002)
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        for step in range(steps):
            idx = (np.arange(batch) + step * batch) % len(train_x)
            xb, yb = train_x[idx], train_y[idx]
            if track == "mc":
                model.train_step(xb, yb)
            else:
                xv = torch.from_numpy(xb).to(torch_device)
                tv = torch.from_numpy(yb).to(torch_device)
                opt.zero_grad()
                loss = TF.cross_entropy(model(xv), tv)
                loss.backward()
                opt.step()
        sync_for(track, device, torch_device)
        train_s = time.perf_counter() - start
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        if track == "mc":
            logits = model.logits(chainer.Variable.constant(mc_tensor(test_x, device))).array
        else:
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(test_x).to(torch_device)).cpu().numpy()
            model.train()
        sync_for(track, device, torch_device)
        return train_s, time.perf_counter() - start, float((logits.argmax(1) == test_y).mean())
    one()
    results = [one() for _ in range(runs)]
    return (
        statistics.median(r[0] for r in results),
        statistics.median(r[1] for r in results),
        results[-1][2],
    )


def benchmark_wiki(track, device, torch_device, tokens, vocab, itos, runs, steps=400, gen=120):
    seq = 32
    def one():
        if track == "mc":
            model = MCTransformer(vocab, device)
        else:
            model = TorchTransformer(vocab).to(torch_device)
            opt = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = np.random.default_rng(21)
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        for _ in range(steps):
            begin = int(rng.integers(0, len(tokens) - seq - 1))
            xb = tokens[begin:begin + seq]
            yb = tokens[begin + 1:begin + seq + 1]
            if track == "mc":
                model.train_step(xb, yb)
            else:
                xv = torch.from_numpy(xb).to(torch_device)
                tv = torch.from_numpy(yb).to(torch_device)
                opt.zero_grad()
                loss = TF.cross_entropy(model(xv.unsqueeze(0)).squeeze(0), tv)
                loss.backward()
                opt.step()
        sync_for(track, device, torch_device)
        train_s = time.perf_counter() - start
        from_char = {ch: idx for idx, ch in itos.items()}
        context = [from_char.get(ch, 1) for ch in "Wikipedia"]
        while len(context) < seq:
            context.insert(0, 0)
        context = context[-seq:]
        sample_rng = np.random.default_rng(31)
        sync_for(track, device, torch_device)
        start = time.perf_counter()
        out = []
        for _ in range(gen):
            arr = np.asarray(context, dtype=np.int64)
            if track == "mc":
                logits = model.logits(arr).array[-1]
            else:
                with torch.no_grad():
                    logits = model(torch.from_numpy(arr).unsqueeze(0).to(torch_device))[0, -1].cpu().numpy()
            weights = np.exp((logits - logits.max()) / 0.7)
            weights /= weights.sum()
            nxt = int(sample_rng.choice(len(weights), p=weights))
            context.pop(0)
            context.append(nxt)
            out.append(itos.get(nxt, ""))
        sync_for(track, device, torch_device)
        return train_s, time.perf_counter() - start, "".join(out)
    one()
    results = [one() for _ in range(runs)]
    return (
        statistics.median(r[0] for r in results),
        statistics.median(r[1] for r in results),
        results[-1][2],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def vulkan_rtx_device():
    if not chainer.vulkan_available():
        return None
    for index, name in enumerate(chainer.vulkan_devices()):
        if "RTX 2070" in name:
            return f"vulkan:{index}"
    return "vulkan:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--workloads", default="iris,mnist,cifar10,wikipedia,cnn")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)

    devices = [("cpu", "cpu")]
    if chainer.cuda_available() and torch.cuda.is_available():
        devices.append(("cuda", "cuda"))
    vk = vulkan_rtx_device()
    if vk is not None:
        devices.append((vk, "cuda" if torch.cuda.is_available() else "cpu"))

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  threads={args.torch_threads}")
    print(f"chainer {chainer.__version__}  devices={[d for d, _ in devices]}")

    loaders = {
        "iris": load_iris,
        "mnist": load_mnist,
        "cifar10": load_cifar,
        "cnn": load_cnn,
    }
    data = {}
    for name, loader in loaders.items():
        start = time.perf_counter()
        data[name] = loader()
        print(f"loaded {name}: {time.perf_counter() - start:.2f} s")
    start = time.perf_counter()
    wiki_tokens, wiki_vocab, wiki_itos = load_wiki()
    print(f"loaded wikipedia: {time.perf_counter() - start:.2f} s")

    rows = []
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    for workload in workloads:
        for device, torch_device in devices:
            for track in ("mc", "torch"):
                try:
                    if workload == "iris":
                        train, ev, acc = benchmark_mlp(
                            track, device, torch_device, data["iris"], (30, 300, 16, 3, 0.01), args.runs
                        )
                    elif workload == "mnist":
                        train, ev, acc = benchmark_mlp(
                            track, device, torch_device, data["mnist"], (64, 300, 128, 10, 0.001), args.runs
                        )
                    elif workload == "cifar10":
                        train, ev, acc = benchmark_mlp(
                            track, device, torch_device, data["cifar10"], (32, 150, 128, 10, 0.001), args.runs
                        )
                    elif workload == "cnn":
                        train, ev, acc = benchmark_conv(
                            track, device, torch_device, data["cnn"], args.runs
                        )
                    elif workload == "wikipedia":
                        train, gen, text = benchmark_wiki(
                            track, device, torch_device, wiki_tokens, wiki_vocab, wiki_itos, args.runs
                        )
                        acc = float("nan")
                        ev = gen
                    else:
                        continue
                    rows.append((workload, device, track, train, ev, acc))
                    print(f"{workload:10} {device:10} {track:5} train={train:.3f}s eval={ev:.3f}s acc={acc:.3f}")
                except Exception as error:
                    print(f"{workload:10} {device:10} {track:5} FAILED: {error}")

    lines = [
        "| workload | mc device | framework | train s | eval/gen s | accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for workload, device, track, train, ev, acc in rows:
        lines.append(f"| {workload} | {device} | {track} | {train:.3f} | {ev:.3f} | {acc:.3f} |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")


if __name__ == "__main__":
    main()
