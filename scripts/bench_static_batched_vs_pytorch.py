"""Compare batched prefill/decode TTFT and tok/s against PyTorch eager.

The model shape matches examples/static_batched_decoder.rs. Weights are
random, so this measures kernels and scheduling, not model quality.
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True

VOCAB = 256
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 128
MAX_SEQ = 128
BATCH = 8
PREFILL = 16
DECODE_TOKENS = 32
DTYPE = torch.float32


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(42)
    hd = D_MODEL // N_HEADS

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=g, dtype=DTYPE) * 0.02

    emb = rand(VOCAB, D_MODEL)
    layers = []
    for _ in range(N_LAYERS):
        layers.append({
            "ln1_g": torch.ones(D_MODEL, device=device),
            "ln1_b": torch.zeros(D_MODEL, device=device),
            "wq": rand(D_MODEL, D_MODEL), "bq": torch.zeros(D_MODEL, device=device),
            "wk": rand(D_MODEL, D_MODEL), "bk": torch.zeros(D_MODEL, device=device),
            "wv": rand(D_MODEL, D_MODEL), "bv": torch.zeros(D_MODEL, device=device),
            "wo": rand(D_MODEL, D_MODEL), "bo": torch.zeros(D_MODEL, device=device),
            "ln2_g": torch.ones(D_MODEL, device=device),
            "ln2_b": torch.zeros(D_MODEL, device=device),
            "w1": rand(D_FF, D_MODEL), "b1": torch.zeros(D_FF, device=device),
            "w2": rand(D_MODEL, D_FF), "b2": torch.zeros(D_MODEL, device=device),
        })
    lnf_g = torch.ones(D_MODEL, device=device)
    lnf_b = torch.zeros(D_MODEL, device=device)
    lm_head = rand(VOCAB, D_MODEL)

    def cache() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(BATCH, N_HEADS, MAX_SEQ, hd, device=device),
            torch.zeros(BATCH, N_HEADS, MAX_SEQ, hd, device=device),
        )

    def layer(layer, x, k_cache, v_cache, start, length, causal):
        norm = F.layer_norm(x, (D_MODEL,), layer["ln1_g"], layer["ln1_b"], 1e-5)
        q = F.linear(norm, layer["wq"], layer["bq"]).view(BATCH, -1, N_HEADS, hd).transpose(1, 2)
        k = F.linear(norm, layer["wk"], layer["bk"]).view(BATCH, -1, N_HEADS, hd).transpose(1, 2)
        v = F.linear(norm, layer["wv"], layer["bv"]).view(BATCH, -1, N_HEADS, hd).transpose(1, 2)
        end = start + length
        k_cache[:, :, start:end] = k
        v_cache[:, :, start:end] = v
        scores = q @ k_cache[:, :, :end].transpose(-2, -1)
        scores = scores * (1.0 / hd ** 0.5)
        if causal and length > 1:
            mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v_cache[:, :, :end]
        out = out.transpose(1, 2).reshape(BATCH, length, D_MODEL)
        x = x + F.linear(out, layer["wo"], layer["bo"])
        norm = F.layer_norm(x, (D_MODEL,), layer["ln2_g"], layer["ln2_b"], 1e-5)
        hidden = F.gelu(F.linear(norm, layer["w1"], layer["b1"]))
        x = x + F.linear(hidden, layer["w2"], layer["b2"])
        return x

    def prefill(tokens: torch.Tensor, caches):
        x = F.embedding(tokens, emb)
        for i, layer_weights in enumerate(layers):
            x = layer(layer_weights, x, caches[i][0], caches[i][1], 0, PREFILL, True)
        x = F.layer_norm(x, (D_MODEL,), lnf_g, lnf_b, 1e-5)
        return F.linear(x[:, -1], lm_head).cpu()

    def decode(token: torch.Tensor, caches, pos: int):
        x = F.embedding(token, emb)
        for i, layer_weights in enumerate(layers):
            x = layer(layer_weights, x, caches[i][0], caches[i][1], pos, 1, False)
        x = F.layer_norm(x, (D_MODEL,), lnf_g, lnf_b, 1e-5)
        return F.linear(x[:, -1], lm_head).cpu()

    prompts = torch.tensor(
        [[(i * 37 + 11) % VOCAB for i in range(BATCH * PREFILL)]],
        device=device, dtype=torch.long,
    ).view(BATCH, PREFILL)

    def fresh_caches():
        return [cache() for _ in range(N_LAYERS)]

    # Warmup.
    for _ in range(5):
        caches = fresh_caches()
        logits = prefill(prompts, caches)
        token = logits.argmax(-1, keepdim=True).to(device)
        decode(token, caches, PREFILL)

    torch.cuda.synchronize()
    prefill_times = []
    for _ in range(20):
        caches = fresh_caches()
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = prefill(prompts, caches)
        torch.cuda.synchronize()
        prefill_times.append(time.perf_counter() - start)
    ttft = statistics.median(prefill_times)

    decode_times = []
    for _ in range(5):
        caches = fresh_caches()
        logits = prefill(prompts, caches)
        token = logits.argmax(-1, keepdim=True).to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(DECODE_TOKENS):
            logits = decode(token, caches, PREFILL + step)
            token = logits.argmax(-1, keepdim=True).to(device)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - start)
    decode_time = statistics.median(decode_times)

    generated = BATCH * DECODE_TOKENS
    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"batch: {BATCH}")
    print(f"prefill_len: {PREFILL}")
    print(f"prefill tokens: {BATCH * PREFILL}")
    print(f"TTFT (batched prefill): {ttft * 1e3:.3f} ms")
    print(f"prefill throughput: {BATCH * PREFILL / ttft:.1f} tok/s")
    print(f"decode throughput: {generated / decode_time:.1f} tok/s "
          f"({decode_time / DECODE_TOKENS * 1e3:.3f} ms per batched step)")
    print(f"per-sequence decode: {generated / decode_time / BATCH:.1f} tok/s")


if __name__ == "__main__":
    main()
