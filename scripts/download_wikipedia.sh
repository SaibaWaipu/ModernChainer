#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
python3 - <<'PY'
import json
import urllib.parse
import urllib.request

titles = [
    "Wikipedia",
    "Artificial intelligence",
    "Machine learning",
    "Rust (programming language)",
    "Neural network",
]
parts = []
for title in titles:
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
    })
    req = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php?" + params,
        headers={"User-Agent": "modern-chainer-example/0.21"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.load(response)
    for page in data["query"]["pages"].values():
        extract = page.get("extract", "")
        if extract:
            parts.append(extract)
with open("data/wikipedia.txt", "w", encoding="utf-8") as handle:
    handle.write("\n\n".join(parts))
print(f"wrote data/wikipedia.txt ({sum(len(p) for p in parts)} chars)")
PY
