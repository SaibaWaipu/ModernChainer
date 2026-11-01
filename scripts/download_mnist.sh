#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/mnist
base=https://ossci-datasets.s3.amazonaws.com/mnist
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "data/mnist/$name" ]; then
    echo "downloading $name"
    curl -L --fail "$base/$name.gz" | gunzip > "data/mnist/$name"
  fi
done
echo "MNIST ready in data/mnist"
