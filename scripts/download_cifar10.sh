#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/cifar10
if [ ! -d data/cifar10/train ] || [ ! -d data/cifar10/test ]; then
  echo "downloading CIFAR-10 (fast.ai S3 mirror)"
  curl -L --fail --max-time 600 https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz -o /tmp/cifar10.tgz
  tar -xzf /tmp/cifar10.tgz -C data/cifar10 --strip-components=1
fi
echo "CIFAR-10 ready in data/cifar10"
