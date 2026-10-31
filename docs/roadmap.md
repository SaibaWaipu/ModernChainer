# Roadmap

## v0.2 — CUDA and Vulkan reductions

- CUDA backend on `cudarc`: dynamic loading, NVRTC kernels, cuBLAS/cuBLASLt,
  im2col + GEMM convolution, memory pool, streams.
- Native Vulkan `sum_all`, `sum_to`, softmax cross entropy and backward.
- CUDA/Vulkan parity tests against CPU; benchmark gates on the existing suite.

## v0.3 — Chainer-shaped training infrastructure

- Python `Trainer`, `Updater`, `Iterator`, `Extension` (LogReport, Evaluator,
  ProgressBar, EarlyStopping, checkpoint).
- `chainer.config` context (train/test, dtype, device, enable_backprop).
- `FunctionNode` subclassing adapter for Python-defined ops.
- Chainer `.npz` reader/writer.

## v0.4 — Performance and precision

- Fused elementwise kernels and bias+activation fusion.
- CUDA Graph capture for static-shape loops.
- fp16/bf16 autocast, loss scaling, mixed-precision optimizers.
- Vulkan cooperative-matrix path where available.
- Close the Linear-backward gap: transposed GEMM kernels, a fused
  `linear_backward` specialization per backend, and a bias+sum fusion.

## v0.5 — Ecosystem

- ONNX export.
- safetensors alongside the native format.
- Vision/NLP example models (MNIST, CIFAR-style CNN, char RNN, small
  Transformer).
- Python type stubs and Sphinx/mdBook documentation site.

## Explicitly deferred

- ChainerX native C++ backend.
- Distributed training (ChainerMN equivalent).
- Full ChainerCV/ChainerRL ports.
- Windows/macOS GPU backends (wgpu already makes Metal/Vulkan portable; not a
  v1 gate).
