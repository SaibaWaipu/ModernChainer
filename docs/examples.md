# Examples

All examples are CPU-first and use synthetic data so they run without dataset
downloads or extra dependencies. Run them from the repository root:

```bash
cargo run --release --example image_classification
cargo run --release --example gan
cargo run --release --example vae
cargo run --release --example vq_vae
cargo run --release --example speech_recognition
cargo run --release --example transformer
```

## Image classification

`examples/image_classification.rs`

Classifies 12×12 synthetic grayscale images containing a horizontal, vertical
or diagonal bar. It builds `data::image::GrayImage` patterns, flattens them and
trains an MLP with softmax cross entropy. This is a dependency-free vision
baseline, not a convolutional network.

## GAN

`examples/gan.rs`

Least-squares GAN on a 2-D ring distribution. Generator and discriminator are
MLPs; the loss is MSE against real/fake targets, so no BCE or sigmoid is needed.
The generated mean radius converges near the target radius of 2.

## VAE

`examples/vae.rs`

Variational autoencoder on the same ring. The encoder predicts `mu` and
`logvar`; sampling uses the reparameterization trick. The loss is reconstruction
MSE plus a `beta * KL` term computed with the new `F::exp` / `F::log` ops.

## VQ-VAE

`examples/vq_vae.rs`

Vector-quantized autoencoder with an 8-entry 2-D codebook. The nearest code is
selected on the host, the straight-through estimator feeds the decoder, and the
codebook is updated with an EMA. The example prints codebook usage.

## Speech recognition

`examples/speech_recognition.rs`

Synthetic two-tone keyword spotting. `data::audio` provides WAV round-tripping,
RMS, zero-crossing rate and Goertzel tone powers. An MLP classifies four
synthetic words from those features.

## Transformer

`examples/transformer.rs`

A single-head Transformer encoder for a toy sequence-classification task. It
uses the new `Embedding`, `LayerNorm`, `F::softmax` and `F::embedding` ops,
scaled dot-product attention, an MLP block and mean pooling via a matmul with a
ones vector. The task is learned to near-perfect accuracy.

## New primitives added for these examples

- `F::exp`, `F::log`, `F::sigmoid`
- `F::softmax`
- `F::layer_norm` with CPU backward
- `F::embedding` with CPU backward
- `Embedding` and `LayerNorm` links
- `modern_chainer::data::image::{GrayImage, PPM}`
- `modern_chainer::data::audio::{Wav, rms, zero_crossing_rate, goertzel_power}`

The new backward kernels are implemented on CPU. CUDA/Vulkan already have
forward embedding/LayerNorm/softmax kernels; their autograd backward is future
work.
