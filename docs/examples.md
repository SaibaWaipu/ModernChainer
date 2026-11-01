# Examples

All examples are CPU-first and use synthetic data so they run without dataset
downloads or extra dependencies. Run them from the repository root:

```bash
cargo run --release --example iris
bash scripts/download_mnist.sh
cargo run --release --example mnist
bash scripts/download_cifar10.sh
cargo run --release --example cifar10
bash scripts/download_wikipedia.sh
cargo run --release --example wikipedia
cargo run --release --example image_classification
cargo run --release --example image_classification_cnn
cargo run --release --example gan
cargo run --release --example vae
cargo run --release --example vq_vae
cargo run --release --example speech_recognition
cargo run --release --example transformer
cargo run --release --example transformer_batch
```

## Iris

`examples/iris.rs`

Loads the committed `data/iris.data`, standardizes the four features, and
trains a 4-16-3 MLP. Typical test accuracy is 29/30.

## MNIST

`examples/mnist.rs`

Downloads via `scripts/download_mnist.sh`, reads the IDX files, and trains a
784-128-10 MLP on a 6000-image subset. It reaches about 88% on 1000 test images
after 300 steps.

## CIFAR-10

`examples/cifar10.rs`

Uses the fast.ai PNG mirror (`scripts/download_cifar10.sh`) and the `image`
crate. A 3072-128-10 MLP baseline reaches about 29% on 1000 test images. This is
a loader/optimization example, not a competitive CIFAR model.

## Wikipedia generation

`examples/wikipedia.rs`

Fetches a few Wikipedia extracts with `scripts/download_wikipedia.sh` (or uses a
built-in sample), trains a character-level causal Transformer for 400 steps and
generates text. The model is intentionally tiny (context 32, d_model 32, one
block), so output is rough but shows the full pipeline.

## Image classification

`examples/image_classification.rs`

Classifies 12×12 synthetic grayscale images containing a horizontal, vertical
or diagonal bar. It builds `data::image::GrayImage` patterns, flattens them and
trains an MLP with softmax cross entropy. This is a dependency-free vision
baseline, not a convolutional network.

## CNN image classification

`examples/image_classification_cnn.rs`

Uses `Conv2d`, `BatchNorm`, `MaxPool2d`, `reshape` and an MLP head. It trains on
the same synthetic 16x16 bar images and reaches 30/30 on the test set.

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

## Batched attention

`examples/transformer_batch.rs`

Runs `MultiHeadAttention` on `[batch, seq, dim]` input, with internal
reshape/flatten projections, and demonstrates `F::reshape` / `F::permute`.

## Data preparation

```bash
bash scripts/download_mnist.sh
bash scripts/download_cifar10.sh
bash scripts/download_wikipedia.sh
```

Iris data is committed. MNIST/CIFAR/Wikipedia files are ignored by git.

## Data utilities

- `data::image::{GrayImage, PatchTokenizer, load_ppm_dir}`
- `data::audio::{Wav, FrameTokenizer, load_wav_dir, tone_features}`
- `data::text::{CharTokenizer, WordTokenizer, load_text_lines}`

## New primitives added for these examples

- `F::exp`, `F::log`, `F::sigmoid`
- `F::softmax`
- `F::layer_norm` with CPU backward
- `F::embedding` with CPU backward
- `Conv2d`, `ConvTranspose2d`, `BatchNorm`, `LayerNorm`, `Embedding`,
  `MultiHeadAttention`, `MaxPool2d`, `AveragePool2d`
- `F::reshape`, `F::permute`, `F::max_pool2d`, `F::average_pool2d`,
  `F::multi_head_attention`
- image/audio/text tokenizers and directory loaders

Convolution, pooling, BatchNorm and attention use correctness-first host
implementations (usable on CPU/CUDA/Vulkan, with gradients). Embedding,
LayerNorm and softmax backward are native CUDA and Vulkan kernels. The same
ops and links are exposed to Python (`F.*`, `L.*`).
