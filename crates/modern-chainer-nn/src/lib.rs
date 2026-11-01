//! Neural network building blocks (`L.*` in Python).

use modern_chainer_autograd::Variable;
use modern_chainer_autograd::ops;
use modern_chainer_core::{Backend, Error, Result, Tensor, he_normal};
use modern_chainer_cpu::CpuBackend;

#[derive(Clone)]
pub struct Parameter<B: Backend> {
    pub name: String,
    pub variable: Variable<B>,
}

impl<B: Backend> Parameter<B> {
    pub fn new(name: impl Into<String>, variable: Variable<B>) -> Self {
        Self {
            name: name.into(),
            variable,
        }
    }
}

/// A module owns parameters and maps a variable to a variable.
pub trait Module<B: Backend>: Send + Sync {
    fn forward(&self, x: &Variable<B>) -> Result<Variable<B>>;
    fn parameters(&self) -> Vec<Parameter<B>>;
}

/// Fully connected layer with Chainer's `(out, in)` weight layout.
pub struct Linear<B: Backend> {
    weight: Variable<B>,
    bias: Option<Variable<B>>,
    in_features: usize,
    out_features: usize,
}

impl Linear<CpuBackend> {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Result<Self> {
        let backend = CpuBackend::new();
        let weight = Tensor::from_f32(
            backend,
            &he_normal([out_features, in_features], in_features, seed),
            [out_features, in_features],
        )?;
        let bias = Tensor::from_f32(backend, &vec![0.0; out_features], [out_features])?;
        Ok(Self {
            weight: Variable::new(weight),
            bias: Some(Variable::new(bias)),
            in_features,
            out_features,
        })
    }
}

impl<B: Backend> Linear<B> {
    pub fn with_backend(
        backend: B,
        in_features: usize,
        out_features: usize,
        seed: u64,
    ) -> Result<Self> {
        let weight = Tensor::from_f32(
            backend.clone(),
            &he_normal([out_features, in_features], in_features, seed),
            [out_features, in_features],
        )?;
        let bias = Tensor::from_f32(backend, &vec![0.0; out_features], [out_features])?;
        Ok(Self {
            weight: Variable::new(weight),
            bias: Some(Variable::new(bias)),
            in_features,
            out_features,
        })
    }

    pub fn from_parts(
        weight: Variable<B>,
        bias: Option<Variable<B>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if weight.data().shape().dims() != [out_features, in_features] {
            return Err(Error::invalid(format!(
                "Linear weight must be [{out_features}, {in_features}], got {}",
                weight.data().shape()
            )));
        }
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn weight(&self) -> &Variable<B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Variable<B>> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::linear(x, &self.weight, self.bias.as_ref())
    }

    fn parameters(&self) -> Vec<Parameter<B>> {
        let mut params = vec![Parameter::new("W", self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(Parameter::new("b", bias.clone()));
        }
        params
    }
}

/// Layer normalization over the last dimension.
pub struct LayerNorm<B: Backend> {
    gamma: Variable<B>,
    beta: Variable<B>,
    eps: f64,
}

impl LayerNorm<CpuBackend> {
    pub fn new(dim: usize, eps: f64) -> Result<Self> {
        Self::with_backend(CpuBackend::new(), dim, eps)
    }
}

impl<B: Backend> LayerNorm<B> {
    pub fn with_backend(backend: B, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            gamma: Variable::new(Tensor::from_f32(backend.clone(), &vec![1.0; dim], [dim])?),
            beta: Variable::new(Tensor::from_f32(backend, &vec![0.0; dim], [dim])?),
            eps,
        })
    }

    pub fn gamma(&self) -> &Variable<B> {
        &self.gamma
    }

    pub fn beta(&self) -> &Variable<B> {
        &self.beta
    }
}

impl<B: Backend> Module<B> for LayerNorm<B> {
    fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::layer_norm(x, &self.gamma, &self.beta, self.eps)
    }

    fn parameters(&self) -> Vec<Parameter<B>> {
        vec![
            Parameter::new("gamma", self.gamma.clone()),
            Parameter::new("beta", self.beta.clone()),
        ]
    }
}

/// Token embedding table.
pub struct Embedding<B: Backend> {
    weight: Variable<B>,
    vocab: usize,
    dim: usize,
}

impl Embedding<CpuBackend> {
    pub fn new(vocab: usize, dim: usize, seed: u64) -> Result<Self> {
        Self::with_backend(CpuBackend::new(), vocab, dim, seed)
    }
}

impl<B: Backend> Embedding<B> {
    pub fn with_backend(backend: B, vocab: usize, dim: usize, seed: u64) -> Result<Self> {
        let data = he_normal([vocab, dim], dim, seed);
        Ok(Self {
            weight: Variable::new(Tensor::from_f32(backend, &data, [vocab, dim])?),
            vocab,
            dim,
        })
    }

    pub fn weight(&self) -> &Variable<B> {
        &self.weight
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn forward(&self, ids: &Variable<B>) -> Result<Variable<B>> {
        ops::embedding(ids, &self.weight)
    }

    pub fn parameters(&self) -> Vec<Parameter<B>> {
        vec![Parameter::new("W", self.weight.clone())]
    }
}

/// 2-D convolution with `(out_channels, in_channels, kh, kw)` weights.
pub struct Conv2d<B: Backend> {
    weight: Variable<B>,
    bias: Option<Variable<B>>,
    stride: usize,
    padding: usize,
}

impl Conv2d<CpuBackend> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::with_backend(
            CpuBackend::new(),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            seed,
        )
    }
}

impl<B: Backend> Conv2d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn with_backend(
        backend: B,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Result<Self> {
        let fan_in = in_channels * kernel * kernel;
        let data = he_normal([out_channels, in_channels, kernel, kernel], fan_in, seed);
        Ok(Self {
            weight: Variable::new(Tensor::from_f32(
                backend.clone(),
                &data,
                [out_channels, in_channels, kernel, kernel],
            )?),
            bias: Some(Variable::new(Tensor::from_f32(
                backend,
                &vec![0.0; out_channels],
                [out_channels],
            )?)),
            stride,
            padding,
        })
    }

    pub fn weight(&self) -> &Variable<B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Variable<B>> {
        self.bias.as_ref()
    }

    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::conv2d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
        )
    }

    pub fn parameters(&self) -> Vec<Parameter<B>> {
        let mut params = vec![Parameter::new("W", self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(Parameter::new("b", bias.clone()));
        }
        params
    }
}

/// Transposed 2-D convolution with `(in_channels, out_channels, kh, kw)` weights.
pub struct ConvTranspose2d<B: Backend> {
    weight: Variable<B>,
    bias: Option<Variable<B>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl ConvTranspose2d<CpuBackend> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::with_backend(
            CpuBackend::new(),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            output_padding,
            seed,
        )
    }
}

impl<B: Backend> ConvTranspose2d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn with_backend(
        backend: B,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        seed: u64,
    ) -> Result<Self> {
        let fan_in = in_channels * kernel * kernel;
        let data = he_normal([in_channels, out_channels, kernel, kernel], fan_in, seed);
        Ok(Self {
            weight: Variable::new(Tensor::from_f32(
                backend.clone(),
                &data,
                [in_channels, out_channels, kernel, kernel],
            )?),
            bias: Some(Variable::new(Tensor::from_f32(
                backend,
                &vec![0.0; out_channels],
                [out_channels],
            )?)),
            stride,
            padding,
            output_padding,
        })
    }

    pub fn weight(&self) -> &Variable<B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Variable<B>> {
        self.bias.as_ref()
    }

    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::conv_transpose2d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
        )
    }

    pub fn parameters(&self) -> Vec<Parameter<B>> {
        let mut params = vec![Parameter::new("W", self.weight.clone())];
        if let Some(bias) = &self.bias {
            params.push(Parameter::new("b", bias.clone()));
        }
        params
    }
}

/// Channel-wise batch normalization for `[N, C, ...]` inputs.
pub struct BatchNorm<B: Backend> {
    gamma: Variable<B>,
    beta: Variable<B>,
    eps: f64,
}

impl BatchNorm<CpuBackend> {
    pub fn new(channels: usize, eps: f64) -> Result<Self> {
        Self::with_backend(CpuBackend::new(), channels, eps)
    }
}

impl<B: Backend> BatchNorm<B> {
    pub fn with_backend(backend: B, channels: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            gamma: Variable::new(Tensor::from_f32(
                backend.clone(),
                &vec![1.0; channels],
                [channels],
            )?),
            beta: Variable::new(Tensor::from_f32(backend, &vec![0.0; channels], [channels])?),
            eps,
        })
    }

    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::batch_norm(x, &self.gamma, &self.beta, self.eps)
    }

    pub fn parameters(&self) -> Vec<Parameter<B>> {
        vec![
            Parameter::new("gamma", self.gamma.clone()),
            Parameter::new("beta", self.beta.clone()),
        ]
    }
}

/// Self-attention with separate Q/K/V and output projections.
pub struct MultiHeadAttention<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    heads: usize,
    causal: bool,
}

impl MultiHeadAttention<CpuBackend> {
    pub fn new(dim: usize, heads: usize, causal: bool) -> Result<Self> {
        Self::with_backend(CpuBackend::new(), dim, heads, causal)
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn with_backend(backend: B, dim: usize, heads: usize, causal: bool) -> Result<Self> {
        Ok(Self {
            wq: Linear::with_backend(backend.clone(), dim, dim, 1)?,
            wk: Linear::with_backend(backend.clone(), dim, dim, 2)?,
            wv: Linear::with_backend(backend.clone(), dim, dim, 3)?,
            wo: Linear::with_backend(backend, dim, dim, 4)?,
            heads,
            causal,
        })
    }

    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        let data = x.data();
        let dims = data.shape().dims();
        if dims.len() != 3 {
            return Err(Error::invalid("MultiHeadAttention expects [N, S, D]"));
        }
        let (batch, seq, dim) = (dims[0], dims[1], dims[2]);
        let flat = ops::reshape(x, [batch * seq, dim])?;
        let q = ops::reshape(&self.wq.forward(&flat)?, [batch, seq, dim])?;
        let k = ops::reshape(&self.wk.forward(&flat)?, [batch, seq, dim])?;
        let v = ops::reshape(&self.wv.forward(&flat)?, [batch, seq, dim])?;
        let context = ops::multi_head_attention(&q, &k, &v, self.heads, self.causal)?;
        let context = ops::reshape(&context, [batch * seq, dim])?;
        let out = self.wo.forward(&context)?;
        ops::reshape(&out, [batch, seq, dim])
    }

    pub fn parameters(&self) -> Vec<Parameter<B>> {
        let mut params = self.wq.parameters();
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params
    }
}

/// Max pooling over `[N, C, H, W]`.
pub struct MaxPool2d {
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool2d {
    pub fn new(kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel,
            stride,
            padding,
        }
    }

    pub fn forward<B: Backend>(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::max_pool2d(x, self.kernel, self.stride, self.padding)
    }
}

/// Average pooling over `[N, C, H, W]`.
pub struct AveragePool2d {
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl AveragePool2d {
    pub fn new(kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel,
            stride,
            padding,
        }
    }

    pub fn forward<B: Backend>(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::average_pool2d(x, self.kernel, self.stride, self.padding)
    }
}

/// Sequential container.
pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B>>>,
}

impl<B: Backend> Sequential<B> {
    pub fn new(layers: Vec<Box<dyn Module<B>>>) -> Self {
        Self { layers }
    }
}

impl<B: Backend> Module<B> for Sequential<B> {
    fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<Parameter<B>> {
        let mut params = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for parameter in layer.parameters() {
                params.push(Parameter::new(
                    format!("{i}/{}", parameter.name),
                    parameter.variable,
                ));
            }
        }
        params
    }
}

pub use modern_chainer_autograd::ops::{
    add, div, matmul, mean, mul, neg, relu, scale, softmax_cross_entropy, sqrt, sub, sum,
};
