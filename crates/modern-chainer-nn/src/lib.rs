//! Neural network building blocks (`L.*` in Python).

use modern_chainer_autograd::Variable;
use modern_chainer_autograd::ops;
use modern_chainer_core::{Backend, Error, Ops, Result, Tensor, he_normal};
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

/// RMSNorm over last dim (modern LLM norm).
pub struct RMSNorm<B: Backend> {
    weight: Variable<B>,
    eps: f64,
}

impl RMSNorm<CpuBackend> {
    pub fn new(dim: usize, eps: f64) -> Result<Self> {
        Self::with_backend(CpuBackend::new(), dim, eps)
    }
}

impl<B: Backend> RMSNorm<B> {
    pub fn with_backend(backend: B, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Variable::new(Tensor::from_f32(backend, &vec![1.0; dim], [dim])?),
            eps,
        })
    }
    pub fn weight(&self) -> &Variable<B> {
        &self.weight
    }
}

impl<B: Backend> Module<B> for RMSNorm<B> {
    fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        ops::rms_norm(x, &self.weight, self.eps)
    }
    fn parameters(&self) -> Vec<Parameter<B>> {
        vec![Parameter::new("weight", self.weight.clone())]
    }
}

/// Dropout with externally supplied mask (training scaling 1/(1-p)).
pub struct Dropout {
    pub p: f64,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        if !(0.0..1.0).contains(&p) {
            panic!("Dropout p must be in [0,1), got {p}");
        }
        Self { p, training: true }
    }
    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }
    pub fn forward<B: Backend>(&self, x: &Variable<B>, mask: &Variable<B>) -> Result<Variable<B>> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }
        ops::dropout(x, mask, self.p)
    }
}

/// Grouped-query attention: `heads` query heads, `kv_heads` KV heads.
pub struct GroupedQueryAttention<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    heads: usize,
    kv_heads: usize,
    causal: bool,
}

impl<B: Backend> GroupedQueryAttention<B> {
    pub fn with_backend(
        backend: B,
        dim: usize,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Self> {
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(Error::invalid("heads must be multiple of kv_heads"));
        }
        if !dim.is_multiple_of(heads) {
            return Err(Error::invalid("dim must divide heads"));
        }
        let kv_dim = dim * kv_heads / heads;
        Ok(Self {
            wq: Linear::with_backend(backend.clone(), dim, dim, 1)?,
            wk: Linear::with_backend(backend.clone(), dim, kv_dim, 2)?,
            wv: Linear::with_backend(backend.clone(), dim, kv_dim, 3)?,
            wo: Linear::with_backend(backend, dim, dim, 4)?,
            heads,
            kv_heads,
            causal,
        })
    }
    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        let data = x.data();
        let dims = data.shape().dims();
        if dims.len() != 3 {
            return Err(Error::invalid("GQA expects [N, S, D]"));
        }
        let (batch, seq, dim) = (dims[0], dims[1], dims[2]);
        let hd = dim / self.heads;
        let flat = ops::reshape(x, [batch * seq, dim])?;
        let q2d = self.wq.forward(&flat)?;
        let kv_dim = dim * self.kv_heads / self.heads;
        let k2d = self.wk.forward(&flat)?;
        let v2d = self.wv.forward(&flat)?;
        let group = self.heads / self.kv_heads;
        let expand = |tensor: &Tensor<B>, kv_heads: usize| -> Result<Vec<f32>> {
            let bytes = tensor.to_bytes()?;
            let (chunks, _) = bytes.as_chunks::<4>();
            let values: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
            if group == 1 {
                return Ok(values);
            }
            let mut out = Vec::with_capacity(batch * self.heads * seq * hd);
            for b in 0..batch {
                for h in 0..kv_heads {
                    let base = (b * kv_heads + h) * seq * hd;
                    let slice = &values[base..base + seq * hd];
                    for _ in 0..group {
                        out.extend_from_slice(slice);
                    }
                }
            }
            let _ = kv_dim;
            Ok(out)
        };
        let dq = {
            let bytes = q2d.data().to_bytes()?;
            let (chunks, _) = bytes.as_chunks::<4>();
            chunks
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect::<Vec<f32>>()
        };
        let dk = expand(&k2d.data(), self.kv_heads)?;
        let dv = expand(&v2d.data(), self.kv_heads)?;
        let backend = data.backend().clone();
        let q = Tensor::from_f32(backend.clone(), &dq, [batch * self.heads, seq, hd])?;
        let ke = Tensor::from_f32(backend.clone(), &dk, [batch * self.heads, seq, hd])?;
        let ve = Tensor::from_f32(backend.clone(), &dv, [batch * self.heads, seq, hd])?;
        let qv = Variable::new(q);
        let kv_var = Variable::new(ke);
        let vv_var = Variable::new(ve);
        let scale = 1.0 / (hd as f64).sqrt();
        let out = {
            let ops = backend.ops();
            let att = ops.attention(
                &qv.data(),
                &kv_var.data(),
                &vv_var.data(),
                scale,
                self.causal,
            )?;
            Variable::new(att)
        };
        let flat_out = ops::reshape(&out, [batch * seq, dim])?;
        self.wo.forward(&flat_out)
    }
    pub fn parameters(&self) -> Vec<Parameter<B>> {
        let mut p = self.wq.parameters();
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
    }
}

/// Pre-norm transformer block: RMSNorm -> GQA/MHA -> residual -> RMSNorm -> MLP(SiLU).
pub struct TransformerBlock<B: Backend> {
    pub norm1: RMSNorm<B>,
    pub norm2: RMSNorm<B>,
    pub attn: MultiHeadAttention<B>,
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub eps: f64,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn with_backend(backend: B, dim: usize, heads: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            norm1: RMSNorm::with_backend(backend.clone(), dim, eps)?,
            norm2: RMSNorm::with_backend(backend.clone(), dim, eps)?,
            attn: MultiHeadAttention::with_backend(backend.clone(), dim, heads, true)?,
            w1: Linear::with_backend(backend.clone(), dim, 4 * dim, 11)?,
            w2: Linear::with_backend(backend, 4 * dim, dim, 12)?,
            eps,
        })
    }
    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        let h = self.norm1.forward(x)?;
        let a = self.attn.forward(&h)?;
        let h = ops::add(x, &a)?;
        let n = self.norm2.forward(&h)?;
        let data = n.data();
        let dims = data.shape().dims();
        if dims.len() != 3 {
            return Err(Error::invalid("TransformerBlock expects [B, T, D]"));
        }
        let (batch, seq, dim) = (dims[0], dims[1], dims[2]);
        let flat = ops::reshape(&n, [batch * seq, dim])?;
        let hidden = ops::linear_activation(
            &flat,
            self.w1.weight(),
            Some(self.w1.bias().unwrap()),
            modern_chainer_core::Activation::Silu,
        )?;
        let out = self.w2.forward(&hidden)?;
        let out = ops::reshape(&out, [batch, seq, dim])?;
        ops::add(&h, &out)
    }
    pub fn parameters(&self) -> Vec<Parameter<B>> {
        let mut p = self.norm1.parameters();
        p.extend(self.norm2.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.w1.parameters());
        p.extend(self.w2.parameters());
        p
    }
}

/// LoRA adapter: frozen base linear + trainable low-rank A/B.
pub struct LoRA<B: Backend> {
    base: Linear<B>,
    a: Variable<B>,
    b: Variable<B>,
    alpha: f64,
}

impl<B: Backend> LoRA<B> {
    pub fn with_backend(
        backend: B,
        base_in: usize,
        base_out: usize,
        rank: usize,
        alpha: f64,
        seed: u64,
    ) -> Result<Self> {
        if rank == 0 || rank > base_in.min(base_out) {
            return Err(Error::invalid("LoRA rank out of range"));
        }
        let base = Linear::with_backend(backend.clone(), base_in, base_out, seed)?;
        let a_data = he_normal([rank, base_in], base_in, seed + 101);
        let a = Variable::new(Tensor::from_f32(backend.clone(), &a_data, [rank, base_in])?);
        let b = Variable::new(Tensor::from_f32(
            backend,
            &vec![0.0; base_out * rank],
            [base_out, rank],
        )?);
        Ok(Self { base, a, b, alpha })
    }
    pub fn forward(&self, x: &Variable<B>) -> Result<Variable<B>> {
        let base = self.base.forward(x)?;
        let scale = self.alpha;
        let low = ops::linear(x, &self.a, None)?;
        let delta = ops::linear(&low, &self.b, None)?;
        let delta = ops::scale(&delta, scale)?;
        ops::add(&base, &delta)
    }
    pub fn parameters(&self) -> Vec<Parameter<B>> {
        vec![
            Parameter::new("lora_A", self.a.clone()),
            Parameter::new("lora_B", self.b.clone()),
        ]
    }
    pub fn base_parameters(&self) -> Vec<Parameter<B>> {
        self.base.parameters()
    }
}
