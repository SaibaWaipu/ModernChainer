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
