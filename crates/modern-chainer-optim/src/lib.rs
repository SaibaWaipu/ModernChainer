//! Optimizers (`chainer.optimizers.*`).

use std::collections::HashMap;

use modern_chainer_core::{AdamConfig, Backend, Ops, Result, Tensor};
use modern_chainer_nn::Parameter;

pub trait Optimizer<B: Backend>: Send {
    fn update(&mut self, params: &[Parameter<B>]) -> Result<()>;
    fn name(&self) -> &'static str;
}

pub fn zero_grad<B: Backend>(params: &[Parameter<B>]) {
    for parameter in params {
        parameter.variable.cleargrad();
    }
}

/// Stochastic gradient descent: `p -= lr * g`.
pub struct SGD {
    pub lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl<B: Backend> Optimizer<B> for SGD {
    fn update(&mut self, params: &[Parameter<B>]) -> Result<()> {
        for parameter in params {
            let Some(grad) = parameter.variable.grad() else {
                continue;
            };
            let ops = grad.backend().ops();
            let delta = ops.scale(&grad, -self.lr)?;
            let next = ops.add(&parameter.variable.data(), &delta)?;
            parameter.variable.set_data(next);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SGD"
    }
}

struct AdamState<B: Backend> {
    m: Tensor<B>,
    v: Tensor<B>,
    t: u32,
}

pub struct Adam<B: Backend> {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    state: HashMap<usize, AdamState<B>>,
}

impl<B: Backend> Adam<B> {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            state: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl<B: Backend> Optimizer<B> for Adam<B> {
    fn update(&mut self, params: &[Parameter<B>]) -> Result<()> {
        for parameter in params {
            let Some(grad) = parameter.variable.grad() else {
                continue;
            };
            let ops = grad.backend().ops();
            let key = parameter.variable.id();
            let entry = self.state.entry(key).or_insert_with(|| AdamState {
                m: ops.zeros_like(&grad).expect("zeros_like"),
                v: ops.zeros_like(&grad).expect("zeros_like"),
                t: 0,
            });
            entry.t += 1;
            let t = entry.t;

            let cfg = AdamConfig {
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                step: t,
            };
            let (next, new_m, new_v) =
                ops.adam_update(&parameter.variable.data(), &grad, &entry.m, &entry.v, &cfg)?;
            entry.m = new_m;
            entry.v = new_v;
            parameter.variable.set_data(next);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Adam"
    }
}

pub struct MomentumSGD<B: Backend> {
    pub lr: f64,
    pub momentum: f64,
    velocity: HashMap<usize, Tensor<B>>,
}

impl<B: Backend> MomentumSGD<B> {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl<B: Backend> Optimizer<B> for MomentumSGD<B> {
    fn update(&mut self, params: &[Parameter<B>]) -> Result<()> {
        for parameter in params {
            let Some(grad) = parameter.variable.grad() else {
                continue;
            };
            let ops = grad.backend().ops();
            let key = parameter.variable.id();
            let velocity = match self.velocity.get(&key) {
                Some(previous) => ops.add(&ops.scale(previous, self.momentum)?, &grad)?,
                None => grad.clone(),
            };
            self.velocity.insert(key, velocity.clone());
            let delta = ops.scale(&velocity, -self.lr)?;
            let next = ops.add(&parameter.variable.data(), &delta)?;
            parameter.variable.set_data(next);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MomentumSGD"
    }
}
