//! Optimizers (`chainer.optimizers.*`).

pub mod amp;

use std::collections::HashMap;

use modern_chainer_core::{AdamConfig, Backend, Ops, Result, Tensor};
use modern_chainer_nn::Parameter;

pub use amp::GradScaler;

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

/// AdamW with decoupled weight decay.
pub struct AdamW<B: Backend> {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    state: HashMap<usize, AdamState<B>>,
}

impl<B: Backend> AdamW<B> {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            state: HashMap::new(),
        }
    }
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
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
            let decay = ops.scale(&parameter.variable.data(), self.weight_decay * self.lr)?;
            let next = ops.sub(&next, &decay)?;
            parameter.variable.set_data(next);
        }
        Ok(())
    }
    fn name(&self) -> &'static str {
        "AdamW"
    }
}

/// Global-norm gradient clipping. Returns the pre-clip norm.
pub fn clip_grad_norm<B: Backend>(params: &[Parameter<B>], max_norm: f64) -> Result<f64> {
    if max_norm <= 0.0 {
        return Err(modern_chainer_core::Error::invalid(
            "max_norm must be positive",
        ));
    }
    let mut total = 0f64;
    for parameter in params {
        if let Some(grad) = parameter.variable.grad() {
            let ops = grad.backend().ops();
            let sq = ops.mul(&grad, &grad)?;
            let sum = ops.sum_all(&sq)?;
            let scalar = ops.to_scalar(&sum)?;
            total += scalar;
        }
    }
    let norm = total.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for parameter in params {
            if let Some(grad) = parameter.variable.grad() {
                let scaled = grad.backend().ops().scale(&grad, scale)?;
                parameter.variable.set_grad(Some(scaled));
            }
        }
    }
    Ok(norm)
}

/// Learning rate schedulers (stateless functions of step).
#[derive(Clone, Copy, Debug)]
pub struct LinearWarmupCosineLR {
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub base_lr: f64,
    pub min_lr: f64,
}

impl LinearWarmupCosineLR {
    pub fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            if self.warmup_steps == 0 {
                return self.base_lr;
            }
            return self.base_lr * (step as f64) / (self.warmup_steps as f64);
        }
        if step >= self.total_steps {
            return self.min_lr;
        }
        let denom = (self.total_steps - self.warmup_steps).max(1) as f64;
        let progress = ((step - self.warmup_steps) as f64) / denom;
        let cos = (std::f64::consts::PI * progress).cos() * 0.5 + 0.5;
        self.min_lr + (self.base_lr - self.min_lr) * cos
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepLR {
    pub base_lr: f64,
    pub step_size: u64,
    pub gamma: f64,
}

impl StepLR {
    pub fn get_lr(&self, step: u64) -> f64 {
        let exponent = (step / self.step_size.max(1)) as i32;
        self.base_lr * self.gamma.powi(exponent)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConstantLR {
    pub lr: f64,
}

impl ConstantLR {
    pub fn get_lr(&self, _step: u64) -> f64 {
        self.lr
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PolynomialLR {
    pub base_lr: f64,
    pub total_steps: u64,
    pub power: f64,
    pub min_lr: f64,
}

impl PolynomialLR {
    pub fn get_lr(&self, step: u64) -> f64 {
        if step == 0 {
            return self.base_lr;
        }
        let progress = (step as f64 / self.total_steps.max(1) as f64).clamp(0.0, 1.0);
        self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress).powf(self.power)
    }
}
