use modern_chainer_autograd::{Variable, ops};
use modern_chainer_core::{Backend, DType, Error, Ops, Result, Tensor};
use modern_chainer_nn::Parameter;

use crate::{Optimizer, zero_grad};

/// Dynamic loss scaler for mixed-precision training.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: u32,
    growth_tracker: u32,
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale.max(1.0);
        self
    }

    pub fn with_growth_interval(mut self, interval: u32) -> Self {
        self.growth_interval = interval.max(1);
        self
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn growth_tracker(&self) -> u32 {
        self.growth_tracker
    }

    pub fn scale_loss<B: Backend>(&self, loss: &Variable<B>) -> Result<Variable<B>> {
        ops::scale(loss, self.scale)
    }

    /// Divide accumulated gradients by the current scale, in place.
    pub fn unscale_<B: Backend>(&self, params: &[Parameter<B>]) -> Result<()> {
        if (self.scale - 1.0).abs() < f64::EPSILON {
            return Ok(());
        }
        let inv = 1.0 / self.scale;
        for parameter in params {
            if let Some(grad) = parameter.variable.grad() {
                let unscaled = grad.backend().ops().scale(&grad, inv)?;
                parameter.variable.set_grad(Some(unscaled));
            }
        }
        Ok(())
    }

    /// Unscale, check for inf/nan, and update.  Returns whether parameters were
    /// updated.
    pub fn step<B: Backend, O: Optimizer<B>>(
        &mut self,
        optimizer: &mut O,
        params: &[Parameter<B>],
    ) -> Result<bool> {
        if self.has_non_finite_grad(params)? {
            self.scale = (self.scale * self.backoff_factor).max(1.0);
            self.growth_tracker = 0;
            zero_grad(params);
            return Ok(false);
        }

        self.unscale_(params)?;
        optimizer.update(params)?;
        self.growth_tracker += 1;
        if self.growth_tracker >= self.growth_interval {
            self.scale *= self.growth_factor;
            self.growth_tracker = 0;
        }
        Ok(true)
    }

    pub fn has_non_finite_grad<B: Backend>(&self, params: &[Parameter<B>]) -> Result<bool> {
        for parameter in params {
            if let Some(grad) = parameter.variable.grad() {
                if grad.dtype() != DType::F32 {
                    return Err(Error::UnsupportedDType(grad.dtype()));
                }
                let bytes = grad.to_bytes()?;
                let (chunks, rest) = bytes.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err(Error::invalid("gradient size is not a multiple of 4"));
                }
                for chunk in chunks {
                    if !f32::from_ne_bytes(*chunk).is_finite() {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience helper: a scaled loss plus the tensor-backed scaler state.
pub fn scaled_loss<B: Backend>(scaler: &GradScaler, loss: &Variable<B>) -> Result<Variable<B>> {
    scaler.scale_loss(loss)
}

/// Convert a gradient tensor to host f32 values (diagnostics only).
pub fn grad_values<B: Backend>(grad: &Tensor<B>) -> Result<Vec<f32>> {
    let bytes = grad.to_bytes()?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::invalid("gradient size is not a multiple of 4"));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect())
}
