//! Differentiable operations exposed as free functions (`F.*` in Python).

use modern_chainer_core::{Backend, Ops, Result, Tensor};

use crate::node::{Op, apply, apply1};
use crate::variable::Variable;

pub use crate::host_ops::{
    average_pool2d, batch_norm, conv_transpose2d, conv2d, max_pool2d, multi_head_attention,
    permute, reshape,
};

fn check_arity<B: Backend>(inputs: &[&Tensor<B>], expected: usize, name: &str) -> Result<()> {
    if inputs.len() != expected {
        return Err(modern_chainer_core::Error::invalid(format!(
            "{name} expects {expected} inputs, got {}",
            inputs.len()
        )));
    }
    Ok(())
}

pub struct Add;

impl<B: Backend> Op<B> for Add {
    const NAME: &'static str = "Add";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Add")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.add(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![
            Some(ops.sum_to(grad, inputs[0].shape())?),
            Some(ops.sum_to(grad, inputs[1].shape())?),
        ])
    }
}

pub struct Sub;

impl<B: Backend> Op<B> for Sub {
    const NAME: &'static str = "Sub";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Sub")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.sub(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let neg = ops.scale(grad, -1.0)?;
        Ok(vec![
            Some(ops.sum_to(grad, inputs[0].shape())?),
            Some(ops.sum_to(&neg, inputs[1].shape())?),
        ])
    }
}

pub struct Mul;

impl<B: Backend> Op<B> for Mul {
    const NAME: &'static str = "Mul";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Mul")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.mul(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let ga = ops.mul(grad, inputs[1])?;
        let gb = ops.mul(grad, inputs[0])?;
        Ok(vec![
            Some(ops.sum_to(&ga, inputs[0].shape())?),
            Some(ops.sum_to(&gb, inputs[1].shape())?),
        ])
    }
}

pub struct Div;

impl<B: Backend> Op<B> for Div {
    const NAME: &'static str = "Div";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Div")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.div(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let ga = ops.div(grad, inputs[1])?;
        let denom = ops.mul(inputs[1], inputs[1])?;
        let numerator = ops.mul(grad, inputs[0])?;
        let gb = ops.div(&numerator, &denom)?;
        let gb = ops.scale(&gb, -1.0)?;
        Ok(vec![
            Some(ops.sum_to(&ga, inputs[0].shape())?),
            Some(ops.sum_to(&gb, inputs[1].shape())?),
        ])
    }
}

pub struct Sqrt;

impl<B: Backend> Op<B> for Sqrt {
    const NAME: &'static str = "Sqrt";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Sqrt")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.sqrt(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let two_y = ops.scale(outputs[0], 2.0)?;
        Ok(vec![Some(ops.div(grad, &two_y)?)])
    }
}

pub struct Relu;

impl<B: Backend> Op<B> for Relu {
    const NAME: &'static str = "ReLU";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Relu")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.relu(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![Some(ops.relu_backward(inputs[0], grad)?)])
    }
}

pub struct MatMul;

impl<B: Backend> Op<B> for MatMul {
    const NAME: &'static str = "MatMul";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "MatMul")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.matmul(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let a_t = ops.transpose2d(inputs[0])?;
        let b_t = ops.transpose2d(inputs[1])?;
        let ga = ops.matmul(grad, &b_t)?;
        let gb = ops.matmul(&a_t, grad)?;
        Ok(vec![Some(ga), Some(gb)])
    }
}

pub struct LinearOp;

impl<B: Backend> Op<B> for LinearOp {
    const NAME: &'static str = "Linear";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        if inputs.len() != 2 && inputs.len() != 3 {
            return Err(modern_chainer_core::Error::invalid(
                "linear expects (x, weight) or (x, weight, bias)",
            ));
        }
        let ops = inputs[0].backend().ops();
        let bias = inputs.get(2).copied();
        Ok(vec![ops.linear(inputs[0], inputs[1], bias)?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        if inputs.len() == 3 {
            let (grad_x, grad_w, grad_b) =
                ops.linear_backward_with_bias(grad, inputs[0], inputs[1], inputs[2].shape())?;
            return Ok(vec![Some(grad_x), Some(grad_w), Some(grad_b)]);
        }
        let (grad_x, grad_w) = ops.linear_backward(grad, inputs[0], inputs[1])?;
        Ok(vec![Some(grad_x), Some(grad_w)])
    }
}

pub struct Transpose2d;

impl<B: Backend> Op<B> for Transpose2d {
    const NAME: &'static str = "Transpose";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Transpose2d")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.transpose2d(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![Some(ops.transpose2d(grad)?)])
    }
}

pub struct Sum;

impl<B: Backend> Op<B> for Sum {
    const NAME: &'static str = "Sum";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Sum")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.sum_all(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![Some(ops.full_like_scalar(inputs[0], grad, 1.0)?)])
    }
}

pub struct Mean;

impl<B: Backend> Op<B> for Mean {
    const NAME: &'static str = "Mean";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Mean")?;
        let ops = inputs[0].backend().ops();
        let sum = ops.sum_all(inputs[0])?;
        let n = inputs[0].numel().max(1) as f64;
        Ok(vec![ops.scale(&sum, 1.0 / n)?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let factor = 1.0 / inputs[0].numel().max(1) as f64;
        Ok(vec![Some(ops.full_like_scalar(inputs[0], grad, factor)?)])
    }
}

pub struct Scale {
    factor: f64,
}

impl<B: Backend> Op<B> for Scale {
    const NAME: &'static str = "Scale";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Scale")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.scale(inputs[0], self.factor)?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![Some(ops.scale(grad, self.factor)?)])
    }
}

pub struct SoftmaxCrossEntropy;

impl<B: Backend> Op<B> for SoftmaxCrossEntropy {
    const NAME: &'static str = "SoftmaxCrossEntropy";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "SoftmaxCrossEntropy")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![ops.softmax_cross_entropy(inputs[0], inputs[1])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let base = ops.softmax_cross_entropy_backward(inputs[0], inputs[1])?;
        let scalar = ops.to_scalar(grad)?;
        Ok(vec![Some(ops.scale(&base, scalar)?), None])
    }
}

// ---------------------------------------------------------------------------
// Public functional API
// ---------------------------------------------------------------------------

pub fn add<B: Backend>(a: &Variable<B>, b: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(Add, &[a.clone(), b.clone()])?;
    Ok(out.remove(0))
}

pub fn sub<B: Backend>(a: &Variable<B>, b: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(Sub, &[a.clone(), b.clone()])?;
    Ok(out.remove(0))
}

pub fn mul<B: Backend>(a: &Variable<B>, b: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(Mul, &[a.clone(), b.clone()])?;
    Ok(out.remove(0))
}

pub fn div<B: Backend>(a: &Variable<B>, b: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(Div, &[a.clone(), b.clone()])?;
    Ok(out.remove(0))
}

pub fn sqrt<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Sqrt, x)
}

pub fn relu<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Relu, x)
}

pub fn matmul<B: Backend>(a: &Variable<B>, b: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(MatMul, &[a.clone(), b.clone()])?;
    Ok(out.remove(0))
}

pub fn transpose<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Transpose2d, x)
}

pub fn sum<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Sum, x)
}

pub fn mean<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Mean, x)
}

pub fn scale<B: Backend>(x: &Variable<B>, factor: f64) -> Result<Variable<B>> {
    apply1(Scale { factor }, x)
}

pub fn neg<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Scale { factor: -1.0 }, x)
}

pub fn softmax_cross_entropy<B: Backend>(
    logits: &Variable<B>,
    target: &Variable<B>,
) -> Result<Variable<B>> {
    let mut out = apply(SoftmaxCrossEntropy, &[logits.clone(), target.clone()])?;
    Ok(out.remove(0))
}

pub fn linear<B: Backend>(
    x: &Variable<B>,
    weight: &Variable<B>,
    bias: Option<&Variable<B>>,
) -> Result<Variable<B>> {
    let mut inputs = vec![x.clone(), weight.clone()];
    if let Some(bias) = bias {
        inputs.push(bias.clone());
    }
    let mut out = apply(LinearOp, &inputs)?;
    Ok(out.remove(0))
}

pub struct Exp;

impl<B: Backend> Op<B> for Exp {
    const NAME: &'static str = "Exp";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Exp")?;
        Ok(vec![inputs[0].backend().ops().exp(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        Ok(vec![Some(grad.backend().ops().mul(grad, outputs[0])?)])
    }
}

pub struct Log;

impl<B: Backend> Op<B> for Log {
    const NAME: &'static str = "Log";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Log")?;
        Ok(vec![inputs[0].backend().ops().log(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        Ok(vec![Some(ops.div(grad, inputs[0])?)])
    }
}

pub struct Sigmoid;

impl<B: Backend> Op<B> for Sigmoid {
    const NAME: &'static str = "Sigmoid";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Sigmoid")?;
        Ok(vec![inputs[0].backend().ops().sigmoid(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let one = ops.ones_like(outputs[0])?;
        let inv = ops.sub(&one, outputs[0])?;
        let local = ops.mul(outputs[0], &inv)?;
        Ok(vec![Some(ops.mul(grad, &local)?)])
    }
}

pub struct Softmax;

impl<B: Backend> Op<B> for Softmax {
    const NAME: &'static str = "Softmax";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Softmax")?;
        Ok(vec![inputs[0].backend().ops().softmax_last_dim(inputs[0])?])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        Ok(vec![Some(
            grad.backend()
                .ops()
                .softmax_last_dim_backward(inputs[0], grad)?,
        )])
    }
}

pub struct LayerNorm {
    eps: f64,
}

impl LayerNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl<B: Backend> Op<B> for LayerNorm {
    const NAME: &'static str = "LayerNorm";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 3, "LayerNorm")?;
        let ops = inputs[0].backend().ops();
        Ok(vec![
            ops.layer_norm(inputs[0], inputs[1], inputs[2], self.eps)?,
        ])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let (dx, dgamma, dbeta) =
            ops.layer_norm_backward(inputs[0], inputs[1], inputs[2], grad, self.eps)?;
        Ok(vec![Some(dx), Some(dgamma), Some(dbeta)])
    }
}

pub struct Embedding;

impl<B: Backend> Op<B> for Embedding {
    const NAME: &'static str = "Embedding";

    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Embedding")?;
        Ok(vec![
            inputs[0].backend().ops().embedding(inputs[0], inputs[1])?,
        ])
    }

    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let grad_weight = grad
            .backend()
            .ops()
            .embedding_backward(inputs[0], inputs[1], grad)?;
        Ok(vec![None, Some(grad_weight)])
    }
}

pub fn exp<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Exp, x)
}

pub fn log<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Log, x)
}

pub fn sigmoid<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Sigmoid, x)
}

pub fn softmax<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Softmax, x)
}

pub fn layer_norm<B: Backend>(
    x: &Variable<B>,
    gamma: &Variable<B>,
    beta: &Variable<B>,
    eps: f64,
) -> Result<Variable<B>> {
    let mut out = apply(
        LayerNorm::new(eps),
        &[x.clone(), gamma.clone(), beta.clone()],
    )?;
    Ok(out.remove(0))
}

pub fn embedding<B: Backend>(ids: &Variable<B>, weight: &Variable<B>) -> Result<Variable<B>> {
    let mut out = apply(Embedding, &[ids.clone(), weight.clone()])?;
    Ok(out.remove(0))
}

pub struct Silu;

impl<B: Backend> Op<B> for Silu {
    const NAME: &'static str = "SiLU";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "SiLU")?;
        Ok(vec![inputs[0].backend().ops().silu(inputs[0])?])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let sig = ops.sigmoid(inputs[0])?;
        let one = ops.ones_like(inputs[0])?;
        let inv = ops.sub(&one, &sig)?;
        let ds = ops.mul(&sig, &inv)?;
        let x_ds = ops.mul(inputs[0], &ds)?;
        let local = ops.add(&sig, &x_ds)?;
        Ok(vec![Some(ops.mul(grad, &local)?)])
    }
}

pub struct Gelu;

impl<B: Backend> Op<B> for Gelu {
    const NAME: &'static str = "GELU";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "GELU")?;
        Ok(vec![inputs[0].backend().ops().gelu(inputs[0])?])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let x = inputs[0];
        let xx = ops.mul(x, x)?;
        let x3 = ops.mul(&xx, x)?;
        let inner = ops.add(x, &ops.scale(&x3, 0.044715)?)?;
        let c = ops.scale(&inner, (2.0 / std::f64::consts::PI).sqrt())?;
        let tanh = {
            let e2 = ops.exp(&ops.scale(&c, -2.0)?)?;
            let denom = ops.add(&ops.ones_like(&c)?, &e2)?;
            let numer = ops.sub(&ops.ones_like(&c)?, &e2)?;
            ops.div(&numer, &denom)?
        };
        let phi = ops.scale(&ops.add(&ops.ones_like(&c)?, &tanh)?, 0.5)?;
        let dphi_dc = {
            let sech = ops.sub(&ops.ones_like(&c)?, &ops.mul(&tanh, &tanh)?)?;
            ops.scale(&sech, 0.5)?
        };
        let three_x = ops.scale(x, 3.0)?;
        let three_xx = ops.mul(&three_x, x)?;
        let d_inner = ops.add(&ops.ones_like(x)?, &ops.scale(&three_xx, 0.044715)?)?;
        let coeff = ops.scale(&d_inner, (2.0 / std::f64::consts::PI).sqrt())?;
        let x_dphi = ops.mul(x, &dphi_dc)?;
        let local = ops.add(&phi, &ops.mul(&x_dphi, &coeff)?)?;
        Ok(vec![Some(ops.mul(grad, &local)?)])
    }
}

pub struct RmsNorm {
    eps: f64,
}

impl RmsNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl<B: Backend> Op<B> for RmsNorm {
    const NAME: &'static str = "RMSNorm";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "RMSNorm")?;
        Ok(vec![
            inputs[0]
                .backend()
                .ops()
                .rms_norm(inputs[0], inputs[1], self.eps)?,
        ])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let (dx, dw) = grad
            .backend()
            .ops()
            .rms_norm_backward(inputs[0], inputs[1], grad, self.eps)?;
        Ok(vec![Some(dx), Some(dw)])
    }
}

pub struct Clamp {
    min: f64,
    max: f64,
}

impl<B: Backend> Op<B> for Clamp {
    const NAME: &'static str = "Clamp";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Clamp")?;
        Ok(vec![
            inputs[0]
                .backend()
                .ops()
                .clamp(inputs[0], self.min, self.max)?,
        ])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let x = inputs[0];
        let bytes = x.to_bytes()?;
        let (chunks, _) = bytes.as_chunks::<4>();
        let mask: Vec<f32> = chunks
            .iter()
            .map(|c| {
                let v = f32::from_ne_bytes(*c) as f64;
                if v >= self.min && v <= self.max {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_f32(x.backend().clone(), &mask, x.shape().clone())?;
        Ok(vec![Some(ops.mul(grad, &mask)?)])
    }
}

pub struct Abs;

impl<B: Backend> Op<B> for Abs {
    const NAME: &'static str = "Abs";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Abs")?;
        Ok(vec![inputs[0].backend().ops().abs(inputs[0])?])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let x = inputs[0];
        let sign = {
            let bytes = x.to_bytes()?;
            let (chunks, _) = bytes.as_chunks::<4>();
            let mask: Vec<f32> = chunks
                .iter()
                .map(|c| {
                    let v = f32::from_ne_bytes(*c);
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            Tensor::from_f32(x.backend().clone(), &mask, x.shape().clone())?
        };
        Ok(vec![Some(ops.mul(grad, &sign)?)])
    }
}

pub struct Rope {
    theta: f64,
    interleaved: bool,
}

impl Rope {
    pub fn new(theta: f64, interleaved: bool) -> Self {
        Self { theta, interleaved }
    }
}

impl<B: Backend> Op<B> for Rope {
    const NAME: &'static str = "RoPE";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 3, "RoPE")?;
        let (q, k) = inputs[0].backend().ops().apply_rope(
            inputs[0],
            inputs[1],
            inputs[2],
            self.theta,
            self.interleaved,
        )?;
        Ok(vec![q, k])
    }
    fn backward(
        &self,
        _grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        Ok(vec![None, None, None])
    }
}

pub struct Cast {
    dtype: modern_chainer_core::DType,
}

impl Cast {
    pub fn new(dtype: modern_chainer_core::DType) -> Self {
        Self { dtype }
    }
}

impl<B: Backend> Op<B> for Cast {
    const NAME: &'static str = "Cast";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 1, "Cast")?;
        Ok(vec![inputs[0].backend().ops().cast(inputs[0], self.dtype)?])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        _inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        Ok(vec![Some(grad.clone())])
    }
}

pub struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl<B: Backend> Op<B> for Dropout {
    const NAME: &'static str = "Dropout";
    fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
        check_arity(inputs, 2, "Dropout")?;
        Ok(vec![
            inputs[0]
                .backend()
                .ops()
                .dropout(inputs[0], inputs[1], self.p)?,
        ])
    }
    fn backward(
        &self,
        grad: &Tensor<B>,
        inputs: &[&Tensor<B>],
        _outputs: &[&Tensor<B>],
    ) -> Result<Vec<Option<Tensor<B>>>> {
        let ops = grad.backend().ops();
        let scale = 1.0 / (1.0 - self.p);
        let g = ops.mul(grad, inputs[1])?;
        Ok(vec![Some(ops.scale(&g, scale)?), None])
    }
}

pub fn silu<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Silu, x)
}

pub fn gelu<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Gelu, x)
}

pub fn rms_norm<B: Backend>(
    x: &Variable<B>,
    weight: &Variable<B>,
    eps: f64,
) -> Result<Variable<B>> {
    let mut out = apply(RmsNorm::new(eps), &[x.clone(), weight.clone()])?;
    Ok(out.remove(0))
}

pub fn clamp<B: Backend>(x: &Variable<B>, min: f64, max: f64) -> Result<Variable<B>> {
    apply1(Clamp { min, max }, x)
}

pub fn abs<B: Backend>(x: &Variable<B>) -> Result<Variable<B>> {
    apply1(Abs, x)
}

pub fn apply_rope<B: Backend>(
    q: &Variable<B>,
    k: &Variable<B>,
    positions: &Variable<B>,
    theta: f64,
    interleaved: bool,
) -> Result<(Variable<B>, Variable<B>)> {
    let out = apply(
        Rope::new(theta, interleaved),
        &[q.clone(), k.clone(), positions.clone()],
    )?;
    Ok((out[0].clone(), out[1].clone()))
}

pub fn cast<B: Backend>(x: &Variable<B>, dtype: modern_chainer_core::DType) -> Result<Variable<B>> {
    apply1(Cast::new(dtype), x)
}

pub fn dropout<B: Backend>(x: &Variable<B>, mask: &Variable<B>, p: f64) -> Result<Variable<B>> {
    let mut out = apply(Dropout::new(p), &[x.clone(), mask.clone()])?;
    Ok(out.remove(0))
}

pub fn linear_activation<B: Backend>(
    x: &Variable<B>,
    weight: &Variable<B>,
    bias: Option<&Variable<B>>,
    activation: modern_chainer_core::Activation,
) -> Result<Variable<B>> {
    struct LinearActOp {
        activation: modern_chainer_core::Activation,
    }
    impl<B: Backend> Op<B> for LinearActOp {
        const NAME: &'static str = "LinearActivation";
        fn forward(&self, inputs: &[&Tensor<B>]) -> Result<Vec<Tensor<B>>> {
            if inputs.len() != 2 && inputs.len() != 3 {
                return Err(modern_chainer_core::Error::invalid(
                    "linear_activation expects (x, weight[, bias])",
                ));
            }
            let bias = inputs.get(2).copied();
            Ok(vec![inputs[0].backend().ops().linear_activation(
                inputs[0],
                inputs[1],
                bias,
                self.activation,
            )?])
        }
        fn backward(
            &self,
            grad: &Tensor<B>,
            inputs: &[&Tensor<B>],
            outputs: &[&Tensor<B>],
        ) -> Result<Vec<Option<Tensor<B>>>> {
            let ops = grad.backend().ops();
            let pre = ops.linear(inputs[0], inputs[1], inputs.get(2).copied())?;
            let _ = outputs;
            let dy = match self.activation {
                modern_chainer_core::Activation::Relu => ops.relu(&pre)?,
                modern_chainer_core::Activation::Gelu => ops.gelu(&pre)?,
                modern_chainer_core::Activation::Silu => ops.silu(&pre)?,
            };
            let _ = dy;
            let grad_pre = match self.activation {
                modern_chainer_core::Activation::Relu => {
                    let mask = ops.relu_backward(&pre, &ops.ones_like(grad)?)?;
                    let _ = mask;
                    let bytes = pre.to_bytes()?;
                    let (chunks, _) = bytes.as_chunks::<4>();
                    let m: Vec<f32> = chunks
                        .iter()
                        .map(|c| {
                            if f32::from_ne_bytes(*c) > 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let t = Tensor::from_f32(pre.backend().clone(), &m, pre.shape().clone())?;
                    ops.mul(grad, &t)?
                }
                modern_chainer_core::Activation::Gelu => {
                    let x = &pre;
                    let xx = ops.mul(x, x)?;
                    let x3 = ops.mul(&xx, x)?;
                    let inner = ops.add(x, &ops.scale(&x3, 0.044715)?)?;
                    let c = ops.scale(&inner, (2.0 / std::f64::consts::PI).sqrt())?;
                    let e2 = ops.exp(&ops.scale(&c, -2.0)?)?;
                    let denom = ops.add(&ops.ones_like(&c)?, &e2)?;
                    let numer = ops.sub(&ops.ones_like(&c)?, &e2)?;
                    let tanh = ops.div(&numer, &denom)?;
                    let phi = ops.scale(&ops.add(&ops.ones_like(&c)?, &tanh)?, 0.5)?;
                    let sech = ops.sub(&ops.ones_like(&c)?, &ops.mul(&tanh, &tanh)?)?;
                    let dphi_dc = ops.scale(&sech, 0.5)?;
                    let three_x = ops.scale(x, 3.0)?;
                    let three_xx = ops.mul(&three_x, x)?;
                    let d_inner = ops.add(&ops.ones_like(x)?, &ops.scale(&three_xx, 0.044715)?)?;
                    let coeff = ops.scale(&d_inner, (2.0 / std::f64::consts::PI).sqrt())?;
                    let x_dphi = ops.mul(x, &dphi_dc)?;
                    let local = ops.add(&phi, &ops.mul(&x_dphi, &coeff)?)?;
                    ops.mul(grad, &local)?
                }
                modern_chainer_core::Activation::Silu => {
                    let sig = ops.sigmoid(&pre)?;
                    let inv = ops.sub(&ops.ones_like(&pre)?, &sig)?;
                    let ds = ops.mul(&sig, &inv)?;
                    let x_ds = ops.mul(&pre, &ds)?;
                    let local = ops.add(&sig, &x_ds)?;
                    ops.mul(grad, &local)?
                }
            };
            if inputs.len() == 3 {
                let (gx, gw, gb) = ops.linear_backward_with_bias(
                    &grad_pre,
                    inputs[0],
                    inputs[1],
                    inputs[2].shape(),
                )?;
                Ok(vec![Some(gx), Some(gw), Some(gb)])
            } else {
                let (gx, gw) = ops.linear_backward(&grad_pre, inputs[0], inputs[1])?;
                Ok(vec![Some(gx), Some(gw)])
            }
        }
    }
    let mut inputs = vec![x.clone(), weight.clone()];
    if let Some(bias) = bias {
        inputs.push(bias.clone());
    }
    let mut out = apply(LinearActOp { activation }, &inputs)?;
    Ok(out.remove(0))
}
