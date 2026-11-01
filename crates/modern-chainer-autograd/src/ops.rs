//! Differentiable operations exposed as free functions (`F.*` in Python).

use modern_chainer_core::{Backend, Ops, Result, Tensor};

use crate::node::{Op, apply, apply1};
use crate::variable::Variable;

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
