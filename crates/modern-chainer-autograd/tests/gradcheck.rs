use modern_chainer_autograd::{Variable, gradcheck, no_grad, ops};
use modern_chainer_core::{Result, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};

fn var(data: &[f32], shape: impl Into<modern_chainer_core::Shape>) -> Variable<CpuBackend> {
    Variable::new(Tensor::from_f32(CpuBackend::new(), data, shape).unwrap())
}

fn check<F>(f: F, inputs: &[Variable<CpuBackend>]) -> Result<()>
where
    F: Fn(&[Variable<CpuBackend>]) -> Result<Variable<CpuBackend>>,
{
    gradcheck(f, inputs, 1e-3, 1e-3, 1e-4)
}

#[test]
fn add_with_broadcast() -> Result<()> {
    let a = var(&[0.2, -0.4, 0.1, 0.3, 0.5, -0.2], [2, 3]);
    let b = var(&[0.1, 0.2, -0.3], [3]);
    check(
        |xs| {
            let y = ops::add(&xs[0], &xs[1])?;
            ops::sum(&y)
        },
        &[a, b],
    )
}

#[test]
fn mul_and_div() -> Result<()> {
    let a = var(&[0.5, 1.2, -0.7, 0.3], [2, 2]);
    let b = var(&[0.4, -0.9, 0.8, 1.1], [2, 2]);
    check(
        |xs| {
            let y = ops::mul(&xs[0], &xs[1])?;
            ops::sum(&y)
        },
        &[a.clone(), b.clone()],
    )?;
    check(
        |xs| {
            let y = ops::div(&xs[0], &xs[1])?;
            ops::sum(&y)
        },
        &[a, b],
    )
}

#[test]
fn matmul() -> Result<()> {
    let a = var(&[0.2, -0.1, 0.4, 0.3, 0.2, -0.5], [2, 3]);
    let b = var(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.2], [3, 2]);
    check(
        |xs| {
            let y = ops::matmul(&xs[0], &xs[1])?;
            ops::sum(&y)
        },
        &[a, b],
    )
}

#[test]
fn relu_and_sqrt() -> Result<()> {
    let a = var(&[-0.5, 0.2, 0.7, -1.2], [2, 2]);
    check(
        |xs| {
            let y = ops::relu(&xs[0])?;
            ops::sum(&y)
        },
        std::slice::from_ref(&a),
    )?;
    let b = var(&[0.25, 0.5, 1.0, 2.0], [4]);
    check(
        |xs| {
            let y = ops::sqrt(&xs[0])?;
            ops::sum(&y)
        },
        &[b],
    )
}

#[test]
fn matmul_relu_chain() -> Result<()> {
    let x = var(&[0.2, -0.3, 0.5, 0.1, 0.7, -0.2], [2, 3]);
    let w = var(&[0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [3, 2]);
    let b = var(&[0.1, -0.1], [2]);
    check(
        |xs| {
            let y = ops::matmul(&xs[0], &xs[1])?;
            let y = ops::add(&y, &xs[2])?;
            let y = ops::relu(&y)?;
            ops::sum(&y)
        },
        &[x, w, b],
    )
}

#[test]
fn softmax_cross_entropy() -> Result<()> {
    let logits = var(&[0.2, 1.3, -0.4, 0.5, -1.0, 0.8], [2, 3]);
    let target = Variable::constant(
        Tensor::from_slice(
            CpuBackend::new(),
            &[1i64, 2],
            [2],
            modern_chainer_core::DType::I64,
        )
        .unwrap(),
    );
    check(
        |xs| ops::softmax_cross_entropy(&xs[0], &xs[1]),
        &[logits, target],
    )
}

#[test]
fn gradients_accumulate() -> Result<()> {
    let a = var(&[1.0, 2.0], [2]);
    let y1 = ops::sum(&a)?;
    let y2 = ops::sum(&ops::mul(&a, &a)?)?;
    y1.backward()?;
    y2.backward()?;
    let grad = a.grad().expect("grad");
    let values = grad.as_slice::<f32>().unwrap();
    assert!((values[0] - 3.0).abs() < 1e-6);
    assert!((values[1] - 5.0).abs() < 1e-6);
    a.cleargrad();
    assert!(a.grad().is_none());
    Ok(())
}

#[test]
fn no_grad_skips_graph() -> Result<()> {
    let a = var(&[1.0, -1.0], [2]);
    let y = no_grad(|| ops::relu(&a).unwrap());
    assert!(y.is_leaf());
    assert!(y.grad_fn_name().is_none());
    Ok(())
}

#[test]
fn unchain_backward_truncates() -> Result<()> {
    let a = var(&[1.0, 2.0], [2]);
    let b = ops::mul(&a, &a)?;
    b.unchain_backward();
    b.backward()?;
    assert!(a.grad().is_none());
    Ok(())
}

#[test]
fn fused_linear() -> Result<()> {
    let x = var(&[0.2, -0.3, 0.5, 0.1, 0.7, -0.2], [2, 3]);
    let w = var(&[0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [2, 3]);
    let b = var(&[0.1, -0.1], [2]);
    check(
        |xs| {
            let y = ops::linear(&xs[0], &xs[1], Some(&xs[2]))?;
            ops::sum(&y)
        },
        &[x, w, b],
    )
}

#[test]
fn softmax_and_layer_norm() -> Result<()> {
    let x = var(&[0.2, -0.4, 0.7, 0.1, 0.3, -0.2], [2, 3]);
    check(
        |xs| {
            let y = ops::softmax(&xs[0])?;
            ops::sum(&y)
        },
        std::slice::from_ref(&x),
    )?;
    let gamma = var(&[1.0, 0.9, 1.1], [3]);
    let beta = var(&[0.1, -0.1, 0.0], [3]);
    check(
        |xs| {
            let y = ops::layer_norm(&xs[0], &xs[1], &xs[2], 1e-5)?;
            let y = ops::mul(&y, &y)?;
            ops::sum(&y)
        },
        &[x, gamma, beta],
    )
}

#[test]
fn embedding_gradient() -> Result<()> {
    let weight = var(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], [3, 2]);
    let ids = Variable::constant(
        Tensor::from_slice(
            CpuBackend::new(),
            &[0i64, 2, 0],
            [3],
            modern_chainer_core::DType::I64,
        )
        .unwrap(),
    );
    check(
        |xs| {
            let y = ops::embedding(&xs[0], &xs[1])?;
            ops::sum(&y)
        },
        &[ids, weight],
    )
}

#[test]
fn exp_log_sigmoid() -> Result<()> {
    let x = var(&[0.2, -0.1, 0.4, 0.3], [2, 2]);
    check(
        |xs| {
            let y = ops::exp(&xs[0])?;
            ops::sum(&y)
        },
        std::slice::from_ref(&x),
    )?;
    let positive = var(&[0.5, 1.2, 0.7, 2.0], [2, 2]);
    check(
        |xs| {
            let y = ops::log(&xs[0])?;
            ops::sum(&y)
        },
        &[positive],
    )?;
    check(
        |xs| {
            let y = ops::sigmoid(&xs[0])?;
            ops::sum(&y)
        },
        &[x],
    )
}
