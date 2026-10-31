use modern_chainer_autograd::Variable;
use modern_chainer_core::Tensor;
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_nn::Parameter;
use modern_chainer_optim::{GradScaler, SGD, zero_grad};

fn param(values: &[f32]) -> Parameter<CpuBackend> {
    Parameter::new(
        "p",
        Variable::new(Tensor::from_f32(CpuBackend::new(), values, [values.len()]).unwrap()),
    )
}

#[test]
fn grad_scaler_unscales_and_updates() {
    let parameter = param(&[1.0]);
    parameter.variable.set_grad(Some(
        Tensor::from_f32(CpuBackend::new(), &[2048.0], [1]).unwrap(),
    ));
    let mut optimizer = SGD::new(1.0);
    let mut scaler = GradScaler::new().with_scale(1024.0);
    assert!(
        scaler
            .step(&mut optimizer, std::slice::from_ref(&parameter))
            .unwrap()
    );
    let value = parameter.variable.data().as_slice::<f32>().unwrap()[0];
    assert!((value - (-1.0)).abs() < 1e-5);
    assert_eq!(scaler.growth_tracker(), 1);
}

#[test]
fn grad_scaler_skips_non_finite_and_backs_off() {
    let parameter = param(&[1.0]);
    parameter.variable.set_grad(Some(
        Tensor::from_f32(CpuBackend::new(), &[f32::INFINITY], [1]).unwrap(),
    ));
    let mut optimizer = SGD::new(1.0);
    let mut scaler = GradScaler::new().with_scale(1024.0);
    assert!(
        !scaler
            .step(&mut optimizer, std::slice::from_ref(&parameter))
            .unwrap()
    );
    assert_eq!(scaler.scale(), 512.0);
    assert!(parameter.variable.grad().is_none());
    let value = parameter.variable.data().as_slice::<f32>().unwrap()[0];
    assert_eq!(value, 1.0);
}

#[test]
fn grad_scaler_grows_after_interval() {
    let parameter = param(&[1.0]);
    parameter.variable.set_grad(Some(
        Tensor::from_f32(CpuBackend::new(), &[0.0], [1]).unwrap(),
    ));
    let mut optimizer = SGD::new(1.0);
    let mut scaler = GradScaler::new().with_scale(1024.0).with_growth_interval(1);
    assert!(
        scaler
            .step(&mut optimizer, std::slice::from_ref(&parameter))
            .unwrap()
    );
    assert_eq!(scaler.scale(), 2048.0);
    zero_grad(std::slice::from_ref(&parameter));
}
