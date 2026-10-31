use modern_chainer_autograd::Variable;
use modern_chainer_autograd::ops as F;
use modern_chainer_core::Tensor;
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_nn::{Linear as LLinear, Module};
use modern_chainer_optim::{Adam, MomentumSGD, Optimizer, SGD, zero_grad};

fn train<O: Optimizer<CpuBackend>>(mut optimizer: O, steps: usize) -> f32 {
    let model = LLinear::new(1, 1, 0).unwrap();
    for _ in 0..steps {
        let x = Variable::new(
            Tensor::from_f32(CpuBackend::new(), &[1.0, 2.0, 3.0, 4.0], [4, 1]).unwrap(),
        );
        let target = Variable::constant(
            Tensor::from_f32(CpuBackend::new(), &[2.0, 4.0, 6.0, 8.0], [4, 1]).unwrap(),
        );
        let prediction = model.forward(&x).unwrap();
        let diff = F::sub(&prediction, &target).unwrap();
        let loss = F::mean(&F::mul(&diff, &diff).unwrap()).unwrap();
        zero_grad(&model.parameters());
        loss.backward().unwrap();
        optimizer.update(&model.parameters()).unwrap();
        if loss.data().scalar_f64().unwrap() < 1e-4 {
            return loss.data().scalar_f64().unwrap() as f32;
        }
    }
    let x =
        Variable::new(Tensor::from_f32(CpuBackend::new(), &[1.0, 2.0, 3.0, 4.0], [4, 1]).unwrap());
    let target = Variable::constant(
        Tensor::from_f32(CpuBackend::new(), &[2.0, 4.0, 6.0, 8.0], [4, 1]).unwrap(),
    );
    let diff = F::sub(&model.forward(&x).unwrap(), &target).unwrap();
    F::mean(&F::mul(&diff, &diff).unwrap())
        .unwrap()
        .data()
        .scalar_f64()
        .unwrap() as f32
}

#[test]
fn sgd_fits_linear_regression() {
    let loss = train(SGD::new(0.05), 200);
    assert!(loss < 1e-3, "loss = {loss}");
}

#[test]
fn adam_fits_linear_regression() {
    let loss = train(Adam::new(0.05), 1200);
    assert!(loss < 1e-2, "loss = {loss}");
}

#[test]
fn momentum_fits_linear_regression() {
    let loss = train(MomentumSGD::new(0.02, 0.9), 500);
    assert!(loss < 1e-2, "loss = {loss}");
}
