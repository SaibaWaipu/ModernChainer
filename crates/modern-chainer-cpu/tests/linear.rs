use modern_chainer_core::{Backend, Ops, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};

#[test]
fn linear_known_values() {
    let b = CpuBackend::new();
    let x = Tensor::from_f32(b, &[1.0, 2.0], [1, 2]).unwrap();
    let w = Tensor::from_f32(b, &[-1.2074714, 0.9428667], [1, 2]).unwrap();
    let bias = Tensor::from_f32(b, &[0.0], [1]).unwrap();
    let y = b.ops().linear(&x, &w, Some(&bias)).unwrap();
    println!("y={:?}", y.to_vec_f32().unwrap());
    assert!((y.to_vec_f32().unwrap()[0] - 0.678262).abs() < 1e-4);
}

#[test]
fn linear_2x3() {
    let b = CpuBackend::new();
    let x = Tensor::from_f32(b, &[0.2, -0.3, 0.5, 0.1, 0.7, -0.2], [2, 3]).unwrap();
    let w = Tensor::from_f32(b, &[0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [2, 3]).unwrap();
    let bias = Tensor::from_f32(b, &[0.1, -0.1], [2]).unwrap();
    let y = b.ops().linear(&x, &w, Some(&bias)).unwrap();
    println!("y={:?}", y.to_vec_f32().unwrap());
}
