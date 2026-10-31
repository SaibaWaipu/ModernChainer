use modern_chainer_core::{Backend, DType, Ops, Shape, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_cuda::{CudaBackend, is_available, probe};
use serial_test::serial;

fn values<B: Backend>(tensor: &Tensor<B>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-4, "element {index}: {x} vs {y}");
    }
}

#[test]
#[serial]
fn cuda_core_ops_match_cpu() {
    let capability = probe();
    eprintln!("cuda capability: {capability:?}");
    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).expect("cuda backend");
    eprintln!("cuda device: {}", backend.name());
    let cpu = CpuBackend::new();

    let a_data = [0.5, -1.0, 2.0, 3.0, -0.5, 0.25];
    let b_data = [1.5, 0.5, -2.0, 0.25, 4.0, -1.0];
    let a = Tensor::from_f32(backend.clone(), &a_data, [2, 3]).unwrap();
    let b = Tensor::from_f32(backend.clone(), &b_data, [2, 3]).unwrap();
    let ca = Tensor::from_f32(cpu, &a_data, [2, 3]).unwrap();
    let cb = Tensor::from_f32(cpu, &b_data, [2, 3]).unwrap();

    assert_close(
        &values(&backend.ops().add(&a, &b).unwrap()),
        &values(&cpu.ops().add(&ca, &cb).unwrap()),
    );
    assert_close(
        &values(&backend.ops().mul(&a, &b).unwrap()),
        &values(&cpu.ops().mul(&ca, &cb).unwrap()),
    );
    assert_close(
        &values(&backend.ops().relu(&a).unwrap()),
        &values(&cpu.ops().relu(&ca).unwrap()),
    );
    assert_close(
        &values(&backend.ops().scale(&a, -2.5).unwrap()),
        &values(&cpu.ops().scale(&ca, -2.5).unwrap()),
    );

    let m_data: Vec<f32> = (0..35).map(|i| i as f32 * 0.1 - 1.0).collect();
    let n_data: Vec<f32> = (0..21).map(|i| 1.0 - i as f32 * 0.07).collect();
    let m = Tensor::from_f32(backend.clone(), &m_data, [5, 7]).unwrap();
    let n = Tensor::from_f32(backend.clone(), &n_data, [7, 3]).unwrap();
    let cm = Tensor::from_f32(cpu, &m_data, [5, 7]).unwrap();
    let cn = Tensor::from_f32(cpu, &n_data, [7, 3]).unwrap();
    assert_close(
        &values(&backend.ops().matmul(&m, &n).unwrap()),
        &values(&cpu.ops().matmul(&cm, &cn).unwrap()),
    );

    let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w_data: Vec<f32> = (0..6).map(|i| (i as f32 * 0.13).sin()).collect();
    let bias_data = [0.1, -0.2, 0.3];
    let x = Tensor::from_f32(backend.clone(), &x_data, [4, 2]).unwrap();
    let w = Tensor::from_f32(backend.clone(), &w_data, [3, 2]).unwrap();
    let bias = Tensor::from_f32(backend.clone(), &bias_data, [3]).unwrap();
    let cx = Tensor::from_f32(cpu, &x_data, [4, 2]).unwrap();
    let cw = Tensor::from_f32(cpu, &w_data, [3, 2]).unwrap();
    let cbias = Tensor::from_f32(cpu, &bias_data, [3]).unwrap();
    assert_close(
        &values(&backend.ops().linear(&x, &w, Some(&bias)).unwrap()),
        &values(&cpu.ops().linear(&cx, &cw, Some(&cbias)).unwrap()),
    );

    let scalar = backend.ops().sum_all(&x).unwrap();
    assert!((backend.ops().to_scalar(&scalar).unwrap() - 36.0).abs() < 1e-4);
    assert_eq!(scalar.shape(), &Shape::scalar());

    let target =
        Tensor::from_slice(backend.clone(), &[0i64, 1i64, 2i64, 1i64], [4], DType::I64).unwrap();
    let logits_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.5).collect();
    let logits = Tensor::from_f32(backend.clone(), &logits_data, [4, 3]).unwrap();
    let closs = cpu
        .ops()
        .softmax_cross_entropy(
            &Tensor::from_f32(cpu, &logits_data, [4, 3]).unwrap(),
            &Tensor::from_slice(cpu, &[0i64, 1, 2, 1], [4], DType::I64).unwrap(),
        )
        .unwrap();
    let loss = backend
        .ops()
        .softmax_cross_entropy(&logits, &target)
        .unwrap();
    assert!((backend.ops().to_scalar(&loss).unwrap() - closs.scalar_f64().unwrap()).abs() < 1e-4);
}

#[test]
#[serial]
fn cuda_linear_and_adam_training_step() {
    use modern_chainer_autograd::Variable;
    use modern_chainer_autograd::ops as F;
    use modern_chainer_nn::{Linear, Module};
    use modern_chainer_optim::{Adam, Optimizer, zero_grad};

    if !is_available() {
        eprintln!("skipping: CUDA is not available");
        return;
    }
    let backend = CudaBackend::new(0).unwrap();
    let weight = Variable::new(Tensor::from_f32(backend.clone(), &[0.2, -0.4], [1, 2]).unwrap());
    let bias = Variable::new(Tensor::from_f32(backend.clone(), &[0.1], [1]).unwrap());
    let model = Linear::from_parts(weight, Some(bias), 2, 1).unwrap();
    let x = Variable::new(
        Tensor::from_f32(
            backend.clone(),
            &[1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0],
            [4, 2],
        )
        .unwrap(),
    );
    let target = Variable::constant(
        Tensor::from_f32(backend.clone(), &[3.0, 4.0, 5.0, 6.0], [4, 1]).unwrap(),
    );
    let mut optimizer = Adam::new(0.05);
    let mut first = None;
    let mut last = 0.0;
    for _ in 0..40 {
        let prediction = model.forward(&x).unwrap();
        let diff = F::sub(&prediction, &target).unwrap();
        let loss = F::mean(&F::mul(&diff, &diff).unwrap()).unwrap();
        last = loss.data().backend().ops().to_scalar(&loss.data()).unwrap();
        first.get_or_insert(last);
        zero_grad(&model.parameters());
        loss.backward().unwrap();
        optimizer.update(&model.parameters()).unwrap();
    }
    let first = first.unwrap();
    eprintln!("cuda adam loss: {first} -> {last}");
    assert!(last < first);
}
