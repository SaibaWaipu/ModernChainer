use modern_chainer_autograd::Variable;
use modern_chainer_autograd::ops as F;
use modern_chainer_core::{Backend, DType, Ops, Shape, Tensor};
use modern_chainer_cpu::{CpuBackend, CpuTensorExt};
use modern_chainer_nn::{Linear, Module};
use modern_chainer_optim::{Optimizer, SGD, zero_grad};
use modern_chainer_vulkan::{VulkanBackend, is_available, probe};
use serial_test::serial;

fn host<B: Backend>(tensor: &Tensor<B>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "element {i}: actual {a} expected {e}");
    }
}

fn vf(backend: &VulkanBackend, data: &[f32], shape: impl Into<Shape>) -> Tensor<VulkanBackend> {
    Tensor::from_f32(backend.clone(), data, shape).unwrap()
}

fn cf(data: &[f32], shape: impl Into<Shape>) -> Tensor<CpuBackend> {
    Tensor::from_f32(CpuBackend::new(), data, shape).unwrap()
}

#[test]
#[serial]
fn probe_reports_adapters() {
    let capability = probe();
    eprintln!("vulkan capability: {capability:?}");
}

#[test]
#[serial]
fn elementwise_and_matmul_match_cpu() {
    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let a_data = [0.5, -1.0, 2.0, 3.0, -0.5, 0.25];
    let b_data = [1.5, 0.5, -2.0, 0.25, 4.0, -1.0];
    let a = vf(&backend, &a_data, [2, 3]);
    let b = vf(&backend, &b_data, [2, 3]);
    let ca = cf(&a_data, [2, 3]);
    let cb = cf(&b_data, [2, 3]);

    assert_close(
        &host(&backend.ops().add(&a, &b).unwrap()),
        &host(&CpuBackend::new().ops().add(&ca, &cb).unwrap()),
        1e-5,
    );
    assert_close(
        &host(&backend.ops().mul(&a, &b).unwrap()),
        &host(&CpuBackend::new().ops().mul(&ca, &cb).unwrap()),
        1e-5,
    );
    assert_close(
        &host(&backend.ops().relu(&a).unwrap()),
        &host(&CpuBackend::new().ops().relu(&ca).unwrap()),
        1e-5,
    );
    assert_close(
        &host(&backend.ops().scale(&a, -2.5).unwrap()),
        &host(&CpuBackend::new().ops().scale(&ca, -2.5).unwrap()),
        1e-5,
    );

    let m_data: Vec<f32> = (0..35).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let n_data: Vec<f32> = (0..21).map(|i| 1.0 - (i as f32) * 0.07).collect();
    let m = vf(&backend, &m_data, [5, 7]);
    let n = vf(&backend, &n_data, [7, 3]);
    assert_close(
        &host(&backend.ops().matmul(&m, &n).unwrap()),
        &host(
            &CpuBackend::new()
                .ops()
                .matmul(&cf(&m_data, [5, 7]), &cf(&n_data, [7, 3]))
                .unwrap(),
        ),
        1e-4,
    );

    let a_data: Vec<f32> = (0..64 * 32).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let b_data: Vec<f32> = (0..32 * 40).map(|i| 0.5 - (i as f32) * 0.005).collect();
    assert_close(
        &host(
            &backend
                .ops()
                .matmul(
                    &vf(&backend, &a_data, [64, 32]),
                    &vf(&backend, &b_data, [32, 40]),
                )
                .unwrap(),
        ),
        &host(
            &CpuBackend::new()
                .ops()
                .matmul(&cf(&a_data, [64, 32]), &cf(&b_data, [32, 40]))
                .unwrap(),
        ),
        1e-2,
    );

    let t_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = vf(&backend, &t_data, [2, 3]);
    assert_close(
        &host(&backend.ops().transpose2d(&t).unwrap()),
        &host(
            &CpuBackend::new()
                .ops()
                .transpose2d(&cf(&t_data, [2, 3]))
                .unwrap(),
        ),
        1e-5,
    );
}

#[test]
#[serial]
fn bias_broadcast_matches_cpu() {
    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let x = vf(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let b = vf(&backend, &[0.5, -1.0, 2.0], [3]);
    let expected = CpuBackend::new()
        .ops()
        .add(
            &cf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
            &cf(&[0.5, -1.0, 2.0], [3]),
        )
        .unwrap();
    assert_close(
        &host(&backend.ops().add(&x, &b).unwrap()),
        &host(&expected),
        1e-5,
    );
}

#[test]
#[serial]
fn reductions_match_cpu() {
    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let x = vf(&backend, &[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let sum = backend.ops().sum_all(&x).unwrap();
    assert!((backend.ops().to_scalar(&sum).unwrap() - 10.0).abs() < 1e-5);

    let reduced = backend.ops().sum_to(&x, &Shape::new([2])).unwrap();
    assert_eq!(reduced.shape().dims(), &[2]);
    assert_close(&host(&reduced), &[4.0, 6.0], 1e-5);

    let logits = vf(&backend, &[0.2, 1.3, -0.4, 0.5, -1.0, 0.8], [2, 3]);
    let target = Tensor::from_bytes(
        backend.clone(),
        &[1i64.to_ne_bytes(), 2i64.to_ne_bytes()].concat(),
        [2],
        DType::I64,
    )
    .unwrap();
    let loss = backend
        .ops()
        .softmax_cross_entropy(&logits, &target)
        .unwrap();
    let cpu_loss = CpuBackend::new()
        .ops()
        .softmax_cross_entropy(
            &cf(&[0.2, 1.3, -0.4, 0.5, -1.0, 0.8], [2, 3]),
            &Tensor::from_slice(CpuBackend::new(), &[1i64, 2], [2], DType::I64).unwrap(),
        )
        .unwrap();
    assert!(
        (backend.ops().to_scalar(&loss).unwrap() - cpu_loss.scalar_f64().unwrap()).abs() < 1e-5
    );
}

#[test]
#[serial]
fn linear_training_step_reduces_loss() {
    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let weight = Variable::new(Tensor::from_f32(backend.clone(), &[0.2, -0.4], [1, 2]).unwrap());
    let bias = Variable::new(Tensor::from_f32(backend.clone(), &[0.1], [1]).unwrap());
    let model = Linear::from_parts(weight, Some(bias), 2, 1).unwrap();
    let x = Variable::new(
        Tensor::from_f32(backend.clone(), &[1.0, 2.0, 2.0, 1.0, 3.0, 1.0], [3, 2]).unwrap(),
    );
    let target =
        Variable::constant(Tensor::from_f32(backend.clone(), &[3.0, 4.0, 5.0], [3, 1]).unwrap());
    let mut optimizer = SGD::new(0.05);

    let mut first = None;
    let mut last = 0.0;
    for _ in 0..100 {
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
    eprintln!("vulkan linear loss: {first} -> {last}");
    assert!(last < first, "loss did not decrease: {first} -> {last}");
}

#[test]
#[serial]
fn generic_sum_to_matches_cpu() {
    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 2.0).collect();
    let x = vf(&backend, &data, [2, 3, 4]);
    let cpu = cf(&data, [2, 3, 4]);
    for shape in [vec![3, 1], vec![2, 1, 4], vec![4], vec![2, 3, 4]] {
        let target = Shape::new(shape.clone());
        let actual = backend.ops().sum_to(&x, &target).unwrap();
        let expected = CpuBackend::new().ops().sum_to(&cpu, &target).unwrap();
        assert_eq!(actual.shape().dims(), expected.shape().dims());
        assert_close(&host(&actual), &host(&expected), 1e-5);
    }
}

#[test]
#[serial]
fn backward_embedding_norm_softmax_match_cpu() {
    use modern_chainer_core::{DType, Ops, Tensor};

    if !is_available() {
        eprintln!("skipping: no Vulkan adapter");
        return;
    }
    let backend = VulkanBackend::new().unwrap();
    let ids = Tensor::from_slice(backend.clone(), &[0i64, 2, 0], [3], DType::I64).unwrap();
    let weight = vf(&backend, &[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], [3, 2]);
    let grad = vf(&backend, &[0.2, 0.1, -0.4, 0.3, 0.5, -0.2], [3, 2]);
    let expected = CpuBackend::new()
        .ops()
        .embedding_backward(
            &Tensor::from_slice(CpuBackend::new(), &[0i64, 2, 0], [3], DType::I64).unwrap(),
            &cf(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], [3, 2]),
            &cf(&[0.2, 0.1, -0.4, 0.3, 0.5, -0.2], [3, 2]),
        )
        .unwrap();
    assert_close(
        &host(
            &backend
                .ops()
                .embedding_backward(&ids, &weight, &grad)
                .unwrap(),
        ),
        &host(&expected),
        1e-5,
    );

    let x = cf(&[0.2, -0.4, 0.7, 0.1, 0.3, -0.2], [2, 3]);
    let gamma = cf(&[1.0, 0.9, 1.1], [3]);
    let beta = cf(&[0.1, -0.1, 0.0], [3]);
    let g = cf(&[0.3, -0.2, 0.4, 0.1, 0.5, -0.1], [2, 3]);
    let (cdx, cdg, cdb) = CpuBackend::new()
        .ops()
        .layer_norm_backward(&x, &gamma, &beta, &g, 1e-5)
        .unwrap();
    let (dx, dg, db) = backend
        .ops()
        .layer_norm_backward(
            &vf(&backend, &[0.2, -0.4, 0.7, 0.1, 0.3, -0.2], [2, 3]),
            &vf(&backend, &[1.0, 0.9, 1.1], [3]),
            &vf(&backend, &[0.1, -0.1, 0.0], [3]),
            &vf(&backend, &[0.3, -0.2, 0.4, 0.1, 0.5, -0.1], [2, 3]),
            1e-5,
        )
        .unwrap();
    assert_close(&host(&dx), &host(&cdx), 1e-5);
    assert_close(&host(&dg), &host(&cdg), 1e-5);
    assert_close(&host(&db), &host(&cdb), 1e-5);

    let soft = cf(&[0.2, -0.4, 0.7, 0.1, 0.3, -0.2], [2, 3]);
    let sg = cf(&[0.3, -0.2, 0.4, 0.1, 0.5, -0.1], [2, 3]);
    let expected = CpuBackend::new()
        .ops()
        .softmax_last_dim_backward(&soft, &sg)
        .unwrap();
    assert_close(
        &host(
            &backend
                .ops()
                .softmax_last_dim_backward(
                    &vf(&backend, &[0.2, -0.4, 0.7, 0.1, 0.3, -0.2], [2, 3]),
                    &vf(&backend, &[0.3, -0.2, 0.4, 0.1, 0.5, -0.1], [2, 3]),
                )
                .unwrap(),
        ),
        &host(&expected),
        1e-5,
    );
}
