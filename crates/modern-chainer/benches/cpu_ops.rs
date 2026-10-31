use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use modern_chainer::Ops;
use modern_chainer::prelude::*;

fn setup(n: usize) -> (Tensor<CpuBackend>, Tensor<CpuBackend>) {
    let backend = CpuBackend::new();
    let a = Tensor::from_f32(backend, &vec![0.5; n], [n]).unwrap();
    let b = Tensor::from_f32(backend, &vec![1.5; n], [n]).unwrap();
    (a, b)
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");
    for n in [1 << 12, 1 << 18, 1 << 22] {
        let (a, b) = setup(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let ops = a.backend().ops();
                black_box(ops.add(&a, &b).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    for (m, k, n) in [
        (64, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 512, 256),
    ] {
        let backend = CpuBackend::new();
        let a = Tensor::from_f32(backend, &vec![0.01; m * k], [m, k]).unwrap();
        let b = Tensor::from_f32(backend, &vec![0.01; k * n], [k, n]).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    let ops = a.backend().ops();
                    black_box(ops.matmul(&a, &b).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn bench_linear_fwd_bwd(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_fwd_bwd");
    for (batch, input, output) in [(32, 128, 128), (128, 784, 256), (256, 512, 512)] {
        let model = L::Linear::new(input, output, 0).unwrap();
        let x = Variable::new(
            Tensor::from_f32(
                CpuBackend::new(),
                &vec![0.01; batch * input],
                [batch, input],
            )
            .unwrap(),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{batch}x{input}x{output}")),
            &(batch, input, output),
            |bench, _| {
                bench.iter(|| {
                    let y = model.forward(&x).unwrap();
                    let loss = F::sum(&y).unwrap();
                    loss.backward().unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise,
    bench_matmul,
    bench_linear_fwd_bwd
);
criterion_main!(benches);
