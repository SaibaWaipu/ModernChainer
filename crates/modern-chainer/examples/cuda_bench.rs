use std::time::Instant;

use modern_chainer::cuda;
use modern_chainer::prelude::*;

fn bench_cpu(mut f: impl FnMut(), warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn bench_cuda(
    backend: &CudaBackend,
    mut f: impl FnMut(),
    warmup: usize,
    iters: usize,
) -> modern_chainer::Result<f64> {
    for _ in 0..warmup {
        f();
    }
    backend.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    backend.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> modern_chainer::Result<()> {
    if !cuda::is_available() {
        println!("CUDA is not available: {}", cuda::probe().reason);
        return Ok(());
    }
    let cpu = CpuBackend::new();
    let gpu = CudaBackend::new(0)?;
    println!("CUDA device: {}", gpu.name());

    let n = 512;
    let a_data = vec![0.01f32; n * n];
    let b_data = vec![0.02f32; n * n];
    let ca = Tensor::from_f32(cpu, &a_data, [n, n])?;
    let cb = Tensor::from_f32(cpu, &b_data, [n, n])?;
    let ga = Tensor::from_f32(gpu.clone(), &a_data, [n, n])?;
    let gb = Tensor::from_f32(gpu.clone(), &b_data, [n, n])?;
    let cpu_ms = bench_cpu(
        || {
            let _ = cpu.ops().matmul(&ca, &cb).unwrap();
        },
        3,
        20,
    );
    let gpu_ms = bench_cuda(
        &gpu,
        || {
            let _ = gpu.ops().matmul(&ga, &gb).unwrap();
        },
        3,
        20,
    )?;
    println!(
        "matmul {n}^3:      cpu {cpu_ms:.3} ms  cuda {gpu_ms:.3} ms  x{:.2}",
        cpu_ms / gpu_ms
    );

    let (batch, input, output) = (128usize, 784usize, 256usize);
    let x_data = vec![0.01f32; batch * input];
    let w_data = vec![0.001f32; output * input];
    let bias_data = vec![0.0f32; output];
    let cx = Tensor::from_f32(cpu, &x_data, [batch, input])?;
    let cw = Tensor::from_f32(cpu, &w_data, [output, input])?;
    let cbias = Tensor::from_f32(cpu, &bias_data, [output])?;
    let gx = Tensor::from_f32(gpu.clone(), &x_data, [batch, input])?;
    let gw = Tensor::from_f32(gpu.clone(), &w_data, [output, input])?;
    let gbias = Tensor::from_f32(gpu.clone(), &bias_data, [output])?;
    let cpu_ms = bench_cpu(
        || {
            let _ = cpu.ops().linear(&cx, &cw, Some(&cbias)).unwrap();
        },
        3,
        20,
    );
    let gpu_ms = bench_cuda(
        &gpu,
        || {
            let _ = gpu.ops().linear(&gx, &gw, Some(&gbias)).unwrap();
        },
        3,
        20,
    )?;
    println!(
        "linear 128x784x256: cpu {cpu_ms:.3} ms  cuda {gpu_ms:.3} ms  x{:.2}",
        cpu_ms / gpu_ms
    );

    let params = 1 << 20;
    let p_data = vec![0.1f32; params];
    let g_data = vec![0.01f32; params];
    let zero = vec![0.0f32; params];
    let cp = Tensor::from_f32(cpu, &p_data, [params])?;
    let cg = Tensor::from_f32(cpu, &g_data, [params])?;
    let cm = Tensor::from_f32(cpu, &zero, [params])?;
    let cv = Tensor::from_f32(cpu, &zero, [params])?;
    let gp = Tensor::from_f32(gpu.clone(), &p_data, [params])?;
    let gg = Tensor::from_f32(gpu.clone(), &g_data, [params])?;
    let gm = Tensor::from_f32(gpu.clone(), &zero, [params])?;
    let gv = Tensor::from_f32(gpu.clone(), &zero, [params])?;
    let cfg = AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        step: 1,
    };
    let cpu_ms = bench_cpu(
        || {
            let _ = cpu.ops().adam_update(&cp, &cg, &cm, &cv, &cfg).unwrap();
        },
        2,
        10,
    );
    let gpu_ms = bench_cuda(
        &gpu,
        || {
            let _ = gpu.ops().adam_update(&gp, &gg, &gm, &gv, &cfg).unwrap();
        },
        2,
        10,
    )?;
    println!(
        "adam 1M params:    cpu {cpu_ms:.3} ms  cuda {gpu_ms:.3} ms  x{:.2}",
        cpu_ms / gpu_ms
    );
    Ok(())
}
