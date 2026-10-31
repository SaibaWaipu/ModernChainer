use modern_chainer::cuda;
use modern_chainer::prelude::*;

fn main() -> Result<()> {
    if !cuda::is_available() {
        println!("CUDA is not available: {}", cuda::probe().reason);
        return Ok(());
    }
    let backend = CudaBackend::new(0)?;
    let config = StaticLinearConfig {
        batch: 32,
        in_features: 16,
        out_features: 4,
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let model = LinearMseAdamStep::new(config, 7)?;
    let mut run = StaticRun::new(model, backend)?;

    let mut x = Vec::with_capacity(32 * 16);
    let mut target = Vec::with_capacity(32 * 4);
    for row in 0..32 {
        let features: Vec<f32> = (0..16)
            .map(|col| ((row * 16 + col) as f32 * 0.01).sin())
            .collect();
        for out in 0..4 {
            let value: f32 = features
                .iter()
                .enumerate()
                .map(|(i, v)| v * ((i + out) as f32 * 0.03).cos())
                .sum();
            target.push(value);
        }
        x.extend_from_slice(&features);
    }
    let input = StaticLinearInput { x, target };
    for step in 0..=100 {
        run.replay(&input)?;
        if step % 20 == 0 {
            let output = run.output()?;
            println!("step {:3}  loss {:.6}", output.step, output.loss);
        }
    }
    Ok(())
}
