use modern_chainer::cuda;
use modern_chainer::prelude::*;

fn main() -> Result<()> {
    if !cuda::is_available() {
        println!("CUDA is not available: {}", cuda::probe().reason);
        return Ok(());
    }
    let backend = CudaBackend::new(0)?;
    let config = GraphTrainerConfig {
        batch: 32,
        in_features: 16,
        out_features: 4,
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let mut trainer = GraphLinearTrainer::new(backend, config, 7)?;
    let mut x = Vec::with_capacity(32 * 16);
    let mut t = Vec::with_capacity(32 * 4);
    for row in 0..32 {
        let features: Vec<f32> = (0..16)
            .map(|col| ((row * 16 + col) as f32 * 0.01).sin())
            .collect();
        for out in 0..4 {
            let target: f32 = features
                .iter()
                .enumerate()
                .map(|(i, value)| value * ((i + out) as f32 * 0.03).cos())
                .sum();
            t.push(target);
        }
        x.extend_from_slice(&features);
    }
    trainer.set_input(&x)?;
    trainer.set_target(&t)?;
    trainer.capture()?;
    for step in 0..=200 {
        if step % 40 == 0 {
            println!(
                "step {:3}  loss {:.6}",
                trainer.step_count()?,
                trainer.loss()?
            );
        }
        trainer.replay()?;
    }
    trainer.synchronize()?;
    println!("final loss {:.6}", trainer.loss()?);
    Ok(())
}
