use modern_chainer::cuda;
use modern_chainer::llm::{Decoder, DecoderConfig};
use modern_chainer::prelude::*;
use modern_chainer::run::RunMode;

fn main() -> Result<()> {
    if !cuda::is_available() {
        println!("CUDA is not available: {}", cuda::probe().reason);
        return Ok(());
    }
    let backend = CudaBackend::new(0)?;
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let decoder = Decoder::new(backend.clone(), config.clone(), 7)?;
    let model = StaticDecoderStep::from_decoder(decoder)?;
    let mut run = StaticRun::new(model, backend)?;
    let _ = RunMode::Inference;

    let prompt = [1u32, 5, 9, 3];
    for token in prompt {
        run.replay(&StaticDecoderInput { token })?;
    }
    let mut logits = run.output()?.logits;
    let mut generated = Vec::new();
    for _ in 0..8 {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap();
        generated.push(token);
        run.replay(&StaticDecoderInput { token })?;
        logits = run.output()?.logits;
    }
    println!("generated: {generated:?}");
    println!("position: {}", run.output()?.position);
    Ok(())
}
