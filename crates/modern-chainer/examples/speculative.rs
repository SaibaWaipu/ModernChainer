use modern_chainer::prelude::*;

fn main() -> Result<()> {
    let target_config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 32,
        eps: 1e-5,
    };
    let draft_config = DecoderConfig {
        vocab: 32,
        d_model: 4,
        n_layers: 1,
        n_heads: 1,
        d_ff: 8,
        max_seq: 32,
        eps: 1e-5,
    };
    let target_model = Decoder::new(CpuBackend::new(), target_config, 1)?;
    let draft_model = Decoder::new(CpuBackend::new(), draft_config, 2)?;
    let prompt = vec![3, 7, 11, 5];

    let mut target = Run::new(
        target_model.clone(),
        CpuBackend::new(),
        RunMode::Inference,
        21,
    )?;
    let mut draft = Run::new(draft_model, CpuBackend::new(), RunMode::Inference, 22)?;
    let speculative = speculative_generate(&mut target, &mut draft, &prompt, 8, 3)?;

    let mut reference = Run::new(target_model, CpuBackend::new(), RunMode::Inference, 21)?;
    let expected = generate(&mut reference, &prompt, 8, Sampler::Greedy)?;

    println!("speculative: {speculative:?}");
    println!("target     : {expected:?}");
    println!(
        "target/draft positions: {}/{}",
        target
            .state()
            .scalar(target.handles().kv.position)?
            .as_i64()?,
        draft
            .state()
            .scalar(draft.handles().kv.position)?
            .as_i64()?
    );
    Ok(())
}
