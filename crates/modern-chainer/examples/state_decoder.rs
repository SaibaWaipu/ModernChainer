use modern_chainer::prelude::*;
use modern_chainer::sample_token;

fn main() -> Result<()> {
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    };
    let decoder = Decoder::new(CpuBackend::new(), config, 42)?;
    let mut run = Run::new(decoder, CpuBackend::new(), RunMode::Inference, 7)?;

    let prompt = vec![1, 2, 3, 4];
    let mut logits = run.step(prompt.clone())?;
    let handles = run.handles().clone();

    println!("explicit state:");
    for (key, id) in run.plan().iter() {
        println!("  {key}  {id}");
    }
    let k = run.state().tensor(handles.kv.layers[0].k)?;
    println!(
        "layers.0.kv.k shape={:?} bytes={}",
        k.shape().dims(),
        run.metrics().state_bytes
    );

    let mut generated = Vec::new();
    for _ in 0..8 {
        let token = sample_token(&logits, &Sampler::Greedy, run.rng_mut())?;
        generated.push(token);
        logits = run.step(vec![token])?;
    }
    println!("generated: {generated:?}");

    let parent_position = run.state().scalar(handles.kv.position)?.as_i64()?;
    let mut draft = run.fork();
    draft.step(vec![9])?;
    let draft_position = draft.state().scalar(handles.kv.position)?.as_i64()?;
    println!("fork: parent position={parent_position}, draft position={draft_position}");
    Ok(())
}
