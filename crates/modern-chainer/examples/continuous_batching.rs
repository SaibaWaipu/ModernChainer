use modern_chainer::prelude::*;

fn main() -> Result<()> {
    let config = DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 32,
        eps: 1e-5,
    };
    let decoder = Decoder::new(CpuBackend::new(), config, 42)?;
    let mut batcher = ContinuousBatcher::new(decoder, CpuBackend::new(), 7);

    let a = batcher.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
    let b = batcher.add_request(vec![4, 5], 3, Sampler::Greedy)?;
    let c = batcher.add_request(vec![6], 6, Sampler::Greedy)?;

    let mut round = 0;
    while batcher.active() > 0 {
        let output = batcher.step()?;
        println!(
            "round {round}: prefill={:?} decode={:?} tokens={:?}",
            output.plan.prefill, output.plan.decode, output.tokens
        );
        round += 1;
    }

    for id in [a, b, c] {
        let state = batcher.state(id).expect("request");
        println!(
            "request {:?}: generated={:?} position={}",
            id,
            state.generated,
            state.position()?
        );
    }
    Ok(())
}
