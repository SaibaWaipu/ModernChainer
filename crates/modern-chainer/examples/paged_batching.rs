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
    let mut runner = PagedBatchRunner::new(decoder, CpuBackend::new(), 2, 16, 7)?;

    let a = runner.add_request(vec![1, 2, 3], 5, Sampler::Greedy)?;
    let b = runner.add_request(vec![4, 5], 3, Sampler::Greedy)?;
    let c = runner.add_request(vec![6], 6, Sampler::Greedy)?;

    let mut round = 0;
    while runner.active() > 0 {
        let output = runner.step()?;
        println!(
            "round {round}: plan={:?} tokens={:?}",
            output.plan, output.tokens
        );
        round += 1;
    }
    for id in [a, b, c] {
        let state = runner.state(id).expect("request");
        println!(
            "request {:?}: generated={:?} position={}",
            id,
            state.generated,
            state.position()
        );
    }
    Ok(())
}
