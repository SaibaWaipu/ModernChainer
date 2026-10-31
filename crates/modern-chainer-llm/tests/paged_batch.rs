use modern_chainer_cpu::CpuBackend;
use modern_chainer_llm::{Decoder, DecoderConfig, PagedBatchRunner, Sampler, generate};
use modern_chainer_run::{Run, RunMode};

fn config() -> DecoderConfig {
    DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 32,
        eps: 1e-5,
    }
}

fn decoder() -> Decoder<CpuBackend> {
    Decoder::new(CpuBackend::new(), config(), 3).unwrap()
}

fn reference(model: &Decoder<CpuBackend>, prompt: &[u32], max_new: usize, seed: u64) -> Vec<u32> {
    let mut run = Run::new(model.clone(), CpuBackend::new(), RunMode::Inference, seed).unwrap();
    generate(&mut run, prompt, max_new, Sampler::Greedy).unwrap()
}

#[test]
fn paged_batched_decode_matches_dense_generation() {
    let model = decoder();
    let mut runner = PagedBatchRunner::new(model.clone(), CpuBackend::new(), 2, 16, 7).unwrap();
    let specs = vec![
        (vec![1u32, 2, 3], 4usize),
        (vec![4u32, 5], 3),
        (vec![6u32], 5),
    ];
    let mut ids = Vec::new();
    for (prompt, max_new) in &specs {
        ids.push(
            runner
                .add_request(prompt.clone(), *max_new, Sampler::Greedy)
                .unwrap(),
        );
    }
    let outputs = runner.run_until_finished().unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(runner.active(), 0);

    for (id, (prompt, max_new)) in ids.iter().zip(specs.iter()) {
        let state = runner.state(*id).unwrap();
        let expected = reference(&model, prompt, *max_new, 7 + id.0);
        assert_eq!(&state.generated, &expected);
        assert_eq!(state.position(), prompt.len() + max_new);
    }
}

#[test]
fn paged_runner_accepts_mid_flight_requests() {
    let model = decoder();
    let mut runner = PagedBatchRunner::new(model.clone(), CpuBackend::new(), 2, 16, 11).unwrap();
    let first = runner
        .add_request(vec![1, 2, 3], 4, Sampler::Greedy)
        .unwrap();
    let output = runner.step().unwrap();
    assert_eq!(output.plan, vec![first]);

    let second = runner.add_request(vec![9, 8], 3, Sampler::Greedy).unwrap();
    let output = runner.step().unwrap();
    assert_eq!(output.plan, vec![first, second]);

    runner.run_until_finished().unwrap();
    let expected = reference(&model, &[9, 8], 3, 11 + second.0);
    assert_eq!(runner.generated(second).unwrap(), expected.as_slice());
}
