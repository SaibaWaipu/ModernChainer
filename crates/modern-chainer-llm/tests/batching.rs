use modern_chainer_core::Result;
use modern_chainer_cpu::CpuBackend;
use modern_chainer_llm::{ContinuousBatcher, Decoder, DecoderConfig, Sampler, generate};
use modern_chainer_run::{Run, RunMode};

fn config() -> DecoderConfig {
    DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 24,
        eps: 1e-5,
    }
}

fn decoder() -> Decoder<CpuBackend> {
    Decoder::new(CpuBackend::new(), config(), 3).unwrap()
}

fn reference(
    model: &Decoder<CpuBackend>,
    prompt: &[u32],
    max_new: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    let mut run = Run::new(model.clone(), CpuBackend::new(), RunMode::Inference, seed)?;
    generate(&mut run, prompt, max_new, Sampler::Greedy)
}

#[test]
fn continuous_batching_matches_individual_generation() {
    let model = decoder();
    let mut batcher = ContinuousBatcher::new(model.clone(), CpuBackend::new(), 7);

    let specs = vec![
        (vec![1u32, 2, 3], 4usize),
        (vec![4u32, 5], 3),
        (vec![6u32], 5),
    ];
    let mut ids = Vec::new();
    for (prompt, max_new) in &specs {
        ids.push(
            batcher
                .add_request(prompt.clone(), *max_new, Sampler::Greedy)
                .unwrap(),
        );
    }

    let outputs = batcher.run_until_finished().unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(batcher.active(), 0);

    for (id, (prompt, max_new)) in ids.iter().zip(specs.iter()) {
        let state = batcher.state(*id).unwrap();
        let expected = reference(&model, prompt, *max_new, 7 + id.0).unwrap();
        assert_eq!(&state.generated, &expected);
        assert_eq!(state.position().unwrap(), prompt.len() + max_new);
    }
}

#[test]
fn requests_added_mid_flight_are_prefilled() {
    let model = decoder();
    let mut batcher = ContinuousBatcher::new(model.clone(), CpuBackend::new(), 11);

    let first = batcher
        .add_request(vec![1, 2, 3], 4, Sampler::Greedy)
        .unwrap();
    let plan = batcher.step().unwrap().plan;
    assert_eq!(plan.prefill, vec![first]);
    assert!(plan.decode.is_empty());

    let plan = batcher.step().unwrap().plan;
    assert!(plan.prefill.is_empty());
    assert_eq!(plan.decode, vec![first]);

    let second = batcher.add_request(vec![9, 8], 3, Sampler::Greedy).unwrap();
    let plan = batcher.step().unwrap().plan;
    assert_eq!(plan.prefill, vec![second]);
    assert_eq!(plan.decode, vec![first]);

    batcher.run_until_finished().unwrap();
    let expected = reference(&model, &[9, 8], 3, 11 + second.0).unwrap();
    assert_eq!(batcher.generated(second).unwrap(), expected.as_slice());
}

#[test]
fn remove_finished_clears_table() {
    let mut batcher = ContinuousBatcher::new(decoder(), CpuBackend::new(), 5);
    batcher.add_request(vec![1, 2], 2, Sampler::Greedy).unwrap();
    batcher.run_until_finished().unwrap();
    assert_eq!(batcher.active(), 0);
    assert_eq!(batcher.len(), 1);
    batcher.remove_finished();
    assert!(batcher.is_empty());
}
