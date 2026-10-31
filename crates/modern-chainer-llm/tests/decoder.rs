use modern_chainer_core::{Backend, Tensor};
use modern_chainer_cpu::CpuBackend;
use modern_chainer_llm::{Decoder, DecoderConfig, Sampler, generate};
use modern_chainer_run::{Run, RunMode};

fn config() -> DecoderConfig {
    DecoderConfig {
        vocab: 32,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        d_ff: 16,
        max_seq: 16,
        eps: 1e-5,
    }
}

fn build() -> Decoder<CpuBackend> {
    Decoder::new(CpuBackend::new(), config(), 1).unwrap()
}

fn run() -> Run<CpuBackend, Decoder<CpuBackend>> {
    Run::new(build(), CpuBackend::new(), RunMode::Inference, 9).unwrap()
}

fn values<B: Backend>(tensor: &Tensor<B>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "element {index}: actual {a}, expected {e}"
        );
    }
}

#[test]
fn prefill_matches_incremental_decode() {
    let decoder = build();
    let prompt = [3u32, 7, 11, 5];

    let mut prefill = Run::new(decoder.clone(), CpuBackend::new(), RunMode::Inference, 9).unwrap();
    let full = prefill.step(prompt.to_vec()).unwrap();

    let mut incremental = Run::new(decoder, CpuBackend::new(), RunMode::Inference, 9).unwrap();
    let mut last = None;
    for token in prompt {
        last = Some(incremental.step(vec![token]).unwrap());
    }
    let incremental = last.unwrap();

    assert_close(&values(&full), &values(&incremental), 1e-3);
    assert_eq!(
        prefill
            .state()
            .scalar(prefill.handles().kv.position)
            .unwrap()
            .as_i64()
            .unwrap(),
        prompt.len() as i64
    );
}

#[test]
fn kv_cache_is_visible_and_grows() {
    let mut run = run();
    run.step(vec![1, 2, 3]).unwrap();
    let handles = run.handles().clone();
    let k = run.state().tensor(handles.kv.layers[0].k).unwrap();
    let v = run.state().tensor(handles.kv.layers[0].v).unwrap();
    assert_eq!(k.shape().dims(), &[2, 3, 4]);
    assert_eq!(v.shape().dims(), &[2, 3, 4]);
    assert_eq!(
        run.state()
            .scalar(handles.kv.position)
            .unwrap()
            .as_i64()
            .unwrap(),
        3
    );
    assert!(run.metrics().state_bytes > 0);
    assert!(run.plan().get("layers.0.kv.k").is_some());
    assert!(run.plan().get("kv.position").is_some());

    run.step(vec![4]).unwrap();
    let k = run.state().tensor(handles.kv.layers[0].k).unwrap();
    assert_eq!(k.shape().dims(), &[2, 4, 4]);
}

#[test]
fn fork_isolates_kv_state() {
    let mut parent = run();
    parent.step(vec![1, 2]).unwrap();
    let handles = parent.handles().clone();
    let parent_shape = parent
        .state()
        .tensor(handles.kv.layers[0].k)
        .unwrap()
        .shape()
        .dims()
        .to_vec();

    let mut draft = parent.fork();
    draft.step(vec![3]).unwrap();

    assert_eq!(
        parent
            .state()
            .scalar(handles.kv.position)
            .unwrap()
            .as_i64()
            .unwrap(),
        2
    );
    assert_eq!(
        draft
            .state()
            .scalar(handles.kv.position)
            .unwrap()
            .as_i64()
            .unwrap(),
        3
    );
    assert_eq!(
        parent
            .state()
            .tensor(handles.kv.layers[0].k)
            .unwrap()
            .shape()
            .dims(),
        parent_shape.as_slice()
    );
    assert_eq!(
        draft
            .state()
            .tensor(handles.kv.layers[0].k)
            .unwrap()
            .shape()
            .dims(),
        &[2, 3, 4]
    );
}

#[test]
fn transaction_rolls_back_kv() {
    let mut run = run();
    run.step(vec![1, 2]).unwrap();
    let handles = run.handles().clone();
    {
        let mut tx = run.transaction();
        tx.run_mut().step(vec![3]).unwrap();
        tx.rollback();
    }
    assert_eq!(
        run.state()
            .scalar(handles.kv.position)
            .unwrap()
            .as_i64()
            .unwrap(),
        2
    );
    assert_eq!(
        run.state()
            .tensor(handles.kv.layers[0].k)
            .unwrap()
            .shape()
            .dims(),
        &[2, 2, 4]
    );
}

#[test]
fn greedy_generation_is_deterministic() {
    let decoder = build();
    let mut a = Run::new(decoder.clone(), CpuBackend::new(), RunMode::Inference, 5).unwrap();
    let mut b = Run::new(decoder, CpuBackend::new(), RunMode::Inference, 5).unwrap();
    let first = generate(&mut a, &[1, 2, 3], 5, Sampler::Greedy).unwrap();
    let second = generate(&mut b, &[1, 2, 3], 5, Sampler::Greedy).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 5);
    assert!(first.iter().all(|&token| (token as usize) < 32));
}

#[test]
fn temperature_sampling_stays_in_vocab() {
    let mut run = run();
    let tokens = generate(&mut run, &[1, 2, 3], 6, Sampler::temperature(1.0, 8)).unwrap();
    assert_eq!(tokens.len(), 6);
    assert!(tokens.iter().all(|&token| (token as usize) < 32));
}

#[test]
fn max_sequence_length_is_enforced() {
    let mut run = Run::new(build(), CpuBackend::new(), RunMode::Inference, 1).unwrap();
    run.step(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        .unwrap();
    let error = run.step(vec![17]).unwrap_err();
    assert!(error.to_string().contains("max_seq"));
}
