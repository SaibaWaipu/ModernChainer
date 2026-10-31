use modern_chainer_cpu::CpuBackend;
use modern_chainer_llm::{Decoder, DecoderConfig, Sampler, generate, speculative_generate};
use modern_chainer_run::{Run, RunMode};

fn target_config() -> DecoderConfig {
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

fn draft_config() -> DecoderConfig {
    DecoderConfig {
        vocab: 32,
        d_model: 4,
        n_layers: 1,
        n_heads: 1,
        d_ff: 8,
        max_seq: 24,
        eps: 1e-5,
    }
}

fn target_decoder() -> Decoder<CpuBackend> {
    Decoder::new(CpuBackend::new(), target_config(), 1).unwrap()
}

fn draft_decoder() -> Decoder<CpuBackend> {
    Decoder::new(CpuBackend::new(), draft_config(), 2).unwrap()
}

#[test]
fn speculative_matches_target_greedy() {
    let target_model = target_decoder();
    let draft_model = draft_decoder();
    let prompt = [3u32, 7, 11, 5];
    let max_new = 8;

    let mut target = Run::new(
        target_model.clone(),
        CpuBackend::new(),
        RunMode::Inference,
        21,
    )
    .unwrap();
    let mut draft = Run::new(draft_model, CpuBackend::new(), RunMode::Inference, 22).unwrap();
    let speculative = speculative_generate(&mut target, &mut draft, &prompt, max_new, 3).unwrap();

    let mut reference = Run::new(target_model, CpuBackend::new(), RunMode::Inference, 21).unwrap();
    let expected = generate(&mut reference, &prompt, max_new, Sampler::Greedy).unwrap();

    assert_eq!(speculative, expected);

    let target_position = target
        .state()
        .scalar(target.handles().kv.position)
        .unwrap()
        .as_i64()
        .unwrap();
    let draft_position = draft
        .state()
        .scalar(draft.handles().kv.position)
        .unwrap()
        .as_i64()
        .unwrap();
    assert_eq!(target_position, (prompt.len() + max_new) as i64);
    assert_eq!(target_position, draft_position);
}

#[test]
fn speculative_caps_proposals_at_remaining_tokens() {
    let target_model = target_decoder();
    let draft_model = draft_decoder();
    let mut target = Run::new(
        target_model.clone(),
        CpuBackend::new(),
        RunMode::Inference,
        1,
    )
    .unwrap();
    let mut draft = Run::new(draft_model, CpuBackend::new(), RunMode::Inference, 2).unwrap();
    let tokens = speculative_generate(&mut target, &mut draft, &[1, 2, 3], 2, 8).unwrap();
    let mut reference = Run::new(target_model, CpuBackend::new(), RunMode::Inference, 1).unwrap();
    let expected = generate(&mut reference, &[1, 2, 3], 2, Sampler::Greedy).unwrap();
    assert_eq!(tokens, expected);
    assert_eq!(tokens.len(), 2);
}

#[test]
fn speculative_rejects_different_vocabularies() {
    let target_model = target_decoder();
    let mut target = Run::new(target_model, CpuBackend::new(), RunMode::Inference, 1).unwrap();
    let mut small = draft_config();
    small.vocab = 16;
    let draft_model = Decoder::new(CpuBackend::new(), small, 2).unwrap();
    let mut draft = Run::new(draft_model, CpuBackend::new(), RunMode::Inference, 2).unwrap();
    let error = speculative_generate(&mut target, &mut draft, &[1, 2], 4, 2).unwrap_err();
    assert!(error.to_string().contains("vocabulary"));
}
