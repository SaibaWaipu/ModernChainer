use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use modern_chainer_core::{Backend, DType, Error, Result, Rng, Tensor};

use crate::sampling::{Sampler, SamplingConfig, sample_token_with_config};

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: Sampler,
    pub timeout: Option<Duration>,
    pub stream: bool,
}

#[derive(Clone, Debug)]
pub struct GenerationOutput {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub ttft_ms: Option<f64>,
    pub latency_ms: f64,
}

#[derive(Clone, Debug)]
pub struct QueueStats {
    pub pending: usize,
    pub active: usize,
    pub completed: usize,
    pub failed: usize,
    pub throughput_tok_per_s: Option<f64>,
}

#[derive(Debug)]
pub struct InferenceQueue {
    pending: VecDeque<GenerationRequest>,
    active: HashMap<u64, GenerationRequest>,
    completed: Vec<GenerationOutput>,
    failed: Vec<(u64, String)>,
    stats_completed_tokens: usize,
    stats_window_start: std::time::Instant,
}

impl Default for InferenceQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceQueue {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            active: HashMap::new(),
            completed: Vec::new(),
            failed: Vec::new(),
            stats_completed_tokens: 0,
            stats_window_start: std::time::Instant::now(),
        }
    }

    pub fn enqueue(&mut self, req: GenerationRequest) -> Result<()> {
        if req.id == 0 {
            return Err(Error::invalid("request id must be non-zero"));
        }
        if self.pending.iter().any(|r| r.id == req.id) || self.active.contains_key(&req.id) {
            return Err(Error::invalid("duplicate request id"));
        }
        if req.max_tokens == 0 {
            return Err(Error::invalid("max_tokens must be > 0"));
        }
        self.pending.push_back(req);
        Ok(())
    }

    pub fn dequeue(&mut self) -> Option<GenerationRequest> {
        self.pending.pop_front()
    }

    pub fn activate(&mut self, req: GenerationRequest) {
        self.active.insert(req.id, req);
    }

    pub fn finish(&mut self, output: GenerationOutput) {
        self.stats_completed_tokens += output.tokens.len();
        self.completed.push(output);
    }

    pub fn fail(&mut self, id: u64, reason: impl Into<String>) {
        self.failed.push((id, reason.into()));
    }

    pub fn stats(&self) -> QueueStats {
        let elapsed = self.stats_window_start.elapsed().as_secs_f64();
        let tput = if elapsed > 0.0 {
            Some(self.stats_completed_tokens as f64 / elapsed)
        } else {
            None
        };
        QueueStats {
            pending: self.pending.len(),
            active: self.active.len(),
            completed: self.completed.len(),
            failed: self.failed.len(),
            throughput_tok_per_s: tput,
        }
    }

    pub fn completed(&self) -> &[GenerationOutput] {
        &self.completed
    }

    pub fn failed(&self) -> &[(u64, String)] {
        &self.failed
    }
}

/// Byte-level tokenizer with add_eos support.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    pub vocab: HashMap<u8, u32>,
    pub inv: HashMap<u32, u8>,
    pub eos: u32,
    pub unk: u32,
}

impl ByteTokenizer {
    pub fn new(eos_byte: u8, unk_byte: u8) -> Self {
        let mut vocab = HashMap::new();
        let mut inv = HashMap::new();
        for b in 0u16..256 {
            let mut id = (b as u32) + 1;
            if b == eos_byte as u16 {
                id = 0;
            } else if b == unk_byte as u16 && b != eos_byte as u16 {
                id = 1;
            }
            vocab.insert(b as u8, id);
            inv.insert(id, b as u8);
        }
        Self {
            vocab,
            inv,
            eos: 0,
            unk: 1,
        }
    }

    pub fn encode(&self, text: &str, add_eos: bool) -> Vec<u32> {
        let mut out: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|&b| *self.vocab.get(&b).unwrap_or(&self.unk))
            .collect();
        if add_eos {
            out.push(self.eos);
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|id| self.inv.get(id).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// KV cache quantisation config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvQuant {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
}

impl KvQuant {
    pub fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Self::F32),
            DType::F16 => Some(Self::F16),
            DType::BF16 => Some(Self::Bf16),
            _ => None,
        }
    }
    pub fn memory_factor(&self) -> f64 {
        match self {
            Self::F32 => 1.0,
            Self::F16 | Self::Bf16 => 0.5,
            Self::Fp8E4m3 => 0.25,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PagedKvConfig {
    pub block_size: usize,
    pub max_blocks: usize,
    pub quant: KvQuant,
}

impl Default for PagedKvConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks: 1024,
            quant: KvQuant::F32,
        }
    }
}

pub fn quantize_f16(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| modern_chainer_core::f32_to_f16_bits(v))
        .collect()
}

pub fn dequantize_f16(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| modern_chainer_core::f16_bits_to_f32(b))
        .collect()
}

pub fn dedup_tokens(history: &[u32], no_repeat_ngram: usize) -> HashSet<u32> {
    let mut banned = HashSet::new();
    if no_repeat_ngram == 0 || history.len() < no_repeat_ngram {
        return banned;
    }
    let n = no_repeat_ngram;
    for i in 0..=history.len() - n {
        let window = &history[i..i + n];
        for j in 0..history.len() - n {
            if history[j..j + n] == *window && j != i {
                banned.insert(history[history.len() - 1]);
                break;
            }
        }
    }
    banned
}

pub fn apply_sampling_pipeline<B: Backend>(
    logits: &Tensor<B>,
    cfg: &SamplingConfig,
    history: &[u32],
    rng: &mut Rng,
) -> Result<u32> {
    sample_token_with_config(logits, cfg, history, rng)
}

pub fn speculative_filter(
    draft_tokens: &[u32],
    target_probs: &[f32],
    rng: &mut Rng,
) -> Result<Vec<u32>> {
    if draft_tokens.len() != target_probs.len() {
        return Err(Error::invalid("draft/target length mismatch"));
    }
    let mut out = Vec::new();
    for (&tok, &p) in draft_tokens.iter().zip(target_probs.iter()) {
        let accept_p = p.clamp(0.0, 1.0);
        if rng.next_f64() < accept_p as f64 {
            out.push(tok);
        } else {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::top_p_filter;

    #[test]
    fn queue_stats_and_byte_tokenizer() {
        let mut q = InferenceQueue::new();
        let tok = ByteTokenizer::new(0, 1);
        let ids = tok.encode("hi\n", false);
        assert_eq!(tok.decode(&ids), "hi\n");
        q.enqueue(GenerationRequest {
            id: 1,
            prompt: ids,
            max_tokens: 8,
            sampler: Sampler::Greedy,
            timeout: None,
            stream: true,
        })
        .unwrap();
        let mut req = q.dequeue().unwrap();
        q.activate(req.clone());
        let stats = q.stats();
        assert_eq!(stats.active, 1);
        assert_eq!(stats.pending, 0);
        q.finish(GenerationOutput {
            id: req.id,
            tokens: vec![1, 2],
            finished: true,
            ttft_ms: Some(1.0),
            latency_ms: 2.0,
        });
        assert_eq!(q.stats().completed, 1);
        req.max_tokens = 0;
        assert!(q.enqueue(req).is_err());
    }

    #[test]
    fn top_p_keeps_highest() {
        let mut probs = vec![(0.7f32, 0usize), (0.2, 1), (0.1, 2)];
        top_p_filter(&mut probs, 0.8, 0.0).unwrap();
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0].1, 0);
    }

    #[test]
    fn kv_quant_memory() {
        assert_eq!(KvQuant::F32.memory_factor(), 1.0);
        assert_eq!(KvQuant::F16.memory_factor(), 0.5);
    }
}
