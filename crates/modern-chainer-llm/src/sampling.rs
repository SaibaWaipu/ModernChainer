use modern_chainer_core::{Backend, DType, Error, Result, Rng, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Greedy,
    Temperature {
        temperature: f32,
        top_k: usize,
    },
    TopP {
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    },
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler::Greedy
    }

    pub fn temperature(temperature: f32, top_k: usize) -> Self {
        Sampler::Temperature { temperature, top_k }
    }

    pub fn top_p(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> Self {
        Sampler::TopP {
            temperature,
            top_k,
            top_p,
            min_p,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub epsilon_cutoff: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            epsilon_cutoff: 0.0,
            repetition_penalty: 1.0,
            no_repeat_ngram: 0,
        }
    }
}

pub fn top_p_filter(probs: &mut Vec<(f32, usize)>, top_p: f32, min_p: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&top_p) || !(0.0..=1.0).contains(&min_p) {
        return Err(Error::invalid("top_p/min_p must be in [0,1]"));
    }
    probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = probs.iter().map(|(p, _)| p).sum();
    if total <= 0.0 {
        return Err(Error::invalid("empty distribution"));
    }
    let mut cum = 0.0;
    let max_p = probs.iter().map(|(p, _)| *p).fold(0.0f32, f32::max) / total;
    let min_threshold = max_p * min_p;
    let mut kept = Vec::with_capacity(probs.len());
    for (p, idx) in probs.drain(..) {
        let norm = p / total;
        if norm < min_threshold {
            continue;
        }
        cum += norm;
        kept.push((p, idx));
        if cum >= top_p {
            break;
        }
    }
    if kept.is_empty() {
        return Err(Error::invalid("no tokens survived top-p/min-p"));
    }
    *probs = kept;
    Ok(())
}

fn logits_values<B: Backend>(logits: &Tensor<B>) -> Result<Vec<f32>> {
    if logits.dtype() != DType::F32 {
        return Err(Error::UnsupportedDType(logits.dtype()));
    }
    let bytes = logits.to_bytes()?;
    let (chunks, _) = bytes.as_chunks::<4>();
    let values: Vec<f32> = chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect();
    if values.is_empty() {
        return Err(Error::invalid("cannot sample from empty logits"));
    }
    Ok(values)
}

pub fn argmax_token<B: Backend>(logits: &Tensor<B>) -> Result<u32> {
    let values = logits_values(logits)?;
    let index = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap();
    u32::try_from(index).map_err(|_| Error::invalid("token id exceeds u32"))
}

fn sample_with(
    values: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut Rng,
) -> Result<usize> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(Error::invalid("temperature must be positive and finite"));
    }
    let threshold = if top_k > 0 && top_k < values.len() {
        let mut sorted: Vec<f32> = values.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Some(sorted[top_k - 1])
    } else {
        None
    };
    let mut scaled = Vec::with_capacity(values.len());
    let mut max = f32::NEG_INFINITY;
    for &value in values {
        let value = if threshold.is_some_and(|t| value < t) {
            f32::NEG_INFINITY
        } else {
            value / temperature
        };
        if value > max {
            max = value;
        }
        scaled.push(value);
    }
    let probs: Vec<(f32, usize)> = scaled
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != f32::NEG_INFINITY)
        .map(|(i, &v)| (v, i))
        .collect();
    if probs.is_empty() {
        return Err(Error::invalid("no logits survived top-k filtering"));
    }
    let exps: Vec<f64> = probs
        .iter()
        .map(|(v, _)| ((*v - max) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        return Err(Error::invalid("no logits survived top-k filtering"));
    }
    let normalized: Vec<(f32, usize)> = probs
        .iter()
        .zip(exps.iter())
        .map(|((_, idx), e)| ((*e / sum) as f32, *idx))
        .collect();
    let mut filtered = normalized;
    if top_p < 1.0 || min_p > 0.0 {
        top_p_filter(&mut filtered, top_p, min_p)?;
    }
    let total: f64 = filtered.iter().map(|(p, _)| *p as f64).sum();
    if total <= 0.0 {
        return Err(Error::invalid("no logits survived top-p/min-p"));
    }
    let mut target = rng.next_f64() * total;
    for (p, idx) in &filtered {
        target -= *p as f64;
        if target <= 0.0 {
            return Ok(*idx);
        }
    }
    Ok(filtered.last().unwrap().1)
}

pub fn sample_token<B: Backend>(
    logits: &Tensor<B>,
    sampler: &Sampler,
    rng: &mut Rng,
) -> Result<u32> {
    let values = logits_values(logits)?;
    let index = match *sampler {
        Sampler::Greedy => return argmax_token(logits),
        Sampler::Temperature { temperature, top_k } => {
            sample_with(&values, temperature, top_k, 1.0, 0.0, rng)?
        }
        Sampler::TopP {
            temperature,
            top_k,
            top_p,
            min_p,
        } => sample_with(&values, temperature, top_k, top_p, min_p, rng)?,
    };
    u32::try_from(index).map_err(|_| Error::invalid("token id exceeds u32"))
}

pub fn sample_token_with_config<B: Backend>(
    logits: &Tensor<B>,
    cfg: &SamplingConfig,
    history: &[u32],
    rng: &mut Rng,
) -> Result<u32> {
    let mut values = logits_values(logits)?;
    if (cfg.repetition_penalty - 1.0).abs() > 1e-6 && cfg.repetition_penalty > 0.0 {
        for &id in history {
            if let Some(v) = values.get_mut(id as usize) {
                if *v > 0.0 {
                    *v /= cfg.repetition_penalty;
                } else {
                    *v *= cfg.repetition_penalty;
                }
            }
        }
    }
    if cfg.epsilon_cutoff > 0.0 && cfg.epsilon_cutoff < 1.0 {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let thresh = max - cfg.epsilon_cutoff * 10.0;
        for v in values.iter_mut() {
            if *v < thresh {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    if cfg.no_repeat_ngram > 0 && history.len() >= cfg.no_repeat_ngram {
        let n = cfg.no_repeat_ngram;
        let mut seen_repeat = false;
        for i in 0..=history.len() - n {
            let window = &history[i..i + n];
            for j in 0..history.len() - n {
                if i != j && history[j..j + n] == *window {
                    seen_repeat = true;
                    break;
                }
            }
        }
        if seen_repeat && !values.is_empty() {
            let last = values.len() - 1;
            values[last] = f32::NEG_INFINITY;
        }
    }
    if cfg.temperature <= 0.0 {
        let index = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        return u32::try_from(index).map_err(|_| Error::invalid("token id exceeds u32"));
    }
    let index = sample_with(
        &values,
        cfg.temperature,
        cfg.top_k,
        cfg.top_p,
        cfg.min_p,
        rng,
    )?;
    u32::try_from(index).map_err(|_| Error::invalid("token id exceeds u32"))
}
