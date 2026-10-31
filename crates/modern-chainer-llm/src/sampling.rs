use modern_chainer_core::{Backend, DType, Error, Result, Rng, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Greedy,
    Temperature { temperature: f32, top_k: usize },
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler::Greedy
    }

    pub fn temperature(temperature: f32, top_k: usize) -> Self {
        Sampler::Temperature { temperature, top_k }
    }
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

pub fn sample_token<B: Backend>(
    logits: &Tensor<B>,
    sampler: &Sampler,
    rng: &mut Rng,
) -> Result<u32> {
    let values = logits_values(logits)?;

    let index = match *sampler {
        Sampler::Greedy => return argmax_token(logits),
        Sampler::Temperature { temperature, top_k } => {
            if temperature <= 0.0 || !temperature.is_finite() {
                return Err(Error::invalid("temperature must be positive and finite"));
            }
            let threshold = if top_k > 0 && top_k < values.len() {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                Some(sorted[top_k - 1])
            } else {
                None
            };
            let mut scaled = Vec::with_capacity(values.len());
            let mut max = f32::NEG_INFINITY;
            for &value in &values {
                let value = if threshold.is_some_and(|threshold| value < threshold) {
                    f32::NEG_INFINITY
                } else {
                    value / temperature
                };
                if value > max {
                    max = value;
                }
                scaled.push(value);
            }
            let mut sum = 0f64;
            for &value in &scaled {
                if value != f32::NEG_INFINITY {
                    sum += ((value - max) as f64).exp();
                }
            }
            if sum <= 0.0 {
                return Err(Error::invalid("no logits survived top-k filtering"));
            }
            let mut target = rng.next_f64() * sum;
            let mut chosen = scaled.len() - 1;
            for (candidate, &value) in scaled.iter().enumerate() {
                if value == f32::NEG_INFINITY {
                    continue;
                }
                target -= ((value - max) as f64).exp();
                if target <= 0.0 {
                    chosen = candidate;
                    break;
                }
            }
            chosen
        }
    };

    u32::try_from(index).map_err(|_| Error::invalid("token id exceeds u32"))
}
