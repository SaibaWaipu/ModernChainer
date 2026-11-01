//! Tiny character-level causal Transformer trained on Wikipedia text.
//!
//! Prepare text with `bash scripts/download_wikipedia.sh`, or the example
//! falls back to a built-in sample.

use modern_chainer::data::text::CharTokenizer;
use modern_chainer::prelude::*;
use modern_chainer_core::Rng;
use modern_chainer_cpu::CpuTensorExt;

const SAMPLE: &str = "Wikipedia is a free online encyclopedia written and maintained by a community of volunteers through open collaboration. It is the largest and most-read reference work in history. Articles are written from a neutral point of view and cite reliable sources. Machine learning is a field of study in artificial intelligence concerned with statistical algorithms that learn from data and generalize to unseen data. A neural network is a network of simple units called neurons that can learn patterns from examples. Rust is a general-purpose programming language emphasizing performance, type safety and concurrency.";

const SEQ: usize = 32;
const DIM: usize = 32;
const HEADS: usize = 4;

struct Model {
    emb: Embedding<CpuBackend>,
    pos: Variable<CpuBackend>,
    attn: MultiHeadAttention<CpuBackend>,
    ln1: LayerNorm<CpuBackend>,
    w1: Linear<CpuBackend>,
    w2: Linear<CpuBackend>,
    ln2: LayerNorm<CpuBackend>,
    head: Linear<CpuBackend>,
}

impl Model {
    fn new(vocab: usize) -> Result<Self> {
        let pos_values = (0..SEQ * DIM)
            .map(|i| ((i * 13 % 97) as f32 / 97.0 - 0.5) * 0.2)
            .collect::<Vec<_>>();
        Ok(Self {
            emb: Embedding::new(vocab, DIM, 1)?,
            pos: Variable::constant(Tensor::from_f32(
                CpuBackend::new(),
                &pos_values,
                [SEQ, DIM],
            )?),
            attn: MultiHeadAttention::new(DIM, HEADS, true)?,
            ln1: LayerNorm::new(DIM, 1e-5)?,
            w1: Linear::new(DIM, DIM * 4, 2)?,
            w2: Linear::new(DIM * 4, DIM, 3)?,
            ln2: LayerNorm::new(DIM, 1e-5)?,
            head: Linear::new(DIM, vocab, 4)?,
        })
    }

    fn forward(&self, tokens: &[i64]) -> Result<Variable<CpuBackend>> {
        let ids = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            tokens,
            [SEQ],
            DType::I64,
        )?);
        let h = F::add(&self.emb.forward(&ids)?, &self.pos)?;
        let h3 = F::reshape(&h, [1, SEQ, DIM])?;
        let a = F::reshape(&self.attn.forward(&h3)?, [SEQ, DIM])?;
        let x = self.ln1.forward(&F::add(&h, &a)?)?;
        let ff = self.w2.forward(&F::relu(&self.w1.forward(&x)?)?)?;
        let x = self.ln2.forward(&F::add(&x, &ff)?)?;
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut params = self.emb.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.w1.parameters());
        params.extend(self.w2.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.head.parameters());
        params
    }
}

fn sample_logits(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits
        .iter()
        .map(|v| ((v - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut pick = rng.next_f64() as f32 * total;
    for (i, w) in weights.iter().enumerate() {
        pick -= w;
        if pick <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

fn main() -> Result<()> {
    let text = std::fs::read_to_string("data/wikipedia.txt").unwrap_or_else(|_| SAMPLE.to_string());
    let text: String = text.chars().take(60_000).collect();
    let tokenizer = CharTokenizer::new(&text);
    let tokens: Vec<u32> = tokenizer.encode(&text);
    let vocab = tokenizer.vocab_size();
    println!("chars: {}, vocab: {}", tokens.len(), vocab);

    let model = Model::new(vocab)?;
    let params = model.parameters();
    let mut optimizer = Adam::new(0.003);
    let mut rng = Rng::seed_from_u64(21);
    for step in 0..=400 {
        let start = (rng.next_u64() as usize) % (tokens.len() - SEQ - 1);
        let input: Vec<i64> = tokens[start..start + SEQ]
            .iter()
            .map(|&t| t as i64)
            .collect();
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &tokens[start + 1..start + SEQ + 1]
                .iter()
                .map(|&t| t as i64)
                .collect::<Vec<_>>(),
            [SEQ],
            DType::I64,
        )?);
        let logits = model.forward(&input)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 50 == 0 {
            println!(
                "step {step:3}  loss {:.4}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let mut context: Vec<i64> = tokenizer
        .encode("Wikipedia")
        .into_iter()
        .map(|t| t as i64)
        .collect();
    while context.len() < SEQ {
        context.insert(0, 0);
    }
    context = context[context.len() - SEQ..].to_vec();
    let mut generated = String::new();
    for _ in 0..240 {
        let logits = model.forward(&context)?;
        let values = logits.data().to_vec_f32()?;
        let row = &values[(SEQ - 1) * vocab..SEQ * vocab];
        let next = sample_logits(row, 0.7, &mut rng) as i64;
        context.remove(0);
        context.push(next);
        generated.push_str(&tokenizer.decode(&[next as u32]));
    }
    println!("\n--- generated ---\n{generated}");
    Ok(())
}
