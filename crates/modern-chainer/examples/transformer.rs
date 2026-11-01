//! A single-head Transformer encoder for a toy sequence-classification task.
//!
//! The task is: label 1 when the first token is in the lower half of the
//! vocabulary and the last token is in the upper half. The model uses
//! embedding, scaled dot-product attention, LayerNorm, an MLP block, and mean
//! pooling implemented as a matmul with a ones vector.

use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

const SEQ: usize = 8;
const VOCAB: usize = 8;
const D: usize = 16;

fn position_encoding() -> Vec<f32> {
    let mut values = vec![0.0; SEQ * D];
    for pos in 0..SEQ {
        for i in 0..D / 2 {
            let angle = pos as f32 / 10_000f32.powf(2.0 * i as f32 / D as f32);
            values[pos * D + 2 * i] = angle.sin();
            values[pos * D + 2 * i + 1] = angle.cos();
        }
    }
    values
}

fn sample(rng: &mut Rng) -> (Vec<i64>, i64) {
    let tokens: Vec<i64> = (0..SEQ)
        .map(|_| (rng.next_u64() % VOCAB as u64) as i64)
        .collect();
    let label = if tokens[0] < 4 && tokens[SEQ - 1] >= 4 {
        1
    } else {
        0
    };
    (tokens, label)
}

fn host_f32(tensor: &Tensor<CpuBackend>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

struct Transformer {
    emb: Embedding<CpuBackend>,
    pos: Variable<CpuBackend>,
    ones: Variable<CpuBackend>,
    wq: Linear<CpuBackend>,
    wk: Linear<CpuBackend>,
    wv: Linear<CpuBackend>,
    wo: Linear<CpuBackend>,
    ln1: LayerNorm<CpuBackend>,
    w1: Linear<CpuBackend>,
    w2: Linear<CpuBackend>,
    ln2: LayerNorm<CpuBackend>,
    head: Linear<CpuBackend>,
}

impl Transformer {
    fn new() -> Result<Self> {
        Ok(Self {
            emb: Embedding::new(VOCAB, D, 1)?,
            pos: Variable::constant(Tensor::from_f32(
                CpuBackend::new(),
                &position_encoding(),
                [SEQ, D],
            )?),
            ones: Variable::constant(Tensor::from_f32(CpuBackend::new(), &[1.0; SEQ], [1, SEQ])?),
            wq: Linear::new(D, D, 2)?,
            wk: Linear::new(D, D, 3)?,
            wv: Linear::new(D, D, 4)?,
            wo: Linear::new(D, D, 5)?,
            ln1: LayerNorm::new(D, 1e-5)?,
            w1: Linear::new(D, D * 4, 6)?,
            w2: Linear::new(D * 4, D, 7)?,
            ln2: LayerNorm::new(D, 1e-5)?,
            head: Linear::new(D, 2, 8)?,
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
        let q = self.wq.forward(&h)?;
        let k = self.wk.forward(&h)?;
        let v = self.wv.forward(&h)?;
        let scores = F::matmul(&q, &F::transpose(&k)?)?;
        let scores = F::scale(&scores, 1.0 / (D as f64).sqrt())?;
        let attn = F::softmax(&scores)?;
        let context = self.wo.forward(&F::matmul(&attn, &v)?)?;
        let x = self.ln1.forward(&F::add(&h, &context)?)?;
        let ff = self.w2.forward(&F::relu(&self.w1.forward(&x)?)?)?;
        let x = self.ln2.forward(&F::add(&x, &ff)?)?;
        let pooled = F::matmul(&self.ones, &x)?;
        self.head.forward(&pooled)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut params = self.emb.parameters();
        params.extend(self.wq.parameters());
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params.extend(self.ln1.parameters());
        params.extend(self.w1.parameters());
        params.extend(self.w2.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.head.parameters());
        params
    }
}

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(41);
    let model = Transformer::new()?;
    let params = model.parameters();
    let mut optimizer = Adam::new(0.003);

    for step in 0..=600 {
        let (tokens, label) = sample(&mut rng);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &[label],
            [1],
            DType::I64,
        )?);
        let logits = model.forward(&tokens)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 100 == 0 {
            println!(
                "step {step:4}  loss {:.4}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let mut correct = 0;
    let total = 100;
    for _ in 0..total {
        let (tokens, label) = sample(&mut rng);
        let logits = model.forward(&tokens)?;
        let values = host_f32(&logits.data());
        let predicted = if values[1] > values[0] { 1 } else { 0 };
        if predicted == label {
            correct += 1;
        }
    }
    println!("test accuracy: {correct}/{total}");
    Ok(())
}
