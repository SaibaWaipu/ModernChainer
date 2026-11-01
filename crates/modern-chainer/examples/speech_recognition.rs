//! Keyword spotting on synthetic two-tone "words".
//!
//! Demonstrates `modern_chainer::data::audio` WAV round-tripping plus simple
//! Goertzel/RMS/ZCR features and an MLP classifier.

use modern_chainer::data::audio::{self, Wav};
use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

const RATE: u32 = 8_000;
const DURATION: usize = 1_600;
const FREQS: [f32; 8] = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1_000.0];
const WORD_FREQS: [(f32, f32); 4] = [
    (300.0, 900.0),
    (400.0, 800.0),
    (500.0, 700.0),
    (600.0, 1_000.0),
];

fn make_word(label: usize, rng: &mut Rng) -> Vec<f32> {
    let (a, b) = WORD_FREQS[label];
    let mut samples = Vec::with_capacity(DURATION);
    for i in 0..DURATION {
        let t = i as f32 / RATE as f32;
        let envelope = (std::f32::consts::PI * i as f32 / DURATION as f32)
            .sin()
            .max(0.0);
        let tone = 0.5 * (2.0 * std::f32::consts::PI * a * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * b * t).sin();
        samples.push(tone * envelope + rng.normal_vec(1, 0.0, 0.05)[0]);
    }
    samples
}

fn features(samples: &[f32]) -> Vec<f32> {
    audio::tone_features(samples, RATE, &FREQS)
}

fn make_dataset(count: usize, rng: &mut Rng) -> (Vec<f32>, Vec<i64>) {
    let mut xs = Vec::with_capacity(count * 10);
    let mut labels = Vec::with_capacity(count);
    for index in 0..count {
        let label = index % 4;
        xs.extend_from_slice(&features(&make_word(label, rng)));
        labels.push(label as i64);
    }
    (xs, labels)
}

fn host_f32(tensor: &Tensor<CpuBackend>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(31);
    let sample = Wav::new(RATE, make_word(1, &mut rng));
    let decoded = Wav::from_bytes(&sample.to_bytes())?;
    println!(
        "wav roundtrip: {} samples @ {} Hz, rms {:.3}",
        decoded.samples.len(),
        decoded.sample_rate,
        audio::rms(&decoded.samples)
    );

    let (train_x, train_t) = make_dataset(240, &mut rng);
    let (test_x, test_t) = make_dataset(80, &mut rng);
    let l1 = Linear::new(10, 32, 1)?;
    let l2 = Linear::new(32, 4, 2)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.005);
    let batch = 32;

    for step in 0..=400 {
        let start = (step * batch) % (train_x.len() / 10);
        let mut x_values = Vec::with_capacity(batch * 10);
        let mut t_values = Vec::with_capacity(batch);
        for i in 0..batch {
            let index = (start + i) % (train_x.len() / 10);
            x_values.extend_from_slice(&train_x[index * 10..(index + 1) * 10]);
            t_values.push(train_t[index]);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &x_values, [batch, 10])?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &t_values,
            [batch],
            DType::I64,
        )?);
        let hidden = F::relu(&l1.forward(&x)?)?;
        let logits = l2.forward(&hidden)?;
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

    let x = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &test_x,
        [test_x.len() / 10, 10],
    )?);
    let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
    let values = host_f32(&logits.data());
    let mut correct = 0;
    for (index, &label) in test_t.iter().enumerate() {
        let row = &values[index * 4..(index + 1) * 4];
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i64)
            .unwrap();
        if predicted == label {
            correct += 1;
        }
    }
    println!("test accuracy: {correct}/{}", test_t.len());
    Ok(())
}
