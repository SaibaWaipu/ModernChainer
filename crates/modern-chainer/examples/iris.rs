//! Iris classification with a small MLP.

use modern_chainer::data::iris::load_iris;
use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

fn host_f32(tensor: &Tensor<CpuBackend>) -> Vec<f32> {
    let bytes = tensor.to_bytes().unwrap();
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

fn main() -> Result<()> {
    let mut samples = load_iris("data/iris.data").or_else(|_| load_iris("../data/iris.data"))?;
    let mut rng = Rng::seed_from_u64(3);
    for i in (1..samples.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        samples.swap(i, j);
    }
    let split = 120;
    let train = &samples[..split];
    let test = &samples[split..];

    let mut mean = [0f64; 4];
    let mut var = [0f64; 4];
    for sample in train {
        for (i, value) in mean.iter_mut().enumerate() {
            *value += sample.features[i] as f64;
        }
    }
    for value in &mut mean {
        *value /= train.len() as f64;
    }
    for sample in train {
        for i in 0..4 {
            let d = sample.features[i] as f64 - mean[i];
            var[i] += d * d;
        }
    }
    let std: Vec<f32> = var
        .iter()
        .map(|v| ((v / train.len() as f64).sqrt().max(1e-6)) as f32)
        .collect();
    let normalize = |features: &[f32; 4]| -> Vec<f32> {
        (0..4)
            .map(|i| (features[i] - mean[i] as f32) / std[i])
            .collect()
    };

    let l1 = Linear::new(4, 16, 1)?;
    let l2 = Linear::new(16, 3, 2)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.01);
    let batch = 30;
    for step in 0..=300 {
        let start = (step * batch) % train.len();
        let mut xs = Vec::with_capacity(batch * 4);
        let mut labels = Vec::with_capacity(batch);
        for i in 0..batch {
            let sample = &train[(start + i) % train.len()];
            xs.extend_from_slice(&normalize(&sample.features));
            labels.push(sample.label as i64);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &xs, [batch, 4])?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &labels,
            [batch],
            DType::I64,
        )?);
        let hidden = F::relu(&l1.forward(&x)?)?;
        let logits = l2.forward(&hidden)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 100 == 0 {
            println!(
                "step {step:3}  loss {:.4}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let mut xs = Vec::with_capacity(test.len() * 4);
    for sample in test {
        xs.extend_from_slice(&normalize(&sample.features));
    }
    let x = Variable::constant(Tensor::from_f32(CpuBackend::new(), &xs, [test.len(), 4])?);
    let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
    let values = host_f32(&logits.data());
    let mut correct = 0;
    for (index, sample) in test.iter().enumerate() {
        let row = &values[index * 3..(index + 1) * 3];
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        if predicted == sample.label {
            correct += 1;
        }
    }
    println!("test accuracy: {correct}/{}", test.len());
    Ok(())
}
