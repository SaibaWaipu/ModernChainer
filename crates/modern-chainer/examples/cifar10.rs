//! CIFAR-10 classification with an MLP baseline.
//!
//! Prepare data with `bash scripts/download_cifar10.sh` (fast.ai PNG mirror).

use modern_chainer::data::cifar10::Cifar10;
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
    let train = Cifar10::load_image_dir("data/cifar10", "train")
        .map_err(|e| Error::invalid(format!("{e}; run scripts/download_cifar10.sh")))?;
    let test = Cifar10::load_image_dir("data/cifar10", "test")?;
    let train_count = train.len().min(4_000);
    let test_count = test.len().min(1_000);
    let mut rng = Rng::seed_from_u64(15);
    let mut order: Vec<usize> = (0..train.len()).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut test_order: Vec<usize> = (0..test.len()).collect();
    for i in (1..test_order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        test_order.swap(i, j);
    }

    let l1 = Linear::new(3072, 128, 1)?;
    let l2 = Linear::new(128, 10, 2)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.001);
    let batch = 32;
    for step in 0..=150 {
        let mut xs = Vec::with_capacity(batch * 3072);
        let mut labels = Vec::with_capacity(batch);
        for i in 0..batch {
            let index = order[(step * batch + i) % train_count];
            xs.extend_from_slice(&train.sample(index));
            labels.push(train.label(index) as i64);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &xs, [batch, 3072])?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &labels,
            [batch],
            DType::I64,
        )?);
        let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 25 == 0 {
            println!(
                "step {step:3}  loss {:.4}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let test_batch = 64;
    let mut correct = 0;
    for start in (0..test_count).step_by(test_batch) {
        let end = (start + test_batch).min(test_count);
        let n = end - start;
        let mut xs = Vec::with_capacity(n * 3072);
        let mut labels = Vec::with_capacity(n);
        for &sample in test_order.iter().take(end).skip(start) {
            xs.extend_from_slice(&test.sample(sample));
            labels.push(test.label(sample));
        }
        let x = Variable::constant(Tensor::from_f32(CpuBackend::new(), &xs, [n, 3072])?);
        let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
        let values = host_f32(&logits.data());
        for (row, &label) in labels.iter().enumerate() {
            let scores = &values[row * 10..(row + 1) * 10];
            let predicted = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u8)
                .unwrap();
            if predicted == label {
                correct += 1;
            }
        }
    }
    println!("test accuracy: {correct}/{test_count}");
    Ok(())
}
