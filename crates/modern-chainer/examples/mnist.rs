//! MNIST digit classification with an MLP.
//!
//! Prepare data with `bash scripts/download_mnist.sh`.

use modern_chainer::data::mnist::Mnist;
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
    let train = Mnist::load(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte",
    )
    .map_err(|e| Error::invalid(format!("{e}; run scripts/download_mnist.sh")))?;
    let test = Mnist::load(
        "data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte",
    )?;
    let train_count = train.len().min(6_000);
    let test_count = test.len().min(1_000);
    let mut rng = Rng::seed_from_u64(9);
    let mut order: Vec<usize> = (0..train_count).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    let l1 = Linear::new(28 * 28, 128, 1)?;
    let l2 = Linear::new(128, 10, 2)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.001);
    let batch = 64;
    for step in 0..=300 {
        let mut xs = Vec::with_capacity(batch * 784);
        let mut labels = Vec::with_capacity(batch);
        for i in 0..batch {
            let index = order[(step * batch + i) % train_count];
            xs.extend_from_slice(&train.sample(index));
            labels.push(train.label(index) as i64);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &xs, [batch, 784])?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &labels,
            [batch],
            DType::I64,
        )?);
        let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
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

    let test_batch = 128;
    let mut correct = 0;
    for start in (0..test_count).step_by(test_batch) {
        let end = (start + test_batch).min(test_count);
        let n = end - start;
        let mut xs = Vec::with_capacity(n * 784);
        let mut labels = Vec::with_capacity(n);
        for index in start..end {
            xs.extend_from_slice(&test.sample(index));
            labels.push(test.label(index));
        }
        let x = Variable::constant(Tensor::from_f32(CpuBackend::new(), &xs, [n, 784])?);
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
