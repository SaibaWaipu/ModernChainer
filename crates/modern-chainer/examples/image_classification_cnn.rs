//! Small CNN image classifier using Conv2d, BatchNorm, pooling and reshape.

use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

fn make_image(label: u32, rng: &mut Rng) -> Vec<f32> {
    let size = 16;
    let mut values = vec![0.0; size * size];
    for y in 0..size {
        for x in 0..size {
            let on = match label {
                0 => y == 7,
                1 => x == 7,
                _ => x == y,
            };
            values[y * size + x] = if on { 1.0 } else { 0.0 } + rng.normal_vec(1, 0.0, 0.1)[0];
        }
    }
    values
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
    let mut rng = Rng::seed_from_u64(5);
    let conv1 = Conv2d::new(1, 4, 3, 1, 1, 1)?;
    let bn1 = BatchNorm::new(4, 1e-5)?;
    let conv2 = Conv2d::new(4, 8, 3, 1, 1, 2)?;
    let bn2 = BatchNorm::new(8, 1e-5)?;
    let fc = Linear::new(8 * 4 * 4, 3, 3)?;
    let mut params = conv1.parameters();
    params.extend(bn1.parameters());
    params.extend(conv2.parameters());
    params.extend(bn2.parameters());
    params.extend(fc.parameters());
    let mut optimizer = Adam::new(0.002);

    let batch = 8;
    let steps = 80;
    for step in 0..=steps {
        let mut xs = Vec::with_capacity(batch * 16 * 16);
        let mut labels = Vec::with_capacity(batch);
        for i in 0..batch {
            let label = (step * batch + i) % 3;
            xs.extend_from_slice(&make_image(label as u32, &mut rng));
            labels.push(label as i64);
        }
        let x = Variable::new(Tensor::from_f32(
            CpuBackend::new(),
            &xs,
            [batch, 1, 16, 16],
        )?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &labels,
            [batch],
            DType::I64,
        )?);
        let h = F::relu(&bn1.forward(&conv1.forward(&x)?)?)?;
        let h = F::max_pool2d(&h, 2, 2, 0)?;
        let h = F::relu(&bn2.forward(&conv2.forward(&h)?)?)?;
        let h = F::max_pool2d(&h, 2, 2, 0)?;
        let h = F::reshape(&h, [batch, 8 * 4 * 4])?;
        let logits = fc.forward(&h)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 20 == 0 {
            println!(
                "step {step:3}  loss {:.4}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let mut xs = Vec::new();
    let mut labels = Vec::new();
    for i in 0..30 {
        let label = i % 3;
        xs.extend_from_slice(&make_image(label as u32, &mut rng));
        labels.push(label as i64);
    }
    let x = Variable::constant(Tensor::from_f32(CpuBackend::new(), &xs, [30, 1, 16, 16])?);
    let h = F::relu(&bn1.forward(&conv1.forward(&x)?)?)?;
    let h = F::max_pool2d(&h, 2, 2, 0)?;
    let h = F::relu(&bn2.forward(&conv2.forward(&h)?)?)?;
    let h = F::max_pool2d(&h, 2, 2, 0)?;
    let h = F::reshape(&h, [30, 8 * 4 * 4])?;
    let logits = fc.forward(&h)?;
    let values = host_f32(&logits.data());
    let mut correct = 0;
    for (index, &label) in labels.iter().enumerate() {
        let row = &values[index * 3..(index + 1) * 3];
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
    println!("test accuracy: {correct}/30");
    Ok(())
}
