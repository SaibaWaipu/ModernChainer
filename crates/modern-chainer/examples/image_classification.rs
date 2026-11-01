//! Tiny image-classification example on synthetic grayscale patterns.
//!
//! Classifies 12x12 images containing a horizontal, vertical or diagonal bar.
//! The model is an MLP over flattened pixels so the example stays dependency
//! free; `modern_chainer::data::image::GrayImage` provides the image container.

use modern_chainer::data::image::GrayImage;
use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

fn make_image(label: u32, rng: &mut Rng) -> GrayImage {
    let size = 12;
    let mut image = GrayImage::zeros(size, size).unwrap();
    for y in 0..size {
        for x in 0..size {
            let on = match label {
                0 => y == 5,
                1 => x == 5,
                _ => x == y,
            };
            let noise = rng.normal_vec(1, 0.0, 0.12)[0];
            image.set_pixel(x, y, if on { 1.0 } else { 0.0 } + noise);
        }
    }
    image
}

fn make_dataset(count: usize, rng: &mut Rng) -> (Vec<f32>, Vec<i64>) {
    let mut xs = Vec::with_capacity(count * 144);
    let mut labels = Vec::with_capacity(count);
    for index in 0..count {
        let label = (index % 3) as u32;
        let image = make_image(label, rng);
        xs.extend_from_slice(image.flatten());
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
    let mut rng = Rng::seed_from_u64(7);
    let (train_x, train_t) = make_dataset(240, &mut rng);
    let (test_x, test_t) = make_dataset(60, &mut rng);

    let l1 = Linear::new(144, 64, 1)?;
    let l2 = Linear::new(64, 3, 2)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.003);

    let batch = 32;
    for step in 0..=300 {
        let start = (step * batch) % train_x.len();
        let mut x_values = Vec::with_capacity(batch * 144);
        let mut t_values = Vec::with_capacity(batch);
        for i in 0..batch {
            let index = (start + i * 7) % (train_x.len() / 144);
            x_values.extend_from_slice(&train_x[index * 144..(index + 1) * 144]);
            t_values.push(train_t[index]);
        }
        let x = Variable::new(Tensor::from_f32(
            CpuBackend::new(),
            &x_values,
            [batch, 144],
        )?);
        let target = Variable::constant(Tensor::from_slice(
            CpuBackend::new(),
            &t_values,
            [batch],
            DType::I64,
        )?);
        let hidden = F::relu(&l1.forward(&x)?)?;
        let logits = l2.forward(&hidden)?;
        let loss = F::softmax_cross_entropy(&logits, &target)?;
        if step % 50 == 0 {
            let value = loss.data().backend().ops().to_scalar(&loss.data())?;
            println!("step {step:4}  loss {value:.4}");
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let x = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &test_x,
        [test_x.len() / 144, 144],
    )?);
    let logits = l2.forward(&F::relu(&l1.forward(&x)?)?)?;
    let values = host_f32(&logits.data());
    let mut correct = 0;
    for (index, &label) in test_t.iter().enumerate() {
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
    println!("test accuracy: {correct}/{}", test_t.len());
    Ok(())
}
