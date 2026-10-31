use modern_chainer::prelude::*;

fn main() -> Result<()> {
    let batch = 64;
    let features = 8;
    let backend = CpuBackend::new();

    let mut x_values = Vec::with_capacity(batch * features);
    let mut t_values = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut sum = 0.0;
        for col in 0..features {
            let value = ((row * features + col) as f32 * 0.031).sin();
            x_values.push(value);
            sum += value;
        }
        t_values.push(sum / features as f32);
    }
    let x = Variable::new(Tensor::from_f32(backend, &x_values, [batch, features])?);
    let target = Variable::constant(Tensor::from_f32(backend, &t_values, [batch, 1])?);

    let l1 = Linear::new(features, 32, 0)?;
    let l2 = Linear::new(32, 1, 1)?;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(0.01);

    for step in 0..=500 {
        let hidden = F::relu(&l1.forward(&x)?)?;
        let prediction = l2.forward(&hidden)?;
        let diff = F::sub(&prediction, &target)?;
        let loss = F::mean(&F::mul(&diff, &diff)?)?;

        if step % 100 == 0 {
            println!(
                "step {step:4}  loss {:.6}",
                loss.data().backend().ops().to_scalar(&loss.data())?
            );
        }

        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }
    Ok(())
}
