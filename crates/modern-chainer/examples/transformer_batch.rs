//! Batched multi-head attention with reshape/permute and a toy training loop.

use modern_chainer::prelude::*;
use modern_chainer_core::Rng;

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(13);
    let batch = 2;
    let seq = 8;
    let dim = 16;
    let attention = MultiHeadAttention::new(dim, 4, false)?;
    let params = attention.parameters();
    let mut optimizer = Adam::new(0.002);
    let x = Variable::new(Tensor::from_f32(
        CpuBackend::new(),
        &rng.normal_vec(batch * seq * dim, 0.0, 0.5),
        [batch, seq, dim],
    )?);
    let target = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &vec![0.0; batch * seq * dim],
        [batch, seq, dim],
    )?);

    for step in 0..=200 {
        let y = attention.forward(&x)?;
        let diff = F::sub(&y, &target)?;
        let loss = F::mean(&F::mul(&diff, &diff)?)?;
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

    let y = attention.forward(&x)?;
    let flattened = F::reshape(&y, [batch * seq, dim])?;
    let permuted = F::permute(&flattened, vec![1, 0])?;
    println!(
        "attention output {:?} -> reshape {:?} -> permute {:?}",
        y.data().shape().dims(),
        flattened.data().shape().dims(),
        permuted.data().shape().dims()
    );
    Ok(())
}
