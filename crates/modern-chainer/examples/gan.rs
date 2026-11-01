//! Least-squares GAN on a synthetic 2-D ring distribution.
//!
//! This uses MLPs and MSE, so it only needs the core autograd ops.

use modern_chainer::prelude::*;
use modern_chainer_core::Rng;
use modern_chainer_cpu::CpuTensorExt;

fn mse(a: &Variable<CpuBackend>, b: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
    let diff = F::sub(a, b)?;
    F::mean(&F::mul(&diff, &diff)?)
}

struct Generator {
    l1: Linear<CpuBackend>,
    l2: Linear<CpuBackend>,
    l3: Linear<CpuBackend>,
}

impl Generator {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 1)?,
            l2: Linear::new(64, 64, 2)?,
            l3: Linear::new(64, 2, 3)?,
        })
    }

    fn forward(&self, z: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
        let h = F::relu(&self.l1.forward(z)?)?;
        let h = F::relu(&self.l2.forward(&h)?)?;
        self.l3.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params.extend(self.l3.parameters());
        params
    }
}

struct Discriminator {
    l1: Linear<CpuBackend>,
    l2: Linear<CpuBackend>,
    l3: Linear<CpuBackend>,
}

impl Discriminator {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 4)?,
            l2: Linear::new(64, 64, 5)?,
            l3: Linear::new(64, 1, 6)?,
        })
    }

    fn forward(&self, x: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
        let h = F::relu(&self.l1.forward(x)?)?;
        let h = F::relu(&self.l2.forward(&h)?)?;
        self.l3.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params.extend(self.l3.parameters());
        params
    }
}

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(11);
    let batch = 64;
    let generator = Generator::new()?;
    let discriminator = Discriminator::new()?;
    let g_params = generator.parameters();
    let d_params = discriminator.parameters();
    let mut opt_g = Adam::new(0.0005);
    let mut opt_d = Adam::new(0.0005);

    let real = {
        let mut values = Vec::with_capacity(batch * 2);
        for _ in 0..batch {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let radius = 2.0 + rng.normal() * 0.08;
            values.push((radius * angle.cos()) as f32);
            values.push((radius * angle.sin()) as f32);
        }
        Variable::constant(Tensor::from_f32(CpuBackend::new(), &values, [batch, 2])?)
    };
    let ones = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &vec![1.0; batch],
        [batch, 1],
    )?);
    let zeros = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &vec![0.0; batch],
        [batch, 1],
    )?);

    for step in 0..=4000 {
        let z = rng.normal_vec(batch * 2, 0.0, 1.0);
        let noise = Variable::constant(Tensor::from_f32(CpuBackend::new(), &z, [batch, 2])?);
        let fake = generator.forward(&noise)?;
        let d_real = discriminator.forward(&real)?;
        let d_fake = discriminator.forward(&fake)?;
        let d_loss = F::scale(&F::add(&mse(&d_real, &ones)?, &mse(&d_fake, &zeros)?)?, 0.5)?;
        zero_grad(&d_params);
        zero_grad(&g_params);
        d_loss.backward()?;
        opt_d.update(&d_params)?;

        let z = rng.normal_vec(batch * 2, 0.0, 1.0);
        let noise = Variable::constant(Tensor::from_f32(CpuBackend::new(), &z, [batch, 2])?);
        let fake = generator.forward(&noise)?;
        let d_fake = discriminator.forward(&fake)?;
        let g_loss = mse(&d_fake, &ones)?;
        zero_grad(&g_params);
        zero_grad(&d_params);
        g_loss.backward()?;
        opt_g.update(&g_params)?;

        if step % 500 == 0 {
            let d = d_loss.data().backend().ops().to_scalar(&d_loss.data())?;
            let g = g_loss.data().backend().ops().to_scalar(&g_loss.data())?;
            println!("step {step:4}  D loss {d:.4}  G loss {g:.4}");
        }
    }

    let z = rng.normal_vec(256 * 2, 0.0, 1.0);
    let noise = Variable::constant(Tensor::from_f32(CpuBackend::new(), &z, [256, 2])?);
    let samples = generator.forward(&noise)?.data().to_vec_f32()?;
    let (points, _) = samples.as_chunks::<2>();
    let mean_radius = points
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .sum::<f32>()
        / 256.0;
    println!("generated mean radius: {mean_radius:.3} (target ~2)");
    Ok(())
}
