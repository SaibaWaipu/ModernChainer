//! Variational autoencoder on a synthetic 2-D ring.
//!
//! Loss = reconstruction MSE + beta * KL(q(z|x) || N(0, I)).
//! Sampling uses the reparameterization trick with a host-generated epsilon.

use modern_chainer::prelude::*;
use modern_chainer_core::Rng;
use modern_chainer_cpu::CpuTensorExt;

fn mse(a: &Variable<CpuBackend>, b: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
    let diff = F::sub(a, b)?;
    F::mean(&F::mul(&diff, &diff)?)
}

struct Encoder {
    l1: Linear<CpuBackend>,
    l2: Linear<CpuBackend>,
    mu: Linear<CpuBackend>,
    logvar: Linear<CpuBackend>,
}

impl Encoder {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 1)?,
            l2: Linear::new(64, 64, 2)?,
            mu: Linear::new(64, 2, 3)?,
            logvar: Linear::new(64, 2, 4)?,
        })
    }

    fn forward(
        &self,
        x: &Variable<CpuBackend>,
    ) -> Result<(Variable<CpuBackend>, Variable<CpuBackend>)> {
        let h = F::relu(&self.l1.forward(x)?)?;
        let h = F::relu(&self.l2.forward(&h)?)?;
        Ok((self.mu.forward(&h)?, self.logvar.forward(&h)?))
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params.extend(self.mu.parameters());
        params.extend(self.logvar.parameters());
        params
    }
}

struct Decoder {
    l1: Linear<CpuBackend>,
    l2: Linear<CpuBackend>,
    l3: Linear<CpuBackend>,
}

impl Decoder {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 5)?,
            l2: Linear::new(64, 64, 6)?,
            l3: Linear::new(64, 2, 7)?,
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

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(17);
    let batch = 64;
    let encoder = Encoder::new()?;
    let decoder = Decoder::new()?;
    let mut params = encoder.parameters();
    params.extend(decoder.parameters());
    let mut optimizer = Adam::new(0.002);

    for step in 0..=3000 {
        let mut values = Vec::with_capacity(batch * 2);
        for _ in 0..batch {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let radius = 2.0 + rng.normal() * 0.1;
            values.push((radius * angle.cos()) as f32);
            values.push((radius * angle.sin()) as f32);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &values, [batch, 2])?);
        let (mu, logvar) = encoder.forward(&x)?;
        let half = F::scale(&logvar, 0.5)?;
        let sigma = F::exp(&half)?;
        let eps_values = rng.normal_vec(batch * 2, 0.0, 1.0);
        let eps = Variable::constant(Tensor::from_f32(
            CpuBackend::new(),
            &eps_values,
            [batch, 2],
        )?);
        let z = F::add(&mu, &F::mul(&sigma, &eps)?)?;
        let recon = decoder.forward(&z)?;
        let recon_loss = mse(&recon, &x)?;

        let mu2 = F::mul(&mu, &mu)?;
        let var = F::exp(&logvar)?;
        let one = Variable::constant(Tensor::from_f32(
            CpuBackend::new(),
            &vec![1.0; batch * 2],
            [batch, 2],
        )?);
        let kl_element = F::scale(&F::sub(&F::add(&mu2, &var)?, &F::add(&one, &logvar)?)?, 0.5)?;
        let kl = F::mean(&kl_element)?;
        let loss = F::add(&recon_loss, &F::scale(&kl, 0.1)?)?;

        if step % 500 == 0 {
            let r = recon_loss
                .data()
                .backend()
                .ops()
                .to_scalar(&recon_loss.data())?;
            let k = kl.data().backend().ops().to_scalar(&kl.data())?;
            println!("step {step:4}  recon {r:.4}  kl {k:.4}");
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;
    }

    let z = Variable::constant(Tensor::from_f32(
        CpuBackend::new(),
        &rng.normal_vec(256 * 2, 0.0, 1.0),
        [256, 2],
    )?);
    let samples = decoder.forward(&z)?.data().to_vec_f32()?;
    let (points, _) = samples.as_chunks::<2>();
    let mean_radius = points
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .sum::<f32>()
        / 256.0;
    println!("sample mean radius: {mean_radius:.3} (target ~2)");
    Ok(())
}
