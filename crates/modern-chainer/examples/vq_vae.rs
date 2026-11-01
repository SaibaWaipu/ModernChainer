//! VQ-VAE on a synthetic 2-D ring with a small EMA-updated codebook.
//!
//! The encoder sees the straight-through estimator; the codebook itself is
//! updated outside autograd with an exponential moving average.

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
}
impl Encoder {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 1)?,
            l2: Linear::new(64, 2, 2)?,
        })
    }
    fn forward(&self, x: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
        let h = F::relu(&self.l1.forward(x)?)?;
        self.l2.forward(&h)
    }
    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p
    }
}

struct Decoder {
    l1: Linear<CpuBackend>,
    l2: Linear<CpuBackend>,
}
impl Decoder {
    fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::new(2, 64, 3)?,
            l2: Linear::new(64, 2, 4)?,
        })
    }
    fn forward(&self, z: &Variable<CpuBackend>) -> Result<Variable<CpuBackend>> {
        let h = F::relu(&self.l1.forward(z)?)?;
        self.l2.forward(&h)
    }
    fn parameters(&self) -> Vec<Parameter<CpuBackend>> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p
    }
}

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(23);
    let batch = 64;
    let codebook_size = 8;
    let encoder = Encoder::new()?;
    let decoder = Decoder::new()?;
    let mut params = encoder.parameters();
    params.extend(decoder.parameters());
    let mut optimizer = Adam::new(0.002);
    let mut codebook: Vec<[f32; 2]> = (0..codebook_size)
        .map(|_| {
            let a = rng.next_f64() * std::f64::consts::TAU;
            [(2.0 * a.cos()) as f32, (2.0 * a.sin()) as f32]
        })
        .collect();

    for step in 0..=3000 {
        let mut values = Vec::with_capacity(batch * 2);
        for _ in 0..batch {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let radius = 2.0 + rng.normal() * 0.1;
            values.push((radius * angle.cos()) as f32);
            values.push((radius * angle.sin()) as f32);
        }
        let x = Variable::new(Tensor::from_f32(CpuBackend::new(), &values, [batch, 2])?);
        let z_e = encoder.forward(&x)?;
        let ze_values = z_e.data().to_vec_f32()?;

        let mut zq_values = Vec::with_capacity(batch * 2);
        let mut assignments = Vec::with_capacity(batch);
        let mut sums = vec![[0.0f32; 2]; codebook_size];
        let mut counts = vec![0usize; codebook_size];
        for row in 0..batch {
            let point = [ze_values[row * 2], ze_values[row * 2 + 1]];
            let (best, _) = codebook
                .iter()
                .enumerate()
                .map(|(index, code)| {
                    let dx = point[0] - code[0];
                    let dy = point[1] - code[1];
                    (index, dx * dx + dy * dy)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assignments.push(best);
            zq_values.push(codebook[best][0]);
            zq_values.push(codebook[best][1]);
            sums[best][0] += point[0];
            sums[best][1] += point[1];
            counts[best] += 1;
        }

        let zq_const =
            Variable::constant(Tensor::from_f32(CpuBackend::new(), &zq_values, [batch, 2])?);
        let ze_const = Variable::constant(z_e.data());
        let straight_through = F::sub(&zq_const, &ze_const)?;
        let z_q = F::add(&z_e, &straight_through)?;
        let recon = decoder.forward(&z_q)?;
        let recon_loss = mse(&recon, &x)?;
        let commitment = F::scale(&mse(&z_e, &zq_const)?, 0.25)?;
        let loss = F::add(&recon_loss, &commitment)?;

        if step % 500 == 0 {
            let r = recon_loss
                .data()
                .backend()
                .ops()
                .to_scalar(&recon_loss.data())?;
            let c = commitment
                .data()
                .backend()
                .ops()
                .to_scalar(&commitment.data())?;
            let used = counts.iter().filter(|&&n| n > 0).count();
            println!("step {step:4}  recon {r:.4}  commit {c:.4}  codes {used}/{codebook_size}");
        }
        zero_grad(&params);
        loss.backward()?;
        optimizer.update(&params)?;

        for code in 0..codebook_size {
            if counts[code] > 0 {
                let n = counts[code] as f32;
                let mean = [sums[code][0] / n, sums[code][1] / n];
                codebook[code][0] = 0.9 * codebook[code][0] + 0.1 * mean[0];
                codebook[code][1] = 0.9 * codebook[code][1] + 0.1 * mean[1];
            }
        }
    }

    println!("codebook: {codebook:?}");
    Ok(())
}
