//! Tiny deterministic RNG used for parameter initialisation.
//!
//! We intentionally avoid a dependency on `rand`: initialisation quality only
//! needs to be good enough for training, and determinism across backends makes
//! numerical parity tests reproducible.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via the polar Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }

    pub fn normal_vec(&mut self, len: usize, mean: f64, std: f64) -> Vec<f32> {
        (0..len)
            .map(|_| (mean + std * self.normal()) as f32)
            .collect()
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::seed_from_u64(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn normal_moments() {
        let mut rng = Rng::seed_from_u64(7);
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.03, "var = {var}");
    }
}
