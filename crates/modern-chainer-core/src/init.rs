use crate::rng::Rng;
use crate::shape::Shape;

/// He normal initialisation (`std = sqrt(2 / fan_in)`), the Chainer default for
/// `L.Linear` and `L.Conv2d`.
pub fn he_normal(shape: impl Into<Shape>, fan_in: usize, seed: u64) -> Vec<f32> {
    let shape = shape.into();
    let std = (2.0 / fan_in.max(1) as f64).sqrt();
    let mut rng = Rng::seed_from_u64(seed);
    rng.normal_vec(shape.numel(), 0.0, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn he_scale() {
        let values = he_normal([64, 32], 32, 1);
        assert_eq!(values.len(), 64 * 32);
        let mean = values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64;
        let var = values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(mean.abs() < 0.01);
        assert!((var - 2.0 / 32.0).abs() < 0.01, "var = {var}");
    }
}
