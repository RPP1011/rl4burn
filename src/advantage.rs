//! Advantage normalization utilities.

/// Normalize advantages to zero mean, unit variance, then clamp.
///
/// Operates on raw f32 slice, returns normalized Vec.
/// Clamping prevents explosive gradients from outlier advantages.
pub fn normalize(advantages: &[f32], clamp: f32) -> Vec<f32> {
    let n = advantages.len();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![0.0]; }

    let mean: f32 = advantages.iter().sum::<f32>() / n as f32;
    let var: f32 = advantages.iter()
        .map(|&a| (a - mean) * (a - mean))
        .sum::<f32>() / n as f32;
    let std = var.sqrt().max(1e-8);

    advantages.iter()
        .map(|&a| ((a - mean) / std).clamp(-clamp, clamp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_mean() {
        let norm = normalize(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0);
        let mean: f32 = norm.iter().sum::<f32>() / norm.len() as f32;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn unit_variance() {
        let norm = normalize(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0);
        let mean: f32 = norm.iter().sum::<f32>() / norm.len() as f32;
        let var: f32 = norm.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norm.len() as f32;
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn clamped() {
        let norm = normalize(&[0.0, 0.0, 0.0, 100.0], 3.0);
        assert!(norm.iter().all(|&x| x >= -3.0 && x <= 3.0));
    }

    #[test]
    fn single_value() {
        let norm = normalize(&[5.0], 10.0);
        assert_eq!(norm, vec![0.0]);
    }

    #[test]
    fn empty() {
        assert!(normalize(&[], 10.0).is_empty());
    }
}
