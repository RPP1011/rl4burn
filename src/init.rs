//! Weight initialization utilities for RL.
//!
//! Provides orthogonal initialization (critical for PPO convergence)
//! as used in CleanRL, OpenAI baselines, and Stable Baselines3.

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::TensorData;

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Mapper that replaces parameter tensors with pre-computed data.
struct WeightReplacer<B: Backend> {
    replacements: Vec<Tensor<B, 1>>, // flattened tensors to inject
    idx: usize,
}

impl<B: Backend> ModuleMapper<B> for WeightReplacer<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        if self.idx < self.replacements.len() {
            let dims = param.val().dims();
            let replacement = self.replacements[self.idx].clone().reshape(dims);
            self.idx += 1;
            Param::initialized(param.id.clone(), replacement).set_require_grad(true)
        } else {
            param
        }
    }
}

/// Create a `Linear` layer with orthogonal weight initialization and zero bias.
///
/// Matches PyTorch's `nn.init.orthogonal_(layer.weight, gain)` +
/// `nn.init.constant_(layer.bias, 0.0)` as used in CleanRL's `layer_init`.
///
/// Uses `Module::map` to replace weights while preserving Burn's autodiff
/// parameter tracking.
pub fn orthogonal_linear<B: Backend>(
    d_in: usize,
    d_out: usize,
    gain: f32,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Linear<B> {
    // Create a standard Linear first (properly registered with autodiff)
    let linear: Linear<B> = LinearConfig::new(d_in, d_out).init(device);

    // Generate orthogonal weight data [d_in, d_out] (Burn's layout)
    let weight_data = orthogonal_matrix(d_in, d_out, gain, rng);
    let weight_flat: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(weight_data, [d_in * d_out]), device);

    // Zero bias [d_out]
    let bias_flat: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(vec![0.0f32; d_out], [d_out]), device);

    // Use ModuleMapper to replace param tensors while preserving tracking
    let mut mapper = WeightReplacer {
        replacements: vec![weight_flat, bias_flat],
        idx: 0,
    };
    linear.map(&mut mapper)
}

/// Generate an orthogonal matrix of shape [rows, cols] scaled by `gain`.
///
/// Uses modified Gram-Schmidt orthogonalization on a random Gaussian matrix.
/// This matches the behavior of `torch.nn.init.orthogonal_`.
fn orthogonal_matrix(rows: usize, cols: usize, gain: f32, rng: &mut impl Rng) -> Vec<f32> {
    let big = rows.max(cols);
    let small = rows.min(cols);

    // Generate [big, small] random matrix (column-major for Gram-Schmidt)
    let mut mat = vec![0.0f32; big * small];
    for i in 0..mat.len() {
        let val: f64 = StandardNormal.sample(rng);
        mat[i] = val as f32;
    }

    // Modified Gram-Schmidt on columns
    for j in 0..small {
        // Subtract projections onto previous orthogonalized columns
        for i in 0..j {
            let mut dot = 0.0f32;
            for r in 0..big {
                dot += mat[r * small + j] * mat[r * small + i];
            }
            for r in 0..big {
                mat[r * small + j] -= dot * mat[r * small + i];
            }
        }
        // Normalize column j
        let mut norm = 0.0f32;
        for r in 0..big {
            norm += mat[r * small + j] * mat[r * small + j];
        }
        norm = norm.sqrt();
        if norm > 1e-10 {
            for r in 0..big {
                mat[r * small + j] /= norm;
            }
        }
    }

    // Extract [rows, cols] and scale by gain
    let mut result = vec![0.0f32; rows * cols];
    if rows >= cols {
        // mat is [big=rows, small=cols], use directly
        for r in 0..rows {
            for c in 0..cols {
                result[r * cols + c] = mat[r * small + c] * gain;
            }
        }
    } else {
        // mat is [big=cols, small=rows], transpose to [rows, cols]
        for r in 0..rows {
            for c in 0..cols {
                result[r * cols + c] = mat[c * small + r] * gain;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn orthogonal_columns_are_unit_norm() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let data = orthogonal_matrix(64, 8, 1.0, &mut rng);
        // Each column should have unit norm
        for c in 0..8 {
            let mut norm_sq = 0.0f32;
            for r in 0..64 {
                norm_sq += data[r * 8 + c] * data[r * 8 + c];
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "column {c} norm² = {norm_sq}, expected 1.0"
            );
        }
    }

    #[test]
    fn orthogonal_columns_are_orthogonal() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let data = orthogonal_matrix(64, 8, 1.0, &mut rng);
        // Dot product between any two columns should be ~0
        for i in 0..8 {
            for j in (i + 1)..8 {
                let mut dot = 0.0f32;
                for r in 0..64 {
                    dot += data[r * 8 + i] * data[r * 8 + j];
                }
                assert!(
                    dot.abs() < 1e-4,
                    "columns {i} and {j} dot = {dot}, expected ~0"
                );
            }
        }
    }

    #[test]
    fn gain_scales_norm() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let gain = 2.0f32;
        let data = orthogonal_matrix(64, 8, gain, &mut rng);
        for c in 0..8 {
            let mut norm_sq = 0.0f32;
            for r in 0..64 {
                norm_sq += data[r * 8 + c] * data[r * 8 + c];
            }
            let expected = gain * gain;
            assert!(
                (norm_sq - expected).abs() < 1e-3,
                "column {c} norm² = {norm_sq}, expected {expected}"
            );
        }
    }

    #[test]
    fn orthogonal_linear_has_zero_bias() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let device = Default::default();
        let linear: Linear<B> = orthogonal_linear(4, 64, 1.0, &device, &mut rng);
        if let Some(bias) = &linear.bias {
            let sum: f32 = bias.val().abs().sum().into_scalar();
            assert!(sum < 1e-8, "bias should be zero, sum={sum}");
        }
    }

    #[test]
    fn wide_matrix_works() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        // rows < cols: [2, 64]
        let data = orthogonal_matrix(2, 64, 1.0, &mut rng);
        assert_eq!(data.len(), 2 * 64);
        // Rows should have unit norm
        for r in 0..2 {
            let mut norm_sq = 0.0f32;
            for c in 0..64 {
                norm_sq += data[r * 64 + c] * data[r * 64 + c];
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "row {r} norm² = {norm_sq}, expected 1.0"
            );
        }
    }
}
