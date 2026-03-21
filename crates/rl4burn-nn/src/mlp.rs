//! Multi-layer perceptron with RMSNorm (R2-Dreamer style).
//!
//! Provides a configurable N-layer MLP with choice of normalization
//! (RMSNorm or LayerNorm) and SiLU activation.  Used throughout
//! R2-Dreamer for prediction heads, encoders, and actor/critic networks.
//!
//! Reference: Nauman & Straffelini, "R2-Dreamer: Redundancy Reduction for
//! Computationally Efficient World Models" (ICLR 2026).

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Configuration for [`RmsNorm`].
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Feature dimension to normalise over.
    pub dim: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-8)]
    pub eps: f64,
}

/// Root Mean Square Layer Normalization.
///
/// `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight`
///
/// Simpler than LayerNorm (no centering), slightly faster.
/// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    #[module(skip)]
    eps: f64,
}

impl RmsNormConfig {
    /// Initialize an [`RmsNorm`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            weight: Tensor::ones([self.dim], device),
            eps: self.eps,
        }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass: `x / sqrt(mean(x^2) + eps) * weight`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // mean(x^2) over last dim -> [batch, 1]
        let ms = x.clone().powf_scalar(2.0).mean_dim(1);
        let rms = (ms + self.eps).sqrt();
        let normed = x / rms;
        normed * self.weight.clone().unsqueeze_dim(0)
    }
}

// ---------------------------------------------------------------------------
// Normalization variant
// ---------------------------------------------------------------------------

/// Which normalisation layer to use inside an MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Root Mean Square normalisation (default in R2-Dreamer).
    Rms,
    /// Standard layer normalisation (default in DreamerV3).
    Layer,
}

// ---------------------------------------------------------------------------
// MLP configuration
// ---------------------------------------------------------------------------

/// Configuration for a multi-layer perceptron.
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Number of hidden layers (default 2).
    #[config(default = 2)]
    pub n_layers: usize,
}

// ---------------------------------------------------------------------------
// MLP module
// ---------------------------------------------------------------------------

/// Multi-layer perceptron with normalization and SiLU activation.
///
/// Architecture per hidden layer: `Linear → Norm → SiLU`.
/// Final layer is a bare `Linear` (no norm, no activation).
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    rms_norms: Vec<RmsNorm<B>>,
    layer_norms: Vec<LayerNorm<B>>,
    output: Linear<B>,
    #[module(skip)]
    use_rms: bool,
}

impl MlpConfig {
    /// Initialize an MLP with the given normalisation kind.
    pub fn init_with_norm<B: Backend>(
        &self,
        norm: NormKind,
        device: &B::Device,
    ) -> Mlp<B> {
        let mut layers = Vec::with_capacity(self.n_layers);
        let mut rms_norms = Vec::new();
        let mut layer_norms = Vec::new();

        for i in 0..self.n_layers {
            let in_dim = if i == 0 { self.input_dim } else { self.hidden_dim };
            layers.push(LinearConfig::new(in_dim, self.hidden_dim).init(device));
            match norm {
                NormKind::Rms => {
                    rms_norms.push(RmsNormConfig::new(self.hidden_dim).init(device));
                }
                NormKind::Layer => {
                    layer_norms.push(LayerNormConfig::new(self.hidden_dim).init(device));
                }
            }
        }

        let output = LinearConfig::new(
            if self.n_layers == 0 { self.input_dim } else { self.hidden_dim },
            self.output_dim,
        )
        .init(device);

        Mlp {
            layers,
            rms_norms,
            layer_norms,
            output,
            use_rms: norm == NormKind::Rms,
        }
    }

    /// Initialize an MLP with RMSNorm (R2-Dreamer default).
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        self.init_with_norm(NormKind::Rms, device)
    }
}

impl<B: Backend> Mlp<B> {
    /// Forward pass through all hidden layers + output projection.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for (i, linear) in self.layers.iter().enumerate() {
            h = linear.forward(h);
            if self.use_rms {
                h = self.rms_norms[i].forward(h);
            } else {
                h = self.layer_norms[i].forward(h);
            }
            h = burn::tensor::activation::silu(h);
        }
        self.output.forward(h)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn rms_norm_output_shape() {
        let norm = RmsNormConfig::new(8).init::<B>(&dev());
        let x = Tensor::<B, 2>::ones([3, 8], &dev());
        let y = norm.forward(x);
        assert_eq!(y.dims(), [3, 8]);
    }

    #[test]
    fn rms_norm_unit_variance_direction() {
        let norm = RmsNormConfig::new(4).init::<B>(&dev());
        let x = Tensor::<B, 2>::from_floats([[2.0, 2.0, 2.0, 2.0]], &dev());
        let y = norm.forward(x);
        let vals: Vec<f32> = y.to_data().to_vec().unwrap();
        // RMS of [2,2,2,2] = 2, so normed = [1,1,1,1] * weight(1) = [1,1,1,1]
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn mlp_output_shape_rms() {
        let mlp = MlpConfig::new(16, 32, 4)
            .with_n_layers(2)
            .init::<B>(&dev());
        let x = Tensor::<B, 2>::zeros([5, 16], &dev());
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [5, 4]);
    }

    #[test]
    fn mlp_output_shape_layernorm() {
        let mlp = MlpConfig::new(16, 32, 4)
            .with_n_layers(3)
            .init_with_norm(NormKind::Layer, &dev());
        let x = Tensor::<B, 2>::zeros([2, 16], &dev());
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [2, 4]);
    }

    #[test]
    fn mlp_zero_hidden_layers() {
        let mlp = MlpConfig::new(8, 16, 3)
            .with_n_layers(0)
            .init::<B>(&dev());
        let x = Tensor::<B, 2>::ones([4, 8], &dev());
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [4, 3]);
    }

    #[test]
    fn mlp_values_finite() {
        let mlp = MlpConfig::new(8, 16, 4)
            .with_n_layers(2)
            .init::<B>(&dev());
        let x = Tensor::<B, 2>::random(
            [3, 8],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let y = mlp.forward(x);
        let vals: Vec<f32> = y.to_data().to_vec().unwrap();
        for &v in &vals {
            assert!(v.is_finite(), "MLP output should be finite, got {v}");
        }
    }
}
