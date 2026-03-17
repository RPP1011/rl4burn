//! FiLM (Feature-wise Linear Modulation) conditioning layer.
//!
//! Modulates feature maps via learned affine transformations conditioned on
//! a context vector:
//!
//! ```text
//! output = γ(context) * input + β(context)
//! ```
//!
//! where γ and β are learned linear projections of the context vector.
//! A residual `+1` is added to γ so that at initialization (when projections
//! output near zero) the layer acts close to identity, improving training
//! stability.
//!
//! Reference: Perez et al., "FiLM: Visual Reasoning with a General
//! Conditioning Layer", AAAI 2018.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// FiLM (Feature-wise Linear Modulation) layer.
///
/// Applies a context-dependent affine transform:
/// `output = (γ(ctx) + 1) * input + β(ctx)`
/// where γ and β are learned linear projections of the context vector.
#[derive(Module, Debug)]
pub struct Film<B: Backend> {
    gamma_proj: Linear<B>,
    beta_proj: Linear<B>,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a [`Film`] layer.
#[derive(Config, Debug)]
pub struct FilmConfig {
    /// Dimension of the context conditioning vector.
    pub context_dim: usize,
    /// Dimension of the features to modulate.
    pub feature_dim: usize,
}

impl FilmConfig {
    /// Initialize a FiLM layer on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Film<B> {
        let gamma_proj = LinearConfig::new(self.context_dim, self.feature_dim).init(device);
        let beta_proj = LinearConfig::new(self.context_dim, self.feature_dim).init(device);
        Film {
            gamma_proj,
            beta_proj,
        }
    }
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

impl<B: Backend> Film<B> {
    /// Apply FiLM conditioning.
    ///
    /// # Arguments
    /// * `input` — Features to modulate `[batch, feature_dim]`
    /// * `context` — Conditioning vector `[batch, context_dim]`
    ///
    /// # Returns
    /// Modulated features `[batch, feature_dim]`
    pub fn forward(&self, input: Tensor<B, 2>, context: Tensor<B, 2>) -> Tensor<B, 2> {
        let gamma = self.gamma_proj.forward(context.clone()); // [batch, feature_dim]
        let beta = self.beta_proj.forward(context); // [batch, feature_dim]

        // output = (gamma + 1) * input + beta
        // The +1 ensures the default behaviour is near-identity when the
        // projection weights are close to zero at initialisation.
        (gamma + 1.0) * input + beta
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

    // -- shape tests ---------------------------------------------------------

    #[test]
    fn output_shape_matches_input() {
        let film = FilmConfig::new(8, 16).init::<B>(&dev());
        let input = Tensor::<B, 2>::zeros([4, 16], &dev());
        let context = Tensor::<B, 2>::zeros([4, 8], &dev());
        let output = film.forward(input, context);
        assert_eq!(output.dims(), [4, 16]);
    }

    #[test]
    fn single_sample_batch() {
        let film = FilmConfig::new(3, 5).init::<B>(&dev());
        let input = Tensor::<B, 2>::ones([1, 5], &dev());
        let context = Tensor::<B, 2>::zeros([1, 3], &dev());
        let output = film.forward(input, context);
        assert_eq!(output.dims(), [1, 5]);
    }

    #[test]
    fn large_batch() {
        let film = FilmConfig::new(4, 6).init::<B>(&dev());
        let input = Tensor::<B, 2>::zeros([128, 6], &dev());
        let context = Tensor::<B, 2>::zeros([128, 4], &dev());
        let output = film.forward(input, context);
        assert_eq!(output.dims(), [128, 6]);
    }

    // -- zero-context near-identity ------------------------------------------

    #[test]
    fn zero_context_zero_input() {
        // With zero input and zero context, output should be close to
        // beta_proj(0).  Since input is 0, the gamma branch vanishes.
        let film = FilmConfig::new(4, 8).init::<B>(&dev());
        let input = Tensor::<B, 2>::zeros([2, 8], &dev());
        let context = Tensor::<B, 2>::zeros([2, 4], &dev());
        let output = film.forward(input, context);
        // Output = (gamma+1)*0 + beta = beta_proj(0) = bias term only.
        // Just verify shape and finite values.
        assert_eq!(output.dims(), [2, 8]);
        let vals: Vec<f32> = output.to_data().to_vec().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    // -- context affects output ----------------------------------------------

    #[test]
    fn different_context_gives_different_output() {
        let film = FilmConfig::new(4, 8).init::<B>(&dev());
        let input = Tensor::<B, 2>::ones([1, 8], &dev());

        let ctx_a = Tensor::<B, 2>::zeros([1, 4], &dev());
        let ctx_b = Tensor::<B, 2>::ones([1, 4], &dev()) * 10.0;

        let out_a = film.forward(input.clone(), ctx_a);
        let out_b = film.forward(input, ctx_b);

        let a_vals: Vec<f32> = out_a.to_data().to_vec().unwrap();
        let b_vals: Vec<f32> = out_b.to_data().to_vec().unwrap();

        // At least one element should differ.
        let any_differ = a_vals
            .iter()
            .zip(b_vals.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            any_differ,
            "outputs should differ for different contexts, got a={a_vals:?}, b={b_vals:?}"
        );
    }

    // -- gradient flow (autodiff) --------------------------------------------

    #[test]
    fn gradient_flow_through_context() {
        use burn::backend::Autodiff;

        type AB = Autodiff<NdArray>;

        let dev = Default::default();
        let film = FilmConfig::new(4, 8).init::<AB>(&dev);
        let input = Tensor::<AB, 2>::ones([2, 8], &dev).require_grad();
        let context = Tensor::<AB, 2>::ones([2, 4], &dev).require_grad();

        let output = film.forward(input.clone(), context.clone());
        let loss = output.sum();
        let grads = loss.backward();

        let ctx_grad = context.grad(&grads).expect("context should have gradients");
        let ctx_grad_vals: Vec<f32> = ctx_grad.to_data().to_vec().unwrap();

        // Gradients should be non-zero (projections are not all-zero at init).
        let any_nonzero = ctx_grad_vals.iter().any(|&g| g.abs() > 1e-8);
        assert!(
            any_nonzero,
            "context gradients should be non-zero: {ctx_grad_vals:?}"
        );

        let inp_grad = input.grad(&grads).expect("input should have gradients");
        let inp_grad_vals: Vec<f32> = inp_grad.to_data().to_vec().unwrap();
        let any_nonzero = inp_grad_vals.iter().any(|&g| g.abs() > 1e-8);
        assert!(
            any_nonzero,
            "input gradients should be non-zero: {inp_grad_vals:?}"
        );
    }

    // -- config roundtrip ----------------------------------------------------

    #[test]
    fn config_dimensions() {
        let cfg = FilmConfig::new(10, 20);
        assert_eq!(cfg.context_dim, 10);
        assert_eq!(cfg.feature_dim, 20);
    }
}
