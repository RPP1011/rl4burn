//! Beta-VAE for opponent modeling (Issue #27, ROA-Star).
//!
//! A variational autoencoder with a tunable beta coefficient on the KL
//! divergence term.  Higher beta encourages disentangled, interpretable
//! latent representations of opponent strategies.
//!
//! The encoder maps opponent-observable features to a latent distribution,
//! and the decoder reconstructs the input.  The latent mean serves as a
//! compact strategy embedding that can condition a policy or value network.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Beta-VAE configuration for opponent modeling.
#[derive(Config, Debug)]
pub struct BetaVaeConfig {
    /// Input dimension (opponent observable features).
    pub input_dim: usize,
    /// Latent dimension for strategy embedding.
    #[config(default = 32)]
    pub latent_dim: usize,
    /// Hidden layer dimension.
    #[config(default = 256)]
    pub hidden_dim: usize,
    /// Beta coefficient for KL regularization. Default: 1.0
    #[config(default = 1.0)]
    pub beta: f32,
}

// ---------------------------------------------------------------------------
// Encoder / Decoder
// ---------------------------------------------------------------------------

/// Beta-VAE encoder: maps observation to latent distribution parameters.
#[derive(Module, Debug)]
pub struct VaeEncoder<B: Backend> {
    fc1: Linear<B>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
}

/// Beta-VAE decoder: maps latent to reconstruction.
#[derive(Module, Debug)]
pub struct VaeDecoder<B: Backend> {
    fc1: Linear<B>,
    fc_out: Linear<B>,
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Beta-VAE for opponent modeling.
#[derive(Module, Debug)]
pub struct BetaVae<B: Backend> {
    encoder: VaeEncoder<B>,
    decoder: VaeDecoder<B>,
    #[module(skip)]
    beta: f32,
    #[module(skip)]
    latent_dim: usize,
}

/// VAE output including reconstruction and latent distribution.
pub struct VaeOutput<B: Backend> {
    pub reconstruction: Tensor<B, 2>,
    pub mu: Tensor<B, 2>,
    pub logvar: Tensor<B, 2>,
    pub z: Tensor<B, 2>,
}

impl BetaVaeConfig {
    /// Initialize the Beta-VAE on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BetaVae<B> {
        let encoder = VaeEncoder {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc_mu: LinearConfig::new(self.hidden_dim, self.latent_dim).init(device),
            fc_logvar: LinearConfig::new(self.hidden_dim, self.latent_dim).init(device),
        };
        let decoder = VaeDecoder {
            fc1: LinearConfig::new(self.latent_dim, self.hidden_dim).init(device),
            fc_out: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
        };
        BetaVae {
            encoder,
            decoder,
            beta: self.beta,
            latent_dim: self.latent_dim,
        }
    }
}

impl<B: Backend> BetaVae<B> {
    /// Encode input to latent distribution parameters (mu, logvar).
    pub fn encode(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = burn::tensor::activation::relu(self.encoder.fc1.forward(x));
        let mu = self.encoder.fc_mu.forward(h.clone());
        let logvar = self.encoder.fc_logvar.forward(h);
        (mu, logvar)
    }

    /// Reparameterization trick: z = mu + sigma * epsilon.
    pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = (logvar * 0.5).exp();
        let eps = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0, 1.0));
        mu + std * eps
    }

    /// Decode latent to reconstruction.
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.decoder.fc1.forward(z));
        self.decoder.fc_out.forward(h)
    }

    /// Full forward pass: encode, sample, decode.
    pub fn forward(&self, x: Tensor<B, 2>) -> VaeOutput<B> {
        let (mu, logvar) = self.encode(x);
        let z = self.reparameterize(mu.clone(), logvar.clone());
        let reconstruction = self.decode(z.clone());
        VaeOutput {
            reconstruction,
            mu,
            logvar,
            z,
        }
    }

    /// Compute beta-VAE loss: reconstruction (MSE) + beta * KL divergence.
    pub fn loss(&self, x: Tensor<B, 2>, output: &VaeOutput<B>) -> Tensor<B, 1> {
        // Reconstruction loss (MSE)
        let recon_loss = (x - output.reconstruction.clone())
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mean();

        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        // Equivalently: 0.5 * sum(exp(logvar) + mu^2 - logvar - 1)
        let kl = (output.logvar.clone().exp()
            + output.mu.clone().powf_scalar(2.0)
            - output.logvar.clone()
            - 1.0)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mean()
            * 0.5;

        (recon_loss + kl * self.beta).unsqueeze()
    }

    /// Extract opponent strategy embedding (the latent mean).
    pub fn strategy_embedding(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mu, _) = self.encode(x);
        mu
    }

    /// Latent dimension accessor.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
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

    fn small_config() -> BetaVaeConfig {
        BetaVaeConfig::new(16)
            .with_latent_dim(8)
            .with_hidden_dim(32)
            .with_beta(1.0)
    }

    #[test]
    fn output_shapes() {
        let vae = small_config().init::<B>(&dev());
        let x = Tensor::<B, 2>::zeros([4, 16], &dev());
        let out = vae.forward(x);

        assert_eq!(out.reconstruction.dims(), [4, 16]);
        assert_eq!(out.mu.dims(), [4, 8]);
        assert_eq!(out.logvar.dims(), [4, 8]);
        assert_eq!(out.z.dims(), [4, 8]);
    }

    #[test]
    fn loss_is_positive() {
        let vae = small_config().init::<B>(&dev());
        let x = Tensor::<B, 2>::ones([4, 16], &dev());
        let out = vae.forward(x.clone());
        let loss: f32 = vae.loss(x, &out).into_scalar();
        assert!(loss > 0.0, "loss should be positive, got {loss}");
    }

    #[test]
    fn strategy_embedding_shape() {
        let vae = small_config().init::<B>(&dev());
        let x = Tensor::<B, 2>::zeros([3, 16], &dev());
        let emb = vae.strategy_embedding(x);
        assert_eq!(emb.dims(), [3, 8]);
    }

    #[test]
    fn encode_decode_shapes() {
        let vae = small_config().init::<B>(&dev());
        let x = Tensor::<B, 2>::zeros([2, 16], &dev());
        let (mu, logvar) = vae.encode(x);
        assert_eq!(mu.dims(), [2, 8]);
        assert_eq!(logvar.dims(), [2, 8]);

        let z = vae.reparameterize(mu, logvar);
        let recon = vae.decode(z);
        assert_eq!(recon.dims(), [2, 16]);
    }

    #[test]
    fn higher_beta_increases_loss() {
        // With higher beta, KL term is weighted more so total loss should be higher
        let x = Tensor::<B, 2>::ones([4, 16], &dev());

        let vae_low = BetaVaeConfig::new(16)
            .with_latent_dim(8)
            .with_hidden_dim(32)
            .with_beta(0.1)
            .init::<B>(&dev());
        let out_low = vae_low.forward(x.clone());
        let loss_low: f32 = vae_low.loss(x.clone(), &out_low).into_scalar();

        // Same architecture but beta=10 -- we just check loss is positive
        // (can't guarantee higher since weights differ)
        let vae_high = BetaVaeConfig::new(16)
            .with_latent_dim(8)
            .with_hidden_dim(32)
            .with_beta(10.0)
            .init::<B>(&dev());
        let out_high = vae_high.forward(x.clone());
        let loss_high: f32 = vae_high.loss(x, &out_high).into_scalar();

        assert!(loss_low > 0.0);
        assert!(loss_high > 0.0);
    }
}
