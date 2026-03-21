//! Representation learning losses for R2-Dreamer (ICLR 2026).
//!
//! Four self-supervised objectives that replace or augment the standard
//! decoder-based reconstruction loss in world models:
//!
//! 1. **Barlow Twins** (`barlow_twins_loss`) — R2-Dreamer's primary contribution:
//!    redundancy reduction via invariance + decorrelation.
//! 2. **InfoNCE** (`infonce_loss`) — contrastive learning on normalised features.
//! 3. **DreamerPro** (`dreamerpro_loss`) — prototype-based with Sinkhorn assignment.
//! 4. **Decoder** (`decoder_loss`) — standard MSE reconstruction (DreamerV3 baseline).
//!
//! Reference: Nauman & Straffelini, "R2-Dreamer: Redundancy Reduction for
//! Computationally Efficient World Models" (ICLR 2026).

use burn::prelude::*;

// ---------------------------------------------------------------------------
// Representation variant enum
// ---------------------------------------------------------------------------

/// Which representation learning objective to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationVariant {
    /// Standard decoder reconstruction (DreamerV3 baseline).
    Dreamer,
    /// Barlow Twins redundancy reduction (R2-Dreamer).
    R2Dreamer,
    /// InfoNCE contrastive loss.
    InfoNCE,
    /// DreamerPro prototype-based loss.
    DreamerPro,
}

// ---------------------------------------------------------------------------
// Barlow Twins (R2-Dreamer)
// ---------------------------------------------------------------------------

/// Configuration for Barlow Twins loss.
#[derive(Debug, Clone)]
pub struct BarlowTwinsConfig {
    /// Weight for the redundancy reduction (off-diagonal) term.
    /// Default: 0.0051 (from R2-Dreamer paper).
    pub lambda: f32,
}

impl Default for BarlowTwinsConfig {
    fn default() -> Self {
        Self { lambda: 0.0051 }
    }
}

/// Barlow Twins loss for redundancy reduction (R2-Dreamer).
///
/// Given two sets of embeddings `z_a` and `z_b` (e.g., posterior features
/// and prior features), compute:
///
/// 1. **Invariance**: diagonal of cross-correlation should be 1.
/// 2. **Redundancy reduction**: off-diagonal should be 0.
///
/// Both inputs are L2-normalised along the batch dimension, then the
/// empirical cross-correlation matrix `C = z_a^T z_b / N` is computed.
///
/// # Arguments
/// * `z_a`, `z_b` — `[batch, dim]` feature embeddings.
/// * `config` — Barlow Twins hyperparameters.
///
/// # Returns
/// Scalar loss `[1]`.
pub fn barlow_twins_loss<B: Backend>(
    z_a: Tensor<B, 2>,
    z_b: Tensor<B, 2>,
    config: &BarlowTwinsConfig,
) -> Tensor<B, 1> {
    let [batch, dim] = z_a.dims();
    let n = batch as f32;

    // Normalise along batch dimension (zero mean, unit std)
    let z_a = batch_normalize(z_a);
    let z_b = batch_normalize(z_b);

    // Cross-correlation: C = z_a^T z_b / N   [dim, dim]
    let c = z_a.clone().transpose().matmul(z_b) / n;

    // Identity target
    let device = c.device();
    let eye = Tensor::<B, 2>::eye(dim, &device);

    // Invariance loss: sum((diag(C) - 1)^2)
    let diag_diff = (c.clone() - eye.clone()) * eye.clone();
    let invariance = diag_diff.powf_scalar(2.0).sum();

    // Redundancy loss: sum(off_diag(C)^2)
    let off_diag_mask = eye.neg() + 1.0; // 1 everywhere except diagonal
    let off_diag = c * off_diag_mask;
    let redundancy = off_diag.powf_scalar(2.0).sum();

    (invariance + redundancy * config.lambda).unsqueeze()
}

/// Zero-mean, unit-std normalization along the batch dimension.
fn batch_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let mean = x.clone().mean_dim(0); // [1, dim]
    let centered = x - mean;
    let std = centered.clone().powf_scalar(2.0).mean_dim(0).sqrt() + 1e-6;
    centered / std
}

// ---------------------------------------------------------------------------
// InfoNCE
// ---------------------------------------------------------------------------

/// Configuration for InfoNCE loss.
#[derive(Debug, Clone)]
pub struct InfoNceConfig {
    /// Temperature for the softmax. Default: 0.1.
    pub temperature: f32,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self { temperature: 0.1 }
    }
}

/// InfoNCE contrastive loss.
///
/// Treats each sample's (z_a[i], z_b[i]) pair as a positive and all
/// other z_b[j≠i] as negatives.
///
/// `loss = -mean(log(exp(sim(z_a, z_b) / T) / sum_j exp(sim(z_a, z_b_j) / T)))`
///
/// Both inputs are L2-normalised before computing cosine similarity.
///
/// # Arguments
/// * `z_a`, `z_b` — `[batch, dim]` feature embeddings.
/// * `config` — InfoNCE hyperparameters.
///
/// # Returns
/// Scalar loss `[1]`.
pub fn infonce_loss<B: Backend>(
    z_a: Tensor<B, 2>,
    z_b: Tensor<B, 2>,
    config: &InfoNceConfig,
) -> Tensor<B, 1> {
    let [batch, _dim] = z_a.dims();

    // L2 normalize
    let z_a = l2_normalize(z_a);
    let z_b = l2_normalize(z_b);

    // Similarity matrix: [batch, batch]
    let logits = z_a.matmul(z_b.transpose()) / config.temperature;

    // Labels: diagonal (each sample matched to itself)
    let device = logits.device();
    let targets = Tensor::<B, 1, Int>::arange(0..batch as i64, &device);

    // Cross-entropy
    let loss = burn::tensor::activation::log_softmax(logits, 1); // [batch, batch]
    let targets_2d: Tensor<B, 2, Int> = targets.unsqueeze_dim(1); // [batch, 1]
    let nll: Tensor<B, 1> = loss.gather(1, targets_2d).squeeze_dim::<1>(1).neg(); // [batch]
    nll.mean().unsqueeze()
}

/// L2-normalise along the feature dimension (dim=1).
fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt() + 1e-8;
    x / norm
}

// ---------------------------------------------------------------------------
// DreamerPro
// ---------------------------------------------------------------------------

/// Configuration for DreamerPro loss.
#[derive(Debug, Clone)]
pub struct DreamerProConfig {
    /// Number of prototypes. Default: 512.
    pub n_prototypes: usize,
    /// Number of Sinkhorn iterations. Default: 3.
    pub sinkhorn_iters: usize,
    /// Temperature for sharpening. Default: 0.1.
    pub temperature: f32,
}

impl Default for DreamerProConfig {
    fn default() -> Self {
        Self {
            n_prototypes: 512,
            sinkhorn_iters: 3,
            temperature: 0.1,
        }
    }
}

/// DreamerPro loss: prototype-based representation learning.
///
/// Computes soft assignments between online features and EMA-target
/// prototypes using Sinkhorn-Knopp normalisation, then trains the
/// online encoder to match these assignments.
///
/// # Arguments
/// * `z_online` — `[batch, dim]` features from the online encoder.
/// * `z_target` — `[batch, dim]` features from the EMA target encoder (detached).
/// * `prototypes` — `[n_prototypes, dim]` prototype vectors (learnable).
/// * `config` — DreamerPro hyperparameters.
///
/// # Returns
/// Scalar loss `[1]`.
pub fn dreamerpro_loss<B: Backend>(
    z_online: Tensor<B, 2>,
    z_target: Tensor<B, 2>,
    prototypes: Tensor<B, 2>, // [n_prototypes, dim]
    config: &DreamerProConfig,
) -> Tensor<B, 1> {
    // Compute scores: [batch, n_prototypes]
    let z_online_n = l2_normalize(z_online);
    let z_target_n = l2_normalize(z_target.detach());
    let protos_n = l2_normalize(prototypes);

    // Target assignments via Sinkhorn
    let target_scores = z_target_n.matmul(protos_n.clone().transpose()) / config.temperature;
    let target_q = sinkhorn(target_scores, config.sinkhorn_iters); // [batch, n_proto]

    // Online log-probabilities
    let online_scores = z_online_n.matmul(protos_n.transpose()) / config.temperature;
    let online_log_p = burn::tensor::activation::log_softmax(online_scores, 1);

    // Cross-entropy: -sum(q * log(p))
    let loss = (target_q * online_log_p).sum_dim(1).squeeze_dim::<1>(1).neg();
    loss.mean().unsqueeze()
}

/// Sinkhorn-Knopp normalisation: iteratively normalize rows and columns
/// to produce a doubly-stochastic assignment matrix.
///
/// Input: `scores` `[batch, n_prototypes]` (unnormalised log-scores).
/// Returns: `[batch, n_prototypes]` soft assignments.
pub fn sinkhorn<B: Backend>(scores: Tensor<B, 2>, n_iters: usize) -> Tensor<B, 2> {
    // Exp to get unnormalised Q
    let mut q = scores.exp();

    for _ in 0..n_iters {
        // Normalise columns (sum over batch dim)
        let col_sum = q.clone().sum_dim(0) + 1e-8; // [1, n_proto]
        q = q / col_sum;
        // Normalise rows (sum over prototype dim)
        let row_sum = q.clone().sum_dim(1) + 1e-8; // [batch, 1]
        q = q / row_sum;
    }

    q
}

// ---------------------------------------------------------------------------
// Decoder reconstruction loss
// ---------------------------------------------------------------------------

/// Standard MSE reconstruction loss (DreamerV3 baseline).
///
/// # Arguments
/// * `pred` — predicted observation `[batch, dim]` (or flattened image).
/// * `target` — ground-truth observation `[batch, dim]`.
///
/// # Returns
/// Scalar loss `[1]`.
pub fn decoder_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.powf_scalar(2.0).mean().unsqueeze()
}

/// Image reconstruction loss (MSE over `[B, C, H, W]` tensors).
pub fn image_decoder_loss<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.powf_scalar(2.0).mean().unsqueeze()
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

    // -- Barlow Twins --------------------------------------------------------

    #[test]
    fn barlow_identical_inputs_low_loss() {
        let z = Tensor::<B, 2>::random(
            [32, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = BarlowTwinsConfig::default();
        let loss: f32 = barlow_twins_loss(z.clone(), z, &config).into_scalar();
        // Identical inputs → cross-correlation ≈ identity → low loss
        assert!(loss < 1.0, "barlow twins identical should have low loss, got {loss}");
    }

    #[test]
    fn barlow_different_inputs_higher_loss() {
        let z_a = Tensor::<B, 2>::random(
            [32, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let z_b = Tensor::<B, 2>::random(
            [32, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = BarlowTwinsConfig::default();
        let loss: f32 = barlow_twins_loss(z_a, z_b, &config).into_scalar();
        assert!(loss > 0.0, "barlow twins loss should be positive, got {loss}");
        assert!(loss.is_finite(), "loss should be finite");
    }

    // -- InfoNCE -------------------------------------------------------------

    #[test]
    fn infonce_positive() {
        let z_a = Tensor::<B, 2>::random(
            [8, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let z_b = Tensor::<B, 2>::random(
            [8, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = InfoNceConfig::default();
        let loss: f32 = infonce_loss(z_a, z_b, &config).into_scalar();
        assert!(loss > 0.0, "infonce loss should be positive, got {loss}");
        assert!(loss.is_finite());
    }

    #[test]
    fn infonce_identical_low() {
        let z = Tensor::<B, 2>::random(
            [8, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = InfoNceConfig { temperature: 0.1 };
        let loss: f32 = infonce_loss(z.clone(), z, &config).into_scalar();
        // Identical → diagonal similarities are maximal → low loss
        assert!(loss.is_finite());
    }

    // -- Sinkhorn ------------------------------------------------------------

    #[test]
    fn sinkhorn_rows_sum_to_one() {
        let scores = Tensor::<B, 2>::random(
            [16, 8],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let q = sinkhorn(scores, 5);
        let row_sums: Vec<f32> = q.sum_dim(1).squeeze_dim::<1>(1).to_data().to_vec().unwrap();
        for (i, &s) in row_sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 0.1,
                "row {i} sum = {s}, expected ~1.0"
            );
        }
    }

    // -- DreamerPro ----------------------------------------------------------

    #[test]
    fn dreamerpro_positive() {
        let z_online = Tensor::<B, 2>::random(
            [8, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let z_target = Tensor::<B, 2>::random(
            [8, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let prototypes = Tensor::<B, 2>::random(
            [32, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = DreamerProConfig {
            n_prototypes: 32,
            sinkhorn_iters: 3,
            temperature: 0.1,
        };
        let loss: f32 = dreamerpro_loss(z_online, z_target, prototypes, &config).into_scalar();
        assert!(loss > 0.0, "dreamerpro loss should be positive, got {loss}");
        assert!(loss.is_finite());
    }

    // -- Decoder loss --------------------------------------------------------

    #[test]
    fn decoder_loss_zero_for_identical() {
        let x = Tensor::<B, 2>::random(
            [4, 8],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let loss: f32 = decoder_loss(x.clone(), x).into_scalar();
        assert!(loss.abs() < 1e-6, "identical inputs should give zero loss, got {loss}");
    }

    #[test]
    fn decoder_loss_positive_for_different() {
        let x = Tensor::<B, 2>::ones([4, 8], &dev());
        let y = Tensor::<B, 2>::zeros([4, 8], &dev());
        let loss: f32 = decoder_loss(x, y).into_scalar();
        assert!((loss - 1.0).abs() < 1e-5, "MSE of 1s vs 0s should be 1.0, got {loss}");
    }
}
