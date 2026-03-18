//! Symlog transform and twohot distributional predictions (DreamerV3 style).
//!
//! **Symlog/symexp** compress large magnitudes while preserving sign, making
//! value prediction easier when returns span many orders of magnitude.
//!
//! **Twohot encoding** represents a scalar as a soft two-hot distribution over
//! uniformly-spaced bins in symlog space.  A neural network predicts logits
//! over the bins and is trained with categorical cross-entropy against the
//! twohot target.  Decoding computes a weighted sum of bin centers and applies
//! `symexp` to recover the original scale.
//!
//! Reference: Hafner et al., "Mastering Diverse Domains through World Models"
//! (DreamerV3), 2023.

use burn::prelude::*;
use burn::tensor::activation::log_softmax;

// ---------------------------------------------------------------------------
// Symlog / symexp
// ---------------------------------------------------------------------------

/// `symlog(x) = sign(x) * ln(|x| + 1)`
///
/// Uses `log1p` for numerical stability.
pub fn symlog<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().sign() * x.abs().log1p()
}

/// `symexp(x) = sign(x) * (exp(|x|) - 1)`
///
/// Inverse of [`symlog`].
pub fn symexp<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().sign() * (x.abs().exp() - 1.0)
}

// ---------------------------------------------------------------------------
// Twohot encoder
// ---------------------------------------------------------------------------

/// Twohot encoder for distributional value predictions.
///
/// Maps scalar values to soft two-hot distributions over `n_bins` bins
/// uniformly spaced in `[low, high]` (in symlog space).  Default parameters
/// match DreamerV3: 255 bins spanning `[-20, 20]`.
pub struct TwohotEncoder {
    /// Number of bins.
    pub n_bins: usize,
    /// Lower bound in symlog space.
    pub low: f32,
    /// Upper bound in symlog space.
    pub high: f32,
}

impl TwohotEncoder {
    /// Create a new encoder with default DreamerV3 parameters (255 bins, [-20, 20]).
    pub fn new() -> Self {
        Self {
            n_bins: 255,
            low: -20.0,
            high: 20.0,
        }
    }

    /// Create an encoder with custom bin count and range.
    pub fn with_bins(n_bins: usize, low: f32, high: f32) -> Self {
        assert!(n_bins >= 2, "need at least 2 bins");
        assert!(low < high, "low must be less than high");
        Self { n_bins, low, high }
    }

    /// Width of each bin.
    fn bin_width(&self) -> f32 {
        (self.high - self.low) / (self.n_bins - 1) as f32
    }

    /// Bin centers as a 1-D float tensor of shape `[n_bins]`.
    pub fn bin_centers<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let w = self.bin_width();
        let centers: Vec<f32> = (0..self.n_bins)
            .map(|i| self.low + i as f32 * w)
            .collect();
        Tensor::from_floats(centers.as_slice(), device)
    }

    /// Encode scalar values into twohot distributions.
    ///
    /// * `values` — shape `[batch]`, raw (un-transformed) scalar values.
    /// * Returns — shape `[batch, n_bins]`, each row sums to 1.
    ///
    /// Values are first mapped through [`symlog`], then clamped to
    /// `[low, high]`.  Each transformed value is represented as weight on the
    /// two nearest bin centers, proportional to proximity.
    pub fn encode<B: Backend>(
        &self,
        values: Tensor<B, 1>,
        _device: &B::Device,
    ) -> Tensor<B, 2> {
        let transformed = symlog(values);
        let clamped = transformed.clamp(self.low, self.high);

        let bin_width = self.bin_width();

        // Normalised position in [0, n_bins-1].
        let normalised = (clamped - self.low) / bin_width;

        // Lower bin index (as float) and fractional weight for upper bin.
        let lower_f = normalised.clone().floor();
        let upper_weight = normalised - lower_f.clone();
        let lower_weight = upper_weight.clone().neg() + 1.0;

        // Upper index, clamped so it doesn't exceed n_bins-1.
        let upper_f = lower_f.clone().clamp(0.0, (self.n_bins - 2) as f32) + 1.0;
        // Also clamp lower to valid range.
        let lower_f = lower_f.clamp(0.0, (self.n_bins - 1) as f32);

        // Build twohot via one_hot.
        // one_hot expects the tensor values to be the index positions (as floats).
        let lower_oh: Tensor<B, 2> = lower_f.one_hot(self.n_bins);
        let upper_oh: Tensor<B, 2> = upper_f.one_hot(self.n_bins);

        // weight * one_hot → [batch, n_bins]
        let lower_weight_2d: Tensor<B, 2> = lower_weight.unsqueeze_dim(1);
        let upper_weight_2d: Tensor<B, 2> = upper_weight.unsqueeze_dim(1);

        lower_weight_2d * lower_oh + upper_weight_2d * upper_oh
    }

    /// Decode a probability distribution back to scalar values.
    ///
    /// * `probs` — shape `[batch, n_bins]`, softmax probabilities over bins.
    /// * Returns — shape `[batch]`, values in original (un-transformed) scale.
    ///
    /// Computes the weighted sum of bin centers (expected value in symlog
    /// space), then applies [`symexp`] to invert the transform.
    pub fn decode<B: Backend>(
        &self,
        probs: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let centers = self.bin_centers::<B>(device); // [n_bins]
        let centers_2d: Tensor<B, 2> = centers.unsqueeze_dim(0); // [1, n_bins]
        let symlog_values = (probs * centers_2d)
            .sum_dim(1)
            .squeeze_dim::<1>(1); // [batch]
        symexp(symlog_values)
    }

    /// Categorical cross-entropy loss against twohot targets.
    ///
    /// * `logits` — shape `[batch, n_bins]`, raw network output (pre-softmax).
    /// * `values` — shape `[batch]`, raw scalar target values.
    /// * Returns — shape `[1]`, mean loss over the batch.
    pub fn loss<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        values: Tensor<B, 1>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let targets = self.encode::<B>(values, device);
        let log_probs = log_softmax(logits, 1);
        // -sum(target * log_softmax(logits), dim=1)  →  mean over batch
        let loss_per_sample: Tensor<B, 1> = (targets * log_probs)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .neg();
        loss_per_sample.mean().unsqueeze()
    }
}

impl Default for TwohotEncoder {
    fn default() -> Self {
        Self::new()
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

    // -- symlog / symexp basics ----------------------------------------------

    #[test]
    fn symlog_zero() {
        let x = Tensor::<B, 1>::from_floats([0.0], &dev());
        let y: f32 = symlog(x).into_scalar();
        assert!(y.abs() < 1e-7, "symlog(0) = {y}, expected 0");
    }

    #[test]
    fn symlog_known_values() {
        let x = Tensor::<B, 1>::from_floats([1.0, -1.0, 1000.0], &dev());
        let y = symlog(x).to_data();
        let vals: Vec<f32> = y.to_vec().unwrap();

        let ln2 = 2.0_f32.ln();
        assert!(
            (vals[0] - ln2).abs() < 1e-5,
            "symlog(1) = {}, expected {ln2}",
            vals[0]
        );
        assert!(
            (vals[1] + ln2).abs() < 1e-5,
            "symlog(-1) = {}, expected {:.5}",
            vals[1],
            -ln2
        );

        let expected = 1001.0_f32.ln();
        assert!(
            (vals[2] - expected).abs() < 1e-3,
            "symlog(1000) = {}, expected {expected}",
            vals[2]
        );
    }

    #[test]
    fn symlog_preserves_sign() {
        let x = Tensor::<B, 1>::from_floats([-5.0, -0.1, 0.1, 5.0], &dev());
        let y = symlog(x).to_data();
        let vals: Vec<f32> = y.to_vec().unwrap();
        assert!(vals[0] < 0.0);
        assert!(vals[1] < 0.0);
        assert!(vals[2] > 0.0);
        assert!(vals[3] > 0.0);
    }

    #[test]
    fn symlog_is_monotonic() {
        let x = Tensor::<B, 1>::from_floats([-10.0, -1.0, 0.0, 1.0, 10.0], &dev());
        let y = symlog(x).to_data();
        let vals: Vec<f32> = y.to_vec().unwrap();
        for i in 0..vals.len() - 1 {
            assert!(
                vals[i] < vals[i + 1],
                "monotonicity violated: y[{}]={} >= y[{}]={}",
                i,
                vals[i],
                i + 1,
                vals[i + 1]
            );
        }
    }

    #[test]
    fn symexp_roundtrip() {
        let inputs = [-100.0_f32, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0, 1e6];
        let x = Tensor::<B, 1>::from_floats(inputs.as_slice(), &dev());
        let roundtrip = symexp(symlog(x));
        let vals: Vec<f32> = roundtrip.to_data().to_vec().unwrap();
        for (i, (&orig, &got)) in inputs.iter().zip(vals.iter()).enumerate() {
            let tol = orig.abs() * 1e-4 + 1e-6;
            assert!(
                (got - orig).abs() < tol,
                "roundtrip[{i}]: symexp(symlog({orig})) = {got}",
            );
        }
    }

    // -- twohot encoder ------------------------------------------------------

    #[test]
    fn twohot_encode_center_bin() {
        // value 0.0 → symlog(0) = 0.0 → falls exactly on center bin.
        let enc = TwohotEncoder::new();
        let v = Tensor::<B, 1>::from_floats([0.0], &dev());
        let dist = enc.encode::<B>(v, &dev());

        let vals: Vec<f32> = dist.to_data().to_vec().unwrap();
        assert_eq!(vals.len(), 255);

        // Center bin is index 127 (midpoint of 0..254).
        let center = 127;
        assert!(
            (vals[center] - 1.0).abs() < 1e-5,
            "center bin weight = {}, expected 1.0",
            vals[center]
        );

        // All other bins should be ~0.
        let rest_sum: f32 = vals.iter().enumerate()
            .filter(|&(i, _)| i != center)
            .map(|(_, v)| v)
            .sum();
        assert!(
            rest_sum.abs() < 1e-5,
            "non-center sum = {rest_sum}, expected 0"
        );
    }

    #[test]
    fn twohot_weights_sum_to_one() {
        let enc = TwohotEncoder::with_bins(11, -5.0, 5.0);
        let v = Tensor::<B, 1>::from_floats([0.3, -2.7, 4.99], &dev());
        let dist = enc.encode::<B>(v, &dev());
        let sums = dist.sum_dim(1).squeeze_dim::<1>(1);
        let sums: Vec<f32> = sums.to_data().to_vec().unwrap();
        for (i, s) in sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {i} sums to {s}, expected 1.0"
            );
        }
    }

    #[test]
    fn twohot_decode_roundtrip() {
        let enc = TwohotEncoder::new();
        // Use values whose symlog falls neatly within the bin range.
        let inputs = [0.0_f32, 1.0, -1.0, 10.0, -10.0, 100.0];
        let v = Tensor::<B, 1>::from_floats(inputs.as_slice(), &dev());
        let dist = enc.encode::<B>(v, &dev());
        let decoded = enc.decode::<B>(dist, &dev());
        let vals: Vec<f32> = decoded.to_data().to_vec().unwrap();

        for (i, (&orig, &got)) in inputs.iter().zip(vals.iter()).enumerate() {
            // Tolerance accounts for bin quantisation.
            let tol = orig.abs() * 0.05 + 0.5;
            assert!(
                (got - orig).abs() < tol,
                "roundtrip[{i}]: encode→decode({orig}) = {got} (tol={tol})",
            );
        }
    }

    #[test]
    fn twohot_uniform_probs_decode_near_zero() {
        // Symmetric bins → uniform probs → expected value in symlog space is 0
        // → symexp(0) = 0.
        let enc = TwohotEncoder::new();
        let uniform = Tensor::<B, 2>::ones([1, 255], &dev()) / 255.0;
        let decoded: f32 = enc.decode::<B>(uniform, &dev()).into_scalar();
        assert!(
            decoded.abs() < 1e-4,
            "uniform decode = {decoded}, expected ≈ 0"
        );
    }

    #[test]
    fn twohot_loss_is_positive() {
        let enc = TwohotEncoder::new();
        let logits = Tensor::<B, 2>::zeros([4, 255], &dev()); // uniform logits
        let targets = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0, 10.0], &dev());
        let loss: f32 = enc.loss::<B>(logits, targets, &dev()).into_scalar();
        assert!(loss > 0.0, "cross-entropy loss should be positive, got {loss}");
    }

    #[test]
    fn twohot_loss_is_scalar() {
        let enc = TwohotEncoder::new();
        let logits = Tensor::<B, 2>::zeros([2, 255], &dev());
        let targets = Tensor::<B, 1>::from_floats([0.0, 5.0], &dev());
        let loss = enc.loss::<B>(logits, targets, &dev());
        assert_eq!(loss.dims(), [1]);
    }
}
