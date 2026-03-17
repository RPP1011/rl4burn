//! Percentile-based return normalization.
//!
//! Tracks an exponential moving average of the 5th-95th percentile range of
//! episode returns and uses it to scale advantages.  The `max(1, S)` floor
//! prevents amplifying noise when rewards are sparse (scale near zero).

use contracts::*;

/// Computes the `p`-th quantile of a *sorted* slice via linear interpolation.
///
/// `p` is in `[0, 1]`.  Behaviour matches NumPy's `np.percentile(…, interpolation='linear')`.
fn quantile_sorted(sorted: &[f32], p: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f32;
    let lo = (idx.floor() as usize).min(n - 1);
    let hi = (idx.ceil() as usize).min(n - 1);
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Percentile-based advantage normalizer with EMA-smoothed scale.
///
/// Instead of per-minibatch mean/std normalization, this tracks the spread of
/// *returns* across rollouts and divides advantages by that spread.  This is
/// more stable when returns are heavy-tailed or sparse.
pub struct PercentileNormalizer {
    low_pct: f32,
    high_pct: f32,
    decay: f32,
    ema_low: f32,
    ema_high: f32,
    initialized: bool,
}

impl PercentileNormalizer {
    /// Create a normalizer with default settings (5th-95th percentile, 0.99 decay).
    pub fn new() -> Self {
        Self {
            low_pct: 0.05,
            high_pct: 0.95,
            decay: 0.99,
            ema_low: 0.0,
            ema_high: 0.0,
            initialized: false,
        }
    }

    /// Create a normalizer with custom percentile bounds.
    #[requires(low >= 0.0 && low < high && high <= 1.0, "percentiles must satisfy 0 <= low < high <= 1")]
    pub fn with_percentiles(low: f32, high: f32) -> Self {
        Self {
            low_pct: low,
            high_pct: high,
            ..Self::new()
        }
    }

    /// Set the EMA decay rate (builder pattern).
    #[requires(decay > 0.0 && decay < 1.0, "decay must be in (0, 1)")]
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay;
        self
    }

    /// Update the running EMA of low/high percentiles with a new batch of returns.
    ///
    /// On the first call the EMA is seeded directly; subsequent calls blend.
    #[requires(!returns.is_empty(), "need at least one return value")]
    pub fn update(&mut self, returns: &[f32]) {
        let mut sorted: Vec<f32> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let low = quantile_sorted(&sorted, self.low_pct);
        let high = quantile_sorted(&sorted, self.high_pct);

        if self.initialized {
            self.ema_low = self.decay * self.ema_low + (1.0 - self.decay) * low;
            self.ema_high = self.decay * self.ema_high + (1.0 - self.decay) * high;
        } else {
            self.ema_low = low;
            self.ema_high = high;
            self.initialized = true;
        }
    }

    /// Current normalization scale: `max(1.0, ema_high - ema_low)`.
    ///
    /// Returns `1.0` before the first [`update`](Self::update) call.
    pub fn scale(&self) -> f32 {
        if !self.initialized {
            return 1.0;
        }
        (self.ema_high - self.ema_low).max(1.0)
    }

    /// Normalize advantages by dividing each element by [`scale`](Self::scale).
    #[ensures(ret.len() == advantages.len(), "output length matches input")]
    pub fn normalize(&self, advantages: &[f32]) -> Vec<f32> {
        let s = self.scale();
        advantages.iter().map(|&a| a / s).collect()
    }

    /// Convenience: update percentiles from `returns`, then normalize `advantages`.
    #[requires(!returns.is_empty(), "need at least one return value")]
    #[ensures(ret.len() == advantages.len(), "output length matches input")]
    pub fn update_and_normalize(
        &mut self,
        returns: &[f32],
        advantages: &[f32],
    ) -> Vec<f32> {
        self.update(returns);
        self.normalize(advantages)
    }
}

impl Default for PercentileNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_returns() {
        let mut n = PercentileNormalizer::new();
        let returns = vec![0.0; 20];
        let advantages = vec![1.0, -1.0, 0.5];
        n.update(&returns);
        assert_eq!(n.scale(), 1.0, "zero spread should floor to 1");
        let out = n.normalize(&advantages);
        assert_eq!(out, advantages, "should pass through unchanged");
    }

    #[test]
    fn constant_returns() {
        let mut n = PercentileNormalizer::new();
        n.update(&vec![42.0; 50]);
        assert_eq!(n.scale(), 1.0, "constant returns have zero spread");
    }

    #[test]
    fn uniform_0_100() {
        let mut n = PercentileNormalizer::new();
        let returns: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        n.update(&returns);
        let s = n.scale();
        assert!(
            (s - 90.0).abs() < 1.0,
            "P95-P5 of [0..100] should be ~90, got {s}"
        );
    }

    #[test]
    fn sparse_rewards() {
        let mut n = PercentileNormalizer::new();
        // Mostly zeros with a single non-zero — percentile range is tiny.
        let mut returns = vec![0.0; 100];
        returns[99] = 0.5;
        n.update(&returns);
        assert_eq!(n.scale(), 1.0, "sparse rewards should floor to 1");
    }

    #[test]
    fn rank_ordering_preserved() {
        let mut n = PercentileNormalizer::new();
        let returns: Vec<f32> = (0..50).map(|i| i as f32).collect();
        n.update(&returns);

        let advantages = vec![-3.0, 0.0, 1.0, 5.0, 10.0];
        let out = n.normalize(&advantages);

        for i in 0..out.len() - 1 {
            assert!(
                out[i] < out[i + 1],
                "rank order broken at index {i}: {} >= {}",
                out[i],
                out[i + 1]
            );
        }
    }

    #[test]
    fn ema_convergence() {
        let mut n = PercentileNormalizer::new().with_decay(0.9);
        // Feed uniform [0, 100] repeatedly; EMA should converge toward 90.
        let returns: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        for _ in 0..200 {
            n.update(&returns);
        }
        let s = n.scale();
        assert!(
            (s - 90.0).abs() < 1.0,
            "after many updates EMA should converge to ~90, got {s}"
        );
    }

    #[test]
    fn scale_before_update() {
        let n = PercentileNormalizer::new();
        assert_eq!(n.scale(), 1.0, "uninitialized scale should be 1");
    }

    #[test]
    fn custom_percentiles() {
        let mut n = PercentileNormalizer::with_percentiles(0.10, 0.90);
        let returns: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        n.update(&returns);
        let s = n.scale();
        assert!(
            (s - 80.0).abs() < 1.0,
            "P90-P10 of [0..100] should be ~80, got {s}"
        );
    }

    #[test]
    fn update_and_normalize_combined() {
        let mut n = PercentileNormalizer::new();
        let returns: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        let advantages = vec![90.0];
        let out = n.update_and_normalize(&returns, &advantages);
        // scale ~90, so 90/90 ~ 1.0
        assert!(
            (out[0] - 1.0).abs() < 0.05,
            "expected ~1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn empty_advantages() {
        let n = PercentileNormalizer::new();
        assert!(n.normalize(&[]).is_empty());
    }
}
