//! Wilcoxon signed-rank test pruner.
//!
//! Prunes a trial if its intermediate values are statistically significantly
//! worse than the best trial's intermediate values at the same steps,
//! using the Wilcoxon signed-rank test.

use crate::study::{Direction, Study};
use crate::trial::{FrozenTrial, TrialState};

use super::Pruner;

/// Pruner based on the Wilcoxon signed-rank test.
///
/// Compares the current trial's intermediate values against the best trial's
/// intermediate values at matching steps. If the p-value is below the
/// threshold, the trial is pruned.
pub struct WilcoxonPruner {
    /// P-value threshold for pruning.
    pub p_threshold: f64,
    /// Minimum number of matching steps required before testing.
    pub n_min_steps: usize,
    /// Number of startup trials before pruning begins.
    pub n_startup_trials: usize,
}

impl WilcoxonPruner {
    /// Create a new Wilcoxon pruner.
    pub fn new(p_threshold: f64, n_min_steps: usize, n_startup_trials: usize) -> Self {
        Self {
            p_threshold,
            n_min_steps,
            n_startup_trials,
        }
    }
}

impl Default for WilcoxonPruner {
    fn default() -> Self {
        Self {
            p_threshold: 0.1,
            n_min_steps: 5,
            n_startup_trials: 5,
        }
    }
}

/// Compute the Wilcoxon signed-rank test statistic and approximate p-value.
///
/// Returns `None` if there are fewer than `n_min` paired observations.
/// Uses the normal approximation for the test statistic.
fn wilcoxon_signed_rank_test(
    differences: &[f64],
    n_min: usize,
) -> Option<f64> {
    // Filter out zeros
    let nonzero: Vec<f64> = differences.iter().copied().filter(|&d| d.abs() > 1e-15).collect();
    let n = nonzero.len();

    if n < n_min {
        return None;
    }

    // Rank by absolute value
    let mut abs_ranked: Vec<(usize, f64)> = nonzero
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d.abs()))
        .collect();
    abs_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute ranks (1-based) with tie handling
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (abs_ranked[j].1 - abs_ranked[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // Average rank for ties
        for k in i..j {
            ranks[abs_ranked[k].0] = avg_rank;
        }
        i = j;
    }

    // W+ = sum of ranks where difference is positive
    let w_plus: f64 = nonzero
        .iter()
        .enumerate()
        .filter(|(_, &d)| d > 0.0)
        .map(|(i, _)| ranks[i])
        .sum();

    // Normal approximation
    let n_f = n as f64;
    let expected = n_f * (n_f + 1.0) / 4.0;
    let variance = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;
    let std_dev = variance.sqrt();

    if std_dev < 1e-15 {
        return Some(1.0); // All differences equal
    }

    let z = (w_plus - expected) / std_dev;

    // Two-tailed p-value using normal CDF approximation
    let p = 2.0 * normal_cdf(-z.abs());
    Some(p)
}

/// Standard normal CDF approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function approximation (Abramowitz & Stegun).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        2.0 - result
    }
}

impl Pruner for WilcoxonPruner {
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool {
        let completed: Vec<&FrozenTrial> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.len() < self.n_startup_trials {
            return false;
        }

        // Find the best completed trial
        let best = match study.best_trial() {
            Some(t) => t,
            None => return false,
        };

        // Compute paired differences at matching steps
        let mut differences = Vec::new();
        for (&step, &trial_val) in &trial.intermediate_values {
            if let Some(&best_val) = best.intermediate_values.get(&step) {
                let diff = match study.direction() {
                    Direction::Minimize => trial_val - best_val, // Positive = trial is worse
                    Direction::Maximize => best_val - trial_val, // Positive = trial is worse
                };
                differences.push(diff);
            }
        }

        // Run the test
        match wilcoxon_signed_rank_test(&differences, self.n_min_steps) {
            Some(p) => {
                // If the test is significant AND the trial is worse, prune
                if p < self.p_threshold {
                    // Check if the trial is actually worse (mean difference > 0)
                    let mean_diff: f64 =
                        differences.iter().sum::<f64>() / differences.len() as f64;
                    mean_diff > 0.0
                } else {
                    false
                }
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::FrozenTrial;

    #[test]
    fn test_wilcoxon_no_prune_during_startup() {
        let pruner = WilcoxonPruner::new(0.1, 5, 10);
        let study = Study::new(Direction::Minimize);
        let trial = FrozenTrial::new(0);
        assert!(!pruner.prune(&study, &trial));
    }

    #[test]
    fn test_wilcoxon_signed_rank_basic() {
        // Clearly positive differences -> small p-value
        let diffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = wilcoxon_signed_rank_test(&diffs, 5).unwrap();
        assert!(p < 0.05, "p={p} should be < 0.05 for all-positive differences");

        // Mixed differences -> higher p-value
        let diffs = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
        let p = wilcoxon_signed_rank_test(&diffs, 5).unwrap();
        assert!(p > 0.5, "p={p} should be > 0.5 for balanced differences");
    }

    #[test]
    fn test_wilcoxon_not_enough_steps() {
        let result = wilcoxon_signed_rank_test(&[1.0, 2.0], 5);
        assert!(result.is_none());
    }
}
