//! PED-ANOVA (Parzen Estimator-based Decomposed ANOVA) importance evaluator.
//!
//! A fast parameter importance method that fits a Parzen estimator for each
//! parameter on the "good" trials (top fraction) and compares against a
//! baseline uniform distribution. More divergent = more important.

use std::collections::HashMap;

use crate::study::Study;
use crate::trial::TrialState;

use super::ImportanceEvaluator;

/// PED-ANOVA importance evaluator.
///
/// Splits trials into "good" and "all" groups, fits per-parameter density
/// estimates on each, and measures how much the good-trial distribution
/// differs from the overall distribution using KL divergence.
pub struct PedAnovaImportanceEvaluator {
    /// Fraction of trials considered "good" (top fraction by objective).
    pub top_fraction: f64,
    /// Number of bins for discretized KL divergence estimation.
    pub n_bins: usize,
}

impl Default for PedAnovaImportanceEvaluator {
    fn default() -> Self {
        Self {
            top_fraction: 0.1,
            n_bins: 20,
        }
    }
}

impl PedAnovaImportanceEvaluator {
    /// Create a new evaluator with the given top fraction.
    pub fn new(top_fraction: f64) -> Self {
        Self {
            top_fraction,
            ..Default::default()
        }
    }

    /// Estimate KL divergence between the "good" distribution and uniform
    /// for a single parameter's values.
    fn parameter_importance(
        &self,
        good_values: &[f64],
        all_values: &[f64],
        low: f64,
        high: f64,
    ) -> f64 {
        if good_values.is_empty() || all_values.len() < 2 || (high - low).abs() < 1e-15 {
            return 0.0;
        }

        let n_bins = self.n_bins;
        let bin_width = (high - low) / n_bins as f64;

        // Build histograms
        let mut good_hist = vec![0.0f64; n_bins];
        let mut all_hist = vec![0.0f64; n_bins];

        for &v in good_values {
            let bin = ((v - low) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            good_hist[bin] += 1.0;
        }

        for &v in all_values {
            let bin = ((v - low) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            all_hist[bin] += 1.0;
        }

        // Normalize to probabilities with Laplace smoothing
        let good_total = good_values.len() as f64 + n_bins as f64 * 0.01;
        let all_total = all_values.len() as f64 + n_bins as f64 * 0.01;

        // KL(good || all)
        let mut kl = 0.0f64;
        for i in 0..n_bins {
            let p = (good_hist[i] + 0.01) / good_total;
            let q = (all_hist[i] + 0.01) / all_total;
            if p > 0.0 && q > 0.0 {
                kl += p * (p / q).ln();
            }
        }

        kl.max(0.0)
    }
}

impl ImportanceEvaluator for PedAnovaImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> HashMap<String, f64> {
        let mut completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < 2 {
            return HashMap::new();
        }

        // Sort by objective value
        completed.sort_by(|a, b| {
            let va = a.value.unwrap();
            let vb = b.value.unwrap();
            match study.direction() {
                crate::study::Direction::Minimize => {
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                }
                crate::study::Direction::Maximize => {
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });

        let n_good = ((completed.len() as f64 * self.top_fraction).ceil() as usize).max(1);
        let good_trials = &completed[..n_good];

        // Collect all parameter names
        let mut param_names: Vec<String> = Vec::new();
        for t in &completed {
            for name in t.params.keys() {
                if !param_names.contains(name) {
                    param_names.push(name.clone());
                }
            }
        }

        // Compute importance for each parameter
        let mut importances = HashMap::new();
        for name in &param_names {
            let all_values: Vec<f64> = completed
                .iter()
                .filter_map(|t| t.params.get(name).copied())
                .collect();

            let good_values: Vec<f64> = good_trials
                .iter()
                .filter_map(|t| t.params.get(name).copied())
                .collect();

            if all_values.len() < 2 {
                importances.insert(name.clone(), 0.0);
                continue;
            }

            let low = all_values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let high = all_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let importance = self.parameter_importance(&good_values, &all_values, low, high);
            importances.insert(name.clone(), importance);
        }

        // Normalize to sum to 1.0
        let total: f64 = importances.values().sum();
        if total > 0.0 {
            for v in importances.values_mut() {
                *v /= total;
            }
        }

        importances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::{FrozenTrial, TrialState};

    #[test]
    fn test_ped_anova_basic() {
        let evaluator = PedAnovaImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        // Add trials where 'x' strongly correlates with objective
        // and 'y' is random noise
        for i in 0..50 {
            let x = (i as f64) / 50.0;
            let y = ((i * 7 + 3) % 50) as f64 / 50.0; // pseudo-random
            let value = x * x; // objective depends only on x

            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(value);
            trial.params.insert("x".to_string(), x);
            trial.params.insert("y".to_string(), y);
            study.add_trial(trial);
        }

        let importances = evaluator.evaluate(&study);
        assert!(importances.contains_key("x"));
        assert!(importances.contains_key("y"));

        // x should be more important than y since objective = x^2
        assert!(
            importances["x"] > importances["y"],
            "x importance {} should be > y importance {}",
            importances["x"],
            importances["y"]
        );
    }

    #[test]
    fn test_ped_anova_empty_study() {
        let evaluator = PedAnovaImportanceEvaluator::default();
        let study = Study::new(Direction::Minimize);
        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }

    #[test]
    fn test_ped_anova_normalizes() {
        let evaluator = PedAnovaImportanceEvaluator::new(0.2);
        let mut study = Study::new(Direction::Minimize);

        for i in 0..30 {
            let x = i as f64;
            let y = (i * 3) as f64;
            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(x + y);
            trial.params.insert("x".to_string(), x);
            trial.params.insert("y".to_string(), y);
            study.add_trial(trial);
        }

        let importances = evaluator.evaluate(&study);
        let total: f64 = importances.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "importances should sum to 1.0, got {total}"
        );
    }
}
