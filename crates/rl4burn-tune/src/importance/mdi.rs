//! Mean Decrease Impurity (MDI) importance evaluator.
//!
//! Builds a simple random forest (ensemble of decision trees) from
//! (params -> objective_value) data and uses the accumulated variance
//! reduction at each split to rank parameter importance.

use std::collections::HashMap;

use crate::study::Study;
use crate::trial::TrialState;

use super::ImportanceEvaluator;

/// MDI (Mean Decrease Impurity) importance evaluator.
///
/// Fits a random forest on completed trials and measures each parameter's
/// importance by the total variance reduction it contributes across all
/// splits, averaged over bootstrap trees.
pub struct MdiImportanceEvaluator {
    /// Number of bootstrap trees in the forest.
    pub n_trees: usize,
    /// Maximum depth of each tree (0 = unlimited).
    pub max_depth: usize,
    /// Minimum number of samples required to attempt a split.
    pub min_samples_split: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for MdiImportanceEvaluator {
    fn default() -> Self {
        Self {
            n_trees: 64,
            max_depth: 6,
            min_samples_split: 3,
            seed: 42,
        }
    }
}

impl MdiImportanceEvaluator {
    /// Create a new evaluator with the given number of trees.
    pub fn new(n_trees: usize) -> Self {
        Self {
            n_trees,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Simple PRNG (xorshift64) for bootstrap sampling
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1)) // avoid zero state
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Return a random index in `0..n`.
    fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Decision tree
// ---------------------------------------------------------------------------

/// A node in a decision tree.
enum Node {
    Leaf {
        mean: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// Accumulated importance for each feature during tree construction.
struct TreeBuilder {
    n_features: usize,
    importance: Vec<f64>,
    max_depth: usize,
    min_samples_split: usize,
}

impl TreeBuilder {
    fn new(n_features: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            n_features,
            importance: vec![0.0; n_features],
            max_depth,
            min_samples_split,
        }
    }

    /// Build a tree on the given subset of data.
    /// `x` is row-major: x[sample][feature], `y` are targets.
    fn build(&mut self, x: &[Vec<f64>], y: &[f64], rng: &mut Rng) -> Node {
        self.build_recursive(x, y, 0, rng)
    }

    fn build_recursive(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        depth: usize,
        rng: &mut Rng,
    ) -> Node {
        let n = y.len();
        let mean = y.iter().sum::<f64>() / n as f64;

        // Stopping conditions
        if n < self.min_samples_split
            || (self.max_depth > 0 && depth >= self.max_depth)
        {
            return Node::Leaf { mean };
        }

        // Variance of current node
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        if var < 1e-15 {
            return Node::Leaf { mean };
        }

        // Try to find the best split.
        // Randomly sample sqrt(n_features) candidate features.
        let n_candidates = (self.n_features as f64).sqrt().ceil() as usize;
        let mut candidate_features = Vec::with_capacity(n_candidates);
        let mut available: Vec<usize> = (0..self.n_features).collect();
        for _ in 0..n_candidates.min(self.n_features) {
            let idx = rng.index(available.len());
            candidate_features.push(available.swap_remove(idx));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        for &feat in &candidate_features {
            // Collect unique sorted values for this feature
            let mut vals: Vec<f64> = x.iter().map(|row| row[feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() < 2 {
                continue;
            }

            // Try midpoints between consecutive unique values
            for pair in vals.windows(2) {
                let threshold = (pair[0] + pair[1]) / 2.0;

                let mut left_sum = 0.0;
                let mut left_sq = 0.0;
                let mut left_n = 0usize;
                let mut right_sum = 0.0;
                let mut right_sq = 0.0;
                let mut right_n = 0usize;

                for i in 0..n {
                    if x[i][feat] <= threshold {
                        left_sum += y[i];
                        left_sq += y[i] * y[i];
                        left_n += 1;
                    } else {
                        right_sum += y[i];
                        right_sq += y[i] * y[i];
                        right_n += 1;
                    }
                }

                if left_n == 0 || right_n == 0 {
                    continue;
                }

                let left_var = left_sq - left_sum * left_sum / left_n as f64;
                let right_var = right_sq - right_sum * right_sum / right_n as f64;
                let reduction = var - left_var - right_var;

                if reduction > best_score {
                    best_score = reduction;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        if best_score <= 0.0 {
            return Node::Leaf { mean };
        }

        // Record the weighted impurity decrease for this feature
        self.importance[best_feature] += best_score;

        // Partition data
        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for i in 0..n {
            if x[i][best_feature] <= best_threshold {
                left_x.push(x[i].clone());
                left_y.push(y[i]);
            } else {
                right_x.push(x[i].clone());
                right_y.push(y[i]);
            }
        }

        let left = self.build_recursive(&left_x, &left_y, depth + 1, rng);
        let right = self.build_recursive(&right_x, &right_y, depth + 1, rng);

        Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

impl ImportanceEvaluator for MdiImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> HashMap<String, f64> {
        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < 2 {
            return HashMap::new();
        }

        // Collect all parameter names in a stable order.
        let mut param_names: Vec<String> = Vec::new();
        for t in &completed {
            for name in t.params.keys() {
                if !param_names.contains(name) {
                    param_names.push(name.clone());
                }
            }
        }
        param_names.sort(); // deterministic order

        if param_names.is_empty() {
            return HashMap::new();
        }

        let n_features = param_names.len();

        // Build data matrix. Missing params get 0.0.
        let x: Vec<Vec<f64>> = completed
            .iter()
            .map(|t| {
                param_names
                    .iter()
                    .map(|name| t.params.get(name).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();

        let y: Vec<f64> = completed
            .iter()
            .map(|t| t.value.unwrap())
            .collect();

        let n = x.len();

        // Accumulate importance over bootstrap trees.
        let mut total_importance = vec![0.0f64; n_features];
        let mut rng = Rng::new(self.seed);

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let mut bx = Vec::with_capacity(n);
            let mut by = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.index(n);
                bx.push(x[idx].clone());
                by.push(y[idx]);
            }

            let mut builder =
                TreeBuilder::new(n_features, self.max_depth, self.min_samples_split);
            let _tree = builder.build(&bx, &by, &mut rng);

            for (i, imp) in builder.importance.iter().enumerate() {
                total_importance[i] += imp;
            }
        }

        // Build result map and normalize.
        let total: f64 = total_importance.iter().sum();
        let mut result = HashMap::new();

        if total > 0.0 {
            for (i, name) in param_names.iter().enumerate() {
                result.insert(name.clone(), total_importance[i] / total);
            }
        } else {
            // All importances are zero — assign uniform
            let uniform = 1.0 / n_features as f64;
            for name in &param_names {
                result.insert(name.clone(), uniform);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::{FrozenTrial, TrialState};

    #[test]
    fn test_mdi_basic() {
        let evaluator = MdiImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        // 'x' strongly determines objective, 'y' is noise
        for i in 0..50 {
            let x = (i as f64) / 50.0;
            let y = ((i * 7 + 3) % 50) as f64 / 50.0;
            let value = x * x;

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

        assert!(
            importances["x"] > importances["y"],
            "x importance {} should be > y importance {}",
            importances["x"],
            importances["y"]
        );
    }

    #[test]
    fn test_mdi_empty_study() {
        let evaluator = MdiImportanceEvaluator::default();
        let study = Study::new(Direction::Minimize);
        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }

    #[test]
    fn test_mdi_normalizes() {
        let evaluator = MdiImportanceEvaluator::new(32);
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

    #[test]
    fn test_mdi_single_trial() {
        let evaluator = MdiImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Complete;
        trial.value = Some(1.0);
        trial.params.insert("x".to_string(), 0.5);
        study.add_trial(trial);

        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }
}
