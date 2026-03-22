//! Functional ANOVA (fANOVA) importance evaluator.
//!
//! Builds a random forest from (params -> objective) data and decomposes
//! the total variance by parameter using weighted variance reductions at
//! each split node. Unlike plain MDI, fANOVA weights each split's
//! contribution by the fraction of total samples that reach the node,
//! giving a proper marginal variance decomposition.

use std::collections::HashMap;

use crate::study::Study;
use crate::trial::TrialState;

use super::ImportanceEvaluator;

/// fANOVA (functional ANOVA) importance evaluator.
///
/// Fits a random forest on completed trials and decomposes variance by
/// parameter. Each split's variance reduction is weighted by the fraction
/// of total training samples passing through that node, yielding marginal
/// importance scores that properly account for sample distribution.
pub struct FanovaImportanceEvaluator {
    /// Number of bootstrap trees in the forest.
    pub n_trees: usize,
    /// Maximum depth of each tree (0 = unlimited).
    pub max_depth: usize,
    /// Minimum number of samples required to attempt a split.
    pub min_samples_split: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for FanovaImportanceEvaluator {
    fn default() -> Self {
        Self {
            n_trees: 64,
            max_depth: 6,
            min_samples_split: 3,
            seed: 42,
        }
    }
}

impl FanovaImportanceEvaluator {
    /// Create a new evaluator with the given number of trees.
    pub fn new(n_trees: usize) -> Self {
        Self {
            n_trees,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Simple PRNG (xorshift64)
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
// fANOVA tree (arena-based)
// ---------------------------------------------------------------------------

/// A node in an fANOVA decision tree stored in an arena (Vec).
enum TreeNode {
    Leaf {
        value: f64,
        n_samples: usize,
    },
    Split {
        param_idx: usize,
        threshold: f64,
        left: usize,
        right: usize,
        variance_reduction: f64,
        n_samples: usize,
    },
}

/// An fANOVA decision tree stored as an arena of nodes.
struct FanovaTree {
    splits: Vec<TreeNode>,
}

impl FanovaTree {
    /// Build a tree from data. Returns the tree and per-feature importance.
    fn build(
        x: &[Vec<f64>],
        y: &[f64],
        n_features: usize,
        max_depth: usize,
        min_samples_split: usize,
        rng: &mut Rng,
    ) -> (Self, Vec<f64>) {
        let mut tree = FanovaTree {
            splits: Vec::new(),
        };
        let total_n = y.len();
        let mut importance = vec![0.0f64; n_features];

        tree.build_recursive(
            x,
            y,
            n_features,
            max_depth,
            min_samples_split,
            total_n,
            0,
            &mut importance,
            rng,
        );

        (tree, importance)
    }

    fn build_recursive(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        n_features: usize,
        max_depth: usize,
        min_samples_split: usize,
        total_n: usize,
        depth: usize,
        importance: &mut [f64],
        rng: &mut Rng,
    ) -> usize {
        let n = y.len();
        let mean = y.iter().sum::<f64>() / n as f64;

        // Stopping conditions
        if n < min_samples_split || (max_depth > 0 && depth >= max_depth) {
            let idx = self.splits.len();
            self.splits.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        // Variance of current node
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        if var < 1e-15 {
            let idx = self.splits.len();
            self.splits.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        // Randomly sample sqrt(n_features) candidate features.
        let n_candidates = (n_features as f64).sqrt().ceil() as usize;
        let mut candidate_features = Vec::with_capacity(n_candidates);
        let mut available: Vec<usize> = (0..n_features).collect();
        for _ in 0..n_candidates.min(n_features) {
            let idx = rng.index(available.len());
            candidate_features.push(available.swap_remove(idx));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        for &feat in &candidate_features {
            let mut vals: Vec<f64> = x.iter().map(|row| row[feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() < 2 {
                continue;
            }

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
            let idx = self.splits.len();
            self.splits.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        // fANOVA key difference: weight variance reduction by fraction of
        // total samples reaching this node (n / total_n), giving a proper
        // marginal variance decomposition.
        let weight = n as f64 / total_n as f64;
        importance[best_feature] += best_score * weight;

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

        // Reserve a slot for this split node
        let node_idx = self.splits.len();
        self.splits.push(TreeNode::Leaf {
            value: 0.0,
            n_samples: 0,
        }); // placeholder

        let left_idx = self.build_recursive(
            &left_x,
            &left_y,
            n_features,
            max_depth,
            min_samples_split,
            total_n,
            depth + 1,
            importance,
            rng,
        );
        let right_idx = self.build_recursive(
            &right_x,
            &right_y,
            n_features,
            max_depth,
            min_samples_split,
            total_n,
            depth + 1,
            importance,
            rng,
        );

        self.splits[node_idx] = TreeNode::Split {
            param_idx: best_feature,
            threshold: best_threshold,
            left: left_idx,
            right: right_idx,
            variance_reduction: best_score,
            n_samples: n,
        };

        node_idx
    }
}

// ---------------------------------------------------------------------------
// ImportanceEvaluator implementation
// ---------------------------------------------------------------------------

impl ImportanceEvaluator for FanovaImportanceEvaluator {
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
        param_names.sort();

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

        let y: Vec<f64> = completed.iter().map(|t| t.value.unwrap()).collect();

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

            let (_tree, imp) = FanovaTree::build(
                &bx,
                &by,
                n_features,
                self.max_depth,
                self.min_samples_split,
                &mut rng,
            );

            for (i, v) in imp.iter().enumerate() {
                total_importance[i] += v;
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
    fn test_fanova_x_squared_dominates() {
        // f(x, y) = x^2 + 0.01 * y^2  =>  x should be far more important than y
        let evaluator = FanovaImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        for i in 0..80 {
            let x = (i as f64) / 80.0;
            let y = ((i * 7 + 3) % 80) as f64 / 80.0;
            let value = x * x + 0.01 * y * y;

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

        // x should dominate significantly
        assert!(
            importances["x"] > 0.7,
            "x importance {} should be > 0.7",
            importances["x"]
        );
    }

    #[test]
    fn test_fanova_empty_study() {
        let evaluator = FanovaImportanceEvaluator::default();
        let study = Study::new(Direction::Minimize);
        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }

    #[test]
    fn test_fanova_normalizes_to_one() {
        let evaluator = FanovaImportanceEvaluator::new(32);
        let mut study = Study::new(Direction::Minimize);

        for i in 0..40 {
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
    fn test_fanova_single_trial() {
        let evaluator = FanovaImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Complete;
        trial.value = Some(1.0);
        trial.params.insert("x".to_string(), 0.5);
        study.add_trial(trial);

        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }

    #[test]
    fn test_fanova_three_params() {
        // f(a, b, c) = 10*a + 0.1*b + 0.001*c
        // a should be most important, c least
        let evaluator = FanovaImportanceEvaluator::new(64);
        let mut study = Study::new(Direction::Minimize);

        for i in 0..100 {
            let a = (i as f64) / 100.0;
            let b = ((i * 13 + 7) % 100) as f64 / 100.0;
            let c = ((i * 31 + 11) % 100) as f64 / 100.0;
            let value = 10.0 * a + 0.1 * b + 0.001 * c;

            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(value);
            trial.params.insert("a".to_string(), a);
            trial.params.insert("b".to_string(), b);
            trial.params.insert("c".to_string(), c);
            study.add_trial(trial);
        }

        let importances = evaluator.evaluate(&study);
        assert!(
            importances["a"] > importances["b"],
            "a ({}) should be > b ({})",
            importances["a"],
            importances["b"]
        );
        assert!(
            importances["b"] > importances["c"],
            "b ({}) should be > c ({})",
            importances["b"],
            importances["c"]
        );
    }
}
