//! fANOVA (functional ANOVA) importance evaluator.
//!
//! Estimates parameter importance by building a random forest of decision trees
//! and computing the mean decrease in impurity (variance) contributed by each
//! parameter. This is Optuna's default importance evaluator.

use std::collections::HashMap;

use rand::prelude::*;
use rand::rngs::StdRng;

use crate::study::Study;
use crate::trial::TrialState;

use super::ImportanceEvaluator;

/// fANOVA importance evaluator using a random forest.
///
/// Builds an ensemble of decision trees on the (params → objective) mapping
/// and estimates importance via mean decrease in impurity (variance reduction).
pub struct FanovaImportanceEvaluator {
    /// Number of trees in the forest.
    pub n_trees: usize,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Minimum number of samples to split a node.
    pub min_samples_split: usize,
    /// Fraction of features to consider at each split.
    pub max_features_frac: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for FanovaImportanceEvaluator {
    fn default() -> Self {
        Self {
            n_trees: 64,
            max_depth: 8,
            min_samples_split: 2,
            max_features_frac: 1.0,
            seed: 42,
        }
    }
}

impl FanovaImportanceEvaluator {
    pub fn new(n_trees: usize, max_depth: usize, seed: u64) -> Self {
        Self {
            n_trees,
            max_depth,
            seed,
            ..Default::default()
        }
    }
}

/// A single node in a decision tree.
#[derive(Debug)]
#[allow(dead_code)]
enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        impurity_decrease: f64,
    },
}

/// A decision tree for regression.
struct DecisionTree {
    #[allow(dead_code)]
    root: TreeNode,
    /// Accumulated impurity decrease per feature.
    feature_importances: Vec<f64>,
}

impl DecisionTree {
    fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        n_features: usize,
        max_depth: usize,
        min_samples_split: usize,
        max_features: usize,
        rng: &mut StdRng,
    ) -> Self {
        let mut feature_importances = vec![0.0; n_features];
        let indices: Vec<usize> = (0..x.len()).collect();
        let root = Self::build_node(
            x,
            y,
            &indices,
            n_features,
            max_depth,
            min_samples_split,
            max_features,
            rng,
            &mut feature_importances,
        );
        DecisionTree {
            root,
            feature_importances,
        }
    }

    fn build_node(
        x: &[Vec<f64>],
        y: &[f64],
        indices: &[usize],
        n_features: usize,
        depth: usize,
        min_samples_split: usize,
        max_features: usize,
        rng: &mut StdRng,
        feature_importances: &mut [f64],
    ) -> TreeNode {
        let n = indices.len();

        // Compute mean value of this node
        let mean_val: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;

        // Leaf conditions
        if depth == 0 || n < min_samples_split {
            return TreeNode::Leaf { value: mean_val };
        }

        // Compute node variance
        let node_var: f64 =
            indices.iter().map(|&i| (y[i] - mean_val).powi(2)).sum::<f64>() / n as f64;
        if node_var < 1e-15 {
            return TreeNode::Leaf { value: mean_val };
        }

        // Select random subset of features to consider
        let mut feature_candidates: Vec<usize> = (0..n_features).collect();
        feature_candidates.shuffle(rng);
        feature_candidates.truncate(max_features);

        // Find best split
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feat in &feature_candidates {
            // Collect unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[i][feat]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue;
            }

            // Try midpoints between consecutive unique values
            let n_candidates = values.len().min(20);
            let step = (values.len() - 1).max(1) / n_candidates.max(1);
            for vi in (0..values.len() - 1).step_by(step.max(1)) {
                let threshold = (values[vi] + values[vi + 1]) / 2.0;

                let left: Vec<usize> = indices
                    .iter()
                    .copied()
                    .filter(|&i| x[i][feat] <= threshold)
                    .collect();
                let right: Vec<usize> = indices
                    .iter()
                    .copied()
                    .filter(|&i| x[i][feat] > threshold)
                    .collect();

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                // Variance reduction
                let left_mean: f64 =
                    left.iter().map(|&i| y[i]).sum::<f64>() / left.len() as f64;
                let right_mean: f64 =
                    right.iter().map(|&i| y[i]).sum::<f64>() / right.len() as f64;
                let left_var: f64 = left
                    .iter()
                    .map(|&i| (y[i] - left_mean).powi(2))
                    .sum::<f64>()
                    / left.len() as f64;
                let right_var: f64 = right
                    .iter()
                    .map(|&i| (y[i] - right_mean).powi(2))
                    .sum::<f64>()
                    / right.len() as f64;

                let weighted_var = (left.len() as f64 * left_var
                    + right.len() as f64 * right_var)
                    / n as f64;
                let score = node_var - weighted_var;

                if score > best_score {
                    best_score = score;
                    best_feature = feat;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_score <= 0.0 || best_left.is_empty() || best_right.is_empty() {
            return TreeNode::Leaf { value: mean_val };
        }

        // Record impurity decrease weighted by number of samples
        let impurity_decrease = best_score * n as f64;
        feature_importances[best_feature] += impurity_decrease;

        let left_node = Self::build_node(
            x,
            y,
            &best_left,
            n_features,
            depth - 1,
            min_samples_split,
            max_features,
            rng,
            feature_importances,
        );
        let right_node = Self::build_node(
            x,
            y,
            &best_right,
            n_features,
            depth - 1,
            min_samples_split,
            max_features,
            rng,
            feature_importances,
        );

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left_node),
            right: Box::new(right_node),
            impurity_decrease,
        }
    }
}

impl ImportanceEvaluator for FanovaImportanceEvaluator {
    fn evaluate(&self, study: &Study) -> HashMap<String, f64> {
        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < 3 {
            return HashMap::new();
        }

        // Collect parameter names (sorted for deterministic ordering)
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

        // Build feature matrix X and target vector y
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

        // Build random forest
        let mut rng = StdRng::seed_from_u64(self.seed);
        let max_features = ((n_features as f64 * self.max_features_frac).ceil() as usize).max(1);
        let mut total_importances = vec![0.0; n_features];

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let bootstrap: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let x_boot: Vec<Vec<f64>> = bootstrap.iter().map(|&i| x[i].clone()).collect();
            let y_boot: Vec<f64> = bootstrap.iter().map(|&i| y[i]).collect();

            let tree = DecisionTree::fit(
                &x_boot,
                &y_boot,
                n_features,
                self.max_depth,
                self.min_samples_split,
                max_features,
                &mut rng,
            );

            for (j, &imp) in tree.feature_importances.iter().enumerate() {
                total_importances[j] += imp;
            }
        }

        // Normalize
        let total: f64 = total_importances.iter().sum();
        let mut result = HashMap::new();
        for (j, name) in param_names.iter().enumerate() {
            let importance = if total > 0.0 {
                total_importances[j] / total
            } else {
                0.0
            };
            result.insert(name.clone(), importance);
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
    fn test_fanova_basic() {
        let evaluator = FanovaImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        // Add trials where 'x' strongly determines objective and 'y' is noise
        for i in 0..100 {
            let x = (i as f64) / 100.0;
            let y = ((i * 7 + 3) % 100) as f64 / 100.0;
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

        // x should be much more important than y
        assert!(
            importances["x"] > importances["y"],
            "x importance {} should be > y importance {}",
            importances["x"],
            importances["y"]
        );
    }

    #[test]
    fn test_fanova_empty() {
        let evaluator = FanovaImportanceEvaluator::default();
        let study = Study::new(Direction::Minimize);
        let importances = evaluator.evaluate(&study);
        assert!(importances.is_empty());
    }

    #[test]
    fn test_fanova_normalizes() {
        let evaluator = FanovaImportanceEvaluator::new(32, 6, 42);
        let mut study = Study::new(Direction::Minimize);

        for i in 0..50 {
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
    fn test_fanova_three_params() {
        let evaluator = FanovaImportanceEvaluator::default();
        let mut study = Study::new(Direction::Minimize);

        // objective = a^2 + 0.1*b, c is noise
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..100 {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let c: f64 = rng.random();
            let value = a * a + 0.1 * b;

            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(value);
            trial.params.insert("a".to_string(), a);
            trial.params.insert("b".to_string(), b);
            trial.params.insert("c".to_string(), c);
            study.add_trial(trial);
        }

        let importances = evaluator.evaluate(&study);
        // 'a' should be most important, then 'b', then 'c'
        assert!(
            importances["a"] > importances["c"],
            "a ({}) should be more important than c ({})",
            importances["a"],
            importances["c"]
        );
    }
}
