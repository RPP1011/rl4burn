//! UCT-based Monte Carlo Tree Search for unit composition / drafting (Issue #29).
//!
//! Provides a generic MCTS implementation that uses Upper Confidence bounds
//! applied to Trees (UCT) for node selection.  The evaluate function is
//! user-supplied, making this suitable for both full rollout evaluation and
//! neural-network value estimation.

use rand::Rng;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A node in the MCTS tree.
#[derive(Debug)]
struct MctsNode {
    /// Number of visits.
    visits: u32,
    /// Total value accumulated.
    total_value: f64,
    /// Children indexed by action (indices into the tree's node pool).
    children: Vec<Option<usize>>,
    /// Number of possible actions from this node.
    n_actions: usize,
    /// Parent node index (None for root).
    parent: Option<usize>,
    /// Action that led to this node (retained for debugging/introspection).
    #[allow(dead_code)]
    action: Option<usize>,
}

impl MctsNode {
    fn new(n_actions: usize, parent: Option<usize>, action: Option<usize>) -> Self {
        Self {
            visits: 0,
            total_value: 0.0,
            children: vec![None; n_actions],
            n_actions,
            parent,
            action,
        }
    }

    fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }

    /// UCT score: mean_value + c * sqrt(ln(parent_visits) / visits)
    fn uct_score(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::MAX; // unvisited = highest priority
        }
        self.mean_value()
            + exploration_constant
                * ((parent_visits as f64).ln() / self.visits as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// MCTS configuration.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of simulations per search. Default: 100
    pub n_simulations: u32,
    /// UCT exploration constant. Default: sqrt(2)
    pub exploration_constant: f64,
    /// Number of possible actions at each node.
    pub n_actions: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            n_simulations: 100,
            exploration_constant: std::f64::consts::SQRT_2,
            n_actions: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// MCTS tree for search.
pub struct MctsTree {
    nodes: Vec<MctsNode>,
    config: MctsConfig,
}

impl MctsTree {
    /// Create a new MCTS tree with a root node.
    pub fn new(config: MctsConfig) -> Self {
        let root = MctsNode::new(config.n_actions, None, None);
        Self {
            nodes: vec![root],
            config,
        }
    }

    /// Run MCTS search from the root.
    ///
    /// # Arguments
    /// * `evaluate` - Function that takes a sequence of actions and returns a value estimate.
    /// * `rng` - Random number generator for tie-breaking and expansion.
    ///
    /// # Returns
    /// Visit counts for each action from the root.
    pub fn search<F>(
        &mut self,
        mut evaluate: F,
        rng: &mut impl Rng,
    ) -> Vec<u32>
    where
        F: FnMut(&[usize]) -> f64,
    {
        for _ in 0..self.config.n_simulations {
            let mut node_idx = 0; // start at root
            let mut action_path = Vec::new();

            // Selection: traverse tree using UCT
            loop {
                let node = &self.nodes[node_idx];

                // Check if any child is unexpanded
                let unexpanded: Vec<usize> = (0..node.n_actions)
                    .filter(|&a| node.children[a].is_none())
                    .collect();

                if !unexpanded.is_empty() {
                    // Expansion: pick a random unexpanded action
                    let action = unexpanded[rng.random_range(0..unexpanded.len())];
                    action_path.push(action);

                    let child_idx = self.nodes.len();
                    let child = MctsNode::new(self.config.n_actions, Some(node_idx), Some(action));
                    self.nodes.push(child);
                    self.nodes[node_idx].children[action] = Some(child_idx);
                    node_idx = child_idx;
                    break;
                }

                // All children expanded: select best by UCT
                let parent_visits = node.visits;
                let n_actions = node.n_actions;
                let children = node.children.clone();
                let best_action = (0..n_actions)
                    .max_by(|&a, &b| {
                        let a_score = self.nodes[children[a].unwrap()]
                            .uct_score(parent_visits, self.config.exploration_constant);
                        let b_score = self.nodes[children[b].unwrap()]
                            .uct_score(parent_visits, self.config.exploration_constant);
                        a_score
                            .partial_cmp(&b_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();

                action_path.push(best_action);
                node_idx = self.nodes[node_idx].children[best_action].unwrap();
            }

            // Simulation: evaluate the leaf
            let value = evaluate(&action_path);

            // Backpropagation: update all nodes on the path
            let mut idx = node_idx;
            loop {
                self.nodes[idx].visits += 1;
                self.nodes[idx].total_value += value;
                if let Some(parent) = self.nodes[idx].parent {
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        // Return visit counts for root's children
        (0..self.config.n_actions)
            .map(|a| {
                self.nodes[0].children[a]
                    .map(|idx| self.nodes[idx].visits)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get the best action (most visited from root).
    pub fn best_action(&self) -> usize {
        (0..self.config.n_actions)
            .max_by_key(|&a| {
                self.nodes[0].children[a]
                    .map(|idx| self.nodes[idx].visits)
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    /// Get action probabilities proportional to visit counts.
    pub fn action_probs(&self) -> Vec<f64> {
        let visits: Vec<u32> = (0..self.config.n_actions)
            .map(|a| {
                self.nodes[0].children[a]
                    .map(|idx| self.nodes[idx].visits)
                    .unwrap_or(0)
            })
            .collect();
        let total: u32 = visits.iter().sum();
        if total == 0 {
            return vec![1.0 / self.config.n_actions as f64; self.config.n_actions];
        }
        visits.iter().map(|&v| v as f64 / total as f64).collect()
    }

    /// Number of nodes in the tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_counts_sum_to_n_simulations() {
        let config = MctsConfig {
            n_simulations: 50,
            n_actions: 4,
            ..Default::default()
        };
        let mut tree = MctsTree::new(config);
        let mut rng = rand::rng();

        let visits = tree.search(|_actions| 0.5, &mut rng);

        let total: u32 = visits.iter().sum();
        assert_eq!(total, 50, "total visits = {total}, expected 50");
    }

    #[test]
    fn best_action_selects_most_visited() {
        let config = MctsConfig {
            n_simulations: 200,
            n_actions: 3,
            ..Default::default()
        };
        let mut tree = MctsTree::new(config);
        let mut rng = rand::rng();

        // Action 1 always returns high value -> should get most visits
        tree.search(
            |actions| {
                if actions.first() == Some(&1) {
                    1.0
                } else {
                    0.0
                }
            },
            &mut rng,
        );

        let best = tree.best_action();
        assert_eq!(best, 1, "best action = {best}, expected 1");
    }

    #[test]
    fn action_probs_sum_to_one() {
        let config = MctsConfig {
            n_simulations: 100,
            n_actions: 5,
            ..Default::default()
        };
        let mut tree = MctsTree::new(config);
        let mut rng = rand::rng();

        tree.search(|_| 0.5, &mut rng);

        let probs = tree.action_probs();
        assert_eq!(probs.len(), 5);

        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "probs sum = {total}, expected 1.0"
        );
    }

    #[test]
    fn empty_tree_action_probs_uniform() {
        let config = MctsConfig {
            n_simulations: 0,
            n_actions: 4,
            ..Default::default()
        };
        let tree = MctsTree::new(config);
        let probs = tree.action_probs();

        for p in &probs {
            assert!(
                (p - 0.25).abs() < 1e-10,
                "expected uniform 0.25, got {p}"
            );
        }
    }

    #[test]
    fn tree_grows_nodes() {
        let config = MctsConfig {
            n_simulations: 50,
            n_actions: 3,
            ..Default::default()
        };
        let mut tree = MctsTree::new(config);
        let mut rng = rand::rng();

        assert_eq!(tree.n_nodes(), 1); // just root
        tree.search(|_| 0.5, &mut rng);
        assert!(tree.n_nodes() > 1, "tree should grow beyond root");
    }
}
