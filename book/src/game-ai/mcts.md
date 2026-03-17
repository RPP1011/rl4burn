# MCTS for Drafting

UCT-based Monte Carlo Tree Search for pre-game decisions like unit composition or hero drafting.

## API

```rust,ignore
use rl4burn::algo::mcts::{MctsTree, MctsConfig};

let mut tree = MctsTree::new(MctsConfig {
    n_simulations: 800,
    exploration_constant: 1.41,
    n_actions: 30,  // number of possible picks
});

let visit_counts = tree.search(|action_path| {
    // Evaluate this sequence of picks.
    // Return estimated win rate (0.0 to 1.0).
    evaluate_composition(action_path)
}, &mut rng);

let best_pick = tree.best_action();
let pick_probs = tree.action_probs();
```

## How UCT works

1. **Select**: walk down the tree, choosing children by UCT score = `mean_value + c * sqrt(ln(parent_visits) / visits)`
2. **Expand**: add a new child for an unexplored action
3. **Evaluate**: call your evaluation function on the action sequence
4. **Backpropagate**: update visit counts and values up to the root

After all simulations, pick the most-visited action (not the highest-value — visit count is more robust).
