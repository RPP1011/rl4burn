# Agent Branching

Clone an agent's weights to create a new specialized agent. Used by SCC for initializing exploiters from the current main agent rather than from scratch.

## API

```rust,ignore
use rl4burn::algo::multi_agent::self_play::branch_agent;

let exploiter = branch_agent(&main_agent);
let mut exploiter_optim = AdamConfig::new().init();  // fresh optimizer!
```

## Why branch?

Starting exploiters from the main agent's current weights (instead of the supervised model) gives them a head start. They already know how to play — they just need to specialize in exploiting weaknesses.

The key: the optimizer state must be reset. `branch_agent` clones the model; you create a fresh optimizer.
