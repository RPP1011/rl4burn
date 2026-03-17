# League Training

AlphaStar-style multi-agent training with role-based specialization. Multiple agents train simultaneously with different objectives.

## Agent Roles

| Role | Opponents | Purpose |
|------|-----------|---------|
| **Main Agent** | 35% self-play + 65% PFSP pool | General strength |
| **Main Exploiter** | Only the main agent | Find main agent's weaknesses |
| **League Exploiter** | PFSP across full pool | Find weaknesses across all strategies |

## API

```rust,ignore
use rl4burn::{League, AgentRole, LeagueAgentConfig};

let mut league = League::new();
league.set_initial_model(initial_model.clone());

// Add agents
league.add_agent(model.clone(), LeagueAgentConfig {
    role: AgentRole::MainAgent,
    checkpoint_interval: 1000,
    reset_threshold: 0,
});
league.add_agent(model.clone(), LeagueAgentConfig {
    role: AgentRole::MainExploiter,
    checkpoint_interval: 2000,
    reset_threshold: 50000,
});

// Training loop
let opponent = league.get_opponent(agent_idx, &mut rng);
// ... play game, update model ...
league.update_agent(agent_idx); // handles checkpointing
```

## Checkpointing

Every `checkpoint_interval` steps, the agent's current weights are frozen and added to the opponent pool. All agents can then play against these frozen snapshots.

## Exploiter resets

Exploiters that stop improving get reset to the initial model weights:

```rust,ignore
league.reset_exploiter(exploiter_idx);
```
