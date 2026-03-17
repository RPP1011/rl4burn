# Multi-Agent Shared-Weight Training

Efficiently control multiple units with a single shared policy network. Used by JueWu for 5 heroes and applicable to any game with multiple controlled units.

## API

```rust,ignore
use rl4burn::{batch_multi_agent_obs, unbatch_actions, broadcast_team_reward};

// Batch observations from all agents across all environments
let (obs_tensor, n_envs, n_agents) = batch_multi_agent_obs::<B>(
    &per_env_per_agent_obs,
    &device,
);
// obs_tensor: [n_envs * n_agents, obs_dim] — one big batch

// Single forward pass for all agents
let output = model.forward(obs_tensor);

// Unbatch actions back to per-env, per-agent
let actions = unbatch_actions(&flat_actions, n_envs, n_agents);
// actions: [n_envs][n_agents]

// Broadcast team reward to all agents
let per_agent_rewards = broadcast_team_reward(&env_rewards, n_agents);
```

## Why shared weights?

With 30 units, running 30 separate forward passes is expensive. With shared weights, batch all 30 observations into one forward pass. The policy generalizes across unit types through the observation encoding.
