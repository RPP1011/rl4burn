# Privileged Critic

The policy sees only what a player would see. The critic sees everything — including enemy positions behind fog of war.

## The Trait

```rust,ignore
use rl4burn::algo::privileged_critic::{PrivilegedActorCritic, make_critic_input};

impl<B: Backend> PrivilegedActorCritic<B> for MyModel<B> {
    fn actor_forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.actor.forward(obs)  // partial observation only
    }

    fn critic_forward(&self, obs: Tensor<B, 2>, privileged: Tensor<B, 2>) -> Tensor<B, 1> {
        let input = make_critic_input(obs, privileged);
        self.critic.forward(input)
    }
}
```

## Why it works

Value estimation under partial observability is noisy — the critic can't tell if you're winning or losing without seeing the full game state. Giving the critic privileged information dramatically reduces variance. At deployment time, only the actor is needed.
