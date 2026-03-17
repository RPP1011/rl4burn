# Environments

The `Env` trait defines how an RL agent interacts with the world. It follows modern [Gymnasium](https://gymnasium.farama.org/) conventions.

## The Env trait

```rust,ignore
pub trait Env {
    type Observation: Clone;
    type Action: Clone;

    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> Space;
    fn action_space(&self) -> Space;
}
```

`step` returns a `Step` struct with separate `terminated` and `truncated` flags:

```rust,ignore
pub struct Step<O> {
    pub observation: O,
    pub reward: f32,
    pub terminated: bool,  // episode ended due to environment dynamics
    pub truncated: bool,   // episode ended due to time limit
}
```

The `done()` method returns `terminated || truncated`.

## Implementing a custom environment

```rust,ignore
use rl4burn::env::{Env, Step};
use rl4burn::space::Space;

struct MyEnv {
    state: f32,
    step_count: usize,
}

impl Env for MyEnv {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.state = 0.0;
        self.step_count = 0;
        vec![self.state]
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        self.state += if action == 0 { -0.1 } else { 0.1 };
        self.step_count += 1;
        Step {
            observation: vec![self.state],
            reward: -self.state.abs(), // reward for staying near 0
            terminated: self.state.abs() > 1.0,
            truncated: self.step_count >= 200,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box { low: vec![-2.0], high: vec![2.0] }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }
}
```

## Built-in environments

| Environment | Obs dim | Actions | Max steps |
|-------------|---------|---------|-----------|
| `CartPole` | 4 | 2 (left/right) | 500 |

```rust,ignore
use rl4burn::envs::CartPole;
use rand::SeedableRng;

let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(42));
```

CartPole is generic over `R: Rng`, so you control the random number generator for reproducibility.
