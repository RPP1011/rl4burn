# Installation

Add rl4burn and Burn to your `Cargo.toml`:

```toml
[dependencies]
rl4burn = { git = "https://github.com/RPP1011/rl4burn" }
burn = { version = "0.20", features = ["std", "ndarray", "autodiff"] }
rand = "0.9"
```

The `ndarray` feature gives you a CPU backend for development and testing. For GPU training, add `wgpu` or `tch` (LibTorch) instead.

## Verify the install

Create a `src/main.rs`:

```rust,ignore
use rl4burn::envs::CartPole;
use rl4burn::env::Env;
use rand::SeedableRng;

fn main() {
    let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(42));
    let obs = env.reset();
    println!("CartPole observation: {:?}", obs);

    let step = env.step(1); // push right
    println!("Reward: {}, Done: {}", step.reward, step.done());
}
```

```bash
cargo run
```

You should see a 4-element observation vector and a reward of 1.0.
