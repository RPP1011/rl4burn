# Spaces

Spaces describe the shape and bounds of observations and actions. They're used for constructing networks (knowing input/output dimensions) and validating data.

## The Space enum

```rust,ignore
pub enum Space {
    Discrete(usize),                     // {0, 1, ..., n-1}
    Box { low: Vec<f32>, high: Vec<f32> }, // continuous, per-dimension bounds
    MultiDiscrete(Vec<usize>),           // multiple independent discrete spaces
}
```

## Methods

- `flat_dim()` — total dimension (one-hot width for Discrete, number of dims for Box)
- `shape()` — shape as a Vec

## Usage

Spaces are returned by `Env::observation_space()` and `Env::action_space()`. Use them to size your network layers:

```rust,ignore
let obs_dim = env.observation_space().flat_dim(); // e.g., 4 for CartPole
let n_actions = match env.action_space() {
    Space::Discrete(n) => n,
    _ => panic!("expected discrete actions"),
};

let fc1 = LinearConfig::new(obs_dim, 64).init(&device);
let out = LinearConfig::new(64, n_actions).init(&device);
```
