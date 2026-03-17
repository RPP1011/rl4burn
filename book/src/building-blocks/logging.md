# Logging

rl4burn provides a lightweight, feature-gated logging system for training metrics. The core `Logger` trait and built-in loggers ship with zero extra dependencies. TensorBoard and JSON output are opt-in via feature flags.

## The Logger trait

All loggers implement `Logger`:

```rust,ignore
pub trait Logger {
    fn log_scalar(&mut self, key: &str, value: f64, step: u64);
    fn log_scalars(&mut self, key: &str, values: &[(&str, f64)], step: u64);
    fn log_text(&mut self, key: &str, text: &str, step: u64);
    fn log_histogram(&mut self, key: &str, values: &[f32], step: u64);
    fn flush(&mut self);
}
```

## Built-in loggers (no feature flags)

**`PrintLogger`** — prints scalars to stderr in a formatted line. Accepts a throttle interval so you don't flood the terminal:

```rust,ignore
use rl4burn::PrintLogger;

// Print at most every 1000 steps
let mut logger = PrintLogger::new(1000);
logger.log_scalar("train/loss", 0.42, 5000);
// stderr: [step     5000] train/loss: 0.4200
```

**`NoopLogger`** — discards everything. Useful as a default when the caller doesn't care about logging.

**`CompositeLogger`** — fans out to multiple loggers simultaneously:

```rust,ignore
use rl4burn::{CompositeLogger, PrintLogger};
use rl4burn::TensorBoardLogger; // requires `tensorboard` feature

let mut logger = CompositeLogger::new(vec![
    Box::new(PrintLogger::new(0)),
    Box::new(TensorBoardLogger::new("runs/ppo_cartpole").unwrap()),
]);
logger.log_scalar("train/loss", 0.5, 100); // goes to both
```

## Logging stats from algorithms

`PpoStats` and `DqnStats` implement the `Loggable` trait, so you can log all their fields in one call:

```rust,ignore
use rl4burn::Loggable;

let (model, stats) = ppo_update(model, &mut optim, &rollout, &config, lr, &device, &mut rng);
stats.log(&mut logger, step);
// Logs: train/policy_loss, train/value_loss, train/entropy, train/approx_kl
```

For DQN:

```rust,ignore
let (online, stats) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);
stats.log(&mut logger, step);
// Logs: train/loss, train/mean_q, train/epsilon
```

## TensorBoard (feature-gated)

Enable the `tensorboard` feature in your `Cargo.toml`:

```toml
[dependencies]
rl4burn = { version = "0.1", features = ["tensorboard"] }
```

Then create a `TensorBoardLogger` pointing at a run directory:

```rust,ignore
use rl4burn::TensorBoardLogger;

let mut logger = TensorBoardLogger::new("runs/experiment_1").unwrap();
logger.log_scalar("train/loss", 0.42, 1000);
logger.log_histogram("weights", &weight_data, 1000);
logger.log_text("info", "training started", 0);
logger.flush();
```

View results with:

```bash
tensorboard --logdir runs/
```

The logger writes standard TFEvent files (`events.out.tfevents.*`) with hand-serialized protobufs — no `prost` or `protobuf` dependency required. Supports scalars, histograms, and text.

## JSON output (feature-gated)

Enable the `json-log` feature:

```toml
[dependencies]
rl4burn = { version = "0.1", features = ["json-log"] }
```

`JsonLogger` writes one JSON object per line (JSONL format) to any `Write` sink:

```rust,ignore
use rl4burn::JsonLogger;

let mut logger = JsonLogger::from_path("train_log.jsonl").unwrap();
logger.log_scalar("train/loss", 0.42, 1000);
logger.flush();
```

Each line looks like:

```json
{"type":"scalar","key":"train/loss","value":0.42,"step":1000,"wall_time":1234567890.123}
```

### Bridging to Weights & Biases

A thin Python bridge script is included at `scripts/wandb_bridge.py`:

```bash
cargo run --example ppo_cartpole --features "ndarray,json-log" 2>&1 \
  | python scripts/wandb_bridge.py
```

The same JSONL format can be ingested by neptune, mlflow, comet, or any custom dashboard.

## Video recording (feature-gated)

Enable the `video` feature to record CartPole episodes as GIFs:

```toml
[dependencies]
rl4burn = { version = "0.1", features = ["video"] }
```

CartPole has a built-in `render()` method that produces RGB frames:

```rust,ignore
use rl4burn::envs::CartPole;
use rl4burn::{write_gif, Env};

let mut env = CartPole::new(rng);
env.reset();

let mut frames = vec![env.render()];
loop {
    let step = env.step(action);
    frames.push(env.render());
    if step.done() { break; }
}

write_gif("episode.gif", &frames, 4).unwrap(); // 4 centiseconds per frame
```

## Putting it all together

A typical training script with logging:

```rust,ignore
use rl4burn::{CompositeLogger, Loggable, Logger, PrintLogger, TensorBoardLogger};

let mut logger = CompositeLogger::new(vec![
    Box::new(PrintLogger::new(5000)),
    Box::new(TensorBoardLogger::new("runs/ppo").unwrap()),
]);

for iter in 0..n_iterations {
    let rollout = ppo_collect::<NdArray, _, _>(&model.valid(), &mut vec_env, &config, &device, &mut rng, &mut ep_acc);

    let step = (iter + 1) as u64 * steps_per_iter as u64;
    if !rollout.episode_returns.is_empty() {
        let avg = rollout.episode_returns.iter().sum::<f32>() / rollout.episode_returns.len() as f32;
        logger.log_scalar("rollout/avg_return", avg as f64, step);
    }

    let (new_model, stats) = ppo_update(model, &mut optim, &rollout, &config, lr, &device, &mut rng);
    model = new_model;
    stats.log(&mut logger, step);
}
logger.flush();
```

## Feature flags summary

| Feature | Dependency | What you get |
|---------|-----------|--------------|
| *(none)* | — | `Logger` trait, `PrintLogger`, `NoopLogger`, `CompositeLogger`, `Loggable` |
| `tensorboard` | `crc32c` | `TensorBoardLogger` (TFEvent files) |
| `json-log` | — | `JsonLogger` (JSONL output) |
| `video` | `gif` | `write_gif()`, `CartPole::render()` |
