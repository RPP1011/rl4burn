# Saving & Sharing

After training, you typically want to save the model weights for later inference and share visualizations of agent behavior. This chapter covers both.

## Saving model weights

Burn models derive `Module`, which gives them `save_file` and `load_file` for free. No rl4burn-specific API is needed — just use Burn's recorder system directly.

### Save a trained model

```rust,ignore
use burn::record::{CompactRecorder, Recorder};

// After training completes:
model
    .save_file("checkpoints/ppo_cartpole", &CompactRecorder::new())
    .expect("failed to save model");
```

This writes `checkpoints/ppo_cartpole.mpk` (MessagePack format). The file contains all learnable parameters.

### Load a saved model

```rust,ignore
use burn::record::{CompactRecorder, Recorder};

// Initialize a fresh model, then load weights into it
let model: ActorCritic<AB> = ActorCritic::new(&device)
    .load_file("checkpoints/ppo_cartpole", &CompactRecorder::new(), &device)
    .expect("failed to load model");
```

The model architecture must match — `load_file` loads parameter values, not the structure.

### Recorder types

Burn provides several recorders. Choose based on your needs:

| Recorder | Format | Good for |
|----------|--------|----------|
| `CompactRecorder` | MessagePack (`.mpk`) | Production — small files, fast I/O |
| `NamedMpkGzFileRecorder` | gzipped MessagePack | Sharing — even smaller files |
| `PrettyJsonFileRecorder` | JSON (`.json`) | Debugging — human-readable weights |
| `BinFileRecorder` | Raw binary (`.bin`) | Maximum speed, no compression |

`CompactRecorder` is the default choice for most use cases.

### Checkpointing during training

Save periodically so you can resume after interruptions or pick the best checkpoint:

```rust,ignore
use burn::record::{CompactRecorder, Recorder};

for iter in 0..n_iterations {
    // ... collect and update ...

    // Save every 50 iterations
    if (iter + 1) % 50 == 0 {
        let path = format!("checkpoints/ppo_step_{}", (iter + 1) * steps_per_iter);
        model
            .save_file(&path, &CompactRecorder::new())
            .expect("failed to save checkpoint");
    }
}

// Always save the final model
model
    .save_file("checkpoints/ppo_final", &CompactRecorder::new())
    .expect("failed to save final model");
```

### DQN: saving online and target networks

For DQN, save both networks so you can resume training correctly:

```rust,ignore
online.save_file("checkpoints/dqn_online", &CompactRecorder::new())?;
target.save_file("checkpoints/dqn_target", &CompactRecorder::new())?;
```

For inference only, you just need the online network.

## Sharing visualizations

### GIF recordings

With the `video` feature, record an episode of your trained agent and save it as a GIF:

```rust,ignore
use rl4burn::envs::CartPole;
use rl4burn::{write_gif, greedy_action, Env, Renderable};

let mut env = CartPole::new(rng);
let mut obs = env.reset();

let mut frames = vec![env.render()];
loop {
    // greedy_action runs a forward pass and returns the argmax action
    let action = greedy_action(&model, &obs, &device);
    let step = env.step(action);
    frames.push(env.render());
    if step.done() { break; }
    obs = step.observation;
}

write_gif("agent_demo.gif", &frames, 4).unwrap();
```

The resulting GIF can be embedded in READMEs, papers, blog posts, or Slack messages.

### TensorBoard

TensorBoard logs are shareable as a directory. Zip the run folder and send it, or use TensorBoard.dev for public sharing:

```bash
# View locally
tensorboard --logdir runs/

# Share publicly (requires Google account)
tensorboard dev upload --logdir runs/ppo_cartpole --name "PPO CartPole"
```

When comparing experiments, use separate run directories:

```rust,ignore
// Each experiment gets its own subdirectory
let logger = TensorBoardLogger::new(format!("runs/ppo_lr{}", config.lr)).unwrap();
```

Then `tensorboard --logdir runs/` overlays all runs for comparison.

### JSONL logs

JSONL files are plain text and easy to share. Post-process them with any tool:

```bash
# Quick plot with Python
python -c "
import json, matplotlib.pyplot as plt
data = [json.loads(l) for l in open('train_log.jsonl') if '\"scalar\"' in l]
returns = [(d['step'], d['value']) for d in data if d['key'] == 'rollout/avg_return']
plt.plot(*zip(*returns))
plt.xlabel('Step'); plt.ylabel('Avg Return')
plt.savefig('training_curve.png')
"

# Send to W&B
python scripts/wandb_bridge.py < train_log.jsonl
```

## Putting it all together

A complete training script that checkpoints the model and records a final evaluation GIF:

```rust,ignore
use burn::record::{CompactRecorder, Recorder};
use rl4burn::{
    CompositeLogger, Loggable, Logger, PrintLogger, TensorBoardLogger,
    envs::CartPole, write_gif, Env, Renderable, greedy_action,
};

let mut logger = CompositeLogger::new(vec![
    Box::new(PrintLogger::new(5000)),
    Box::new(TensorBoardLogger::new("runs/ppo").unwrap()),
]);

// Training loop
for iter in 0..n_iterations {
    let rollout = ppo_collect::<NdArray, _, _>(
        &model.valid(), &mut vec_env, &config, &device, &mut rng, &mut current_obs, &mut ep_acc,
    );

    let (new_model, stats) = ppo_update(
        model, &mut optim, &rollout, &config, lr, &device, &mut rng,
    );
    model = new_model;

    let step = (iter + 1) as u64 * steps_per_iter as u64;
    stats.log(&mut logger, step);

    // Checkpoint every 100 iterations
    if (iter + 1) % 100 == 0 {
        model.save_file(
            &format!("checkpoints/ppo_{step}"),
            &CompactRecorder::new(),
        ).unwrap();
    }
}
logger.flush();

// Save final weights
model.save_file("checkpoints/ppo_final", &CompactRecorder::new()).unwrap();

// Record evaluation episode
let mut env = CartPole::new(rng);
let mut obs = env.reset();
let mut frames = vec![env.render()];
loop {
    let action = greedy_action(&model.valid(), &obs, &device);
    let step = env.step(action);
    frames.push(env.render());
    if step.done() { break; }
    obs = step.observation;
}
write_gif("evaluation.gif", &frames, 4).unwrap();
```
