# Sequence Replay Buffer

A FIFO buffer that samples contiguous sequences of a fixed length, respecting episode boundaries. Used by DreamerV3 for world model training.

## API

```rust,ignore
use rl4burn::{SequenceReplayBuffer, SequenceStep};

let mut buffer = SequenceReplayBuffer::new(1_000_000, 64);
// capacity: 1M steps, sequence_length: 64

// Add transitions
buffer.push(SequenceStep {
    observation: obs.clone(),
    action: vec![1.0, 0.0],
    reward: 1.0,
    done: false,
});

// Sample batch of sequences
let sequences = buffer.sample(16, &mut rng);
// sequences: Vec<Vec<SequenceStep<O>>>, each of length 64
```

## Episode boundaries

Sampled sequences never cross episode boundaries. If a `done=true` step appears in the buffer, no sequence will start before it and end after it.

## FIFO eviction

When the buffer exceeds capacity, the oldest steps are removed first. Episode start indices are automatically adjusted.

## Difference from ReplayBuffer

| Feature | ReplayBuffer | SequenceReplayBuffer |
|---------|-------------|---------------------|
| Sample unit | Single step | Contiguous sequence |
| Episode boundaries | Not tracked | Enforced |
| Primary use | DQN, off-policy | DreamerV3 world models |
