# SCC (StarCraft Commander)

## The 30-second version

SCC (inspir.ai, ICML 2021) reaches GrandMaster in StarCraft II with **10x less compute** than AlphaStar. Its trick: a more efficient architecture (49M vs 139M parameters) and smarter training (agent branching instead of training exploiters from scratch).

## Key innovations

### Group Transformer

Instead of processing all game units with one big attention layer, SCC groups them:
- **Intra-group self-attention**: ally units attend to each other, enemy units attend to each other
- **Inter-group cross-attention**: ally representations attend to enemy representations

This is more efficient for games with natural groupings (teams, unit types).

rl4burn provides the building blocks: `TransformerEncoder` for self-attention, `MultiHeadAttention` for cross-attention. See [Transformer Encoder](../nn/transformer.md) and [Attention Mechanisms](../nn/attention.md).

### Attention-based pooling

Variable numbers of units get aggregated into fixed-size vectors using learned query vectors. Better than mean-pooling because the model learns *which* units matter most.

```rust,ignore
use rl4burn::{AttentionPool, AttentionPoolConfig};

let pool = AttentionPoolConfig::new(128, 4, 2).init(&device);
// 128-dim entity embeddings, 4 learned queries, 2 attention heads
// Output: [batch, 4 * 128] = [batch, 512]
```

See [Attention Mechanisms](../nn/attention.md).

### FiLM conditioning

The target position head is conditioned on the action type using FiLM: `output = gamma(ctx) * input + beta(ctx)`. This lets the same network produce different spatial distributions depending on whether you're attacking, moving, or casting.

```rust,ignore
use rl4burn::{Film, FilmConfig};
let film = FilmConfig::new(action_embed_dim, spatial_feature_dim).init(&device);
```

See [FiLM Conditioning](../nn/film.md).

### Agent branching

When creating a new exploiter, SCC clones the *current main agent's weights* instead of starting from the supervised model. The optimizer state is reset. This gives exploiters a head start.

```rust,ignore
use rl4burn::algo::self_play::branch_agent;
let exploiter = branch_agent(&main_agent);
// Create a fresh optimizer for the exploiter
```

See [Agent Branching](../game-ai/agent-branching.md).

### Pointer networks

For selecting "which of my units should do this?", SCC uses pointer networks — attention over encoder outputs producing a selection distribution.

```rust,ignore
use rl4burn::{PointerNet, PointerNetConfig};
```

## The architecture in one sentence

Group Transformer encodes entities → attention pooling aggregates → residual LSTM sequences → FiLM-conditioned heads output → pointer networks select.

## Further reading

- [SCC paper](https://arxiv.org/abs/2106.07564) (ICML, 2021)
