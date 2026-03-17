# Attention Mechanisms

Three specialized attention modules for game AI architectures.

## Target Attention

Scaled dot-product attention for selecting a target entity. The LSTM output serves as query; encoded entities serve as keys. Returns a probability distribution over entities.

```rust,ignore
use rl4burn::{TargetAttention, TargetAttentionConfig};

let attn = TargetAttentionConfig::new(256, 128).init(&device);
// query_dim=256 (LSTM output), key_dim=128 (entity embedding)

let probs = attn.forward(query, keys, Some(mask));
// query: [batch, 256], keys: [batch, n_entities, 128]
// mask: [batch, n_entities] (true = valid target)
// probs: [batch, n_entities] (sums to 1 over valid targets)
```

## Attention Pooling

Aggregates variable-count entity embeddings into a fixed-size vector using learned query vectors. Superior to mean/max pooling.

```rust,ignore
use rl4burn::{AttentionPool, AttentionPoolConfig};

let pool = AttentionPoolConfig::new(128, 4, 2).init(&device);
// embed_dim=128, 4 learned queries, 2 attention heads

let pooled = pool.forward(entities, None);
// entities: [batch, n_entities, 128]
// pooled: [batch, 512] (4 queries * 128 dims)
```

## Pointer Network

Additive (Bahdanau) attention for entity selection: `score = v^T * tanh(W_q * query + W_k * keys)`. Used by AlphaStar and SCC for selecting subsets of units.

```rust,ignore
use rl4burn::{PointerNet, PointerNetConfig};

let ptr = PointerNetConfig::new(256, 128, 64).init(&device);
// query_dim=256, key_dim=128, hidden_dim=64

let probs = ptr.forward(query, keys, Some(mask));
// probs: [batch, n_entities] (selection probabilities)
```

All three modules support masking for absent/dead entities.
