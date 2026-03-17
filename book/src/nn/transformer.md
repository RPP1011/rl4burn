# Transformer Encoder

Reusable multi-head self-attention blocks for entity processing. Used by ROA-Star and SCC to encode sets of game units.

## Multi-Head Attention

```rust,ignore
use rl4burn::{MultiHeadAttention, MultiHeadAttentionConfig};

let attn = MultiHeadAttentionConfig::new(128, 4).init(&device);
// d_model=128, 4 heads (d_k = 32 per head)

let output = attn.forward(query, key, value, None);
// All inputs: [batch, seq_len, 128]
// Optional mask: [batch, seq_len] (true = attend, false = ignore)
```

## Transformer Block

Pre-norm residual block: self-attention + feedforward.

```rust,ignore
use rl4burn::{TransformerBlock, TransformerBlockConfig};

let block = TransformerBlockConfig::new(128, 4, 512).init(&device);
// d_model=128, 4 heads, d_ff=512
let output = block.forward(input, None);  // residual: output ≈ input + attention + ffn
```

## Stacked Encoder

```rust,ignore
use rl4burn::{TransformerEncoder, TransformerEncoderConfig};

let encoder = TransformerEncoderConfig::new(128, 4, 2, 512).init(&device);
// 2 layers of transformer blocks
let encoded = encoder.forward(entities, None);
```

## Properties

- **Permutation equivariant**: reordering input tokens reorders output tokens identically (no positional encoding).
- **Variable-length**: use masking for padded sequences.
- For 30 entities with 128-dim embeddings, a 2-layer encoder runs in microseconds on CPU.
