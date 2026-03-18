//! Attention-based neural network modules for RL.
//!
//! This module provides several attention mechanisms commonly used in
//! reinforcement learning architectures:
//!
//! - [`MultiHeadAttention`] — Multi-head scaled dot-product attention.
//! - [`TransformerBlock`] — Pre-norm transformer encoder block.
//! - [`TransformerEncoder`] — Stack of transformer encoder blocks.
//! - [`TargetAttention`] — Scaled dot-product attention for entity selection.
//! - [`AttentionPool`] — Learned-query cross-attention pooling for variable-count inputs.
//! - [`PointerNet`] — Additive (Bahdanau) attention for pointer networks.

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};

// ====================== Multi-Head Attention ======================

/// Configuration for a [`MultiHeadAttention`] module.
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// Model dimension (must be divisible by `n_heads`).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

/// Multi-head scaled dot-product attention.
///
/// Projects queries, keys, and values through separate linear layers,
/// splits into multiple heads, computes scaled dot-product attention
/// per head, concatenates the results, and projects to the output.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    w_q: Linear<B>,
    w_k: Linear<B>,
    w_v: Linear<B>,
    w_out: Linear<B>,
    #[module(skip)]
    n_heads: usize,
    #[module(skip)]
    d_k: usize,
}

impl MultiHeadAttentionConfig {
    /// Initialize a multi-head attention module on the given device.
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `n_heads`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model {} must be divisible by n_heads {}",
            self.d_model,
            self.n_heads
        );
        let d_k = self.d_model / self.n_heads;
        MultiHeadAttention {
            w_q: LinearConfig::new(self.d_model, self.d_model).init(device),
            w_k: LinearConfig::new(self.d_model, self.d_model).init(device),
            w_v: LinearConfig::new(self.d_model, self.d_model).init(device),
            w_out: LinearConfig::new(self.d_model, self.d_model).init(device),
            n_heads: self.n_heads,
            d_k,
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `query` — `[batch, seq_q, d_model]`
    /// * `key` — `[batch, seq_k, d_model]`
    /// * `value` — `[batch, seq_k, d_model]`
    /// * `mask` — Optional `[batch, seq_q, seq_k]` boolean mask where `true`
    ///   means the position should be **masked out** (ignored).
    ///
    /// # Returns
    /// Output tensor `[batch, seq_q, d_model]`.
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_q, _] = query.dims();
        let [_, seq_k, _] = key.dims();
        let n_heads = self.n_heads;
        let d_k = self.d_k;

        // Project and reshape to [batch, n_heads, seq, d_k]
        let q = self
            .w_q
            .forward(query)
            .reshape([batch, seq_q, n_heads, d_k])
            .swap_dims(1, 2); // [batch, n_heads, seq_q, d_k]
        let k = self
            .w_k
            .forward(key)
            .reshape([batch, seq_k, n_heads, d_k])
            .swap_dims(1, 2); // [batch, n_heads, seq_k, d_k]
        let v = self
            .w_v
            .forward(value)
            .reshape([batch, seq_k, n_heads, d_k])
            .swap_dims(1, 2); // [batch, n_heads, seq_k, d_k]

        // Scaled dot-product attention
        let scale = (d_k as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale; // [batch, n_heads, seq_q, seq_k]

        // Apply mask: set masked positions to -1e9 before softmax
        let scores = if let Some(m) = mask {
            // Expand mask from [batch, seq_q, seq_k] to [batch, n_heads, seq_q, seq_k]
            let m = m.unsqueeze_dim::<4>(1).expand([batch, n_heads, seq_q, seq_k]);
            let neg_inf = Tensor::<B, 4>::ones_like(&scores) * (-1e9);
            scores.mask_where(m, neg_inf)
        } else {
            scores
        };

        let attn_weights = softmax(scores, 3); // [batch, n_heads, seq_q, seq_k]
        let attn_output = attn_weights.matmul(v); // [batch, n_heads, seq_q, d_k]

        // Concatenate heads: [batch, seq_q, d_model]
        let attn_output = attn_output
            .swap_dims(1, 2) // [batch, seq_q, n_heads, d_k]
            .reshape([batch, seq_q, n_heads * d_k]);

        self.w_out.forward(attn_output)
    }
}

// ====================== Transformer Block ======================

/// Configuration for a [`TransformerBlock`].
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feed-forward hidden dimension. Default: `4 * d_model`.
    #[config(default = 0)]
    pub d_ff: usize,
}

/// Pre-norm transformer encoder block.
///
/// Applies:
/// ```text
/// x = x + self_attn(norm1(x))
/// x = x + ff2(relu(ff1(norm2(x))))
/// ```
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl TransformerBlockConfig {
    /// Initialize a transformer block on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let d_ff = if self.d_ff == 0 {
            4 * self.d_model
        } else {
            self.d_ff
        };
        TransformerBlock {
            self_attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),
            ff1: LinearConfig::new(self.d_model, d_ff).init(device),
            ff2: LinearConfig::new(d_ff, self.d_model).init(device),
            norm1: LayerNormConfig::new(self.d_model).init(device),
            norm2: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` — Input `[batch, seq, d_model]`
    /// * `mask` — Optional attention mask `[batch, seq, seq]`
    ///
    /// # Returns
    /// Output `[batch, seq, d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        // Pre-norm self-attention with residual
        let normed = self.norm1.forward(x.clone());
        let attn_out = self
            .self_attn
            .forward(normed.clone(), normed.clone(), normed, mask);
        let x = x + attn_out;

        // Pre-norm feed-forward with residual
        let normed = self.norm2.forward(x.clone());
        let ff_out = self.ff2.forward(relu(self.ff1.forward(normed)));
        x + ff_out
    }
}

// ====================== Transformer Encoder ======================

/// Configuration for a [`TransformerEncoder`].
#[derive(Config, Debug)]
pub struct TransformerEncoderConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Feed-forward hidden dimension. Default: `4 * d_model`.
    #[config(default = 0)]
    pub d_ff: usize,
}

/// Stack of pre-norm transformer encoder blocks.
#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Vec<TransformerBlock<B>>,
}

impl TransformerEncoderConfig {
    /// Initialize a transformer encoder on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoder<B> {
        let d_ff = if self.d_ff == 0 {
            4 * self.d_model
        } else {
            self.d_ff
        };
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(self.d_model, self.n_heads)
                    .with_d_ff(d_ff)
                    .init(device)
            })
            .collect();
        TransformerEncoder { layers }
    }
}

impl<B: Backend> TransformerEncoder<B> {
    /// Forward pass through all layers.
    ///
    /// # Arguments
    /// * `x` — Input `[batch, seq, d_model]`
    /// * `mask` — Optional attention mask `[batch, seq, seq]`
    ///
    /// # Returns
    /// Output `[batch, seq, d_model]`.
    pub fn forward(&self, mut x: Tensor<B, 3>, mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }
        x
    }
}

// ====================== Target Attention ======================

/// Configuration for a [`TargetAttention`] module.
#[derive(Config, Debug)]
pub struct TargetAttentionConfig {
    /// Query dimension (e.g., LSTM hidden size).
    pub query_dim: usize,
    /// Key dimension (e.g., entity embedding size).
    pub key_dim: usize,
}

/// Scaled dot-product attention for entity selection.
///
/// Projects a query vector and a set of key vectors to a common dimension,
/// computes scaled dot-product scores, and returns a probability distribution
/// over entities. Supports optional masking for invalid entities.
#[derive(Module, Debug)]
pub struct TargetAttention<B: Backend> {
    w_q: Linear<B>,
    w_k: Linear<B>,
    #[module(skip)]
    d_k: usize,
}

impl TargetAttentionConfig {
    /// Initialize a target attention module on the given device.
    ///
    /// The projection dimension is `min(query_dim, key_dim)`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TargetAttention<B> {
        let d_k = self.query_dim.min(self.key_dim);
        TargetAttention {
            w_q: LinearConfig::new(self.query_dim, d_k).init(device),
            w_k: LinearConfig::new(self.key_dim, d_k).init(device),
            d_k,
        }
    }
}

impl<B: Backend> TargetAttention<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `query` — `[batch, query_dim]`
    /// * `keys` — `[batch, n_entities, key_dim]`
    /// * `mask` — Optional `[batch, n_entities]` boolean mask where `true`
    ///   means the entity should be **masked out**.
    ///
    /// # Returns
    /// Attention weights `[batch, n_entities]` (probability distribution).
    pub fn forward(
        &self,
        query: Tensor<B, 2>,
        keys: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, n_entities, _] = keys.dims();

        // Project query: [batch, d_k] -> [batch, 1, d_k]
        let q = self
            .w_q
            .forward(query)
            .unsqueeze_dim::<3>(1); // [batch, 1, d_k]

        // Project keys: [batch, n_entities, d_k]
        let k = self.w_k.forward(keys);

        // Scaled dot-product: [batch, 1, d_k] x [batch, d_k, n_entities] -> [batch, 1, n_entities]
        let scale = (self.d_k as f32).sqrt();
        let scores = q
            .matmul(k.transpose())
            .squeeze_dim::<2>(1) / scale; // [batch, n_entities]

        // Apply mask
        let scores = if let Some(m) = mask {
            let neg_inf = Tensor::<B, 2>::ones([batch, n_entities], &scores.device()) * (-1e9);
            scores.mask_where(m, neg_inf)
        } else {
            scores
        };

        softmax(scores, 1) // [batch, n_entities]
    }
}

// ====================== Attention Pool ======================

/// Configuration for an [`AttentionPool`] module.
#[derive(Config, Debug)]
pub struct AttentionPoolConfig {
    /// Entity embedding dimension.
    pub embed_dim: usize,
    /// Number of learned query vectors (output size = `n_queries * embed_dim`).
    pub n_queries: usize,
    /// Number of attention heads. Default: 1.
    #[config(default = 1)]
    pub n_heads: usize,
}

/// Attention-based pooling for variable-count entity embeddings.
///
/// Uses learned query vectors that cross-attend over a set of entity
/// embeddings. The attended outputs are concatenated into a fixed-size
/// vector of dimension `n_queries * embed_dim`.
#[derive(Module, Debug)]
pub struct AttentionPool<B: Backend> {
    query_embed: Embedding<B>,
    attn: MultiHeadAttention<B>,
}

impl AttentionPoolConfig {
    /// Initialize an attention pool module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionPool<B> {
        AttentionPool {
            query_embed: EmbeddingConfig::new(self.n_queries, self.embed_dim).init(device),
            attn: MultiHeadAttentionConfig::new(self.embed_dim, self.n_heads).init(device),
        }
    }
}

impl<B: Backend> AttentionPool<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `entities` — `[batch, n_entities, embed_dim]`
    /// * `mask` — Optional `[batch, n_entities]` boolean mask where `true`
    ///   means the entity should be **masked out**.
    ///
    /// # Returns
    /// Pooled output `[batch, n_queries * embed_dim]`.
    pub fn forward(
        &self,
        entities: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, n_entities, embed_dim] = entities.dims();
        let n_queries = self.query_embed.weight.shape().dims::<2>()[0];

        // Get learned queries: [n_queries, embed_dim]
        // Create indices [0, 1, ..., n_queries-1] and look up embeddings
        let indices: Tensor<B, 2, Int> = Tensor::arange(0..n_queries as i64, &entities.device())
            .unsqueeze_dim(0); // [1, n_queries]
        let queries = self.query_embed.forward(indices); // [1, n_queries, embed_dim]
        let queries = queries.expand([batch, n_queries, embed_dim]); // [batch, n_queries, embed_dim]

        // Build cross-attention mask: [batch, n_queries, n_entities]
        let attn_mask = mask.map(|m| {
            // m: [batch, n_entities] -> [batch, 1, n_entities] -> [batch, n_queries, n_entities]
            m.unsqueeze_dim::<3>(1)
                .expand([batch, n_queries, n_entities])
        });

        // Cross-attend: queries attend over entities
        let attended = self.attn.forward(queries, entities.clone(), entities, attn_mask);
        // attended: [batch, n_queries, embed_dim]

        // Flatten to [batch, n_queries * embed_dim]
        attended.reshape([batch, n_queries * embed_dim])
    }
}

// ====================== Pointer Network ======================

/// Configuration for a [`PointerNet`] module.
#[derive(Config, Debug)]
pub struct PointerNetConfig {
    /// Query dimension.
    pub query_dim: usize,
    /// Key dimension.
    pub key_dim: usize,
    /// Hidden dimension for additive attention.
    pub hidden_dim: usize,
}

/// Pointer network using additive (Bahdanau) attention.
///
/// Computes attention scores as:
/// ```text
/// score = v^T * tanh(W_q * query + W_k * keys)
/// ```
///
/// Returns a probability distribution over entities.
#[derive(Module, Debug)]
pub struct PointerNet<B: Backend> {
    w_query: Linear<B>,
    w_key: Linear<B>,
    v: Linear<B>,
}

impl PointerNetConfig {
    /// Initialize a pointer network module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PointerNet<B> {
        PointerNet {
            w_query: LinearConfig::new(self.query_dim, self.hidden_dim).init(device),
            w_key: LinearConfig::new(self.key_dim, self.hidden_dim).init(device),
            v: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> PointerNet<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `query` — `[batch, query_dim]`
    /// * `keys` — `[batch, n_entities, key_dim]`
    /// * `mask` — Optional `[batch, n_entities]` boolean mask where `true`
    ///   means the entity should be **masked out**.
    ///
    /// # Returns
    /// Selection probabilities `[batch, n_entities]`.
    pub fn forward(
        &self,
        query: Tensor<B, 2>,
        keys: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, n_entities, _] = keys.dims();

        // Project query: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        let q_proj = self
            .w_query
            .forward(query)
            .unsqueeze_dim::<3>(1); // [batch, 1, hidden_dim]

        // Project keys: [batch, n_entities, hidden_dim]
        let k_proj = self.w_key.forward(keys);

        // Additive attention: tanh(q + k) -> v
        let combined = (q_proj + k_proj).tanh(); // [batch, n_entities, hidden_dim]
        let scores = self
            .v
            .forward(combined)
            .squeeze_dim::<2>(2); // [batch, n_entities]

        // Apply mask
        let scores = if let Some(m) = mask {
            let neg_inf = Tensor::<B, 2>::ones([batch, n_entities], &scores.device()) * (-1e9);
            scores.mask_where(m, neg_inf)
        } else {
            scores
        };

        softmax(scores, 1) // [batch, n_entities]
    }
}

// ====================== Tests ======================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // -- MultiHeadAttention tests --------------------------------------------

    #[test]
    fn mha_output_shape() {
        let mha = MultiHeadAttentionConfig::new(16, 4).init::<B>(&dev());
        let q = Tensor::<B, 3>::zeros([2, 5, 16], &dev());
        let k = Tensor::<B, 3>::zeros([2, 7, 16], &dev());
        let v = Tensor::<B, 3>::zeros([2, 7, 16], &dev());

        let out = mha.forward(q, k, v, None);
        assert_eq!(out.dims(), [2, 5, 16]);
    }

    #[test]
    fn mha_self_attention_shape() {
        let mha = MultiHeadAttentionConfig::new(32, 8).init::<B>(&dev());
        let x = Tensor::<B, 3>::ones([4, 10, 32], &dev());

        let out = mha.forward(x.clone(), x.clone(), x, None);
        assert_eq!(out.dims(), [4, 10, 32]);
    }

    #[test]
    fn mha_with_mask() {
        let mha = MultiHeadAttentionConfig::new(8, 2).init::<B>(&dev());
        let x = Tensor::<B, 3>::ones([1, 3, 8], &dev());

        // Mask out position 1 and 2 for all queries
        let mask_data = Tensor::<B, 3, Int>::from_data(
            [[[0i64, 1, 1]]],
            &dev(),
        );
        let mask = mask_data.greater_elem(0);

        let out = mha.forward(x.clone(), x.clone(), x, Some(mask));
        assert_eq!(out.dims(), [1, 3, 8]);
        // Output should be finite
        let vals: Vec<f32> = out.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "MHA masked output is not finite: {v}");
        }
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn mha_panics_on_indivisible() {
        let _ = MultiHeadAttentionConfig::new(10, 3).init::<B>(&dev());
    }

    // -- TransformerBlock tests ----------------------------------------------

    #[test]
    fn transformer_block_output_shape() {
        let block = TransformerBlockConfig::new(16, 4).init::<B>(&dev());
        let x = Tensor::<B, 3>::zeros([2, 5, 16], &dev());

        let out = block.forward(x, None);
        assert_eq!(out.dims(), [2, 5, 16]);
    }

    #[test]
    fn transformer_block_residual_connection() {
        // With zero input, the residual should keep output finite and reasonable
        let block = TransformerBlockConfig::new(8, 2).init::<B>(&dev());
        let x = Tensor::<B, 3>::zeros([1, 3, 8], &dev());

        let out = block.forward(x, None);
        assert_eq!(out.dims(), [1, 3, 8]);
        let vals: Vec<f32> = out.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "TransformerBlock output is not finite: {v}");
        }
    }

    // -- TransformerEncoder tests --------------------------------------------

    #[test]
    fn transformer_encoder_output_shape() {
        let encoder = TransformerEncoderConfig::new(16, 4, 3).init::<B>(&dev());
        let x = Tensor::<B, 3>::zeros([2, 5, 16], &dev());

        let out = encoder.forward(x, None);
        assert_eq!(out.dims(), [2, 5, 16]);
    }

    #[test]
    fn transformer_encoder_multiple_layers() {
        let encoder = TransformerEncoderConfig::new(8, 2, 4)
            .with_d_ff(16)
            .init::<B>(&dev());
        let x = Tensor::<B, 3>::ones([1, 4, 8], &dev());

        let out = encoder.forward(x, None);
        assert_eq!(out.dims(), [1, 4, 8]);
        let vals: Vec<f32> = out.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(
                v.is_finite(),
                "TransformerEncoder output is not finite: {v}"
            );
        }
    }

    #[test]
    fn self_attention_permutation_equivariance() {
        // If we permute the input sequence, the output should be permuted
        // in the same way (self-attention is equivariant to permutations).
        let mha = MultiHeadAttentionConfig::new(8, 2).init::<B>(&dev());

        // Create input with distinct vectors per position
        let x = Tensor::<B, 3>::from_data(
            [[[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            &dev(),
        );

        // Permuted input: swap positions 0 and 2
        let x_perm = Tensor::<B, 3>::from_data(
            [[[0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            &dev(),
        );

        let out = mha.forward(x.clone(), x.clone(), x, None);
        let out_perm = mha.forward(x_perm.clone(), x_perm.clone(), x_perm, None);

        // out[0, 0, :] should equal out_perm[0, 2, :] (swapped positions)
        let out_vals: Vec<f32> = out.to_data().to_vec().unwrap();
        let out_perm_vals: Vec<f32> = out_perm.to_data().to_vec().unwrap();

        let d = 8;
        for i in 0..d {
            let orig_pos0 = out_vals[0 * d + i]; // position 0 of original
            let perm_pos2 = out_perm_vals[2 * d + i]; // position 2 of permuted
            assert!(
                (orig_pos0 - perm_pos2).abs() < 1e-4,
                "Equivariance violated at dim {i}: {orig_pos0} vs {perm_pos2}"
            );
        }
    }

    // -- TargetAttention tests -----------------------------------------------

    #[test]
    fn target_attention_output_shape() {
        let ta = TargetAttentionConfig::new(16, 8).init::<B>(&dev());
        let query = Tensor::<B, 2>::zeros([2, 16], &dev());
        let keys = Tensor::<B, 3>::zeros([2, 5, 8], &dev());

        let weights = ta.forward(query, keys, None);
        assert_eq!(weights.dims(), [2, 5]);
    }

    #[test]
    fn target_attention_sums_to_one() {
        let ta = TargetAttentionConfig::new(8, 8).init::<B>(&dev());
        let query = Tensor::<B, 2>::ones([3, 8], &dev());
        let keys = Tensor::<B, 3>::ones([3, 4, 8], &dev());

        let weights = ta.forward(query, keys, None);
        let sums: Vec<f32> = weights.sum_dim(1).to_data().to_vec().unwrap();
        for s in &sums {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "Target attention weights should sum to 1, got {s}"
            );
        }
    }

    #[test]
    fn target_attention_single_valid_target() {
        // Mask all but one entity; that entity should get prob ~1.0
        let ta = TargetAttentionConfig::new(4, 4).init::<B>(&dev());
        let query = Tensor::<B, 2>::ones([1, 4], &dev());
        let keys = Tensor::<B, 3>::ones([1, 3, 4], &dev());

        // Mask out entities 0 and 1; only entity 2 is valid
        let mask_data = Tensor::<B, 2, Int>::from_data([[1i64, 1, 0]], &dev());
        let mask = mask_data.greater_elem(0);

        let weights = ta.forward(query, keys, Some(mask));
        let w_vals: Vec<f32> = weights.to_data().to_vec().unwrap();

        assert!(
            w_vals[2] > 0.99,
            "Single valid target should get prob ~1.0, got {}",
            w_vals[2]
        );
        assert!(
            w_vals[0] < 0.01,
            "Masked target should get prob ~0.0, got {}",
            w_vals[0]
        );
    }

    // -- AttentionPool tests -------------------------------------------------

    #[test]
    fn attention_pool_output_shape() {
        let pool = AttentionPoolConfig::new(8, 3).init::<B>(&dev());
        let entities = Tensor::<B, 3>::zeros([2, 5, 8], &dev());

        let pooled = pool.forward(entities, None);
        assert_eq!(pooled.dims(), [2, 3 * 8]);
    }

    #[test]
    fn attention_pool_with_mask() {
        let pool = AttentionPoolConfig::new(8, 2)
            .with_n_heads(2)
            .init::<B>(&dev());
        let entities = Tensor::<B, 3>::ones([1, 4, 8], &dev());

        // Mask out entities 2 and 3
        let mask_data = Tensor::<B, 2, Int>::from_data([[0i64, 0, 1, 1]], &dev());
        let mask = mask_data.greater_elem(0);

        let pooled = pool.forward(entities, Some(mask));
        assert_eq!(pooled.dims(), [1, 2 * 8]);
        let vals: Vec<f32> = pooled.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "AttentionPool output is not finite: {v}");
        }
    }

    #[test]
    fn attention_pool_single_query() {
        let pool = AttentionPoolConfig::new(16, 1).init::<B>(&dev());
        let entities = Tensor::<B, 3>::ones([3, 7, 16], &dev());

        let pooled = pool.forward(entities, None);
        assert_eq!(pooled.dims(), [3, 16]);
    }

    // -- PointerNet tests ----------------------------------------------------

    #[test]
    fn pointer_net_output_shape() {
        let pn = PointerNetConfig::new(16, 8, 32).init::<B>(&dev());
        let query = Tensor::<B, 2>::zeros([2, 16], &dev());
        let keys = Tensor::<B, 3>::zeros([2, 5, 8], &dev());

        let probs = pn.forward(query, keys, None);
        assert_eq!(probs.dims(), [2, 5]);
    }

    #[test]
    fn pointer_net_sums_to_one() {
        let pn = PointerNetConfig::new(8, 8, 16).init::<B>(&dev());
        let query = Tensor::<B, 2>::ones([3, 8], &dev());
        let keys = Tensor::<B, 3>::ones([3, 4, 8], &dev());

        let probs = pn.forward(query, keys, None);
        let sums: Vec<f32> = probs.sum_dim(1).to_data().to_vec().unwrap();
        for s in &sums {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "PointerNet probs should sum to 1, got {s}"
            );
        }
    }

    #[test]
    fn pointer_net_single_valid_target() {
        let pn = PointerNetConfig::new(4, 4, 8).init::<B>(&dev());
        let query = Tensor::<B, 2>::ones([1, 4], &dev());
        let keys = Tensor::<B, 3>::ones([1, 3, 4], &dev());

        // Mask out entities 0 and 2; only entity 1 is valid
        let mask_data = Tensor::<B, 2, Int>::from_data([[1i64, 0, 1]], &dev());
        let mask = mask_data.greater_elem(0);

        let probs = pn.forward(query, keys, Some(mask));
        let p_vals: Vec<f32> = probs.to_data().to_vec().unwrap();

        assert!(
            p_vals[1] > 0.99,
            "Single valid target should get prob ~1.0, got {}",
            p_vals[1]
        );
        assert!(
            p_vals[0] < 0.01,
            "Masked target should get prob ~0.0, got {}",
            p_vals[0]
        );
        assert!(
            p_vals[2] < 0.01,
            "Masked target should get prob ~0.0, got {}",
            p_vals[2]
        );
    }

    #[test]
    fn pointer_net_masked_targets_get_low_prob() {
        let pn = PointerNetConfig::new(8, 8, 16).init::<B>(&dev());
        let query = Tensor::<B, 2>::ones([1, 8], &dev());
        let keys = Tensor::<B, 3>::ones([1, 5, 8], &dev());

        // Mask out entities 1, 3, 4
        let mask_data = Tensor::<B, 2, Int>::from_data([[0i64, 1, 0, 1, 1]], &dev());
        let mask = mask_data.greater_elem(0);

        let probs = pn.forward(query, keys, Some(mask));
        let p_vals: Vec<f32> = probs.to_data().to_vec().unwrap();

        // Masked positions should have near-zero probability
        assert!(p_vals[1] < 0.01, "Masked entity 1 should have ~0 prob");
        assert!(p_vals[3] < 0.01, "Masked entity 3 should have ~0 prob");
        assert!(p_vals[4] < 0.01, "Masked entity 4 should have ~0 prob");

        // Unmasked positions should share the probability
        assert!(p_vals[0] > 0.1, "Unmasked entity 0 should have significant prob");
        assert!(p_vals[2] > 0.1, "Unmasked entity 2 should have significant prob");
    }
}
