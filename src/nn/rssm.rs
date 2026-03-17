//! RSSM (Recurrent State-Space Model) for DreamerV3 (Issue #21).
//!
//! The RSSM models environment dynamics with a split state:
//! - **Deterministic** (`h`): GRU hidden state that captures temporal structure.
//! - **Stochastic** (`z`): Categorical variables that capture per-step uncertainty.
//!
//! Components:
//! 1. **Sequence model** (GRU): `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`
//! 2. **Encoder** (posterior): `z_t ~ Cat(MLP(h_t, obs_t))`
//! 3. **Dynamics** (prior): `z_hat_t ~ Cat(MLP(h_t))`
//! 4. **Reward predictor**: `MLP(h_t, z_t) -> twohot reward`
//! 5. **Continue predictor**: `MLP(h_t, z_t) -> Bernoulli`
//!
//! Reference: Hafner et al., "Mastering Diverse Domains through World Models"
//! (DreamerV3), 2023.

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::nn::rnn::{GruCell, GruCellConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// RSSM configuration.
#[derive(Config, Debug)]
pub struct RssmConfig {
    /// Observation encoding dimension.
    pub obs_dim: usize,
    /// Action dimension (one-hot or embedded).
    pub action_dim: usize,
    /// Deterministic state size (GRU hidden state).
    #[config(default = 512)]
    pub deterministic_size: usize,
    /// Number of categorical groups for stochastic state.
    #[config(default = 32)]
    pub n_categories: usize,
    /// Number of classes per categorical group.
    #[config(default = 32)]
    pub n_classes: usize,
    /// Hidden size for MLPs.
    #[config(default = 512)]
    pub hidden_size: usize,
    /// Uniform mixture fraction for categoricals (unimix). Default: 0.01
    #[config(default = 0.01)]
    pub unimix: f32,
}

impl RssmConfig {
    /// Stochastic state dimension (flattened).
    pub fn stoch_size(&self) -> usize {
        self.n_categories * self.n_classes
    }

    /// Full state dimension (deterministic + stochastic).
    pub fn state_size(&self) -> usize {
        self.deterministic_size + self.stoch_size()
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// RSSM state: deterministic `h` + stochastic `z`.
#[derive(Clone, Debug)]
pub struct RssmState<B: Backend> {
    /// Deterministic state from GRU: `[batch, deterministic_size]`.
    pub h: Tensor<B, 2>,
    /// Stochastic state (one-hot categoricals): `[batch, n_categories * n_classes]`.
    pub z: Tensor<B, 2>,
}

impl<B: Backend> RssmState<B> {
    /// Create a zero-initialized state.
    pub fn zeros(
        batch_size: usize,
        det_size: usize,
        stoch_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            h: Tensor::zeros([batch_size, det_size], device),
            z: Tensor::zeros([batch_size, stoch_size], device),
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// RSSM world model.
#[derive(Module, Debug)]
pub struct Rssm<B: Backend> {
    // Sequence model input projection
    seq_input_proj: Linear<B>,
    // Sequence model (GRU)
    seq_gru: GruCell<B>,
    seq_norm: LayerNorm<B>,

    // Encoder (posterior): obs + h -> z logits
    enc_mlp1: Linear<B>,
    enc_mlp2: Linear<B>,
    enc_norm: LayerNorm<B>,

    // Dynamics (prior): h -> z logits
    dyn_mlp1: Linear<B>,
    dyn_mlp2: Linear<B>,
    dyn_norm: LayerNorm<B>,

    // Reward predictor: h + z -> twohot
    reward_mlp1: Linear<B>,
    reward_mlp2: Linear<B>,
    reward_norm: LayerNorm<B>,

    // Continue predictor: h + z -> 1
    cont_mlp1: Linear<B>,
    cont_mlp2: Linear<B>,
    cont_norm: LayerNorm<B>,

    // Config values stored individually (Module derive requires Default for
    // skipped fields, so we store the primitives directly).
    #[module(skip)]
    deterministic_size: usize,
    #[module(skip)]
    n_categories: usize,
    #[module(skip)]
    n_classes: usize,
    #[module(skip)]
    unimix: f32,
}

impl RssmConfig {
    /// Initialize the RSSM on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Rssm<B> {
        let stoch_size = self.stoch_size();
        let hz_size = self.deterministic_size + stoch_size;

        // Sequence model: input is [z_{t-1}, a_{t-1}]
        let seq_input_dim = stoch_size + self.action_dim;
        let seq_input_proj =
            LinearConfig::new(seq_input_dim, self.hidden_size).init(device);
        let seq_gru =
            GruCellConfig::new(self.hidden_size, self.deterministic_size).init(device);
        let seq_norm = LayerNormConfig::new(self.deterministic_size).init(device);

        // Encoder: obs + h -> n_categories * n_classes logits
        let enc_mlp1 = LinearConfig::new(
            self.obs_dim + self.deterministic_size,
            self.hidden_size,
        )
        .init(device);
        let enc_mlp2 = LinearConfig::new(self.hidden_size, stoch_size).init(device);
        let enc_norm = LayerNormConfig::new(self.hidden_size).init(device);

        // Dynamics: h -> n_categories * n_classes logits
        let dyn_mlp1 =
            LinearConfig::new(self.deterministic_size, self.hidden_size).init(device);
        let dyn_mlp2 = LinearConfig::new(self.hidden_size, stoch_size).init(device);
        let dyn_norm = LayerNormConfig::new(self.hidden_size).init(device);

        // Reward: h + z -> 255 bins (twohot)
        let reward_mlp1 =
            LinearConfig::new(hz_size, self.hidden_size).init(device);
        let reward_mlp2 = LinearConfig::new(self.hidden_size, 255).init(device);
        let reward_norm = LayerNormConfig::new(self.hidden_size).init(device);

        // Continue: h + z -> 1
        let cont_mlp1 =
            LinearConfig::new(hz_size, self.hidden_size).init(device);
        let cont_mlp2 = LinearConfig::new(self.hidden_size, 1).init(device);
        let cont_norm = LayerNormConfig::new(self.hidden_size).init(device);

        Rssm {
            seq_input_proj,
            seq_gru,
            seq_norm,
            enc_mlp1,
            enc_mlp2,
            enc_norm,
            dyn_mlp1,
            dyn_mlp2,
            dyn_norm,
            reward_mlp1,
            reward_mlp2,
            reward_norm,
            cont_mlp1,
            cont_mlp2,
            cont_norm,
            deterministic_size: self.deterministic_size,
            n_categories: self.n_categories,
            n_classes: self.n_classes,
            unimix: self.unimix,
        }
    }
}

impl<B: Backend> Rssm<B> {
    /// Stochastic state dimension (flattened).
    fn stoch_size(&self) -> usize {
        self.n_categories * self.n_classes
    }

    /// Sequence model step: advance the deterministic state.
    ///
    /// `h_t = GRU(h_{t-1}, proj([z_{t-1}, a_{t-1}]))`
    pub fn sequence_step(
        &self,
        prev_state: &RssmState<B>,
        action: Tensor<B, 2>, // [batch, action_dim]
    ) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![prev_state.z.clone(), action], 1);
        let projected = self.seq_input_proj.forward(input);
        let projected = burn::tensor::activation::silu(projected);
        let h = self.seq_gru.forward(projected, prev_state.h.clone());
        self.seq_norm.forward(h)
    }

    /// Encoder (posterior): compute z distribution from observation and h.
    ///
    /// Returns logits `[batch, n_categories * n_classes]`.
    pub fn encode(&self, h: Tensor<B, 2>, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![h, obs], 1);
        let hidden = self.enc_mlp1.forward(input);
        let hidden = self.enc_norm.forward(hidden);
        let hidden = burn::tensor::activation::silu(hidden);
        self.enc_mlp2.forward(hidden)
    }

    /// Dynamics (prior): predict z distribution from h alone.
    ///
    /// Returns logits `[batch, n_categories * n_classes]`.
    pub fn dynamics(&self, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.dyn_mlp1.forward(h);
        let hidden = self.dyn_norm.forward(hidden);
        let hidden = burn::tensor::activation::silu(hidden);
        self.dyn_mlp2.forward(hidden)
    }

    /// Sample stochastic state from logits using straight-through.
    ///
    /// Returns one-hot z `[batch, n_categories * n_classes]`.
    pub fn sample_stochastic(
        &self,
        logits: Tensor<B, 2>, // [batch, n_cat * n_cls]
    ) -> Tensor<B, 2> {
        let [batch, _] = logits.dims();
        let n_cat = self.n_categories;
        let n_cls = self.n_classes;

        // Reshape to [batch * n_cat, n_cls]
        let logits_flat = logits.reshape([batch * n_cat, n_cls]);

        // Apply unimix: mix softmax probs with uniform
        let probs = burn::tensor::activation::softmax(logits_flat, 1);
        let uniform = 1.0 / n_cls as f32;
        let mixed =
            probs.clone() * (1.0 - self.unimix) + uniform * self.unimix;

        // Straight-through: one_hot(argmax(probs)) + probs - probs.detach()
        let indices: Tensor<B, 1, Int> =
            mixed.clone().argmax(1).squeeze_dim::<1>(1); // [batch * n_cat]

        // Build one-hot using index comparison
        let indices_2d = indices.unsqueeze_dim::<2>(1); // [batch*n_cat, 1]
        let device = mixed.device();
        let arange_data: Vec<i32> = (0..n_cls as i32).collect();
        let arange: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(arange_data, [1, n_cls]),
            &device,
        )
        .repeat_dim(0, batch * n_cat);

        let one_hot_mask = arange.equal(indices_2d.repeat_dim(1, n_cls));
        let one_hot_float: Tensor<B, 2> =
            Tensor::zeros([batch * n_cat, n_cls], &device)
                .mask_fill(one_hot_mask, 1.0);

        // Straight-through: gradient flows through mixed probs
        let st = one_hot_float + mixed.clone() - mixed.detach();

        st.reshape([batch, n_cat * n_cls])
    }

    /// Predict reward distribution from state.
    ///
    /// Returns logits `[batch, 255]`.
    pub fn predict_reward(
        &self,
        h: Tensor<B, 2>,
        z: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![h, z], 1);
        let hidden = self.reward_mlp1.forward(input);
        let hidden = self.reward_norm.forward(hidden);
        let hidden = burn::tensor::activation::silu(hidden);
        self.reward_mlp2.forward(hidden)
    }

    /// Predict continue probability from state.
    ///
    /// Returns logits `[batch, 1]`.
    pub fn predict_continue(
        &self,
        h: Tensor<B, 2>,
        z: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![h, z], 1);
        let hidden = self.cont_mlp1.forward(input);
        let hidden = self.cont_norm.forward(hidden);
        let hidden = burn::tensor::activation::silu(hidden);
        self.cont_mlp2.forward(hidden)
    }

    /// Full observation step: advance state and compute posterior.
    ///
    /// Returns `(new_state, posterior_logits, prior_logits)`.
    pub fn obs_step(
        &self,
        prev_state: &RssmState<B>,
        action: Tensor<B, 2>,
        obs: Tensor<B, 2>,
    ) -> (RssmState<B>, Tensor<B, 2>, Tensor<B, 2>) {
        // Advance deterministic state
        let h = self.sequence_step(prev_state, action);

        // Compute posterior and prior
        let post_logits = self.encode(h.clone(), obs);
        let prior_logits = self.dynamics(h.clone());

        // Sample z from posterior
        let z = self.sample_stochastic(post_logits.clone());

        let state = RssmState { h, z };
        (state, post_logits, prior_logits)
    }

    /// Imagination step: advance state using prior only (no observation).
    pub fn imagine_step(
        &self,
        prev_state: &RssmState<B>,
        action: Tensor<B, 2>,
    ) -> RssmState<B> {
        let h = self.sequence_step(prev_state, action);
        let prior_logits = self.dynamics(h.clone());
        let z = self.sample_stochastic(prior_logits);
        RssmState { h, z }
    }

    /// Get initial state (zeros).
    pub fn initial_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> RssmState<B> {
        RssmState::zeros(
            batch_size,
            self.deterministic_size,
            self.stoch_size(),
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    fn small_config() -> RssmConfig {
        RssmConfig::new(16, 4)
            .with_deterministic_size(32)
            .with_n_categories(4)
            .with_n_classes(4)
            .with_hidden_size(32)
    }

    #[test]
    fn initial_state_zeros() {
        let rssm = small_config().init::<B>(&dev());
        let state = rssm.initial_state(2, &dev());
        assert_eq!(state.h.dims(), [2, 32]);
        assert_eq!(state.z.dims(), [2, 16]); // 4 * 4

        let h_vals: Vec<f32> = state.h.to_data().to_vec().unwrap();
        let z_vals: Vec<f32> = state.z.to_data().to_vec().unwrap();
        for v in h_vals.iter().chain(z_vals.iter()) {
            assert_eq!(*v, 0.0, "initial state should be all zeros");
        }
    }

    #[test]
    fn sequence_step_output_shape() {
        let rssm = small_config().init::<B>(&dev());
        let state = rssm.initial_state(3, &dev());
        let action = Tensor::<B, 2>::zeros([3, 4], &dev());

        let h = rssm.sequence_step(&state, action);
        assert_eq!(h.dims(), [3, 32]);
    }

    #[test]
    fn encode_output_shape() {
        let rssm = small_config().init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 32], &dev());
        let obs = Tensor::<B, 2>::zeros([3, 16], &dev());

        let logits = rssm.encode(h, obs);
        assert_eq!(logits.dims(), [3, 16]); // n_cat * n_cls = 4 * 4
    }

    #[test]
    fn dynamics_output_shape() {
        let rssm = small_config().init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 32], &dev());

        let logits = rssm.dynamics(h);
        assert_eq!(logits.dims(), [3, 16]); // n_cat * n_cls = 4 * 4
    }

    #[test]
    fn sample_stochastic_output_shape() {
        let rssm = small_config().init::<B>(&dev());
        let logits = Tensor::<B, 2>::zeros([3, 16], &dev());

        let z = rssm.sample_stochastic(logits);
        assert_eq!(z.dims(), [3, 16]);
    }

    #[test]
    fn obs_step_returns_valid_state() {
        let rssm = small_config().init::<B>(&dev());
        let state = rssm.initial_state(2, &dev());
        let action = Tensor::<B, 2>::ones([2, 4], &dev());
        let obs = Tensor::<B, 2>::ones([2, 16], &dev());

        let (new_state, post_logits, prior_logits) =
            rssm.obs_step(&state, action, obs);

        assert_eq!(new_state.h.dims(), [2, 32]);
        assert_eq!(new_state.z.dims(), [2, 16]);
        assert_eq!(post_logits.dims(), [2, 16]);
        assert_eq!(prior_logits.dims(), [2, 16]);

        // Check values are finite
        let h_vals: Vec<f32> = new_state.h.to_data().to_vec().unwrap();
        for v in &h_vals {
            assert!(v.is_finite(), "h contains non-finite value: {v}");
        }
        let z_vals: Vec<f32> = new_state.z.to_data().to_vec().unwrap();
        for v in &z_vals {
            assert!(v.is_finite(), "z contains non-finite value: {v}");
        }
    }

    #[test]
    fn imagine_step_returns_valid_state() {
        let rssm = small_config().init::<B>(&dev());
        let state = rssm.initial_state(2, &dev());
        let action = Tensor::<B, 2>::ones([2, 4], &dev());

        let new_state = rssm.imagine_step(&state, action);

        assert_eq!(new_state.h.dims(), [2, 32]);
        assert_eq!(new_state.z.dims(), [2, 16]);

        let h_vals: Vec<f32> = new_state.h.to_data().to_vec().unwrap();
        for v in &h_vals {
            assert!(v.is_finite(), "h contains non-finite value: {v}");
        }
    }

    #[test]
    fn reward_predictor_shape() {
        let rssm = small_config().init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 32], &dev());
        let z = Tensor::<B, 2>::zeros([3, 16], &dev());

        let reward_logits = rssm.predict_reward(h, z);
        assert_eq!(reward_logits.dims(), [3, 255]);
    }

    #[test]
    fn continue_predictor_shape() {
        let rssm = small_config().init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 32], &dev());
        let z = Tensor::<B, 2>::zeros([3, 16], &dev());

        let cont_logits = rssm.predict_continue(h, z);
        assert_eq!(cont_logits.dims(), [3, 1]);
    }

    #[test]
    fn config_stoch_and_state_size() {
        let config = small_config();
        assert_eq!(config.stoch_size(), 16); // 4 * 4
        assert_eq!(config.state_size(), 48); // 32 + 16
    }

    #[test]
    fn multi_step_imagination() {
        let rssm = small_config().init::<B>(&dev());
        let mut state = rssm.initial_state(2, &dev());

        // Run 5 imagination steps
        for _ in 0..5 {
            let action = Tensor::<B, 2>::ones([2, 4], &dev());
            state = rssm.imagine_step(&state, action);
        }

        assert_eq!(state.h.dims(), [2, 32]);
        assert_eq!(state.z.dims(), [2, 16]);

        let h_vals: Vec<f32> = state.h.to_data().to_vec().unwrap();
        for v in &h_vals {
            assert!(v.is_finite(), "h diverged after multi-step: {v}");
        }
    }
}
