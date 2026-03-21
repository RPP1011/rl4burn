//! R2-Dreamer agent: world-model-based RL with multiple representation losses.
//!
//! This module provides:
//! - [`DreamerConfig`] — top-level configuration.
//! - [`DreamerAgent`] — composed world model + actor-critic.
//! - [`dreamer_world_model_loss`] — trains the world model on observed sequences.
//! - [`dreamer_actor_critic_loss`] — trains actor-critic via imagination.
//! - [`DreamerStats`] — loggable training statistics.
//!
//! The agent supports four representation learning variants:
//! - **Dreamer** (decoder-based reconstruction, DreamerV3 baseline)
//! - **R2-Dreamer** (Barlow Twins redundancy reduction)
//! - **InfoNCE** (contrastive learning)
//! - **DreamerPro** (prototype-based with Sinkhorn)
//!
//! Reference: Nauman & Straffelini, "R2-Dreamer: Redundancy Reduction for
//! Computationally Efficient World Models" (ICLR 2026).

use burn::prelude::*;

use rl4burn_nn::mlp::{Mlp, MlpConfig, NormKind};
use rl4burn_nn::rssm::{Rssm, RssmConfig, RssmState};
use rl4burn_nn::symlog::TwohotEncoder;

use crate::loss::kl_balance::{kl_balanced_loss, KlBalanceConfig};
use crate::loss::representation::{
    self, BarlowTwinsConfig, DreamerProConfig, InfoNceConfig, RepresentationVariant,
};
use crate::planning::imagination::{imagine_rollout, lambda_returns, ImaginedTrajectory};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for the R2-Dreamer agent.
#[derive(Debug, Clone)]
pub struct DreamerConfig {
    /// RSSM world model configuration.
    pub rssm: RssmConfig,
    /// Representation learning variant.
    pub rep_variant: RepresentationVariant,
    /// Actor MLP hidden size.
    pub actor_hidden: usize,
    /// Critic MLP hidden size.
    pub critic_hidden: usize,
    /// Number of hidden layers in actor/critic MLPs.
    pub ac_layers: usize,
    /// Number of action dimensions.
    pub action_dim: usize,
    /// Whether actions are discrete (true) or continuous (false).
    pub discrete_actions: bool,
    /// Imagination horizon for actor-critic training.
    pub horizon: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Lambda for lambda-returns.
    pub lambda: f32,
    /// Entropy bonus coefficient for the actor.
    pub entropy_coef: f32,
    /// KL balancing configuration.
    pub kl_config: KlBalanceConfig,
    /// Barlow Twins configuration (used when `rep_variant == R2Dreamer`).
    pub barlow_config: BarlowTwinsConfig,
    /// InfoNCE configuration (used when `rep_variant == InfoNCE`).
    pub infonce_config: InfoNceConfig,
    /// DreamerPro configuration (used when `rep_variant == DreamerPro`).
    pub dreamerpro_config: DreamerProConfig,
    /// Slow critic EMA decay rate.
    pub slow_critic_decay: f32,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            rssm: RssmConfig::new(256, 4)
                .with_deterministic_size(512)
                .with_n_categories(32)
                .with_n_classes(32)
                .with_hidden_size(512),
            rep_variant: RepresentationVariant::R2Dreamer,
            actor_hidden: 512,
            critic_hidden: 512,
            ac_layers: 2,
            action_dim: 4,
            discrete_actions: false,
            horizon: 15,
            gamma: 0.997,
            lambda: 0.95,
            entropy_coef: 3e-4,
            kl_config: KlBalanceConfig::default(),
            barlow_config: BarlowTwinsConfig::default(),
            infonce_config: InfoNceConfig::default(),
            dreamerpro_config: DreamerProConfig::default(),
            slow_critic_decay: 0.98,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent (plain struct, not Module-derived)
// ---------------------------------------------------------------------------

/// R2-Dreamer agent: world model + actor-critic.
///
/// This is a plain struct (not `#[derive(Module)]`) because it holds a
/// heterogeneous `DreamerConfig`.  Call individual sub-modules for training.
pub struct DreamerAgent<B: Backend> {
    /// RSSM world model.
    pub rssm: Rssm<B>,
    /// Actor MLP: state → action logits (discrete) or mean (continuous).
    pub actor: Mlp<B>,
    /// Critic MLP: state → twohot value logits.
    pub critic: Mlp<B>,
    /// Representation projection head for self-supervised losses.
    pub rep_projector: Option<Mlp<B>>,
    /// DreamerPro prototypes (only allocated if variant == DreamerPro).
    pub prototypes: Option<Tensor<B, 2>>,
    /// Runtime config.
    pub config: DreamerConfig,
}

impl DreamerConfig {
    /// Initialize a [`DreamerAgent`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> DreamerAgent<B> {
        let rssm = self.rssm.init(device);
        let state_dim = self.rssm.state_size();

        let actor = MlpConfig::new(state_dim, self.actor_hidden, self.action_dim)
            .with_n_layers(self.ac_layers)
            .init_with_norm(NormKind::Rms, device);

        let critic = MlpConfig::new(state_dim, self.critic_hidden, 255)
            .with_n_layers(self.ac_layers)
            .init_with_norm(NormKind::Rms, device);

        let rep_projector = match self.rep_variant {
            RepresentationVariant::Dreamer => None,
            _ => Some(
                MlpConfig::new(state_dim, self.actor_hidden, 256)
                    .with_n_layers(1)
                    .init_with_norm(NormKind::Rms, device),
            ),
        };

        let prototypes = match self.rep_variant {
            RepresentationVariant::DreamerPro => Some(Tensor::random(
                [self.dreamerpro_config.n_prototypes, 256],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                device,
            )),
            _ => None,
        };

        DreamerAgent {
            rssm,
            actor,
            critic,
            rep_projector,
            prototypes,
            config: self.clone(),
        }
    }
}

impl<B: Backend> DreamerAgent<B> {
    /// Compute the full RSSM state feature vector `[h, z]`.
    pub fn state_features(&self, state: &RssmState<B>) -> Tensor<B, 2> {
        Tensor::cat(vec![state.h.clone(), state.z.clone()], 1)
    }

    /// Actor forward: state → action logits/mean.
    pub fn act(&self, state: &RssmState<B>) -> Tensor<B, 2> {
        let features = self.state_features(state);
        self.actor.forward(features)
    }

    /// Critic forward: state → twohot value logits `[batch, 255]`.
    pub fn value_logits(&self, state: &RssmState<B>) -> Tensor<B, 2> {
        let features = self.state_features(state);
        self.critic.forward(features)
    }

    /// Decode critic logits to scalar values.
    pub fn value(&self, state: &RssmState<B>) -> Tensor<B, 1> {
        let logits = self.value_logits(state);
        let probs = burn::tensor::activation::softmax(logits, 1);
        let device = state.h.device();
        TwohotEncoder::new().decode(probs, &device)
    }
}

// ---------------------------------------------------------------------------
// World model loss
// ---------------------------------------------------------------------------

/// Statistics from world model training.
#[derive(Debug, Clone)]
pub struct WorldModelStats {
    pub kl_loss: f32,
    pub rep_loss: f32,
    pub reward_loss: f32,
    pub continue_loss: f32,
    pub total_loss: f32,
}

/// Train the world model on a batch of observed sequences.
///
/// # Arguments
/// * `agent` — the Dreamer agent (RSSM + heads).
/// * `observations` — `[batch, seq_len, obs_dim]` encoded observation embeddings.
/// * `actions` — `[batch, seq_len, action_dim]` actions taken.
/// * `rewards` — `[batch, seq_len]` observed rewards.
/// * `continues` — `[batch, seq_len]` continue flags (1.0 = not done).
///
/// # Returns
/// `(total_loss, stats)`.
pub fn dreamer_world_model_loss<B: Backend>(
    agent: &DreamerAgent<B>,
    observations: Tensor<B, 3>,
    actions: Tensor<B, 3>,
    rewards: Tensor<B, 2>,
    continues: Tensor<B, 2>,
) -> (Tensor<B, 1>, WorldModelStats) {
    let [batch, seq_len, _obs_dim] = observations.dims();
    let device = observations.device();

    let mut state = agent.rssm.initial_state(batch, &device);

    let mut all_post_logits = Vec::with_capacity(seq_len);
    let mut all_prior_logits = Vec::with_capacity(seq_len);
    let mut all_states = Vec::with_capacity(seq_len);

    // Roll through sequence
    for t in 0..seq_len {
        let obs_t: Tensor<B, 2> = observations
            .clone()
            .slice([0..batch, t..t + 1])
            .squeeze::<2>();
        let act_t: Tensor<B, 2> = actions
            .clone()
            .slice([0..batch, t..t + 1])
            .squeeze::<2>();

        let (new_state, post_logits, prior_logits) =
            agent.rssm.obs_step(&state, act_t, obs_t);

        all_post_logits.push(post_logits);
        all_prior_logits.push(prior_logits);
        all_states.push(new_state.clone());
        state = new_state;
    }

    // 1. KL loss (averaged over time)
    let mut kl_total = Tensor::<B, 1>::zeros([1], &device);
    for t in 0..seq_len {
        kl_total = kl_total
            + kl_balanced_loss(
                all_post_logits[t].clone(),
                all_prior_logits[t].clone(),
                &agent.config.kl_config,
            );
    }
    let kl_loss = kl_total / seq_len as f32;

    // 2. Representation loss
    let rep_loss = match agent.config.rep_variant {
        RepresentationVariant::Dreamer => Tensor::<B, 1>::zeros([1], &device),
        RepresentationVariant::R2Dreamer => {
            let proj = agent.rep_projector.as_ref().unwrap();
            let mut loss = Tensor::<B, 1>::zeros([1], &device);
            for t in 0..seq_len {
                let post_feat = agent.state_features(&all_states[t]);
                let z_post = proj.forward(post_feat.clone());
                let z_prior = proj.forward(post_feat.detach());
                loss = loss
                    + representation::barlow_twins_loss(
                        z_post,
                        z_prior,
                        &agent.config.barlow_config,
                    );
            }
            loss / seq_len as f32
        }
        RepresentationVariant::InfoNCE => {
            let proj = agent.rep_projector.as_ref().unwrap();
            let mut loss = Tensor::<B, 1>::zeros([1], &device);
            for t in 0..seq_len {
                let post_feat = agent.state_features(&all_states[t]);
                let z_post = proj.forward(post_feat.clone());
                let z_prior = proj.forward(post_feat.detach());
                loss = loss
                    + representation::infonce_loss(
                        z_post,
                        z_prior,
                        &agent.config.infonce_config,
                    );
            }
            loss / seq_len as f32
        }
        RepresentationVariant::DreamerPro => {
            let proj = agent.rep_projector.as_ref().unwrap();
            let protos = agent.prototypes.as_ref().unwrap();
            let mut loss = Tensor::<B, 1>::zeros([1], &device);
            for t in 0..seq_len {
                let post_feat = agent.state_features(&all_states[t]);
                let z_online = proj.forward(post_feat.clone());
                let z_target = proj.forward(post_feat.detach());
                loss = loss
                    + representation::dreamerpro_loss(
                        z_online,
                        z_target,
                        protos.clone(),
                        &agent.config.dreamerpro_config,
                    );
            }
            loss / seq_len as f32
        }
    };

    // 3. Reward prediction loss (twohot cross-entropy)
    let twohot = TwohotEncoder::new();
    let mut reward_loss = Tensor::<B, 1>::zeros([1], &device);
    for t in 0..seq_len {
        let r_logits = agent
            .rssm
            .predict_reward(all_states[t].h.clone(), all_states[t].z.clone());
        let r_target: Tensor<B, 1> = rewards
            .clone()
            .slice([0..batch, t..t + 1])
            .squeeze::<1>();
        reward_loss = reward_loss + twohot.loss(r_logits, r_target, &device);
    }
    let reward_loss = reward_loss / seq_len as f32;

    // 4. Continue prediction loss (binary cross-entropy)
    let mut cont_loss = Tensor::<B, 1>::zeros([1], &device);
    for t in 0..seq_len {
        let c_logits = agent
            .rssm
            .predict_continue(all_states[t].h.clone(), all_states[t].z.clone());
        let c_target: Tensor<B, 2> = continues.clone().slice([0..batch, t..t + 1]);
        let c_prob = burn::tensor::activation::sigmoid(c_logits);
        let eps = 1e-7;
        let bce = (c_target.clone() * (c_prob.clone() + eps).log()
            + (c_target.neg() + 1.0) * ((c_prob.neg() + 1.0) + eps).log())
        .neg()
        .mean();
        cont_loss = cont_loss + bce.unsqueeze();
    }
    let cont_loss = cont_loss / seq_len as f32;

    let total = kl_loss.clone() + rep_loss.clone() + reward_loss.clone() + cont_loss.clone();

    let stats = WorldModelStats {
        kl_loss: kl_loss.clone().into_scalar().elem(),
        rep_loss: rep_loss.clone().into_scalar().elem(),
        reward_loss: reward_loss.clone().into_scalar().elem(),
        continue_loss: cont_loss.clone().into_scalar().elem(),
        total_loss: total.clone().into_scalar().elem(),
    };

    (total, stats)
}

// ---------------------------------------------------------------------------
// Actor-critic loss
// ---------------------------------------------------------------------------

/// Statistics from actor-critic training.
#[derive(Debug, Clone)]
pub struct ActorCriticStats {
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub entropy: f32,
}

/// Combined statistics for logging.
#[derive(Debug, Clone)]
pub struct DreamerStats {
    pub wm: WorldModelStats,
    pub ac: ActorCriticStats,
}

impl rl4burn_core::log::Loggable for DreamerStats {
    fn log(&self, logger: &mut dyn rl4burn_core::log::Logger, step: u64) {
        logger.log_scalar("train/kl_loss", self.wm.kl_loss as f64, step);
        logger.log_scalar("train/rep_loss", self.wm.rep_loss as f64, step);
        logger.log_scalar("train/reward_loss", self.wm.reward_loss as f64, step);
        logger.log_scalar("train/continue_loss", self.wm.continue_loss as f64, step);
        logger.log_scalar("train/wm_total_loss", self.wm.total_loss as f64, step);
        logger.log_scalar("train/actor_loss", self.ac.actor_loss as f64, step);
        logger.log_scalar("train/critic_loss", self.ac.critic_loss as f64, step);
        logger.log_scalar("train/entropy", self.ac.entropy as f64, step);
    }
}

/// Train actor and critic via imagination rollouts.
///
/// # Arguments
/// * `agent` — the Dreamer agent.
/// * `initial_states` — starting RSSM states from the replay buffer.
///
/// # Returns
/// `(actor_loss, critic_loss, stats)`.
pub fn dreamer_actor_critic_loss<B: Backend>(
    agent: &DreamerAgent<B>,
    initial_states: RssmState<B>,
) -> (Tensor<B, 1>, Tensor<B, 1>, ActorCriticStats) {
    let device = initial_states.h.device();
    let config = &agent.config;

    // 1. Imagination rollout
    let traj: ImaginedTrajectory<B> = imagine_rollout(
        &agent.rssm,
        initial_states,
        |h, z| {
            let state = RssmState { h, z };
            agent.act(&state)
        },
        config.horizon,
    );

    // 2. Compute values at each imagined state
    let twohot = TwohotEncoder::new();
    let mut values: Vec<Tensor<B, 1>> = Vec::with_capacity(traj.states.len());
    for state in &traj.states {
        let v_logits = agent.value_logits(state);
        let v_probs = burn::tensor::activation::softmax(v_logits, 1);
        values.push(twohot.decode(v_probs, &device));
    }

    // 3. Decode rewards and continues
    let mut rewards: Vec<Tensor<B, 1>> = Vec::with_capacity(config.horizon);
    let mut continues: Vec<Tensor<B, 1>> = Vec::with_capacity(config.horizon);
    for t in 0..config.horizon {
        let r_probs =
            burn::tensor::activation::softmax(traj.reward_logits[t].clone(), 1);
        rewards.push(twohot.decode(r_probs, &device));

        let c_prob = burn::tensor::activation::sigmoid(traj.continue_logits[t].clone());
        continues.push(c_prob.squeeze_dim::<1>(1));
    }

    // 4. Lambda-returns
    let returns = lambda_returns(&rewards, &values, &continues, config.gamma, config.lambda);

    // 5. Actor loss
    let mut actor_loss = Tensor::<B, 1>::zeros([1], &device);
    let mut total_entropy = 0.0f32;

    for t in 0..config.horizon {
        let advantage = returns[t].clone() - values[t].clone().detach();
        let action_logits = agent.act(&traj.states[t]);

        if config.discrete_actions {
            let log_probs =
                burn::tensor::activation::log_softmax(action_logits.clone(), 1);
            let probs = burn::tensor::activation::softmax(action_logits, 1);
            let entropy: Tensor<B, 1> = (probs * log_probs.clone())
                .sum_dim(1)
                .squeeze_dim::<1>(1)
                .neg();

            let action_idx: Tensor<B, 2, Int> = traj.actions[t].clone().argmax(1);
            let action_lp: Tensor<B, 1> =
                log_probs.gather(1, action_idx).squeeze_dim::<1>(1);

            let step_loss = (action_lp * advantage.detach()).neg()
                - entropy.clone() * config.entropy_coef;
            actor_loss = actor_loss + step_loss.mean().unsqueeze();

            let ent_val: f32 = entropy.mean().into_scalar().elem();
            total_entropy += ent_val;
        } else {
            let step_loss = (advantage.detach().unsqueeze_dim::<2>(1) * action_logits)
                .sum_dim(1)
                .squeeze_dim::<1>(1)
                .neg();
            actor_loss = actor_loss + step_loss.mean().unsqueeze();
        }
    }
    let actor_loss = actor_loss / config.horizon as f32;
    let avg_entropy = total_entropy / config.horizon as f32;

    // 6. Critic loss
    let mut critic_loss = Tensor::<B, 1>::zeros([1], &device);
    for t in 0..config.horizon {
        let v_logits = agent.value_logits(&traj.states[t]);
        let target = returns[t].clone().detach();
        critic_loss = critic_loss + twohot.loss(v_logits, target, &device);
    }
    let critic_loss = critic_loss / config.horizon as f32;

    let stats = ActorCriticStats {
        actor_loss: actor_loss.clone().into_scalar().elem(),
        critic_loss: critic_loss.clone().into_scalar().elem(),
        entropy: avg_entropy,
    };

    (actor_loss, critic_loss, stats)
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

    fn small_config() -> DreamerConfig {
        DreamerConfig {
            rssm: RssmConfig::new(16, 4)
                .with_deterministic_size(32)
                .with_n_categories(4)
                .with_n_classes(4)
                .with_hidden_size(32),
            rep_variant: RepresentationVariant::R2Dreamer,
            actor_hidden: 32,
            critic_hidden: 32,
            ac_layers: 1,
            action_dim: 4,
            discrete_actions: true,
            horizon: 3,
            gamma: 0.99,
            lambda: 0.95,
            entropy_coef: 1e-3,
            kl_config: KlBalanceConfig::default(),
            barlow_config: BarlowTwinsConfig::default(),
            infonce_config: InfoNceConfig::default(),
            dreamerpro_config: DreamerProConfig {
                n_prototypes: 8,
                sinkhorn_iters: 2,
                temperature: 0.1,
            },
            slow_critic_decay: 0.98,
        }
    }

    #[test]
    fn agent_init_and_act() {
        let config = small_config();
        let agent = config.init::<B>(&dev());
        let state = agent.rssm.initial_state(2, &dev());
        let action = agent.act(&state);
        assert_eq!(action.dims(), [2, 4]);
    }

    #[test]
    fn agent_value() {
        let config = small_config();
        let agent = config.init::<B>(&dev());
        let state = agent.rssm.initial_state(2, &dev());
        let v = agent.value(&state);
        assert_eq!(v.dims(), [2]);
        let vals: Vec<f32> = v.to_data().to_vec().unwrap();
        for &val in &vals {
            assert!(val.is_finite(), "value should be finite, got {val}");
        }
    }

    #[test]
    fn world_model_loss_runs() {
        let config = small_config();
        let agent = config.init::<B>(&dev());

        let obs = Tensor::<B, 3>::zeros([2, 5, 16], &dev());
        let acts = Tensor::<B, 3>::zeros([2, 5, 4], &dev());
        let rews = Tensor::<B, 2>::zeros([2, 5], &dev());
        let conts = Tensor::<B, 2>::ones([2, 5], &dev());

        let (loss, stats) = dreamer_world_model_loss(&agent, obs, acts, rews, conts);
        assert_eq!(loss.dims(), [1]);
        assert!(stats.total_loss.is_finite());
    }

    #[test]
    fn actor_critic_loss_runs() {
        let config = small_config();
        let agent = config.init::<B>(&dev());
        let initial = agent.rssm.initial_state(2, &dev());

        let (a_loss, c_loss, stats) = dreamer_actor_critic_loss(&agent, initial);
        assert_eq!(a_loss.dims(), [1]);
        assert_eq!(c_loss.dims(), [1]);
        assert!(stats.actor_loss.is_finite());
        assert!(stats.critic_loss.is_finite());
    }

    #[test]
    fn all_rep_variants_compile() {
        for variant in [
            RepresentationVariant::Dreamer,
            RepresentationVariant::R2Dreamer,
            RepresentationVariant::InfoNCE,
            RepresentationVariant::DreamerPro,
        ] {
            let mut config = small_config();
            config.rep_variant = variant;
            let agent = config.init::<B>(&dev());

            let obs = Tensor::<B, 3>::zeros([2, 3, 16], &dev());
            let acts = Tensor::<B, 3>::zeros([2, 3, 4], &dev());
            let rews = Tensor::<B, 2>::zeros([2, 3], &dev());
            let conts = Tensor::<B, 2>::ones([2, 3], &dev());

            let (loss, stats) = dreamer_world_model_loss(&agent, obs, acts, rews, conts);
            assert!(
                stats.total_loss.is_finite(),
                "variant {variant:?} produced non-finite loss: {}",
                stats.total_loss
            );
            let _val: f32 = loss.into_scalar().elem();
        }
    }
}
