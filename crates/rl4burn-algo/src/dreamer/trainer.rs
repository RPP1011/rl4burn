//! DreamerTrainer: gradient-based training loop with optimizer integration.
//!
//! Provides a complete training step that:
//! 1. Computes world model loss and applies gradients.
//! 2. Computes actor-critic loss via imagination and applies gradients.
//! 3. Updates the slow (EMA) critic via Polyak averaging.

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rl4burn_nn::clip::clip_grad_norm;
use rl4burn_nn::mlp::{Mlp, MlpConfig, NormKind};
use rl4burn_nn::polyak::polyak_update;
use rl4burn_nn::rssm::{Rssm, RssmState};
use rl4burn_nn::symlog::TwohotEncoder;

use crate::dreamer::{
    dreamer_actor_critic_loss, dreamer_world_model_loss, ActorCriticStats, DreamerAgent,
    DreamerConfig, DreamerStats, WorldModelStats,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Training hyperparameters for the Dreamer agent.
#[derive(Debug, Clone)]
pub struct DreamerTrainerConfig {
    /// Agent configuration.
    pub agent: DreamerConfig,
    /// World model learning rate.
    pub wm_lr: f64,
    /// Actor learning rate.
    pub actor_lr: f64,
    /// Critic learning rate.
    pub critic_lr: f64,
    /// Maximum gradient norm for global clipping (0.0 = disabled).
    pub max_grad_norm: f32,
}

impl Default for DreamerTrainerConfig {
    fn default() -> Self {
        Self {
            agent: DreamerConfig::default(),
            wm_lr: 1e-4,
            actor_lr: 3e-5,
            critic_lr: 3e-5,
            max_grad_norm: 100.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Dreamer trainer with separate optimizers for world model, actor, and critic.
///
/// Manages gradient-based training, EMA slow critic updates, and provides
/// a single `train_step` method for the full update pipeline.
pub struct DreamerTrainer<B: AutodiffBackend> {
    /// The Dreamer agent (RSSM + actor + critic + projector).
    pub agent: DreamerAgent<B>,
    /// Slow (EMA) critic for stable value targets.
    pub slow_critic: Mlp<B::InnerBackend>,
    /// World model optimizer (RSSM).
    wm_optim: OptimizerAdaptor<Adam, Rssm<B>, B>,
    /// Actor optimizer.
    actor_optim: OptimizerAdaptor<Adam, Mlp<B>, B>,
    /// Critic optimizer.
    critic_optim: OptimizerAdaptor<Adam, Mlp<B>, B>,
    /// Training config.
    pub config: DreamerTrainerConfig,
    /// Training step counter.
    pub step: u64,
}

impl DreamerTrainerConfig {
    /// Initialize a [`DreamerTrainer`] on the given device.
    pub fn init<B: AutodiffBackend>(&self, device: &B::Device) -> DreamerTrainer<B> {
        let agent = self.agent.init::<B>(device);

        // Initialize slow critic on the inner backend
        let slow_critic = MlpConfig::new(
            self.agent.rssm.state_size(),
            self.agent.critic_hidden,
            255,
        )
        .with_n_layers(self.agent.ac_layers)
        .init_with_norm(NormKind::Rms, device);

        // Copy initial critic weights to slow critic
        let slow_critic = polyak_update(slow_critic, &agent.critic.valid(), 1.0);

        let adam_config = AdamConfig::new().with_epsilon(1e-8);
        let wm_optim = adam_config.init();
        let actor_optim = adam_config.init();
        let critic_optim = adam_config.init();

        DreamerTrainer {
            agent,
            slow_critic,
            wm_optim,
            actor_optim,
            critic_optim,
            config: self.clone(),
            step: 0,
        }
    }
}

/// Collect all parameter values from a module, flattened.
#[cfg(test)]
fn collect_param_values<B: Backend, M: burn::module::Module<B>>(module: &M) -> Vec<f32> {
    struct Collector {
        values: Vec<f32>,
    }
    impl<B: Backend> burn::module::ModuleVisitor<B> for Collector {
        fn visit_float<const D: usize>(
            &mut self,
            param: &burn::module::Param<Tensor<B, D>>,
        ) {
            let data: Vec<f32> = param.val().to_data().to_vec().unwrap();
            self.values.extend(data);
        }
    }
    let mut c = Collector { values: Vec::new() };
    module.visit(&mut c);
    c.values
}

impl<B: AutodiffBackend> DreamerTrainer<B> {
    /// Perform a full training step: world model + actor-critic + EMA update.
    ///
    /// # Arguments
    /// * `observations` — `[batch, seq_len, obs_dim]` encoded observations.
    /// * `actions` — `[batch, seq_len, action_dim]` actions.
    /// * `rewards` — `[batch, seq_len]` rewards.
    /// * `continues` — `[batch, seq_len]` continue flags.
    ///
    /// # Returns
    /// Combined training statistics.
    pub fn train_step(
        &mut self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        rewards: Tensor<B, 2>,
        continues: Tensor<B, 2>,
    ) -> DreamerStats {
        let wm_lr = self.config.wm_lr;
        let actor_lr = self.config.actor_lr;
        let critic_lr = self.config.critic_lr;

        // 1. World model update
        let wm_stats = self.update_world_model(
            observations.clone(),
            actions,
            rewards,
            continues,
            wm_lr,
        );

        // 2. Actor-critic update via imagination
        let [batch, ..] = observations.dims();
        let device = observations.device();
        let initial_states = self.agent.rssm.initial_state(batch, &device);
        let ac_stats = self.update_actor_critic(initial_states, actor_lr, critic_lr);

        // 3. EMA update for slow critic
        self.update_slow_critic();

        self.step += 1;

        DreamerStats {
            wm: wm_stats,
            ac: ac_stats,
        }
    }

    /// Update the world model (RSSM + representation projector).
    pub fn update_world_model(
        &mut self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        rewards: Tensor<B, 2>,
        continues: Tensor<B, 2>,
        lr: f64,
    ) -> WorldModelStats {
        let (loss, stats) = dreamer_world_model_loss(
            &self.agent,
            observations,
            actions,
            rewards,
            continues,
        );

        // Backward pass on RSSM
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.agent.rssm);

        // Clip gradients
        let grads = if self.config.max_grad_norm > 0.0 {
            clip_grad_norm(&self.agent.rssm, grads, self.config.max_grad_norm)
        } else {
            grads
        };

        // Apply gradients to RSSM
        self.agent.rssm = self.wm_optim.step(lr, self.agent.rssm.clone(), grads);

        stats
    }

    /// Update actor and critic via imagination rollouts.
    ///
    /// Actor and critic are updated in separate forward passes to avoid
    /// consuming the autodiff graph twice.
    pub fn update_actor_critic(
        &mut self,
        initial_states: RssmState<B>,
        actor_lr: f64,
        critic_lr: f64,
    ) -> ActorCriticStats {
        // --- Actor update (separate forward pass) ---
        let (actor_loss, _, partial_stats) =
            dreamer_actor_critic_loss(&self.agent, initial_states.clone());

        let actor_grads = actor_loss.backward();
        let actor_grads = GradientsParams::from_grads(actor_grads, &self.agent.actor);
        let actor_grads = if self.config.max_grad_norm > 0.0 {
            clip_grad_norm(&self.agent.actor, actor_grads, self.config.max_grad_norm)
        } else {
            actor_grads
        };
        self.agent.actor =
            self.actor_optim
                .step(actor_lr, self.agent.actor.clone(), actor_grads);

        // --- Critic update (separate forward pass) ---
        let (_, critic_loss, stats) =
            dreamer_actor_critic_loss(&self.agent, initial_states);

        let critic_grads = critic_loss.backward();
        let critic_grads = GradientsParams::from_grads(critic_grads, &self.agent.critic);
        let critic_grads = if self.config.max_grad_norm > 0.0 {
            clip_grad_norm(
                &self.agent.critic,
                critic_grads,
                self.config.max_grad_norm,
            )
        } else {
            critic_grads
        };
        self.agent.critic =
            self.critic_optim
                .step(critic_lr, self.agent.critic.clone(), critic_grads);

        // Return stats from the first pass (actor loss is from before update)
        ActorCriticStats {
            actor_loss: partial_stats.actor_loss,
            critic_loss: stats.critic_loss,
            entropy: partial_stats.entropy,
        }
    }

    /// Polyak-average the critic weights into the slow critic.
    pub fn update_slow_critic(&mut self) {
        let decay = self.agent.config.slow_critic_decay;
        // slow = decay * slow + (1 - decay) * critic
        // polyak_update(target, source, tau) → target = tau * source + (1-tau) * target
        let ema_tau = 1.0 - decay;
        self.slow_critic = polyak_update(
            self.slow_critic.clone(),
            &self.agent.critic.valid(),
            ema_tau,
        );
    }

    /// Get the slow critic's value estimate for given states.
    pub fn slow_value(&self, state: &RssmState<B::InnerBackend>) -> Tensor<B::InnerBackend, 1> {
        let features = Tensor::cat(vec![state.h.clone(), state.z.clone()], 1);
        let logits = self.slow_critic.forward(features);
        let probs = burn::tensor::activation::softmax(logits, 1);
        let device = state.h.device();
        TwohotEncoder::new().decode(probs, &device)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    use crate::loss::kl_balance::KlBalanceConfig;
    use crate::loss::representation::{
        BarlowTwinsConfig, DreamerProConfig, InfoNceConfig, RepresentationVariant,
    };
    use rl4burn_nn::rssm::RssmConfig;

    type AB = Autodiff<NdArray>;

    fn small_trainer_config() -> DreamerTrainerConfig {
        DreamerTrainerConfig {
            agent: DreamerConfig {
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
                encoder_config: None,
            },
            wm_lr: 1e-4,
            actor_lr: 3e-5,
            critic_lr: 3e-5,
            max_grad_norm: 100.0,
        }
    }

    #[test]
    fn trainer_init() {
        let config = small_trainer_config();
        let device = <AB as Backend>::Device::default();
        let _trainer = config.init::<AB>(&device);
    }

    #[test]
    fn world_model_update_changes_params() {
        let config = small_trainer_config();
        let device = <AB as Backend>::Device::default();
        let mut trainer = config.init::<AB>(&device);

        let before = collect_param_values(&trainer.agent.rssm);

        let obs = Tensor::<AB, 3>::random(
            [2, 5, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let acts = Tensor::<AB, 3>::random(
            [2, 5, 4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let rews = Tensor::<AB, 2>::zeros([2, 5], &device);
        let conts = Tensor::<AB, 2>::ones([2, 5], &device);

        let stats = trainer.update_world_model(obs, acts, rews, conts, 1e-3);
        assert!(stats.total_loss.is_finite());

        let after = collect_param_values(&trainer.agent.rssm);

        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "RSSM params should change after gradient step");
    }

    #[test]
    fn actor_critic_update_changes_params() {
        let config = small_trainer_config();
        let device = <AB as Backend>::Device::default();
        let mut trainer = config.init::<AB>(&device);

        let before = collect_param_values(&trainer.agent.actor);

        let initial = trainer.agent.rssm.initial_state(2, &device);
        let stats = trainer.update_actor_critic(initial, 1e-3, 1e-3);
        assert!(stats.actor_loss.is_finite());
        assert!(stats.critic_loss.is_finite());

        let after = collect_param_values(&trainer.agent.actor);

        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "actor params should change after gradient step");
    }

    #[test]
    fn slow_critic_ema_updates() {
        let config = small_trainer_config();
        let device = <AB as Backend>::Device::default();
        let mut trainer = config.init::<AB>(&device);

        // Do a critic update to create divergence between critic and slow_critic
        let initial = trainer.agent.rssm.initial_state(2, &device);
        let _stats = trainer.update_actor_critic(initial, 1e-2, 1e-2);

        let before = collect_param_values(&trainer.slow_critic);

        trainer.update_slow_critic();

        let after = collect_param_values(&trainer.slow_critic);

        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "slow critic should update via EMA");
    }

    #[test]
    fn full_train_step() {
        let config = small_trainer_config();
        let device = <AB as Backend>::Device::default();
        let mut trainer = config.init::<AB>(&device);

        let obs = Tensor::<AB, 3>::random(
            [2, 5, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let acts = Tensor::<AB, 3>::random(
            [2, 5, 4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let rews = Tensor::<AB, 2>::zeros([2, 5], &device);
        let conts = Tensor::<AB, 2>::ones([2, 5], &device);

        let stats = trainer.train_step(obs, acts, rews, conts);
        assert!(stats.wm.total_loss.is_finite());
        assert!(stats.ac.actor_loss.is_finite());
        assert!(stats.ac.critic_loss.is_finite());
        assert_eq!(trainer.step, 1);
    }
}
