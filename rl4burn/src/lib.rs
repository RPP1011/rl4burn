//! Reinforcement learning algorithms for the Burn ML framework.
//!
//! `rl4burn` provides generic, backend-agnostic RL building blocks that
//! exploit Burn's type system: write `PPO<B: AutodiffBackend>` once and
//! run on WGPU, CUDA, NdArray, or LibTorch.
//!
//! # Crates
//!
//! - [`rl4burn_core`] — Environment trait, spaces, vectorized environments, wrappers, logging
//! - [`rl4burn_nn`] — Neural network utilities (init, gradient clipping, polyak, policy traits)
//! - [`rl4burn_collect`] — Data collection (GAE, V-trace, UPGO, replay buffer, advantage normalization)
//! - [`rl4burn_algo`] — Algorithms (PPO, DQN, AC, imitation, multi-agent, planning)
//! - [`rl4burn_envs`] — Built-in environments (CartPole, Pendulum, GridWorld)

// Re-export sub-crates as modules for discoverability
pub use rl4burn_core as core;
pub use rl4burn_nn as nn;
pub use rl4burn_collect as collect;
pub use rl4burn_algo as algo;
pub use rl4burn_envs as envs;

// Also expose the env and log modules directly from core
pub mod env {
    pub use rl4burn_core::env::*;
}
pub mod log {
    pub use rl4burn_core::log::*;
}

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// Error types
pub use rl4burn_core::error::{Rl4BurnError, Result as Rl4BurnResult};

// Environment
pub use rl4burn_core::env::adapter::DiscreteEnvAdapter;
pub use rl4burn_core::env::space::Space;
pub use rl4burn_core::env::vec_env::SyncVecEnv;
pub use rl4burn_core::env::wrapper;
pub use rl4burn_core::env::{Env, Step};

// Algorithms
pub use rl4burn_algo::base::dqn::{dqn_update, epsilon_greedy, epsilon_schedule, DqnConfig, DqnStats, QNetwork, Transition};
pub use rl4burn_algo::base::ppo::{ppo_collect, ppo_update, PpoConfig, PpoRollout, PpoStats};
pub use rl4burn_algo::base::ppo_masked::{
    masked_ppo_collect, masked_ppo_update, MaskedActorCritic, MaskedPpoRollout,
};
pub use rl4burn_algo::base::ac::{ac_vtrace_update, AcStats};
pub use rl4burn_algo::imitation::behavioral_cloning::{bc_loss_discrete, bc_loss_multi_head, bc_step};
pub use rl4burn_algo::imitation::distillation::{distillation_loss, value_distillation_loss, DistillationConfig};
pub use rl4burn_algo::multi_agent::league::{AgentRole, League, LeagueAgentConfig};
pub use rl4burn_algo::multi_agent::utils::{batch_multi_agent_obs, broadcast_team_reward, unbatch_actions, MultiAgentRolloutData};
pub use rl4burn_algo::multi_agent::pfsp::{PfspConfig, PfspMatchmaking, PlayerRecord};
pub use rl4burn_algo::multi_agent::self_play::{branch_agent, SelfPlayPool};
pub use rl4burn_algo::planning::imagination::{imagine_rollout, lambda_returns, ImaginedTrajectory};
pub use rl4burn_algo::planning::mcts::{MctsConfig, MctsTree};
pub use rl4burn_algo::distributed::{
    DistributedConfig, GradientSync, LocalSync, ReduceStrategy, scale_gradients,
};
pub use rl4burn_algo::privileged_critic::{make_critic_input, PrivilegedActorCritic};
pub use rl4burn_algo::z_conditioning::{z_reward, ZConditioning, ZConditioningConfig};

// Data collection
pub use rl4burn_collect::actor_learner::{actor_learner_collect, batched_inference};
pub use rl4burn_collect::centralized_inference::{
    inference_channel, serve_inference_batch, InferenceHandle, InferenceReceiver,
    InferenceRequest, InferenceResponse,
};
pub use rl4burn_collect::trajectory::{trajectory_queue, Trajectory, TrajectoryConsumer, TrajectoryProducer};
pub use rl4burn_collect::cspl::{CsplConfig, CsplPhase, CsplPipeline};
pub use rl4burn_collect::advantage::normalize;
pub use rl4burn_collect::gae::gae;
pub use rl4burn_collect::intrinsic::{combine_rewards, CountBasedReward, EntropyReductionReward, IntrinsicReward};
pub use rl4burn_collect::percentile_normalize::PercentileNormalizer;
pub use rl4burn_collect::replay::ReplayBuffer;
pub use rl4burn_collect::sequence_replay::{SequenceReplayBuffer, SequenceStep};
pub use rl4burn_collect::upgo::upgo as upgo_advantages;
pub use rl4burn_collect::vtrace::vtrace_targets;

// Action distributions
pub use rl4burn_nn::autoregressive::{ActionHead, CompositeDistribution};
pub use rl4burn_nn::dist::{ActionDist, LogStdMode};

// Neural network utilities
pub use rl4burn_nn::attention::{
    AttentionPool, AttentionPoolConfig, MultiHeadAttention, MultiHeadAttentionConfig, PointerNet,
    PointerNetConfig, TargetAttention, TargetAttentionConfig, TransformerBlock,
    TransformerBlockConfig, TransformerEncoder, TransformerEncoderConfig,
};
pub use rl4burn_nn::clip::clip_grad_norm;
pub use rl4burn_nn::film::{Film, FilmConfig};
pub use rl4burn_nn::init::orthogonal_linear;
pub use rl4burn_nn::policy::{greedy_action, DiscreteAcOutput, DiscreteActorCritic};
pub use rl4burn_nn::polyak::polyak_update;
pub use rl4burn_nn::rnn::{
    BlockGruCell, BlockGruCellConfig, GruCell, GruCellConfig, LstmCell, LstmCellConfig, LstmState,
};
pub use rl4burn_nn::conv::{ConvDecoder, ConvDecoderConfig, ConvEncoder, ConvEncoderConfig};
pub use rl4burn_nn::mlp::{Mlp, MlpConfig, NormKind, RmsNorm, RmsNormConfig};
pub use rl4burn_nn::multi_encoder::{MultiDecoder, MultiDecoderConfig, MultiEncoder, MultiEncoderConfig};
pub use rl4burn_nn::rssm::{Rssm, RssmConfig, RssmState};
pub use rl4burn_nn::symlog::{symexp, symlog, TwohotEncoder};
pub use rl4burn_nn::vae::{BetaVae, BetaVaeConfig, VaeOutput};

// Loss functions
pub use rl4burn_algo::loss::kl_balance::{
    categorical_kl, categorical_kl_groups, kl_balanced_loss, kl_balanced_loss_groups,
    KlBalanceConfig,
};
pub use rl4burn_algo::loss::policy::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use rl4burn_algo::loss::multi_head_value::{multi_head_gae, multi_head_value_loss, MultiHeadGaeResult, MultiHeadValueConfig};
pub use rl4burn_algo::loss::representation::{
    barlow_twins_loss, infonce_loss, dreamerpro_loss, decoder_loss, image_decoder_loss,
    sinkhorn, BarlowTwinsConfig, InfoNceConfig, DreamerProConfig, RepresentationVariant,
};

// Dreamer agent
pub use rl4burn_algo::dreamer::{
    DreamerAgent, DreamerConfig, DreamerStats, WorldModelStats, ActorCriticStats,
    dreamer_world_model_loss, dreamer_actor_critic_loss,
};

// Cloud deployment
#[cfg(feature = "cloud")]
pub mod cloud {
    pub use rl4burn_cloud::*;
}

// Rendering
pub use rl4burn_core::env::render::{Renderable, RgbFrame};

// Logging
pub use rl4burn_core::log::{CompositeLogger, Loggable, Logger, NoopLogger, PrintLogger};
#[cfg(feature = "tensorboard")]
pub use rl4burn_core::log::TensorBoardLogger;
#[cfg(feature = "json-log")]
pub use rl4burn_core::log::JsonLogger;
#[cfg(feature = "video")]
pub use rl4burn_core::log::write_gif;
