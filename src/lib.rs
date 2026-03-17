//! Reinforcement learning algorithms for the Burn ML framework.
//!
//! `rl4burn` provides generic, backend-agnostic RL building blocks that
//! exploit Burn's type system: write `PPO<B: AutodiffBackend>` once and
//! run on WGPU, CUDA, NdArray, or LibTorch.
//!
//! # Modules
//!
//! - [`env`] — Environment trait, spaces, vectorized environments, wrappers
//! - [`envs`] — Built-in environments (CartPole)
//! - [`algo`] — Algorithms (PPO, DQN)
//! - [`nn`] — Neural network utilities (init, gradient clipping, polyak, losses, policy traits)
//! - [`collect`] — Data collection (GAE, V-trace, UPGO, replay buffer, advantage normalization)

/// Environment abstractions: trait, spaces, vectorized envs, wrappers.
pub mod env;

/// Built-in environments (CartPole).
pub mod envs;

/// RL algorithms (PPO, DQN).
pub mod algo;

/// Neural network utilities for RL.
pub mod nn;

/// Data collection and advantage estimation.
pub mod collect;

/// Logging infrastructure for training metrics.
pub mod log;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// Environment
pub use env::adapter::DiscreteEnvAdapter;
pub use env::space::Space;
pub use env::vec_env::SyncVecEnv;
pub use env::wrapper;
pub use env::{Env, Step};

// Algorithms
pub use algo::dqn::{dqn_update, epsilon_greedy, epsilon_schedule, DqnConfig, DqnStats, QNetwork, Transition};
pub use algo::ppo::{ppo_collect, ppo_update, PpoConfig, PpoRollout, PpoStats};
pub use algo::ppo_masked::{
    masked_ppo_collect, masked_ppo_update, MaskedActorCritic, MaskedPpoRollout,
};
pub use algo::behavioral_cloning::{bc_loss_discrete, bc_loss_multi_head, bc_step};
pub use algo::distillation::{distillation_loss, value_distillation_loss, DistillationConfig};
pub use algo::cspl::{CsplConfig, CsplPhase, CsplPipeline};
pub use algo::league::{AgentRole, League, LeagueAgentConfig};
pub use algo::multi_agent::{batch_multi_agent_obs, broadcast_team_reward, unbatch_actions, MultiAgentRolloutData};
pub use algo::pfsp::{PfspConfig, PfspMatchmaking, PlayerRecord};
pub use algo::privileged_critic::{make_critic_input, PrivilegedActorCritic};
pub use algo::self_play::{branch_agent, SelfPlayPool};
pub use algo::z_conditioning::{z_reward, ZConditioning, ZConditioningConfig};
pub use algo::imagination::{imagine_rollout, lambda_returns, ImaginedTrajectory};
pub use algo::mcts::{MctsConfig, MctsTree};
pub use algo::distributed::{
    DistributedConfig, GradientSync, LocalSync, ReduceStrategy, scale_gradients,
};

// Action distributions
pub use nn::autoregressive::{ActionHead, CompositeDistribution};
pub use nn::dist::{ActionDist, LogStdMode};

// Neural network utilities
pub use nn::attention::{
    AttentionPool, AttentionPoolConfig, MultiHeadAttention, MultiHeadAttentionConfig, PointerNet,
    PointerNetConfig, TargetAttention, TargetAttentionConfig, TransformerBlock,
    TransformerBlockConfig, TransformerEncoder, TransformerEncoderConfig,
};
pub use nn::clip::clip_grad_norm;
pub use nn::film::{Film, FilmConfig};
pub use nn::init::orthogonal_linear;
pub use nn::kl_balance::{
    categorical_kl, categorical_kl_groups, kl_balanced_loss, kl_balanced_loss_groups,
    KlBalanceConfig,
};
pub use nn::loss::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use nn::rnn::{
    BlockGruCell, BlockGruCellConfig, GruCell, GruCellConfig, LstmCell, LstmCellConfig, LstmState,
};
pub use nn::rssm::{Rssm, RssmConfig, RssmState};
pub use nn::policy::{greedy_action, DiscreteAcOutput, DiscreteActorCritic};
pub use nn::symlog::{symexp, symlog, TwohotEncoder};
pub use nn::vae::{BetaVae, BetaVaeConfig, VaeOutput};

// Rendering
pub use env::render::{Renderable, RgbFrame};
pub use nn::multi_head_value::{multi_head_gae, multi_head_value_loss, MultiHeadGaeResult, MultiHeadValueConfig};
pub use nn::polyak::polyak_update;

// Data collection
pub use collect::advantage::normalize;
pub use collect::gae::gae;
pub use collect::intrinsic::{combine_rewards, CountBasedReward, EntropyReductionReward, IntrinsicReward};
pub use collect::percentile_normalize::PercentileNormalizer;
pub use collect::replay::ReplayBuffer;
pub use collect::sequence_replay::{SequenceReplayBuffer, SequenceStep};
pub use collect::upgo::upgo as upgo_advantages;
pub use collect::vtrace::vtrace_targets;

// Logging
pub use log::{CompositeLogger, Loggable, Logger, NoopLogger, PrintLogger};
#[cfg(feature = "tensorboard")]
pub use log::TensorBoardLogger;
#[cfg(feature = "json-log")]
pub use log::JsonLogger;
#[cfg(feature = "video")]
pub use log::write_gif;
