//! # Example 5 — PPO for Continuous Control
//!
//! Trains PPO with a **Gaussian (diagonal Normal) policy** on the Pendulum-v1
//! environment.  This is the continuous-action counterpart of the discrete
//! CartPole quickstart.
//!
//! ## Key concepts demonstrated
//!
//! - `ActionDist::Continuous` and `LogStdMode` for Gaussian policies
//! - `MaskedActorCritic` trait (also used for continuous, despite the name)
//! - `NormalizeObservation` and `NormalizeReward` wrappers
//! - Why continuous PPO differs from discrete PPO in several hyperparameters
//!
//! Run:
//! ```sh
//! cargo run -p ppo-continuous --release
//! ```

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::Pendulum;
use rl4burn::wrapper::{NormalizeObservation, NormalizeReward};
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, ActionDist, Loggable, LogStdMode, Logger,
    MaskedActorCritic, PpoConfig, SyncVecEnv,
};
use rl4burn::log::{CompositeLogger, PrintLogger};

// ---------------------------------------------------------------------------
// Model: Continuous Actor-Critic with Gaussian Policy
// ---------------------------------------------------------------------------
//
// ## Gaussian policy overview
//
// For continuous control the policy outputs the parameters of a Normal
// distribution for each action dimension:
//
//   pi(a | s) = N(mean(s), std(s))
//
// The actor network outputs **mean** and **log_std** (log of the standard
// deviation).  We use log_std rather than std directly because:
//   1. std must be positive — exp(log_std) guarantees this.
//   2. The unconstrained log_std is easier for gradient-based optimizers.
//
// ## LogStdMode::ModelOutput vs LogStdMode::Separate
//
// There are two common ways to parameterize log_std:
//
// - **ModelOutput** (used here): the network's final layer outputs
//   `[mean_1, ..., mean_d, log_std_1, ..., log_std_d]` — i.e. 2*action_dim
//   outputs.  log_std is **state-dependent**: different observations can
//   produce different exploration noise.  This is more expressive but can be
//   harder to train (the network must learn both when to explore and when to
//   exploit).
//
// - **Separate** (CleanRL default): log_std is a standalone learnable
//   parameter vector, independent of the observation.  The network only
//   outputs `mean`.  This is simpler and often sufficient for locomotion
//   tasks where a global exploration level works fine.
//
// Choose Separate for stability on simpler tasks; choose ModelOutput when
// you need state-dependent exploration (e.g., the agent should be cautious
// near cliffs but exploratory in safe regions).

#[derive(Module, Debug)]
struct ContinuousAC<B: Backend> {
    actor_fc1: Linear<B>,
    actor_fc2: Linear<B>,
    // Outputs [mean, log_std] for the 1-d action (2 outputs total).
    // With LogStdMode::ModelOutput, the framework splits this tensor
    // into means and log_stds automatically.
    actor_out: Linear<B>,

    critic_fc1: Linear<B>,
    critic_fc2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> ContinuousAC<B> {
    fn new(device: &B::Device) -> Self {
        // Pendulum obs = [cos(theta), sin(theta), angular_velocity] => 3 dims
        // Pendulum action = [torque] => 1 dim
        // actor_out produces 2 values: mean and log_std (for 1-d action)
        Self {
            actor_fc1: LinearConfig::new(3, 64).init(device),
            actor_fc2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 2).init(device), // mean + log_std
            critic_fc1: LinearConfig::new(3, 64).init(device),
            critic_fc2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

// The `MaskedActorCritic` trait is used for both discrete (with action masks)
// and continuous policies.  For continuous control the "logits" tensor holds
// the raw Gaussian parameters (means and optionally log_stds depending on
// LogStdMode).
//
// `forward` returns `(logits, values)`:
//   - logits: [batch, 2 * action_dim] when LogStdMode::ModelOutput
//             [batch, action_dim]     when LogStdMode::Separate
//   - values: [batch]
impl<B: Backend> MaskedActorCritic<B> for ContinuousAC<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        // Actor: obs -> hidden -> hidden -> [mean, log_std]
        let a = self.actor_fc1.forward(obs.clone()).tanh();
        let a = self.actor_fc2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        // Critic: obs -> hidden -> hidden -> scalar value
        let c = self.critic_fc1.forward(obs).tanh();
        let c = self.critic_fc2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        (logits, values)
    }
}

// ---------------------------------------------------------------------------
// Action sampling and log-probability (handled by the framework)
// ---------------------------------------------------------------------------
//
// Under the hood, when `ActionDist::Continuous` is used:
//
// 1. **Sampling**: the framework splits the logits into means and log_stds,
//    computes std = exp(log_std), then samples:
//      action_raw = mean + std * epsilon,   epsilon ~ N(0, 1)
//
// 2. **Action squashing**: the raw sample is passed through tanh to bound it
//    to [-1, 1], matching the Pendulum's action range.  The environment then
//    scales to its native range (e.g., [-2, 2] torque).
//
// 3. **Log-probability**: for the unsquashed Normal:
//      log p(a) = -0.5 * ((a - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
//    With tanh squashing, a correction term is applied:
//      log p_squashed(a) = log p(a_raw) - sum(log(1 - tanh(a_raw)^2))
//    This correction is critical for correct policy gradients when actions
//    are bounded.
//
// All of this is handled automatically by `masked_ppo_collect` and
// `masked_ppo_update` — you just need to specify `ActionDist::Continuous`.

// ---------------------------------------------------------------------------
// Main: training loop
// ---------------------------------------------------------------------------

type AutodiffB = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // --- Environment setup ---------------------------------------------------
    //
    // We use 4 parallel environments and wrap each with:
    //
    // - `NormalizeObservation`: keeps a running mean/std of observations and
    //   normalizes them to roughly zero-mean, unit-variance.  This is crucial
    //   for continuous control where observation scales can vary wildly
    //   (e.g., angles in [-pi, pi] vs angular velocity in [-8, 8]).
    //   The `clip` parameter (10.0) prevents extreme normalized values.
    //
    // - `NormalizeReward`: keeps a running estimate of the return's standard
    //   deviation and divides rewards by it.  This stabilizes value function
    //   training when reward magnitudes change during learning.  The `gamma`
    //   parameter (0.99) must match the PPO discount factor.

    let n_envs = 4;
    let envs: Vec<NormalizeReward<NormalizeObservation<Pendulum<rand::rngs::SmallRng>>>> =
        (0..n_envs)
            .map(|i| {
                let env = Pendulum::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64));
                let env = NormalizeObservation::new(env, 10.0).expect("Box obs space");
                NormalizeReward::new(env, 0.99, 10.0)
            })
            .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // --- Model and action distribution ---------------------------------------

    let mut model: ContinuousAC<AutodiffB> = ContinuousAC::new(&device);

    // Tell the framework this is a continuous 1-d action space with
    // model-output log_std.  The framework uses this to:
    //   - split logits into means / log_stds
    //   - sample from Normal(mean, exp(log_std))
    //   - compute log-probabilities and entropy
    let action_dist = ActionDist::Continuous {
        action_dim: 1,
        log_std_mode: LogStdMode::ModelOutput,
    };

    // --- Hyperparameters -----------------------------------------------------
    //
    // Several hyperparameters differ from discrete PPO:
    //
    // - `ent_coef = 0.0`: For discrete policies, an entropy bonus encourages
    //   exploration by preventing premature collapse to a deterministic policy.
    //   For continuous Gaussian policies, exploration is inherently driven by
    //   the learned std — the policy is *always* stochastic during training.
    //   Adding entropy bonus on top can destabilize training by fighting the
    //   optimizer's attempt to reduce std as the policy improves.  Most
    //   continuous PPO implementations (CleanRL, SB3) use ent_coef = 0.0.
    //
    // - `update_epochs = 10` (vs 4 for discrete): continuous control benefits
    //   from more gradient steps per rollout.  The Gaussian policy's smooth
    //   parameterization means it is less prone to catastrophic updates from
    //   extra epochs.  The KL divergence between old and new Gaussians changes
    //   gradually, so the clipping mechanism remains effective even with more
    //   epochs.  (CleanRL uses 10 for continuous, 4 for discrete.)
    //
    // - `n_steps = 256`: longer rollouts to capture the Pendulum's dynamics
    //   over multiple time steps (each episode is up to 200 steps).
    //
    // - `minibatch_size = 64`: smaller minibatches with more epochs helps
    //   continuous PPO extract more signal from each rollout.

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.0,       // no entropy bonus for continuous (see above)
        update_epochs: 10,    // more epochs than discrete (see above)
        minibatch_size: 64,
        n_steps: 256,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    // --- Training loop -------------------------------------------------------

    let total_timesteps = 1_000_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    let mut logger = CompositeLogger::new(vec![
        Box::new(PrintLogger::new(0)) as Box<dyn Logger>,
    ]);

    eprintln!(
        "Training continuous PPO on Pendulum-v1 ({total_timesteps} timesteps, {n_envs} envs)"
    );
    eprintln!("Action distribution: Gaussian with ModelOutput log_std");
    eprintln!("Wrappers: NormalizeObservation + NormalizeReward");
    eprintln!("{:-<80}", "");

    for iter in 0..n_iterations {
        // Linear learning rate annealing
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // Collect rollout: the framework handles action sampling from the
        // Gaussian, tanh squashing, and log-prob computation internally.
        let inference_model = model.valid();
        let rollout = masked_ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );
        recent_returns.extend_from_slice(&rollout.episode_returns);

        // PPO update: computes GAE advantages, then runs `update_epochs`
        // passes of clipped surrogate + value loss over minibatches.
        let stats;
        (model, stats) = masked_ppo_update(
            model,
            &mut optim,
            &rollout,
            &action_dist,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        // Keep a sliding window of recent episode returns for reporting
        if recent_returns.len() > 30 {
            let start = recent_returns.len() - 30;
            recent_returns = recent_returns[start..].to_vec();
        }

        let timestep = ((iter + 1) * steps_per_iter) as u64;

        if !recent_returns.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 =
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            logger.log_scalar("rollout/avg_return", avg as f64, timestep);
            stats.log(&mut logger, timestep);
        }
    }
    logger.flush();

    // --- Summary -------------------------------------------------------------
    if !recent_returns.is_empty() {
        let avg: f32 =
            recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        eprintln!("{:-<80}", "");
        eprintln!("Final average return (last 30 episodes): {avg:.1}");
        eprintln!();
        eprintln!("Pendulum-v1 returns range from ~-1600 (random) to ~-200 (good).");
        eprintln!("A well-trained agent should reach around -200 to -300.");
    }
}
