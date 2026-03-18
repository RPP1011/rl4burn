//! # PPO for Discrete Actions — Fully Annotated
//!
//! A self-contained Proximal Policy Optimization (Schulman et al., 2017) agent
//! that solves CartPole-v1. Every implementation detail is annotated with inline
//! comments following CleanRL's practice of documenting each "implementation
//! matter" (Huang et al., 2022).
//!
//! Run: `cargo run -p ppo-annotated --release`
//!
//! References:
//!   - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
//!   - Schulman et al., "High-Dimensional Continuous Control Using GAE" (2015)
//!   - Engstrom et al., "Implementation Matters in Deep RL" (2020)
//!   - Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning?" (2021)
//!   - Huang et al., "The 37 Implementation Details of PPO" (2022)

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    orthogonal_linear, ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, PpoConfig,
    SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ===========================================================================
// Model: Separate Actor-Critic with Orthogonal Initialization
// ===========================================================================
//
// IMPLEMENTATION DETAIL: Separate actor and critic networks.
//
// CleanRL (and the original PPO paper's Atari experiments) use *separate*
// networks for the actor and critic rather than a shared backbone. Why?
//
//   1. Shared backbones create a tug-of-war: the policy loss and value loss
//      push the shared features in different directions, which can destabilize
//      training. Separate networks avoid this interference entirely.
//
//   2. For simple environments like CartPole with small observation spaces,
//      the parameter overhead of separate networks is negligible.
//
//   3. Andrychowicz et al. (2021) found that shared networks require careful
//      tuning of the value loss coefficient to balance the two objectives,
//      while separate networks are more robust.
//
// In practice, shared backbones are mainly used when feature extraction is
// expensive (e.g., CNN encoders for pixel observations) and even then, the
// policy and value heads typically have separate final layers.

#[derive(Module, Debug)]
struct Agent<B: Backend> {
    // Actor network: obs -> 64 (tanh) -> 64 (tanh) -> n_actions (logits)
    actor_fc1: burn::nn::Linear<B>,
    actor_fc2: burn::nn::Linear<B>,
    actor_out: burn::nn::Linear<B>,
    // Critic network: obs -> 64 (tanh) -> 64 (tanh) -> 1 (scalar value)
    critic_fc1: burn::nn::Linear<B>,
    critic_fc2: burn::nn::Linear<B>,
    critic_out: burn::nn::Linear<B>,
}

impl<B: Backend> Agent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        // IMPLEMENTATION DETAIL: Orthogonal initialization with specific gains.
        //
        // Standard Xavier/Kaiming init assumes the activation is ReLU or linear.
        // Orthogonal initialization (Saxe et al., 2013) creates weight matrices
        // whose singular values are all equal to `gain`, preserving gradient
        // norms across layers and enabling better signal propagation in deep nets.
        //
        // The gain values match CleanRL's `layer_init()`:
        //
        //   - Hidden layers: gain = sqrt(2) ≈ 1.414
        //     This compensates for the variance reduction caused by tanh
        //     activations. tanh squashes inputs to [-1, 1], roughly halving
        //     the variance of the output. sqrt(2) counteracts this so that
        //     the signal magnitude is preserved layer-to-layer.
        //
        //   - Actor output: gain = 0.01
        //     A tiny gain makes the initial logits near-zero, so the initial
        //     policy is close to uniform over actions. This ensures the agent
        //     explores broadly at the start of training rather than committing
        //     to arbitrary actions based on random initialization. This is a
        //     critical detail — large initial logits can cause the policy to
        //     be overconfident before any learning has occurred.
        //
        //   - Critic output: gain = 1.0
        //     The value function has no activation after it, so we use the
        //     default orthogonal gain. Starting with moderate value predictions
        //     (not too large, not too small) helps the value loss converge
        //     smoothly. There is no reason to shrink the output like the actor.
        let sqrt2 = std::f32::consts::SQRT_2;

        Self {
            actor_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            actor_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            actor_out: orthogonal_linear(64, 2, 0.01, device, rng),

            critic_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            critic_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            critic_out: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for Agent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        // IMPLEMENTATION DETAIL: Tanh activations (not ReLU).
        //
        // CleanRL's PPO uses tanh throughout the MLP, matching the original
        // OpenAI baselines. Why tanh instead of ReLU?
        //
        //   1. Tanh is bounded in [-1, 1], which naturally limits the magnitude
        //      of hidden activations. This acts as a soft form of normalization,
        //      preventing activations from growing unboundedly.
        //
        //   2. Tanh is zero-centered (unlike ReLU), so the mean activation is
        //      near zero. This avoids the "bias shift" problem where all
        //      activations are positive, making subsequent layers' gradients
        //      all have the same sign.
        //
        //   3. For the small MLPs used in classic control tasks, the "dying
        //      ReLU" problem (where neurons permanently output zero) can be
        //      significant. Tanh has non-zero gradients everywhere.
        //
        //   4. Empirically, Andrychowicz et al. (2021) found that tanh works
        //      at least as well as ReLU for on-policy RL across many tasks.

        // Actor: produces unnormalized logits for each action
        let a = self.actor_fc1.forward(obs.clone()).tanh();
        let a = self.actor_fc2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        // Critic: produces a single scalar V(s) estimate
        let c = self.critic_fc1.forward(obs).tanh();
        let c = self.critic_fc2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        DiscreteAcOutput { logits, values }
    }
}

// ===========================================================================
// Main: Training Loop
// ===========================================================================

fn main() {
    let device = NdArrayDevice::Cpu;

    // Seed for reproducibility. CleanRL defaults to seed=1.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    // -----------------------------------------------------------------------
    // Vectorized environments
    // -----------------------------------------------------------------------
    // IMPLEMENTATION DETAIL: Multiple parallel environments (n_envs=4).
    //
    // Running multiple environments in parallel serves two purposes:
    //   1. It increases the effective batch size per rollout (n_steps * n_envs
    //      = 128 * 4 = 512 transitions), providing more diverse data for each
    //      optimization step.
    //   2. It decorrelates the training data. A single environment produces
    //      highly autocorrelated transitions; with multiple environments, each
    //      minibatch mixes transitions from independent episodes.
    //
    // SyncVecEnv steps all environments synchronously. When an environment
    // terminates, it auto-resets and returns the first observation of the new
    // episode (matching Gymnasium's behavior).
    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // -----------------------------------------------------------------------
    // Model and optimizer
    // -----------------------------------------------------------------------
    let mut model: Agent<AutodiffB> = Agent::new(&device, &mut rng);

    // IMPLEMENTATION DETAIL: Adam with eps=1e-5 (not the default 1e-8).
    //
    // CleanRL uses eps=1e-5 for the Adam optimizer. A larger epsilon adds more
    // numerical stability to the denominator of Adam's update rule:
    //   θ_t = θ_{t-1} - lr * m_t / (sqrt(v_t) + eps)
    //
    // With eps=1e-8 (PyTorch default), very small v_t values can cause
    // enormous parameter updates. eps=1e-5 dampens these extreme updates,
    // which matters in RL where the loss landscape is non-stationary.
    //
    // NOTE: We do NOT configure per-parameter gradient clipping on the
    // optimizer. Instead, rl4burn's ppo_update applies *global* gradient norm
    // clipping (see max_grad_norm below), matching PyTorch's clip_grad_norm_.
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // -----------------------------------------------------------------------
    // PPO hyperparameters (CleanRL defaults)
    // -----------------------------------------------------------------------
    let config = PpoConfig {
        // Base learning rate for Adam. Will be linearly annealed to 0.
        lr: 2.5e-4,

        // Discount factor gamma=0.99: future rewards are worth 99% of
        // immediate rewards per timestep. For CartPole (max 500 steps),
        // this means rewards ~100 steps out are still worth 0.99^100 ≈ 0.37
        // of their face value — the agent cares about long-term survival.
        gamma: 0.99,

        // IMPLEMENTATION DETAIL: GAE lambda=0.95.
        //
        // Generalized Advantage Estimation (Schulman et al., 2015) computes
        // advantages as an exponentially-weighted sum of TD residuals:
        //
        //   A_t^GAE = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        //   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        //
        // Lambda controls the bias-variance tradeoff:
        //   - lambda=0 gives TD(0): A_t = delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        //     Low variance (one-step bootstrap) but high bias (relies entirely
        //     on the critic's accuracy, which is poor early in training).
        //   - lambda=1 gives Monte Carlo returns: A_t = sum of discounted rewards - V(s_t)
        //     Zero bias but high variance (the full trajectory is noisy).
        //   - lambda=0.95 is a sweet spot: it looks ~20 steps ahead
        //     (1/(1-0.95) = 20) before the exponential weighting decays to
        //     near-zero, balancing the critic's bootstrap with actual returns.
        //
        // The GAE computation itself walks backward through the rollout:
        //   last_gae = 0
        //   for t = T-1, ..., 0:
        //       delta = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        //       last_gae = delta + gamma * lambda * (1 - done_t) * last_gae
        //       advantages[t] = last_gae
        //   returns[t] = advantages[t] + values[t]
        //
        // The (1 - done_t) terms zero out bootstrapping across episode
        // boundaries — when done=true, we do not bootstrap from the next
        // state's value because the episode has ended.
        gae_lambda: 0.95,

        // IMPLEMENTATION DETAIL: PPO clip epsilon=0.2.
        //
        // The clipped surrogate objective is the heart of PPO:
        //
        //   L^CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
        //
        // where r_t = pi_new(a|s) / pi_old(a|s) is the probability ratio.
        //
        // Why min of clipped and unclipped?
        //   - When A_t > 0 (good action): we want to increase pi(a|s), so
        //     r_t grows. But the clip caps the benefit at (1+eps)*A_t, so
        //     there is no incentive to increase the ratio beyond 1.2x.
        //   - When A_t < 0 (bad action): we want to decrease pi(a|s), so
        //     r_t shrinks. The clip caps the benefit at (1-eps)*A_t, so
        //     there is no incentive to decrease the ratio below 0.8x.
        //   - The min ensures we take the *pessimistic* bound: the objective
        //     only gets credit for changes within the trust region [0.8, 1.2].
        //
        // This prevents catastrophically large policy updates that plagued
        // vanilla policy gradient and even TRPO (which required expensive
        // conjugate gradient solves to enforce the trust region).
        clip_eps: 0.2,

        // IMPLEMENTATION DETAIL: Value loss coefficient = 0.5.
        //
        // The total loss is: L = L_policy + vf_coef * L_value - ent_coef * H
        //
        // vf_coef=0.5 downweights the value loss relative to the policy loss.
        // This prevents the value function gradient from dominating the shared
        // parameter updates (even with separate networks, the total loss is
        // still a single scalar that drives the backward pass).
        vf_coef: 0.5,

        // IMPLEMENTATION DETAIL: Entropy bonus coefficient = 0.01.
        //
        // The entropy H = -sum(pi(a|s) * log pi(a|s)) measures how "spread
        // out" the policy distribution is. Adding ent_coef * H to the objective
        // (subtracting it from the loss) encourages exploration:
        //
        //   - High entropy = uniform policy = maximum exploration
        //   - Low entropy = deterministic policy = pure exploitation
        //
        // 0.01 is a gentle nudge toward exploration. Too large (e.g., 0.1)
        // and the agent never commits to a good action; too small and the
        // policy collapses early to a suboptimal deterministic strategy.
        //
        // For CartPole with 2 actions, max entropy = ln(2) ≈ 0.693.
        // A healthy training run starts near 0.69 and gradually decreases
        // to ~0.2-0.4 as the agent learns which action is correct.
        ent_coef: 0.01,

        // IMPLEMENTATION DETAIL: 4 optimization epochs per rollout.
        //
        // After collecting 512 transitions, we train on them for 4 full
        // passes. More epochs extract more learning from each rollout (sample
        // efficiency), but too many epochs cause the policy to drift far from
        // the collection policy, violating the trust region assumption and
        // destabilizing training. 4 is the standard CleanRL default.
        update_epochs: 4,

        // Minibatch size = batch_size / num_minibatches = 512 / 4 = 128.
        // Each epoch splits the 512 transitions into 4 random minibatches.
        minibatch_size: 128,

        // Number of steps each environment runs per rollout.
        // Total transitions per rollout = n_steps * n_envs = 128 * 4 = 512.
        n_steps: 128,

        // IMPLEMENTATION DETAIL: Value loss clipping (Engstrom et al., 2020).
        //
        // When clip_vloss=true, the value loss becomes:
        //   V_clipped = V_old + clip(V_new - V_old, -eps, +eps)
        //   L_value = 0.5 * max((V_new - V_target)^2, (V_clipped - V_target)^2)
        //
        // This mirrors the policy clipping: it prevents the value function
        // from changing too much in a single update. If V_new moves far from
        // V_old, the clipped version is used instead, limiting the update.
        //
        // Engstrom et al. (2020) found this is one of the "implementation
        // matters" that contributes to PPO's empirical performance, though
        // its effect is environment-dependent. max() (not min) is used because
        // we are clipping a *loss* — we take the pessimistic (larger) loss.
        clip_vloss: true,

        // IMPLEMENTATION DETAIL: Global gradient norm clipping = 0.5.
        //
        // After computing gradients for the total loss, we rescale all
        // parameter gradients so their combined L2 norm does not exceed 0.5:
        //
        //   total_norm = sqrt(sum(||g_i||^2 for all parameter tensors g_i))
        //   if total_norm > max_grad_norm:
        //       g_i *= max_grad_norm / total_norm   (for all i)
        //
        // This is *global* clipping (PyTorch's clip_grad_norm_), NOT
        // per-parameter clipping. Global clipping preserves the relative
        // magnitude of gradients across parameters — if one layer has 10x
        // larger gradients than another, that ratio is maintained after
        // clipping. Per-parameter clipping would destroy this relationship.
        //
        // Gradient clipping prevents catastrophic updates caused by rare
        // high-reward trajectories or numerical instability, which can cause
        // gradient norms to spike by 100x or more.
        max_grad_norm: 0.5,

        // IMPLEMENTATION DETAIL: Target KL early stopping (optional).
        //
        // When Some(threshold), PPO monitors the approximate KL divergence
        // between the old and new policies:
        //
        //   approx_kl = 0.5 * mean((log_pi_new - log_pi_old)^2)
        //
        // If approx_kl exceeds the threshold, the remaining update epochs
        // are skipped. This provides an adaptive trust region: when the policy
        // has already changed enough, stop optimizing to prevent overshooting.
        //
        // CleanRL defaults to None (no early stopping). Setting it to
        // Some(0.01-0.02) can improve stability in harder environments at
        // the cost of some sample efficiency (fewer gradient steps per rollout).
        //
        // When to use target_kl:
        //   - Training is unstable (oscillating returns, KL spikes)
        //   - Environment has sparse or deceptive rewards
        //   - You want to trade sample efficiency for stability
        target_kl: None,

        // Dual clipping is disabled (standard PPO behavior).
        dual_clip_coef: None,
    };

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    let total_timesteps = 500_000;
    let steps_per_iter = config.n_steps * n_envs; // 128 * 4 = 512
    let n_iterations = total_timesteps / steps_per_iter;

    // Track recent episode returns to detect convergence.
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = 0.0f32;

    // `current_obs` persists across rollouts so we never lose the environment
    // state between collection phases. Initialize with the first observations.
    let mut current_obs = vec_env.reset();

    // `ep_acc` tracks per-environment cumulative reward across rollout boundaries.
    // An episode that starts in rollout N and ends in rollout N+1 will have its
    // full return correctly computed because we accumulate rewards here.
    let mut ep_acc = vec![0.0f32; n_envs];

    println!("PPO CartPole — {} total timesteps, {} iterations", total_timesteps, n_iterations);
    println!("Rollout: {} steps x {} envs = {} transitions/iter", config.n_steps, n_envs, steps_per_iter);
    println!("Update: {} epochs x {} minibatches of {}", config.update_epochs, steps_per_iter / config.minibatch_size, config.minibatch_size);
    println!();

    for iter in 0..n_iterations {
        // -------------------------------------------------------------------
        // IMPLEMENTATION DETAIL: Linear learning rate annealing.
        //
        // The learning rate decays linearly from lr to 0 over all iterations:
        //   current_lr = lr * (1 - iter / total_iters)
        //
        // Why anneal the LR?
        //   - Early in training, large LRs help escape bad initial policies.
        //   - Late in training, small LRs allow fine-tuning around the optimum
        //     without overshooting.
        //   - Without annealing, PPO often oscillates in later stages as it
        //     keeps making large updates even when the policy is nearly optimal.
        //
        // Linear annealing is the simplest schedule and works surprisingly
        // well. CleanRL uses it by default. More sophisticated schedules
        // (cosine, warmup-then-decay) rarely improve results for PPO.
        // -------------------------------------------------------------------
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // -------------------------------------------------------------------
        // Rollout collection phase
        // -------------------------------------------------------------------
        // ppo_collect runs the current policy in all environments for n_steps,
        // collecting (obs, action, log_prob, value, reward, done) at each step.
        //
        // Internally it:
        //   1. Builds an observation tensor [n_envs, obs_dim] from current_obs
        //   2. Calls model.forward() to get logits and values
        //   3. Converts logits to probabilities via softmax
        //   4. Samples actions from the categorical distribution
        //   5. Records log_prob = log_softmax(logits)[action] for each sample
        //   6. Steps all environments with the sampled actions
        //   7. Tracks episode returns across rollout boundaries using ep_acc
        //   8. After all steps, bootstraps V(s_T) for the final observations
        //   9. Computes GAE advantages and returns for each environment's
        //      trajectory slice
        //
        // No computation graph is retained — this is pure inference.
        // We use model.valid() to get the inference-mode model (drops autodiff).
        let inference_model = model.valid();
        let rollout = ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        // Collect completed episode returns for logging.
        recent_returns.extend_from_slice(&rollout.episode_returns);

        // -------------------------------------------------------------------
        // PPO update phase
        // -------------------------------------------------------------------
        // ppo_update performs multiple epochs of minibatch gradient descent on
        // the collected rollout data. Internally it:
        //
        //   1. Creates an index array [0..total_steps] and shuffles it each
        //      epoch (Fisher-Yates) to form random minibatches.
        //
        //   2. For each minibatch:
        //      a. Gathers the batch's obs, actions, old_log_probs, advantages,
        //         returns, and old_values from the rollout.
        //
        //      b. PER-MINIBATCH ADVANTAGE NORMALIZATION:
        //         Normalizes advantages to zero mean and unit variance *within
        //         the minibatch*, not globally. Why per-minibatch?
        //           - Global normalization can leave individual minibatches
        //             with skewed advantage distributions (e.g., all positive),
        //             causing the policy gradient to push in one direction.
        //           - Per-minibatch normalization ensures each gradient step
        //             sees a balanced distribution of "good" and "bad" actions.
        //           - Uses sample std (N-1 denominator) matching CleanRL.
        //
        //      c. Runs the model forward on the batch observations to get
        //         new logits and values.
        //
        //      d. Computes new_log_probs = log_softmax(logits)[action] and
        //         the probability ratio r = exp(new_log_prob - old_log_prob).
        //
        //      e. CLIPPED SURROGATE OBJECTIVE:
        //         surr1 = ratio * advantage
        //         surr2 = clip(ratio, 1-eps, 1+eps) * advantage
        //         policy_loss = -mean(min(surr1, surr2))
        //         The negative sign is because we minimize the loss (gradient
        //         descent), but want to maximize the objective.
        //
        //      f. ENTROPY BONUS:
        //         entropy = -mean(sum(pi * log_pi, dim=actions))
        //
        //      g. VALUE LOSS (optionally clipped):
        //         If clip_vloss: uses max of unclipped and clipped MSE
        //         Otherwise: simple MSE = 0.5 * mean((V_new - returns)^2)
        //
        //      h. TOTAL LOSS = policy_loss + vf_coef * value_loss - ent_coef * entropy
        //
        //      i. Backward pass, then GLOBAL gradient norm clipping to
        //         max_grad_norm, then optimizer step with current_lr.
        //
        //   3. After each epoch, checks approximate KL divergence. If
        //      target_kl is set and KL exceeds it, remaining epochs are skipped.
        //
        // Returns the updated model and training statistics.
        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, current_lr, &device, &mut rng);

        // -------------------------------------------------------------------
        // Logging
        // -------------------------------------------------------------------
        // Keep only the last 20 episode returns for a rolling average.
        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 25 == 0 {
                println!(
                    "iter {:>4}/{}: lr={:.2e}  avg_return={:>6.1} (best={:>6.1})  \
                     ploss={:>8.4}  vloss={:>8.1}  entropy={:.3}  approx_kl={:.4}",
                    iter + 1,
                    n_iterations,
                    current_lr,
                    avg,
                    best_avg,
                    stats.policy_loss,
                    stats.value_loss,
                    stats.entropy,
                    stats.approx_kl,
                );
            }

            // CartPole-v1 is "solved" when average return exceeds 450 over
            // recent episodes. The maximum episode length is 500 steps.
            if best_avg > 450.0 {
                println!(
                    "Solved at iteration {} ({} timesteps)! Best avg return: {:.1}",
                    iter + 1,
                    (iter + 1) * steps_per_iter,
                    best_avg,
                );
                break;
            }
        }
    }

    if best_avg <= 450.0 {
        println!(
            "Training complete. Best avg return: {:.1} (did not reach 450 threshold)",
            best_avg,
        );
    }
}
