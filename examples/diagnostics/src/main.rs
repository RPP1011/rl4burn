//! Debugging and diagnostics for PPO training.
//!
//! Runs CartPole PPO with comprehensive metric tracking and interpretation.
//! After training, prints a diagnostic report that flags common issues.
//!
//! This is the example you want when your agent is not learning and you
//! need to figure out why.
//!
//! # Metrics tracked
//!
//! | Metric             | What it means                                      | Healthy range       |
//! |--------------------|----------------------------------------------------|---------------------|
//! | policy_loss        | Clipped surrogate objective (negated)               | Decreases, ~0       |
//! | value_loss         | MSE between value predictions and returns           | Decreases over time |
//! | entropy            | Policy randomness (H = -sum(p*log(p)))              | Slow decrease       |
//! | approx_kl          | How much the policy changed this update             | < 0.02 typical      |
//! | clip_fraction      | Fraction of ratios that were clipped                | 0.05-0.20           |
//! | explained_variance | 1 - Var(returns-values)/Var(returns)                | Approaches 1.0      |
//! | episode_return     | Total reward per episode (the primary metric)       | Increases           |

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, PpoConfig, PpoStats, SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Model (same as quickstart, CartPole actor-critic)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    actor1: Linear<B>,
    actor2: Linear<B>,
    actor_out: Linear<B>,
    critic1: Linear<B>,
    critic2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            actor1: LinearConfig::new(4, 64).init(device),
            actor2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 2).init(device),
            critic1: LinearConfig::new(4, 64).init(device),
            critic2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for ActorCritic<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        let a = self.actor1.forward(obs.clone()).tanh();
        let a = self.actor2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        let c = self.critic1.forward(obs).tanh();
        let c = self.critic2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        DiscreteAcOutput { logits, values }
    }
}

// ---------------------------------------------------------------------------
// Diagnostic metrics beyond PpoStats
// ---------------------------------------------------------------------------

/// Extended diagnostics computed per rollout.
#[allow(dead_code)]
struct DiagnosticSnapshot {
    /// Metrics from the PPO update.
    stats: PpoStats,
    /// Average episode return in this rollout.
    avg_return: f32,
    /// Number of completed episodes.
    num_episodes: usize,

    /// Explained variance: 1 - Var(returns - values) / Var(returns).
    ///
    /// Measures how well the value function predicts actual returns:
    ///   - 1.0 = perfect prediction (value function explains all variance)
    ///   - 0.0 = value function is no better than predicting the mean
    ///   - negative = value function is WORSE than predicting the mean
    ///
    /// If negative, your value function is actively hurting training.
    /// This often means the learning rate is too high or the network
    /// is too small to represent the value function.
    explained_variance: f32,

    /// Fraction of policy ratios that were clipped by PPO's epsilon.
    ///
    /// Measures how aggressively the policy is changing:
    ///   - 0.0 = no clipping (policy barely changed)
    ///   - 0.05-0.20 = healthy range
    ///   - > 0.30 = policy changing too fast, consider lowering LR
    ///
    /// Note: this is computed as fraction of |ratio - 1| > clip_eps.
    clip_fraction: f32,
}

/// Compute explained variance: 1 - Var(residual) / Var(returns).
fn explained_variance(values: &[f32], returns: &[f32]) -> f32 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f32;
    let ret_mean = returns.iter().sum::<f32>() / n;
    let ret_var = returns.iter().map(|r| (r - ret_mean).powi(2)).sum::<f32>() / n;

    if ret_var < 1e-8 {
        // Returns are constant; any prediction is equally good/bad.
        return 0.0;
    }

    let residuals: Vec<f32> = values.iter().zip(returns).map(|(v, r)| r - v).collect();
    let res_mean = residuals.iter().sum::<f32>() / n;
    let res_var = residuals.iter().map(|r| (r - res_mean).powi(2)).sum::<f32>() / n;

    1.0 - res_var / ret_var
}

/// Estimate clip fraction from rollout data by re-evaluating the policy.
///
/// Since we do not have the updated policy's log_probs during collection,
/// we approximate clip fraction from the PpoStats.approx_kl:
///   clip_frac ~ 2 * approx_kl / clip_eps^2
/// This is a rough proxy. In production, you would track actual clip
/// fraction inside the ppo_update function.
fn estimate_clip_fraction(approx_kl: f32, clip_eps: f32) -> f32 {
    // Heuristic: larger KL means more clipping occurred.
    // Exact computation would require tracking inside ppo_update.
    let estimate = (2.0 * approx_kl).sqrt() / clip_eps;
    estimate.min(1.0)
}

// ---------------------------------------------------------------------------
// Diagnostic history for trend analysis
// ---------------------------------------------------------------------------

struct DiagnosticHistory {
    entries: Vec<DiagnosticSnapshot>,
}

impl DiagnosticHistory {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn push(&mut self, snapshot: DiagnosticSnapshot) {
        self.entries.push(snapshot);
    }

    /// Print the final diagnostic report.
    ///
    /// This is the most important function in this example. It checks for
    /// common failure modes and provides actionable recommendations.
    fn print_report(&self) {
        eprintln!();
        eprintln!("{:=^80}", "");
        eprintln!("{:^80}", "DIAGNOSTIC REPORT");
        eprintln!("{:=^80}", "");

        if self.entries.is_empty() {
            eprintln!("No data collected.");
            return;
        }

        let n = self.entries.len();
        let last = &self.entries[n - 1];

        // Summary statistics
        eprintln!();
        eprintln!("--- Final Metrics (last update) ---");
        eprintln!("  policy_loss:        {:.4}", last.stats.policy_loss);
        eprintln!("  value_loss:         {:.4}", last.stats.value_loss);
        eprintln!("  entropy:            {:.4}", last.stats.entropy);
        eprintln!("  approx_kl:          {:.6}", last.stats.approx_kl);
        eprintln!("  clip_fraction:      {:.4}", last.clip_fraction);
        eprintln!("  explained_variance: {:.4}", last.explained_variance);
        eprintln!("  avg_return:         {:.1}", last.avg_return);

        // Trend analysis: compare first quarter to last quarter
        let q1_end = n / 4;
        let q4_start = 3 * n / 4;

        let avg_metric = |f: fn(&DiagnosticSnapshot) -> f32, start: usize, end: usize| -> f32 {
            let slice = &self.entries[start..end];
            if slice.is_empty() {
                return 0.0;
            }
            slice.iter().map(|s| f(s)).sum::<f32>() / slice.len() as f32
        };

        let early_return = avg_metric(|s| s.avg_return, 0, q1_end.max(1));
        let late_return = avg_metric(|s| s.avg_return, q4_start, n);
        let early_entropy = avg_metric(|s| s.stats.entropy, 0, q1_end.max(1));
        let late_entropy = avg_metric(|s| s.stats.entropy, q4_start, n);
        let early_vloss = avg_metric(|s| s.stats.value_loss, 0, q1_end.max(1));
        let late_vloss = avg_metric(|s| s.stats.value_loss, q4_start, n);

        eprintln!();
        eprintln!("--- Trends (Q1 vs Q4 averages) ---");
        eprintln!(
            "  avg_return:    {:.1} -> {:.1} ({})",
            early_return,
            late_return,
            if late_return > early_return { "improving" } else { "NOT improving" }
        );
        eprintln!(
            "  entropy:       {:.3} -> {:.3} ({})",
            early_entropy,
            late_entropy,
            if late_entropy < early_entropy { "decreasing (normal)" } else { "increasing (unusual)" }
        );
        eprintln!(
            "  value_loss:    {:.4} -> {:.4} ({})",
            early_vloss,
            late_vloss,
            if late_vloss < early_vloss { "decreasing (good)" } else { "increasing (concerning)" }
        );

        // Health checks
        eprintln!();
        eprintln!("--- Health Checks ---");
        let mut issues = 0;

        // Check 1: Entropy collapse
        //
        // Entropy measures policy randomness. For CartPole with 2 actions,
        // maximum entropy is ln(2) = 0.693. If entropy collapses to near 0,
        // the policy has become deterministic too early and stopped exploring.
        //
        // Fix: increase ent_coef (e.g., 0.01 -> 0.05) to penalize
        // deterministic policies.
        if last.stats.entropy < 0.1 {
            eprintln!("  [WARNING] Entropy collapse detected ({:.4} < 0.1)", last.stats.entropy);
            eprintln!("    -> Policy is too deterministic. Increase ent_coef.");
            eprintln!("    -> For 2 actions, max entropy = ln(2) = 0.693");
            issues += 1;
        } else {
            eprintln!("  [OK] Entropy: {:.4} (healthy, not collapsed)", last.stats.entropy);
        }

        // Check 2: KL divergence too high
        //
        // approx_kl measures how much the policy changed in one update.
        // Large KL means the policy is changing too aggressively, which
        // can cause training instability (policy oscillation).
        //
        // Fix: lower learning rate, or enable target_kl early stopping
        // (e.g., target_kl = Some(0.015)).
        if last.stats.approx_kl > 0.05 {
            eprintln!(
                "  [WARNING] KL too high ({:.4} > 0.05)",
                last.stats.approx_kl
            );
            eprintln!("    -> Policy changing too aggressively. Lower learning rate.");
            eprintln!("    -> Or set target_kl = Some(0.015) for early stopping.");
            issues += 1;
        } else {
            eprintln!(
                "  [OK] approx_kl: {:.6} (within healthy range < 0.05)",
                last.stats.approx_kl
            );
        }

        // Check 3: Explained variance negative
        //
        // Negative explained variance means the value function's predictions
        // are worse than just predicting the mean return. The value function
        // is actively providing bad baselines for advantage estimation.
        //
        // Fix: increase network capacity, lower learning rate, or check
        // that rewards are not changing scale during training.
        if last.explained_variance < 0.0 {
            eprintln!(
                "  [WARNING] Explained variance negative ({:.4})",
                last.explained_variance
            );
            eprintln!("    -> Value function worse than constant prediction.");
            eprintln!("    -> Increase critic network capacity or lower learning rate.");
            issues += 1;
        } else {
            eprintln!(
                "  [OK] Explained variance: {:.4} (positive, value function is useful)",
                last.explained_variance
            );
        }

        // Check 4: Value loss increasing
        //
        // Value loss should generally decrease as the value function improves.
        // If it increases over training, the critic may be overfitting to
        // recent data or the reward scale may be changing.
        //
        // Fix: reduce vf_coef, add value function clipping (clip_vloss=true),
        // or normalize rewards.
        if late_vloss > early_vloss * 1.5 && n > 10 {
            eprintln!(
                "  [WARNING] Value loss increasing ({:.4} -> {:.4})",
                early_vloss, late_vloss
            );
            eprintln!("    -> Critic may be overfitting or reward scale is changing.");
            eprintln!("    -> Try clip_vloss=true, lower vf_coef, or normalize rewards.");
            issues += 1;
        } else {
            eprintln!("  [OK] Value loss trend: stable or decreasing");
        }

        // Check 5: No improvement in returns
        if late_return <= early_return && n > 10 {
            eprintln!(
                "  [WARNING] Returns not improving ({:.1} -> {:.1})",
                early_return, late_return
            );
            eprintln!("    -> Agent may not be learning. Check all other diagnostics.");
            eprintln!("    -> Try different hyperparameters (see hyperparameter-tuning example).");
            issues += 1;
        } else {
            eprintln!("  [OK] Returns improving: {:.1} -> {:.1}", early_return, late_return);
        }

        // Check 6: Clip fraction
        if last.clip_fraction > 0.3 {
            eprintln!(
                "  [WARNING] High clip fraction ({:.2})",
                last.clip_fraction
            );
            eprintln!("    -> Too many ratios being clipped. Policy changing too fast.");
            eprintln!("    -> Lower learning rate or increase clip_eps.");
            issues += 1;
        } else {
            eprintln!("  [OK] Clip fraction: {:.4} (reasonable)", last.clip_fraction);
        }

        eprintln!();
        if issues == 0 {
            eprintln!("All health checks passed. Training appears healthy.");
        } else {
            eprintln!(
                "{} issue(s) detected. See recommendations above.",
                issues
            );
        }

        // Reference: known-good metrics for CartPole
        eprintln!();
        eprintln!("--- Reference: Expected CartPole Metrics at Convergence ---");
        eprintln!("  avg_return:         ~475-500 (max 500)");
        eprintln!("  policy_loss:        ~-0.01 to 0.0");
        eprintln!("  value_loss:         ~1-10 (depends on return scale)");
        eprintln!("  entropy:            ~0.3-0.5 (some stochasticity is fine)");
        eprintln!("  approx_kl:          ~0.005-0.02");
        eprintln!("  explained_variance: ~0.8-1.0");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

type B = Autodiff<NdArray>;

fn main() {
    eprintln!("=== PPO Diagnostics and Debugging ===");
    eprintln!();

    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: ActorCritic<B> = ActorCritic::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 2.5e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 128,
        n_steps: 128,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    let total_timesteps = 100_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut diagnostics = DiagnosticHistory::new();

    eprintln!("Training PPO on CartPole ({total_timesteps} steps, {n_envs} envs)");
    eprintln!(
        "Logging every iteration with comprehensive diagnostics."
    );
    eprintln!();
    eprintln!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Step", "Return", "Pi Loss", "V Loss", "Entropy", "KL", "ClipFrac", "ExplVar"
    );
    eprintln!("{:-<88}", "");

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        let rollout = ppo_collect::<NdArray, _, _>(
            &model.valid(),
            &mut vec_env,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        // Compute explained variance BEFORE the update, using the rollout's
        // value predictions vs the GAE-computed returns.
        //
        // explained_variance = 1 - Var(returns - values) / Var(returns)
        //
        // This tells us how well the current value function predicts the
        // actual discounted returns. It should approach 1.0 as training
        // progresses.
        let ev = explained_variance(&rollout.values, &rollout.returns);

        recent_returns.extend_from_slice(&rollout.episode_returns);
        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        let stats;
        (model, stats) = ppo_update(
            model,
            &mut optim,
            &rollout,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        let avg_return = if recent_returns.is_empty() {
            0.0
        } else {
            recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
        };

        // Clip fraction: estimated from approx_kl.
        // In a production system you would track this inside ppo_update.
        let clip_frac = estimate_clip_fraction(stats.approx_kl, config.clip_eps);

        let snapshot = DiagnosticSnapshot {
            stats: stats.clone(),
            avg_return,
            num_episodes: recent_returns.len(),
            explained_variance: ev,
            clip_fraction: clip_frac,
        };

        // Print a detailed row every 5 iterations
        if (iter + 1) % 5 == 0 {
            eprintln!(
                "{:<8} {:>10.1} {:>10.4} {:>10.4} {:>10.4} {:>10.6} {:>10.4} {:>10.4}",
                (iter + 1) * steps_per_iter,
                snapshot.avg_return,
                snapshot.stats.policy_loss,
                snapshot.stats.value_loss,
                snapshot.stats.entropy,
                snapshot.stats.approx_kl,
                snapshot.clip_fraction,
                snapshot.explained_variance,
            );
        }

        diagnostics.push(snapshot);
    }

    // Print the final diagnostic report with health checks
    diagnostics.print_report();
}
