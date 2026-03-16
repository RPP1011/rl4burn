//! V-trace off-policy correction (Espeholt et al., 2018).
//!
//! Pure f32 computation — no tensors, no backend dependency.
//! Operates on a single trajectory.

use contracts::*;

/// V-trace off-policy correction for a single trajectory.
///
/// # Arguments
/// - `log_rhos`: log importance ratios log(π_new/π_old), one per step
/// - `discounts`: per-step discount factor γ (usually constant, but can vary for terminal steps)
/// - `rewards`: per-step rewards
/// - `values`: V(s_t) from critic, one per step
/// - `bootstrap`: V(s_T) for non-terminal episodes, 0 for terminal
/// - `clip_rho`: clipping threshold for importance weights (typically 1.0)
/// - `clip_c`: clipping threshold for trace accumulation (typically 1.0)
///
/// # Returns
/// `(value_targets, advantages)` — both same length as inputs.
#[requires(!log_rhos.is_empty(), "trajectory must have at least one step")]
#[requires(log_rhos.len() == discounts.len(), "all inputs must have same length")]
#[requires(log_rhos.len() == rewards.len(), "all inputs must have same length")]
#[requires(log_rhos.len() == values.len(), "all inputs must have same length")]
#[requires(clip_rho > 0.0, "clip_rho must be positive")]
#[requires(clip_c > 0.0, "clip_c must be positive")]
#[requires(bootstrap.is_finite(), "bootstrap must be finite")]
#[ensures(ret.0.len() == log_rhos.len(), "value targets length matches input")]
#[ensures(ret.1.len() == log_rhos.len(), "advantages length matches input")]
#[ensures(ret.0.iter().all(|v| v.is_finite()), "value targets must be finite")]
#[ensures(ret.1.iter().all(|a| a.is_finite()), "advantages must be finite")]
pub fn vtrace_targets(
    log_rhos: &[f32],
    discounts: &[f32],
    rewards: &[f32],
    values: &[f32],
    bootstrap: f32,
    clip_rho: f32,
    clip_c: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = log_rhos.len();

    let rhos: Vec<f32> = log_rhos.iter()
        .map(|&lr| lr.clamp(-20.0, 20.0).exp())
        .collect();
    let rho_bar: Vec<f32> = rhos.iter().map(|&r| r.min(clip_rho)).collect();
    let c_bar: Vec<f32> = rhos.iter().map(|&r| r.min(clip_c)).collect();

    // V(s_{t+1})
    let next_values: Vec<f32> = values[1..].iter().copied()
        .chain(std::iter::once(bootstrap))
        .collect();

    // TD errors: δ_t = ρ̄_t * (r_t + γ_t * V(s_{t+1}) - V(s_t))
    let deltas: Vec<f32> = (0..len)
        .map(|t| rho_bar[t] * (rewards[t] + discounts[t] * next_values[t] - values[t]))
        .collect();

    // Backward accumulation: vs - V = Σ (γc)^k * δ_{t+k}
    let mut vs_minus_v = vec![0.0f32; len];
    let mut acc = 0.0f32;
    for t in (0..len).rev() {
        acc = deltas[t] + discounts[t] * c_bar[t] * acc;
        vs_minus_v[t] = acc;
    }

    // V-trace targets
    let vs: Vec<f32> = (0..len).map(|t| vs_minus_v[t] + values[t]).collect();

    // Policy gradient advantages: ρ̄_t * (r_t + γ * vs_{t+1} - V(s_t))
    let next_vs: Vec<f32> = vs[1..].iter().copied()
        .chain(std::iter::once(bootstrap))
        .collect();
    let advantages: Vec<f32> = (0..len)
        .map(|t| rho_bar[t] * (rewards[t] + discounts[t] * next_vs[t] - values[t]))
        .collect();

    (vs, advantages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_policy_zero_rewards_terminal() {
        // On-policy, zero rewards, terminal: targets correct toward 0
        // (no future reward to sustain the values)
        let (vs, adv) = vtrace_targets(
            &[0.0, 0.0, 0.0], &[0.99, 0.99, 0.99],
            &[0.0, 0.0, 0.0], &[0.5, 0.3, 0.1], 0.0, 1.0, 1.0,
        );
        assert_eq!(vs.len(), 3);
        assert_eq!(adv.len(), 3);
        // Last step: target = 0 + 0.99*0 - 0.1 + 0.1 = 0 (corrected toward bootstrap)
        // Advantages should be negative (values are overestimates)
        assert!(adv[2] < 0.0, "advantage should be negative when values overestimate");
    }

    #[test]
    fn single_step_td0() {
        // Single step: target = r + γ * bootstrap
        let (vs, adv) = vtrace_targets(
            &[0.0], &[0.99], &[1.0], &[0.0], 0.5, 1.0, 1.0,
        );
        let expected = 1.0 + 0.99 * 0.5;
        assert!((vs[0] - expected).abs() < 1e-5, "vs={}, expected={}", vs[0], expected);
        assert!((adv[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn off_policy_clipping() {
        let (vs_clipped, _) = vtrace_targets(
            &[5.0, 5.0], &[0.99, 0.99], &[1.0, 0.0], &[0.0, 0.0], 0.0, 1.0, 1.0,
        );
        let (vs_unclipped, _) = vtrace_targets(
            &[5.0, 5.0], &[0.99, 0.99], &[1.0, 0.0], &[0.0, 0.0], 0.0, 200.0, 200.0,
        );
        assert!(vs_clipped[0] <= vs_unclipped[0] + 1e-6);
    }

    #[test]
    fn terminal_vs_nonterminal() {
        let (vs_term, _) = vtrace_targets(
            &[0.0], &[0.99], &[0.0], &[0.0], 0.0, 1.0, 1.0,
        );
        let (vs_nonterm, _) = vtrace_targets(
            &[0.0], &[0.99], &[0.0], &[0.0], 1.0, 1.0, 1.0,
        );
        assert!(vs_nonterm[0] > vs_term[0]);
    }

    #[test]
    fn output_lengths() {
        for n in [1, 3, 10] {
            let (vs, adv) = vtrace_targets(
                &vec![0.0; n], &vec![0.99; n], &vec![0.0; n], &vec![0.0; n], 0.0, 1.0, 1.0,
            );
            assert_eq!(vs.len(), n);
            assert_eq!(adv.len(), n);
        }
    }
}
