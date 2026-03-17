//! Generalized Advantage Estimation (Schulman et al., 2015).
//!
//! Pure f32 computation — no tensors, no backend dependency.

use contracts::*;

/// Compute GAE advantages and value targets for a single rollout.
///
/// # Arguments
/// - `rewards`: per-step rewards, length T
/// - `values`: V(s_t) from critic, length T
/// - `dones`: whether the episode ended at step t, length T
/// - `last_value`: V(s_T) bootstrap for non-terminal final state
/// - `gamma`: discount factor
/// - `lambda`: GAE smoothing parameter (0 = TD(0), 1 = Monte Carlo)
///
/// # Returns
/// `(advantages, returns)` — both length T.
#[requires(!rewards.is_empty(), "must have at least one step")]
#[requires(rewards.len() == values.len(), "rewards and values must match")]
#[requires(rewards.len() == dones.len(), "rewards and dones must match")]
#[requires(gamma >= 0.0 && gamma <= 1.0, "gamma must be in [0, 1]")]
#[requires(lambda >= 0.0 && lambda <= 1.0, "lambda must be in [0, 1]")]
#[ensures(ret.0.len() == rewards.len(), "advantages length matches")]
#[ensures(ret.1.len() == rewards.len(), "returns length matches")]
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = rewards.len();
    let mut advantages = vec![0.0f32; len];
    let mut last_gae = 0.0f32;

    for t in (0..len).rev() {
        let next_value = if t == len - 1 { last_value } else { values[t + 1] };
        let not_done = if dones[t] { 0.0 } else { 1.0 };
        let delta = rewards[t] + gamma * next_value * not_done - values[t];
        last_gae = delta + gamma * lambda * not_done * last_gae;
        advantages[t] = last_gae;
    }

    let returns: Vec<f32> = advantages
        .iter()
        .zip(values)
        .map(|(a, v)| a + v)
        .collect();
    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_terminal() {
        let (adv, ret) = gae(&[1.0], &[0.0], &[true], 0.0, 0.99, 0.95);
        assert!((adv[0] - 1.0).abs() < 1e-5);
        assert!((ret[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn single_step_nonterminal() {
        // δ = r + γ*V_next - V = 1.0 + 0.99*0.5 - 0.0 = 1.495
        let (adv, ret) = gae(&[1.0], &[0.0], &[false], 0.5, 0.99, 0.95);
        let expected = 1.0 + 0.99 * 0.5;
        assert!((adv[0] - expected).abs() < 1e-5);
        assert!((ret[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn multi_step_all_terminal_at_end() {
        let rewards = [1.0, 1.0, 1.0];
        let values = [0.0, 0.0, 0.0];
        let dones = [false, false, true];
        let (adv, _ret) = gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
        // Last step (terminal): δ=1, gae=1
        assert!((adv[2] - 1.0).abs() < 1e-5);
        // Earlier steps accumulate discounted advantages
        assert!(adv[0] > adv[1]);
        assert!(adv[1] > adv[2]);
    }

    #[test]
    fn zero_lambda_gives_td0() {
        // λ=0: advantage = δ_t only (no accumulation)
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 0.5, 0.5];
        let dones = [false, false, false];
        let (adv, _) = gae(&rewards, &values, &dones, 1.0, 0.99, 0.0);
        // δ_0 = 1.0 + 0.99*0.5 - 0.5 = 0.995
        assert!((adv[0] - 0.995).abs() < 1e-5);
        // δ_1 = 2.0 + 0.99*0.5 - 0.5 = 1.995
        assert!((adv[1] - 1.995).abs() < 1e-5);
        // δ_2 = 3.0 + 0.99*1.0 - 0.5 = 3.49
        assert!((adv[2] - 3.49).abs() < 1e-5);
    }

    #[test]
    fn done_resets_accumulation() {
        let rewards = [1.0, 1.0, 1.0, 1.0];
        let values = [0.0; 4];
        let dones = [false, true, false, false];
        let (adv, _) = gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
        // Step 1 is terminal: δ=1, gae=1
        assert!((adv[1] - 1.0).abs() < 1e-5);
        // Step 0: δ = 1 + 0.99*0*0 - 0 = 1 (done resets next_value and gae)
        // Since done[0] is false but done[1] is true, step 0 uses next_value = values[1] = 0
        // δ_0 = 1.0 + 0.99*0.0 - 0.0 = 1.0, but wait, not_done[0]=1.0 (done[0]=false)
        // δ_0 = 1.0 + 0.99 * values[1] * 1.0 - 0.0 = 1.0
        // gae_0 = 1.0 + 0.99 * 0.95 * 1.0 * gae_1
        // gae_1 = 1.0 (terminal step)
        // gae_0 = 1.0 + 0.99*0.95*1.0 = 1.0 + 0.9405 = 1.9405
        assert!((adv[0] - 1.9405).abs() < 1e-3);
    }

    #[test]
    fn output_lengths() {
        for n in [1, 5, 20] {
            let (adv, ret) = gae(
                &vec![0.0; n],
                &vec![0.0; n],
                &vec![false; n],
                0.0,
                0.99,
                0.95,
            );
            assert_eq!(adv.len(), n);
            assert_eq!(ret.len(), n);
        }
    }
}
