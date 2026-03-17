//! UPGO (Upgoing Policy Gradient) for self-imitation learning.
//!
//! Pure f32 computation — no tensors, no backend dependency.
//! Reinforces trajectories where the TD error is positive.

use contracts::*;

/// Compute UPGO advantages for self-imitation learning.
///
/// UPGO (Upgoing Policy Gradient) reinforces trajectories where
/// the TD error is positive, i.e., where the actual return exceeded
/// the value estimate. Used by ROA-Star alongside V-trace.
///
/// At each timestep, the "upgoing" condition checks whether the
/// immediate TD was non-negative. When it is, the return is propagated
/// backward. When it isn't, the trajectory is truncated to the value
/// baseline.
///
/// # Arguments
/// - `rewards`: per-step rewards, length T
/// - `values`: V(s_t) from critic, length T
/// - `dones`: whether the episode ended at step t, length T
/// - `last_value`: V(s_T) bootstrap for non-terminal final state
/// - `gamma`: discount factor
///
/// # Returns
/// UPGO advantages (targets − values), length T.
#[requires(!rewards.is_empty(), "must have at least one step")]
#[requires(rewards.len() == values.len(), "rewards and values must match")]
#[requires(rewards.len() == dones.len(), "rewards and dones must match")]
#[requires(gamma >= 0.0 && gamma <= 1.0, "gamma must be in [0, 1]")]
#[ensures(ret.len() == rewards.len(), "advantages length matches")]
pub fn upgo(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
) -> Vec<f32> {
    let n = rewards.len();
    let mut targets = vec![0.0f32; n];

    // Bootstrap: last target
    let mut next_target = last_value;

    // Backward pass
    for t in (0..n).rev() {
        let not_done = if dones[t] { 0.0 } else { 1.0 };
        let next_val = if t + 1 < n { values[t + 1] } else { last_value };

        // TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        let td = rewards[t] + gamma * next_val * not_done - values[t];

        if td >= 0.0 {
            // Upgoing: use actual return (propagate)
            targets[t] = rewards[t] + gamma * not_done * next_target;
        } else {
            // Not upgoing: use value estimate (truncate)
            targets[t] = values[t];
        }

        next_target = targets[t];
    }

    // Return advantages = targets - values
    targets
        .iter()
        .zip(values.iter())
        .map(|(t, v)| t - v)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_positive_td_equals_nstep_returns() {
        // All TD errors positive: UPGO targets = n-step returns
        // rewards=[1, 1, 1], values=[0, 0, 0], last_value=0, gamma=0.99
        // All TDs = 1 + 0.99*0 - 0 = 1 > 0, so targets propagate fully
        // target[2] = 1 + 0.99 * 0 * 0 = 1 (last_value=0, not_done=1 but done=false... wait)
        let rewards = [1.0, 1.0, 1.0];
        let values = [0.0, 0.0, 0.0];
        let dones = [false, false, true];

        let adv = upgo(&rewards, &values, &dones, 0.0, 0.99);

        // Step 2 (terminal): td = 1 + 0.99*0*0 - 0 = 1 >= 0
        //   target[2] = 1 + 0.99 * 0.0 * next_target = 1.0
        // Step 1: td = 1 + 0.99*0*1 - 0 = 1 >= 0
        //   target[1] = 1 + 0.99 * 1.0 * target[2] = 1 + 0.99*1.0 = 1.99
        // Step 0: td = 1 + 0.99*0*1 - 0 = 1 >= 0
        //   target[0] = 1 + 0.99 * 1.0 * target[1] = 1 + 0.99*1.99 = 2.9701
        assert!((adv[2] - 1.0).abs() < 1e-5, "adv[2]={}", adv[2]);
        assert!((adv[1] - 1.99).abs() < 1e-5, "adv[1]={}", adv[1]);
        assert!((adv[0] - 2.9701).abs() < 1e-4, "adv[0]={}", adv[0]);
    }

    #[test]
    fn all_negative_td_advantages_zero() {
        // Values overestimate everything: TD < 0, targets = values, advantages = 0
        let rewards = [0.0, 0.0, 0.0];
        let values = [10.0, 10.0, 10.0];
        let dones = [false, false, false];

        let adv = upgo(&rewards, &values, &dones, 10.0, 0.99);

        for (i, &a) in adv.iter().enumerate() {
            assert!(a.abs() < 1e-5, "adv[{}]={}, expected 0.0", i, a);
        }
    }

    #[test]
    fn mixed_truncation() {
        // Step 0: positive TD -> propagate, Step 1: negative TD -> truncate
        let rewards = [2.0, 0.0];
        let values = [0.0, 5.0];
        let dones = [false, false];
        let last_value = 5.0;
        let gamma = 1.0;

        let adv = upgo(&rewards, &values, &dones, last_value, gamma);

        // Step 1: td = 0 + 1*5 - 5 = 0 >= 0, target[1] = 0 + 1*5 = 5, adv[1] = 0
        // Step 0: td = 2 + 1*5 - 0 = 7 >= 0, target[0] = 2 + 1*5 = 7, adv[0] = 7
        assert!((adv[1]).abs() < 1e-5, "adv[1]={}", adv[1]);
        assert!((adv[0] - 7.0).abs() < 1e-5, "adv[0]={}", adv[0]);
    }

    #[test]
    fn mixed_truncation_negative_step() {
        // Step 1 has negative TD, so it truncates and its target = value
        let rewards = [1.0, 0.0];
        let values = [0.0, 5.0];
        let dones = [false, false];
        let last_value = 0.0;
        let gamma = 1.0;

        let adv = upgo(&rewards, &values, &dones, last_value, gamma);

        // Step 1: td = 0 + 1*0 - 5 = -5 < 0, target[1] = 5, adv[1] = 0
        // Step 0: td = 1 + 1*5 - 0 = 6 >= 0, target[0] = 1 + 1*target[1] = 1+5 = 6, adv[0] = 6
        assert!((adv[1]).abs() < 1e-5, "adv[1]={}", adv[1]);
        assert!((adv[0] - 6.0).abs() < 1e-5, "adv[0]={}", adv[0]);
    }

    #[test]
    fn episode_boundary_resets() {
        // Done at step 1 should cut off future returns
        let rewards = [1.0, 1.0, 1.0, 1.0];
        let values = [0.0, 0.0, 0.0, 0.0];
        let dones = [false, true, false, false];
        let gamma = 0.99;

        let adv = upgo(&rewards, &values, &dones, 0.0, gamma);

        // Step 3: td = 1 + 0.99*0 - 0 = 1 >= 0, target[3] = 1 + 0.99*0 = 1
        // Step 2: td = 1 + 0.99*0 - 0 = 1 >= 0, target[2] = 1 + 0.99*1 = 1.99
        // Step 1 (done): td = 1 + 0.99*0*0 - 0 = 1 >= 0, target[1] = 1 + 0.99*0*target[2] = 1
        // Step 0: td = 1 + 0.99*0 - 0 = 1 >= 0, target[0] = 1 + 0.99*1*target[1] = 1 + 0.99 = 1.99
        assert!((adv[1] - 1.0).abs() < 1e-5, "done step: adv[1]={}", adv[1]);
        assert!((adv[0] - 1.99).abs() < 1e-5, "pre-done step: adv[0]={}", adv[0]);
        // Step 2 and 3 form an independent trajectory
        assert!((adv[3] - 1.0).abs() < 1e-5, "adv[3]={}", adv[3]);
        assert!((adv[2] - 1.99).abs() < 1e-5, "adv[2]={}", adv[2]);
    }

    #[test]
    fn single_step_positive() {
        // r=2, V(s)=1, V(s')=0 (terminal), gamma=1
        // td = 2 + 0 - 1 = 1 >= 0, target = 2 + 0 = 2, adv = 1
        let adv = upgo(&[2.0], &[1.0], &[true], 0.0, 1.0);
        assert!((adv[0] - 1.0).abs() < 1e-5, "adv={}", adv[0]);
    }

    #[test]
    fn single_step_negative() {
        // r=0, V(s)=5, V(s')=0 (terminal), gamma=1
        // td = 0 + 0 - 5 = -5 < 0, target = 5, adv = 0
        let adv = upgo(&[0.0], &[5.0], &[true], 0.0, 1.0);
        assert!(adv[0].abs() < 1e-5, "adv={}", adv[0]);
    }

    #[test]
    fn output_lengths() {
        for n in [1, 5, 20] {
            let adv = upgo(
                &vec![0.0; n],
                &vec![0.0; n],
                &vec![false; n],
                0.0,
                0.99,
            );
            assert_eq!(adv.len(), n);
        }
    }
}
