//! Imagination rollout engine for DreamerV3 (Issue #23).
//!
//! Generates trajectories entirely within the RSSM latent space. Starting from
//! initial states, the actor samples actions, the RSSM advances through the
//! dynamics model (no observations), and reward/continue are predicted.

use burn::prelude::*;

use rl4burn_nn::rssm::{Rssm, RssmState};

// ---------------------------------------------------------------------------
// Imagined trajectory
// ---------------------------------------------------------------------------

/// Result of an imagination rollout.
pub struct ImaginedTrajectory<B: Backend> {
    /// States at each timestep: [horizon+1] states (including initial).
    pub states: Vec<RssmState<B>>,
    /// Actions at each timestep: [horizon][batch, action_dim].
    pub actions: Vec<Tensor<B, 2>>,
    /// Predicted reward logits: [horizon][batch, 255].
    pub reward_logits: Vec<Tensor<B, 2>>,
    /// Predicted continue logits: [horizon][batch, 1].
    pub continue_logits: Vec<Tensor<B, 2>>,
}

// ---------------------------------------------------------------------------
// Imagination rollout
// ---------------------------------------------------------------------------

/// Generate imagined trajectories within the RSSM latent space.
///
/// Starting from initial states, the actor samples actions, the RSSM advances
/// through the dynamics model (no observations), and reward/continue are predicted.
///
/// # Arguments
/// * `rssm` - The world model
/// * `initial_states` - Starting states [batch]
/// * `actor_fn` - Function that maps (h, z) -> action tensor [batch, action_dim]
/// * `horizon` - Number of imagination steps (DreamerV3 uses 15)
///
/// # Returns
/// Imagined trajectory with states, actions, rewards, continues
pub fn imagine_rollout<B, F>(
    rssm: &Rssm<B>,
    initial_states: RssmState<B>,
    mut actor_fn: F,
    horizon: usize,
) -> ImaginedTrajectory<B>
where
    B: Backend,
    F: FnMut(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
{
    let mut states = vec![initial_states.clone()];
    let mut actions = Vec::with_capacity(horizon);
    let mut reward_logits = Vec::with_capacity(horizon);
    let mut continue_logits = Vec::with_capacity(horizon);

    let mut state = initial_states;

    for _t in 0..horizon {
        // Actor selects action from current state
        let action = actor_fn(state.h.clone(), state.z.clone());

        // RSSM imagines next state (dynamics only, no observation)
        let next_state = rssm.imagine_step(&state, action.clone());

        // Predict reward and continue
        let r_logits = rssm.predict_reward(next_state.h.clone(), next_state.z.clone());
        let c_logits = rssm.predict_continue(next_state.h.clone(), next_state.z.clone());

        actions.push(action);
        reward_logits.push(r_logits);
        continue_logits.push(c_logits);
        states.push(next_state.clone());
        state = next_state;
    }

    ImaginedTrajectory {
        states,
        actions,
        reward_logits,
        continue_logits,
    }
}

// ---------------------------------------------------------------------------
// Lambda returns
// ---------------------------------------------------------------------------

/// Compute lambda-returns from imagined trajectories.
///
/// DreamerV3 computes lambda-returns backward from the critic's bootstrap value.
///
/// # Arguments
/// * `rewards` - Predicted rewards per step [horizon]
/// * `values` - Critic values per step [horizon+1] (including bootstrap)
/// * `continues` - Continue probabilities per step [horizon]
/// * `gamma` - Discount factor
/// * `lambda` - Lambda for lambda-returns
///
/// # Returns
/// Lambda-returns: one tensor per timestep [horizon][batch]
pub fn lambda_returns<B: Backend>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    continues: &[Tensor<B, 1>],
    gamma: f32,
    lambda: f32,
) -> Vec<Tensor<B, 1>> {
    let horizon = rewards.len();
    let mut returns = vec![values[horizon].clone(); horizon]; // initialize with bootstrap

    // Backward pass
    for t in (0..horizon).rev() {
        // lambda-return: R_t = r_t + gamma * c_t * ((1-lambda) * V_{t+1} + lambda * R_{t+1})
        let next_return = if t + 1 < horizon {
            returns[t + 1].clone()
        } else {
            values[horizon].clone()
        };

        let next_val = values[t + 1].clone();
        let cont = continues[t].clone();

        let target = next_val * (1.0 - lambda) + next_return * lambda;
        returns[t] = rewards[t].clone() + cont * gamma * target;
    }

    returns
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use rl4burn_nn::rssm::RssmConfig;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    fn small_rssm() -> Rssm<B> {
        RssmConfig::new(16, 4)
            .with_deterministic_size(32)
            .with_n_categories(4)
            .with_n_classes(4)
            .with_hidden_size(32)
            .init::<B>(&dev())
    }

    #[test]
    fn trajectory_length_matches_horizon() {
        let rssm = small_rssm();
        let initial = rssm.initial_state(2, &dev());
        let horizon = 5;

        let traj = imagine_rollout(
            &rssm,
            initial,
            |_h, _z| Tensor::<B, 2>::zeros([2, 4], &dev()),
            horizon,
        );

        assert_eq!(traj.actions.len(), horizon);
        assert_eq!(traj.reward_logits.len(), horizon);
        assert_eq!(traj.continue_logits.len(), horizon);
    }

    #[test]
    fn states_count_is_horizon_plus_one() {
        let rssm = small_rssm();
        let initial = rssm.initial_state(3, &dev());
        let horizon = 7;

        let traj = imagine_rollout(
            &rssm,
            initial,
            |_h, _z| Tensor::<B, 2>::zeros([3, 4], &dev()),
            horizon,
        );

        assert_eq!(traj.states.len(), horizon + 1);
    }

    #[test]
    fn reward_logits_shape() {
        let rssm = small_rssm();
        let initial = rssm.initial_state(2, &dev());

        let traj = imagine_rollout(
            &rssm,
            initial,
            |_h, _z| Tensor::<B, 2>::zeros([2, 4], &dev()),
            3,
        );

        for rl in &traj.reward_logits {
            assert_eq!(rl.dims(), [2, 255]);
        }
    }

    #[test]
    fn continue_logits_shape() {
        let rssm = small_rssm();
        let initial = rssm.initial_state(2, &dev());

        let traj = imagine_rollout(
            &rssm,
            initial,
            |_h, _z| Tensor::<B, 2>::zeros([2, 4], &dev()),
            3,
        );

        for cl in &traj.continue_logits {
            assert_eq!(cl.dims(), [2, 1]);
        }
    }

    // -- lambda returns -------------------------------------------------------

    #[test]
    fn lambda_zero_gives_td0() {
        // lambda=0 => R_t = r_t + gamma * c_t * V_{t+1}  (one-step TD)
        let r0 = Tensor::<B, 1>::from_floats([1.0], &dev());
        let r1 = Tensor::<B, 1>::from_floats([2.0], &dev());
        let v0 = Tensor::<B, 1>::from_floats([10.0], &dev());
        let v1 = Tensor::<B, 1>::from_floats([20.0], &dev());
        let v2 = Tensor::<B, 1>::from_floats([30.0], &dev());
        let c0 = Tensor::<B, 1>::from_floats([1.0], &dev());
        let c1 = Tensor::<B, 1>::from_floats([1.0], &dev());

        let returns = lambda_returns(
            &[r0, r1],
            &[v0, v1, v2],
            &[c0, c1],
            0.99, // gamma
            0.0,  // lambda = 0
        );

        // R_1 = r1 + gamma * c1 * V2 = 2 + 0.99*1*30 = 31.7
        let ret1: f32 = returns[1].clone().into_scalar();
        assert!((ret1 - 31.7).abs() < 1e-4, "R_1 = {ret1}, expected 31.7");

        // R_0 = r0 + gamma * c0 * V1 = 1 + 0.99*1*20 = 20.8
        let ret0: f32 = returns[0].clone().into_scalar();
        assert!((ret0 - 20.8).abs() < 1e-4, "R_0 = {ret0}, expected 20.8");
    }

    #[test]
    fn lambda_one_gives_mc_returns() {
        // lambda=1 => full Monte-Carlo-like returns (bootstrapping from V_H)
        // R_1 = r1 + gamma * c1 * V2  (last step always bootstraps)
        // R_0 = r0 + gamma * c0 * R_1
        let r0 = Tensor::<B, 1>::from_floats([1.0], &dev());
        let r1 = Tensor::<B, 1>::from_floats([2.0], &dev());
        let v0 = Tensor::<B, 1>::from_floats([10.0], &dev());
        let v1 = Tensor::<B, 1>::from_floats([20.0], &dev());
        let v2 = Tensor::<B, 1>::from_floats([30.0], &dev());
        let c0 = Tensor::<B, 1>::from_floats([1.0], &dev());
        let c1 = Tensor::<B, 1>::from_floats([1.0], &dev());

        let returns = lambda_returns(
            &[r0, r1],
            &[v0, v1, v2],
            &[c0, c1],
            0.99, // gamma
            1.0,  // lambda = 1
        );

        // R_1 = r1 + gamma * c1 * V2 = 2 + 0.99 * 30 = 31.7
        let ret1: f32 = returns[1].clone().into_scalar();
        assert!((ret1 - 31.7).abs() < 1e-4, "R_1 = {ret1}, expected 31.7");

        // R_0 = r0 + gamma * c0 * R_1 = 1 + 0.99 * 31.7 = 32.383
        let expected_r0 = 1.0 + 0.99 * 31.7;
        let ret0: f32 = returns[0].clone().into_scalar();
        assert!(
            (ret0 - expected_r0).abs() < 1e-3,
            "R_0 = {ret0}, expected {expected_r0}"
        );
    }

    // -- Continuation flag tests ----------------------------------------------

    #[test]
    fn continuation_zero_blocks_future_rewards() {
        let rewards: Vec<Tensor<B, 1>> = [1.0, 1.0, 1.0, 100.0, 100.0]
            .iter()
            .map(|&r| Tensor::from_floats([r], &dev()))
            .collect();
        let values: Vec<Tensor<B, 1>> = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .map(|&v| Tensor::from_floats([v], &dev()))
            .collect();
        let continues: Vec<Tensor<B, 1>> = [1.0, 1.0, 0.0, 1.0, 1.0]
            .iter()
            .map(|&c| Tensor::from_floats([c], &dev()))
            .collect();

        let rets = lambda_returns(&rewards, &values, &continues, 0.99, 0.95);

        // Return at step 2: r2 + gamma * c2 * (...) = 1.0 + 0.99 * 0 * ... = 1.0
        let ret2: f32 = rets[2].clone().into_scalar();
        assert!(
            (ret2 - 1.0).abs() < 1e-4,
            "return at terminal step should equal immediate reward, got {ret2}"
        );

        // Return at step 0 should NOT include the 100.0 rewards from steps 3-4
        let ret0: f32 = rets[0].clone().into_scalar();
        assert!(
            ret0 < 10.0,
            "return at step 0 should not include post-terminal rewards, got {ret0}"
        );
    }

    #[test]
    fn all_continues_one_equals_no_termination() {
        let rewards: Vec<Tensor<B, 1>> = [1.0, 1.0, 1.0]
            .iter()
            .map(|&r| Tensor::from_floats([r], &dev()))
            .collect();
        let values: Vec<Tensor<B, 1>> = [0.0, 0.0, 0.0, 0.0]
            .iter()
            .map(|&v| Tensor::from_floats([v], &dev()))
            .collect();
        let continues: Vec<Tensor<B, 1>> = [1.0, 1.0, 1.0]
            .iter()
            .map(|&c| Tensor::from_floats([c], &dev()))
            .collect();

        let rets = lambda_returns(&rewards, &values, &continues, 0.99, 1.0);

        let ret0: f32 = rets[0].clone().into_scalar();
        let expected = 1.0 + 0.99 * (1.0 + 0.99 * 1.0);
        assert!(
            (ret0 - expected).abs() < 1e-3,
            "R_0 = {ret0}, expected {expected}"
        );
    }

    #[test]
    fn all_continues_zero_gives_immediate_reward() {
        let rewards: Vec<Tensor<B, 1>> = [5.0, 3.0, 7.0]
            .iter()
            .map(|&r| Tensor::from_floats([r], &dev()))
            .collect();
        let values: Vec<Tensor<B, 1>> = [10.0, 20.0, 30.0, 40.0]
            .iter()
            .map(|&v| Tensor::from_floats([v], &dev()))
            .collect();
        let continues: Vec<Tensor<B, 1>> = [0.0, 0.0, 0.0]
            .iter()
            .map(|&c| Tensor::from_floats([c], &dev()))
            .collect();

        let rets = lambda_returns(&rewards, &values, &continues, 0.99, 0.95);

        for (t, &expected) in [5.0f32, 3.0, 7.0].iter().enumerate() {
            let ret: f32 = rets[t].clone().into_scalar();
            assert!(
                (ret - expected).abs() < 1e-4,
                "R_{t} = {ret}, expected {expected} (immediate reward only)"
            );
        }
    }

    #[test]
    fn lambda_returns_batch_dimension() {
        let batch = 3;
        let rewards = vec![
            Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev()),
            Tensor::from_floats([4.0, 5.0, 6.0], &dev()),
        ];
        let values = vec![
            Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0], &dev()),
            Tensor::from_floats([0.0, 0.0, 0.0], &dev()),
            Tensor::from_floats([0.0, 0.0, 0.0], &dev()),
        ];
        let continues = vec![
            Tensor::<B, 1>::from_floats([1.0, 1.0, 1.0], &dev()),
            Tensor::from_floats([1.0, 1.0, 1.0], &dev()),
        ];

        let rets = lambda_returns(&rewards, &values, &continues, 0.99, 1.0);
        assert_eq!(rets.len(), 2);
        for ret in &rets {
            assert_eq!(ret.dims(), [batch]);
            let data: Vec<f32> = ret.clone().into_data().to_vec().unwrap();
            for &v in &data {
                assert!(v.is_finite(), "return should be finite, got {v}");
                assert!(v > 0.0, "return should be positive for positive rewards");
            }
        }
    }
}
