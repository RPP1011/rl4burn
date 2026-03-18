//! Behavioral cloning / supervised pre-training (Issue #16).
//!
//! Cross-entropy imitation learning from expert demonstrations. Supports
//! single-head discrete actions and multi-head (hierarchical) actions.

use burn::prelude::*;
use burn::tensor::activation::log_softmax;

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

/// Compute behavioral cloning loss (cross-entropy) for discrete actions.
///
/// # Arguments
/// * `logits` - Model output logits `[batch, n_actions]`
/// * `expert_actions` - Expert action indices `[batch]` as i32
///
/// # Returns
/// Scalar cross-entropy loss `[1]`
pub fn bc_loss_discrete<B: Backend>(
    logits: Tensor<B, 2>,
    expert_actions: Tensor<B, 1, Int>,
    _device: &B::Device,
) -> Tensor<B, 1> {
    let log_probs = log_softmax(logits, 1);

    let action_indices = expert_actions.unsqueeze_dim(1); // [batch, 1]
    let action_log_probs = log_probs.gather(1, action_indices).squeeze_dim::<1>(1);

    action_log_probs.neg().mean().unsqueeze()
}

/// Compute behavioral cloning loss for multi-head / hierarchical actions.
///
/// Each head has its own cross-entropy loss, summed together.
///
/// # Arguments
/// * `logits` - `[batch, total_logits]`
/// * `expert_actions` - `[batch, n_heads]` as i32 indices
/// * `head_sizes` - Number of categories per head
///
/// # Returns
/// Scalar loss `[1]`
pub fn bc_loss_multi_head<B: Backend>(
    logits: Tensor<B, 2>,
    expert_actions: Tensor<B, 2, Int>,
    head_sizes: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch, _] = logits.dims();
    let mut total_loss: Tensor<B, 1> = Tensor::zeros([1], device);

    let mut offset = 0;
    for (h, &size) in head_sizes.iter().enumerate() {
        let head_logits = logits.clone().slice([0..batch, offset..offset + size]);
        let head_log_probs = log_softmax(head_logits, 1);

        let head_actions = expert_actions.clone().slice([0..batch, h..h + 1]); // [batch, 1]
        let head_lp = head_log_probs.gather(1, head_actions).squeeze_dim::<1>(1);

        total_loss = total_loss + head_lp.neg().mean().unsqueeze();
        offset += size;
    }

    total_loss
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

/// Behavioral cloning training step.
///
/// Performs one gradient descent step on the cross-entropy loss between
/// the model's action distribution and the expert actions.
///
/// # Returns
/// `(updated_model, loss_value)`
pub fn bc_step<B, M, O>(
    mut model: M,
    optim: &mut O,
    obs: Tensor<B, 2>,
    expert_actions: Tensor<B, 1, Int>,
    lr: f64,
    device: &B::Device,
) -> (M, f32)
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    M: rl4burn_nn::policy::DiscreteActorCritic<B>,
    O: burn::optim::Optimizer<M, B>,
{
    let output = model.forward(obs);
    let loss = bc_loss_discrete(output.logits, expert_actions, device);
    let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

    let grads = loss.backward();
    let grads = burn::optim::GradientsParams::from_grads(grads, &model);
    model = optim.step(lr, model, grads);

    (model, loss_val)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn uniform_policy_loss_is_ln_k() {
        // For a uniform policy over K actions, cross-entropy = ln(K)
        let k = 4;
        let batch = 64;

        // Uniform logits (all zeros → softmax gives 1/K for each)
        let logits: Tensor<B, 2> = Tensor::zeros([batch, k], &dev());

        // Random expert actions
        let actions_data: Vec<i32> = (0..batch).map(|i| (i % k) as i32).collect();
        let expert_actions: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(actions_data, [batch]), &dev());

        let loss = bc_loss_discrete(logits, expert_actions, &dev());
        let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];

        let expected = (k as f32).ln();
        assert!(
            (loss_val - expected).abs() < 0.01,
            "uniform loss {loss_val:.4} should be ln({k}) = {expected:.4}"
        );
    }

    #[test]
    fn perfect_logits_give_low_loss() {
        let batch = 16;
        let k = 3;

        // Expert always picks action 1. Set logit for action 1 very high.
        let mut logits_data = vec![0.0f32; batch * k];
        for i in 0..batch {
            logits_data[i * k + 1] = 10.0; // strong preference for action 1
        }
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data, [batch, k]), &dev());

        let actions_data = vec![1i32; batch];
        let expert_actions: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(actions_data, [batch]), &dev());

        let loss = bc_loss_discrete(logits, expert_actions, &dev());
        let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];

        assert!(
            loss_val < 0.01,
            "loss for near-perfect logits should be near 0, got {loss_val:.4}"
        );
    }

    #[test]
    fn multi_head_loss_sums_heads() {
        let batch = 8;
        let head_sizes = [3, 4]; // 2 heads: 3 actions, 4 actions
        let total_logits = 7;

        // Uniform logits
        let logits: Tensor<B, 2> = Tensor::zeros([batch, total_logits], &dev());

        let actions_data: Vec<i32> = (0..batch * 2)
            .map(|i| if i % 2 == 0 { 0 } else { 1 })
            .collect();
        let expert_actions: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(actions_data, [batch, 2]), &dev());

        let loss = bc_loss_multi_head(logits, expert_actions, &head_sizes, &dev());
        let loss_val = loss.into_data().to_vec::<f32>().unwrap()[0];

        // Expected: ln(3) + ln(4)
        let expected = (3.0f32).ln() + (4.0f32).ln();
        assert!(
            (loss_val - expected).abs() < 0.05,
            "multi-head loss {loss_val:.4} should be ln(3)+ln(4) = {expected:.4}"
        );
    }

    #[test]
    fn bc_loss_backward_produces_gradients() {
        use burn::backend::Autodiff;
        use burn::nn::{Linear, LinearConfig};

        type AB = Autodiff<NdArray>;

        let dev: <AB as Backend>::Device = Default::default();
        let model: Linear<AB> = LinearConfig::new(4, 3).init::<AB>(&dev);

        let obs: Tensor<AB, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f32; 8 * 4], [8, 4]),
            &dev,
        );
        let actions: Tensor<AB, 1, Int> = Tensor::from_data(
            TensorData::new(vec![1i32; 8], [8]),
            &dev,
        );

        let logits = model.forward(obs);
        let loss = bc_loss_discrete(logits, actions, &dev);
        let grads = loss.backward();

        // Verify gradients exist for the model weight
        let weight_grad = model.weight.grad(&grads);
        assert!(weight_grad.is_some(), "bc_loss_discrete should produce gradients");

        let grad_data: Vec<f32> = weight_grad.unwrap().into_data().to_vec().unwrap();
        let has_nonzero = grad_data.iter().any(|v| v.abs() > 1e-10);
        assert!(has_nonzero, "gradients should be nonzero");
    }

    #[test]
    fn bc_loss_wrong_action_higher_than_correct() {
        // Logits strongly prefer action 0, but expert picks action 2
        // Loss should be higher than when expert picks the preferred action
        let batch = 8;
        let k = 3;

        let mut logits_data = vec![0.0f32; batch * k];
        for i in 0..batch {
            logits_data[i * k] = 5.0; // model prefers action 0
        }

        let logits_wrong: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(logits_data.clone(), [batch, k]),
            &dev(),
        );
        let logits_right: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(logits_data, [batch, k]),
            &dev(),
        );

        let wrong_actions: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(vec![2i32; batch], [batch]),
            &dev(),
        );
        let right_actions: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(vec![0i32; batch], [batch]),
            &dev(),
        );

        let loss_wrong = bc_loss_discrete(logits_wrong, wrong_actions, &dev())
            .into_data().to_vec::<f32>().unwrap()[0];
        let loss_right = bc_loss_discrete(logits_right, right_actions, &dev())
            .into_data().to_vec::<f32>().unwrap()[0];

        assert!(
            loss_wrong > loss_right,
            "loss for wrong action ({loss_wrong:.4}) should exceed loss for correct action ({loss_right:.4})"
        );
    }
}
