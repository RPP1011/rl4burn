//! Policy and actor-critic traits.
//!
//! These traits define the interface between RL algorithms and neural network
//! models. Users implement these on their Burn modules.

use burn::prelude::*;
use burn::tensor::TensorData;

/// Output of a discrete actor-critic forward pass.
pub struct DiscreteAcOutput<B: Backend> {
    /// Raw logits (pre-softmax) for each action. Shape: `[batch, n_actions]`.
    pub logits: Tensor<B, 2>,
    /// State value estimates. Shape: `[batch]`.
    pub values: Tensor<B, 1>,
}

/// Actor-critic model for discrete action spaces.
///
/// Used by PPO and A2C. The model takes a batch of observations and returns
/// action logits and value estimates in a single forward pass.
///
/// # Example
///
/// ```ignore
/// #[derive(Module)]
/// struct MyActorCritic<B: Backend> {
///     shared: Linear<B>,
///     policy_head: Linear<B>,
///     value_head: Linear<B>,
/// }
///
/// impl<B: Backend> DiscreteActorCritic<B> for MyActorCritic<B> {
///     fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
///         let h = self.shared.forward(obs);
///         DiscreteAcOutput {
///             logits: self.policy_head.forward(h.clone()),
///             values: self.value_head.forward(h).squeeze(1),
///         }
///     }
/// }
/// ```
pub trait DiscreteActorCritic<B: Backend> {
    /// Forward pass producing action logits and value estimates.
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B>;
}

/// Run a forward pass and return the greedy (argmax) action for a single observation.
///
/// Convenience function for inference and evaluation loops. Handles tensor
/// conversion internally so the caller works with plain `&[f32]` and `usize`.
///
/// # Example
///
/// ```ignore
/// let action = greedy_action(&model, &obs, &device);
/// let step = env.step(action);
/// ```
pub fn greedy_action<B: Backend, M: DiscreteActorCritic<B>>(
    model: &M,
    obs: &[f32],
    device: &B::Device,
) -> usize {
    let obs_tensor: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), device);
    let output = model.forward(obs_tensor);
    let action_data: Vec<f32> = output
        .logits
        .argmax(1)
        .float()
        .into_data()
        .to_vec()
        .unwrap();
    action_data[0] as usize
}
