//! Policy and actor-critic traits.
//!
//! These traits define the interface between RL algorithms and neural network
//! models. Users implement these on their Burn modules.

use burn::prelude::*;

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

/// A deterministic policy for inference and deployment.
///
/// Works with plain `Backend` (no autodiff needed), enabling deployment
/// on NdArray, WGPU, or WASM backends without training infrastructure.
pub trait Policy<B: Backend> {
    /// Select actions deterministically given a batch of observations.
    ///
    /// Returns action tensor. Shape depends on the action space:
    /// - Discrete: `[batch, 1]` (action indices as floats)
    /// - Continuous: `[batch, action_dim]`
    fn act(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Convenience: get the greedy (argmax) action from an actor-critic model.
pub fn greedy_action<B: Backend>(output: &DiscreteAcOutput<B>) -> Tensor<B, 2> {
    output.logits.clone().argmax(1).float()
}
