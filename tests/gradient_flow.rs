//! Regression test: verify Burn's autodiff tracks gradients through
//! orthogonal_linear (Param::from_data + load_record).

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::Linear;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::init::orthogonal_linear;
use rl4burn::policy::{DiscreteAcOutput, DiscreteActorCritic};

type AB = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct Agent<B: Backend> {
    actor_fc1: Linear<B>,
    actor_out: Linear<B>,
    critic_fc1: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> DiscreteActorCritic<B> for Agent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        let logits = self.actor_out.forward(self.actor_fc1.forward(obs.clone()).tanh());
        let values = self
            .critic_out
            .forward(self.critic_fc1.forward(obs).tanh())
            .squeeze_dim::<1>(1);
        DiscreteAcOutput { logits, values }
    }
}

#[test]
fn orthogonal_init_gradients_flow() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let sqrt2 = std::f32::consts::SQRT_2;

    let model: Agent<AB> = Agent {
        actor_fc1: orthogonal_linear(4, 16, sqrt2, &device, &mut rng),
        actor_out: orthogonal_linear(16, 2, 0.01, &device, &mut rng),
        critic_fc1: orthogonal_linear(4, 16, sqrt2, &device, &mut rng),
        critic_out: orthogonal_linear(16, 1, 1.0, &device, &mut rng),
    };

    let before: Vec<f32> = model
        .actor_out
        .weight
        .val()
        .into_data()
        .to_vec()
        .unwrap();

    let obs: Tensor<AB, 2> = Tensor::from_data(
        burn::tensor::TensorData::new(vec![0.1f32, -0.2, 0.3, -0.1], [1, 4]),
        &device,
    );
    let output = model.forward(obs);
    let loss = output.logits.sum() + output.values.sum();

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let mut optim = AdamConfig::new().init();
    let model = optim.step(1e-3, model, grads);

    let after: Vec<f32> = model
        .actor_out
        .weight
        .val()
        .into_data()
        .to_vec()
        .unwrap();
    let diff: f32 = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    assert!(
        diff > 1e-6,
        "orthogonal_linear weights must update via autodiff, diff={diff}"
    );
}
