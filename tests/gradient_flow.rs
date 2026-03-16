//! Minimal test: does Burn's autodiff propagate gradients through
//! log_softmax → gather → policy gradient loss → backward → optimizer step?

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::TensorData;

use rl4burn::init::orthogonal_linear;
use rl4burn::policy::{DiscreteAcOutput, DiscreteActorCritic};

type AB = Autodiff<NdArray>;

// Exact same Agent struct from ppo_cartpole test
#[derive(Module, Debug)]
struct Agent<B: Backend> {
    actor_fc1: Linear<B>,
    actor_fc2: Linear<B>,
    actor_out: Linear<B>,
    critic_fc1: Linear<B>,
    critic_fc2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> DiscreteActorCritic<B> for Agent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        let a = self.actor_fc1.forward(obs.clone()).tanh();
        let a = self.actor_fc2.forward(a).tanh();
        let logits = self.actor_out.forward(a);
        let c = self.critic_fc1.forward(obs).tanh();
        let c = self.critic_fc2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);
        DiscreteAcOutput { logits, values }
    }
}

#[test]
fn agent_gradient_flows() {
    use rand::SeedableRng;
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let sqrt2 = std::f32::consts::SQRT_2;

    let model: Agent<AB> = Agent {
        actor_fc1: orthogonal_linear(4, 64, sqrt2, &device, &mut rng),
        actor_fc2: orthogonal_linear(64, 64, sqrt2, &device, &mut rng),
        actor_out: orthogonal_linear(64, 2, 0.01, &device, &mut rng),
        critic_fc1: orthogonal_linear(4, 64, sqrt2, &device, &mut rng),
        critic_fc2: orthogonal_linear(64, 64, sqrt2, &device, &mut rng),
        critic_out: orthogonal_linear(64, 1, 1.0, &device, &mut rng),
    };

    let actor_before: f32 = model.actor_out.weight.val().into_data().to_vec::<f32>().unwrap()
        .iter().map(|x| x*x).sum();
    let critic_before: f32 = model.critic_out.weight.val().into_data().to_vec::<f32>().unwrap()
        .iter().map(|x| x*x).sum();

    let obs: Tensor<AB, 2> = Tensor::from_data(
        TensorData::new(vec![0.1f32, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.4], [2, 4]),
        &device,
    );

    let output = model.forward(obs);
    let loss = output.logits.sum() + output.values.sum();

    eprintln!("agent loss = {:?}", loss.clone().into_data().to_vec::<f32>());

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    let mut optim = AdamConfig::new().init();
    let model = optim.step(1e-3, model, grads);

    let actor_after: f32 = model.actor_out.weight.val().into_data().to_vec::<f32>().unwrap()
        .iter().map(|x| x*x).sum();
    let critic_after: f32 = model.critic_out.weight.val().into_data().to_vec::<f32>().unwrap()
        .iter().map(|x| x*x).sum();

    eprintln!("agent: actor_out L2 before={actor_before:.8} after={actor_after:.8}");
    eprintln!("agent: critic_out L2 before={critic_before:.8} after={critic_after:.8}");

    assert!((actor_after - actor_before).abs() > 1e-8, "actor_out should change");
    assert!((critic_after - critic_before).abs() > 1e-8, "critic_out should change");
}

#[test]
fn policy_gradient_updates_weights() {
    let device = NdArrayDevice::Cpu;
    let model: Linear<AB> = LinearConfig::new(4, 2).init(&device);
    let orig_weight: Vec<f32> = model.weight.val().into_data().to_vec().unwrap();

    let obs: Tensor<AB, 2> = Tensor::from_data(
        TensorData::new(vec![0.1f32, -0.2, 0.3, -0.1], [1, 4]), &device);
    let advantage: Tensor<AB, 1> =
        Tensor::from_data(TensorData::new(vec![1.0f32], [1]), &device);
    let action: Tensor<AB, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32], [1, 1]), &device);

    let logits = model.forward(obs);
    let log_probs = log_softmax(logits, 1);
    let action_lp: Tensor<AB, 1> = log_probs.gather(1, action).squeeze_dim::<1>(1);
    let loss: Tensor<AB, 1> = (action_lp * advantage).neg().mean().unsqueeze();

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let model = optim.step(1e-3, model, grads);

    let new_weight: Vec<f32> = model.weight.val().into_data().to_vec().unwrap();
    let max_diff: f32 = orig_weight.iter().zip(&new_weight).map(|(a, b)| (a-b).abs()).fold(0.0, f32::max);
    eprintln!("max weight diff = {max_diff}");
    assert!(max_diff > 1e-6);
}
