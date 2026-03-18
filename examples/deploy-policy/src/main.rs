//! Deploying trained policies — export and run inference without training deps.
//!
//! This example demonstrates the full train-then-deploy workflow:
//!
//! 1. **Train**: Run PPO on CartPole for a short training run.
//! 2. **Save**: Serialize the model using burn's `CompactRecorder`.
//! 3. **Load**: Load the model back using only the inference backend (no autodiff).
//! 4. **Deploy**: Run greedy evaluation episodes with `greedy_action`.
//!
//! ## Key concepts
//!
//! ### `model.valid()` — stripping the autodiff graph
//!
//! During training, models use an `Autodiff<B>` backend that tracks operations
//! for gradient computation. For inference, you strip this overhead by calling
//! `model.valid()`, which returns the same model on the inner backend `B`.
//! This is more efficient and what you'd use in production.
//!
//! ### Burn's recorder system
//!
//! Burn provides `Module::save_file` and `Module::load_file` for serialization.
//! The recorder type controls the format:
//!
//! - `CompactRecorder` — MessagePack (`.mpk`). Small files, fast I/O.
//!   Best for production deployment.
//! - `PrettyJsonFileRecorder` — Human-readable JSON. Good for debugging
//!   or inspecting weights, but much larger files.
//! - `BinFileRecorder` — Raw binary. Maximum speed, no compression.
//!
//! The model architecture must match between save and load — the recorder
//! stores parameter values, not the network structure.
//!
//! ### GPU inference with WGPU
//!
//! To deploy on GPU, just swap the backend type:
//! ```ignore
//! use burn::backend::Wgpu;
//! type InferB = Wgpu;
//! let device = burn::backend::wgpu::WgpuDevice::default();
//! let model: Model<InferB> = Model::new(&device)
//!     .load_file("trained_model", &CompactRecorder::new(), &device)
//!     .unwrap();
//! ```
//! The saved weights are backend-agnostic — train on CPU with NdArray,
//! deploy on GPU with WGPU (or vice versa).
//!
//! ### WASM deployment
//!
//! Burn's WGPU backend compiles to WebAssembly, enabling browser-based
//! inference. The workflow is the same: load weights into a WGPU-backed
//! model, then run forward passes. The `.mpk` file can be served as a
//! static asset. See burn's WASM examples for the full setup.
//!
//! Run: `cargo run --release -p deploy-policy`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    greedy_action, ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, Env, PpoConfig,
    SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Model — identical architecture used for both training and deployment.
//
// In a real project, you'd define this once in a shared crate so that the
// training binary and the inference binary both use the same struct.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Model<B: Backend> {
    actor1: Linear<B>,
    actor2: Linear<B>,
    actor_out: Linear<B>,
    critic1: Linear<B>,
    critic2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> Model<B> {
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

impl<B: Backend> DiscreteActorCritic<B> for Model<B> {
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
// Phase 1: Training
// ---------------------------------------------------------------------------

type TrainB = Autodiff<NdArray>;

/// Train a CartPole agent and return the trained model (on the inference backend).
fn train(device: &NdArrayDevice) -> Model<NdArray> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs: usize = 4;
    let envs: Vec<_> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: Model<TrainB> = Model::new(device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new();

    let total_steps = 50_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent: Vec<f32> = Vec::new();

    eprintln!("=== Phase 1: Training ===");
    eprintln!("Training PPO on CartPole ({total_steps} steps, {n_envs} envs)");
    eprintln!("{:-<60}", "");

    for iter in 0..n_iters {
        let lr = config.lr * (1.0 - iter as f64 / n_iters as f64);
        let rollout = ppo_collect::<NdArray, _, _>(
            &model.valid(),
            &mut vec_env,
            &config,
            device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );
        recent.extend_from_slice(&rollout.episode_returns);
        if recent.len() > 20 {
            recent = recent[recent.len() - 20..].to_vec();
        }

        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, lr, device, &mut rng);

        if !recent.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            eprintln!(
                "step {:>6} | avg return {:>6.1} | policy_loss {:.4} | entropy {:.4}",
                (iter + 1) * steps_per_iter,
                avg,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    if !recent.is_empty() {
        let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        eprintln!("Training avg return (last 20 eps): {avg:.1}");
    }

    // Strip the autodiff graph. `model.valid()` returns `Model<NdArray>` —
    // the same weights but without gradient tracking overhead.
    // This is what you'd use in production inference.
    model.valid()
}

// ---------------------------------------------------------------------------
// Phase 2: Save the model
// ---------------------------------------------------------------------------

fn save_model(model: Model<NdArray>, path: &str) {
    eprintln!();
    eprintln!("=== Phase 2: Saving ===");

    // `save_file` serializes all learnable parameters to disk.
    // CompactRecorder uses MessagePack format — small and fast.
    // The file extension `.mpk` is added automatically.
    //
    // Other recorder options:
    //   PrettyJsonFileRecorder — human-readable, good for debugging
    //   BinFileRecorder — raw binary, maximum speed
    model
        .save_file(path, &CompactRecorder::new())
        .expect("Failed to save model");

    eprintln!("Model saved to {path}.mpk");
    eprintln!("(CompactRecorder uses MessagePack — small file, fast I/O)");
}

// ---------------------------------------------------------------------------
// Phase 3: Load for inference (no autodiff!)
// ---------------------------------------------------------------------------

/// The inference backend — just NdArray, no Autodiff wrapper.
/// This is lighter weight and what you'd use in production.
type InferB = NdArray;

fn load_model(path: &str, device: &NdArrayDevice) -> Model<InferB> {
    eprintln!();
    eprintln!("=== Phase 3: Loading for inference ===");

    // Initialize a fresh model with the same architecture, then load weights.
    // The architecture must match exactly — load_file fills in parameter values,
    // it does not reconstruct the network structure.
    //
    // Note: we use `NdArray` directly (no `Autodiff` wrapper). This is the key
    // deployment optimization — no gradient tracking overhead.
    //
    // The saved weights are backend-agnostic. You could load them into a
    // `Wgpu`-backed model for GPU inference:
    //   let model: Model<Wgpu> = Model::new(&wgpu_device)
    //       .load_file(path, &CompactRecorder::new(), &wgpu_device)?;
    let model: Model<InferB> = Model::new(device)
        .load_file(path, &CompactRecorder::new(), device)
        .expect("Failed to load model");

    eprintln!("Model loaded from {path}.mpk (inference-only backend, no autodiff)");
    model
}

// ---------------------------------------------------------------------------
// Phase 4: Greedy evaluation
// ---------------------------------------------------------------------------

fn evaluate(model: &Model<InferB>, device: &NdArrayDevice, n_episodes: usize) {
    eprintln!();
    eprintln!("=== Phase 4: Deployment (greedy evaluation) ===");
    eprintln!("Running {n_episodes} episodes with greedy (argmax) policy...");
    eprintln!("{:-<60}", "");

    let mut total_return = 0.0f32;
    let mut total_steps = 0usize;

    for ep in 0..n_episodes {
        let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(2000 + ep as u64));
        let mut obs = env.reset();
        let mut ep_return = 0.0f32;
        let mut steps = 0;

        loop {
            // `greedy_action` runs a forward pass and returns argmax(logits).
            // This is deterministic — no sampling, no exploration noise.
            // Perfect for deployment where you want the best known action.
            let action = greedy_action(model, &obs, device);
            let step = env.step(action);
            ep_return += step.reward;
            steps += 1;

            if step.done() {
                break;
            }
            obs = step.observation;
        }

        eprintln!("  Episode {ep:>2}: return = {ep_return:>6.1}, steps = {steps}");
        total_return += ep_return;
        total_steps += steps;
    }

    eprintln!("{:-<60}", "");
    let avg_return = total_return / n_episodes as f32;
    let avg_steps = total_steps as f32 / n_episodes as f32;
    eprintln!("Average return: {avg_return:.1}");
    eprintln!("Average steps:  {avg_steps:.1}");

    if avg_return > 200.0 {
        eprintln!("Policy performs well! Ready for production deployment.");
    } else {
        eprintln!("Policy needs more training (50k steps is short for CartPole).");
        eprintln!("Try increasing total_steps in the train phase.");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let device = NdArrayDevice::Cpu;
    let model_path = "/tmp/rl4burn_trained_model";

    // Phase 1: Train the model (short run for demonstration).
    let trained_model = train(&device);

    // Phase 2: Save the trained model to disk.
    save_model(trained_model, model_path);

    // Phase 3: Load the model back using inference-only backend.
    // In a real deployment, phases 1-2 and 3-4 would be separate binaries.
    // The inference binary only needs burn (no autodiff feature) and your
    // model definition.
    let deployed_model = load_model(model_path, &device);

    // Phase 4: Run greedy evaluation episodes.
    evaluate(&deployed_model, &device, 10);

    // Cleanup the temp file.
    let _ = std::fs::remove_file(format!("{model_path}.mpk"));
}
