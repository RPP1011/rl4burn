//! Global gradient norm clipping matching PyTorch's `clip_grad_norm_`.
//!
//! Burn's built-in `GradientClippingConfig::Norm` clips per-parameter,
//! not globally. This module provides PyTorch-compatible global clipping.

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

/// Clip gradients by global L2 norm, matching PyTorch's `clip_grad_norm_`.
///
/// Computes the L2 norm across ALL parameters. If it exceeds `max_norm`,
/// all gradients are scaled by `max_norm / (global_norm + 1e-6)`.
pub fn clip_grad_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    model: &M,
    grads: GradientsParams,
    max_norm: f32,
) -> GradientsParams {
    // Phase 1: Extract gradients and compute global norm.
    // Gradients in GradientsParams are stored on B::InnerBackend.
    let inner_model = model.valid();
    let mut collector = GradNormCalc {
        grads,
        global_norm_sq: 0.0,
        grad_data: Vec::new(),
    };
    inner_model.visit(&mut collector);

    let global_norm = collector.global_norm_sq.sqrt();
    let clip_coef = (max_norm / (global_norm + 1e-6)).min(1.0);

    if clip_coef >= 1.0 {
        return collector.grads;
    }

    // Phase 2: Re-register scaled gradients with correct dimensions.
    let mut scaler = GradScaler {
        grads: collector.grads,
        clip_coef,
        grad_data: collector.grad_data,
    };
    inner_model.map(&mut scaler);
    scaler.grads
}

/// Phase 1: Extract gradients, compute L2 norms.
struct GradNormCalc {
    grads: GradientsParams,
    global_norm_sq: f32,
    grad_data: Vec<(ParamId, TensorData)>,
}

impl<B: Backend> ModuleVisitor<B> for GradNormCalc {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let id = param.id.clone();
        if let Some(grad) = self.grads.remove::<B, D>(id.clone()) {
            let data = grad.into_data();
            let flat: Vec<f32> = data.to_vec::<f32>().unwrap();
            let norm_sq: f32 = flat.iter().map(|x| x * x).sum();
            self.global_norm_sq += norm_sq;
            self.grad_data.push((id, data));
        }
    }
}

/// Phase 2: Re-register scaled gradients with correct dimensions.
struct GradScaler {
    grads: GradientsParams,
    clip_coef: f32,
    grad_data: Vec<(ParamId, TensorData)>,
}

impl<B: Backend> ModuleMapper<B> for GradScaler {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let id = param.id.clone();
        if let Some(pos) = self.grad_data.iter().position(|(gid, _)| *gid == id) {
            let (_, data) = self.grad_data.remove(pos);
            let scaled: Vec<f32> = data
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| v * self.clip_coef)
                .collect();
            let shape = data.shape;
            let device = param.val().device();
            let grad_tensor: Tensor<B, D> =
                Tensor::from_data(TensorData::new(scaled, shape), &device);
            self.grads.register::<B, D>(id, grad_tensor);
        }
        param
    }
}
