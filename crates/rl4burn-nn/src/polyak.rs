//! Polyak (soft) target network updates.
//!
//! `target = τ * source + (1 - τ) * target`
//!
//! Used by DQN, SAC, TD3, DDPG for stable target networks.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;

/// Collect all parameter tensors from a module, flattened to 1D.
struct ParamCollector<B: Backend> {
    params: Vec<Tensor<B, 1>>,
}

impl<B: Backend> ParamCollector<B> {
    fn new() -> Self {
        Self { params: Vec::new() }
    }
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        let n: usize = tensor.dims().iter().product();
        self.params.push(tensor.reshape([n]));
    }
}

/// Map each parameter tensor: target = τ * source + (1 - τ) * target.
struct PolyakMapper<B: Backend> {
    tau: f32,
    source_params: Vec<Tensor<B, 1>>,
    idx: usize,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let tensor = param.val();
        let dims = tensor.dims();
        let source = self.source_params[self.idx].clone().reshape(dims);
        self.idx += 1;
        let updated = tensor * (1.0 - self.tau) + source * self.tau;
        Param::initialized(param.id.clone(), updated)
    }
}

/// Perform a Polyak (soft) update: `target = τ * source + (1 - τ) * target`.
///
/// `tau = 1.0` copies source into target.
/// `tau = 0.0` leaves target unchanged.
///
/// Both modules must have the same architecture (same parameter shapes in
/// the same traversal order).
pub fn polyak_update<B: Backend, M: Module<B>>(target: M, source: &M, tau: f32) -> M {
    assert!(
        (0.0..=1.0).contains(&tau),
        "tau must be in [0, 1], got {tau}"
    );

    // Collect source parameters
    let mut collector = ParamCollector::<B>::new();
    source.visit(&mut collector);

    // Map target parameters
    let mut mapper = PolyakMapper {
        tau,
        source_params: collector.params,
        idx: 0,
    };
    target.map(&mut mapper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::{Linear, LinearConfig};

    type B = NdArray;

    fn make_linear(device: &<B as Backend>::Device) -> Linear<B> {
        LinearConfig::new(4, 2).init(device)
    }

    #[test]
    fn tau_zero_preserves_target() {
        let device = Default::default();
        let target = make_linear(&device);
        let source = make_linear(&device);

        let mut orig = ParamCollector::<B>::new();
        target.visit(&mut orig);

        let updated = polyak_update(target, &source, 0.0);

        let mut after = ParamCollector::<B>::new();
        updated.visit(&mut after);

        for (o, a) in orig.params.iter().zip(after.params.iter()) {
            let diff: f32 = (o.clone() - a.clone())
                .abs()
                .sum()
                .into_scalar();
            assert!(diff < 1e-6, "tau=0 should preserve target, diff={diff}");
        }
    }

    #[test]
    fn tau_one_copies_source() {
        let device = Default::default();
        let target = make_linear(&device);
        let source = make_linear(&device);

        let mut src_params = ParamCollector::<B>::new();
        source.visit(&mut src_params);

        let updated = polyak_update(target, &source, 1.0);

        let mut upd_params = ParamCollector::<B>::new();
        updated.visit(&mut upd_params);

        for (s, u) in src_params.params.iter().zip(upd_params.params.iter()) {
            let diff: f32 = (s.clone() - u.clone())
                .abs()
                .sum()
                .into_scalar();
            assert!(diff < 1e-6, "tau=1 should copy source, diff={diff}");
        }
    }
}
