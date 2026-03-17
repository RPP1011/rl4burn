# Global Gradient Clipping

`clip_grad_norm` clips gradients by their **global** L2 norm across all parameters. This matches PyTorch's `torch.nn.utils.clip_grad_norm_`.

## Why not use Burn's built-in clipping?

Burn's `GradientClippingConfig::Norm` clips each parameter tensor **independently**. PyTorch clips the **global** norm across all parameters at once. These produce different behavior:

- **Per-parameter** (Burn): A large gradient in the critic doesn't affect clipping of the actor's gradient.
- **Global** (PyTorch/rl4burn): The total gradient norm is computed, then all gradients are scaled by the same factor.

For PPO with shared optimizer over actor + critic, global clipping is standard.

## API

```rust,ignore
pub fn clip_grad_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    model: &M,
    grads: GradientsParams,
    max_norm: f32,
) -> GradientsParams
```

Call between `backward()` and `optim.step()`:

```rust,ignore
let grads = loss.backward();
let mut grads = GradientsParams::from_grads(grads, &model);
grads = clip_grad_norm(&model, grads, 0.5);  // max_norm = 0.5
model = optim.step(lr, model, grads);
```

PPO handles this automatically via `PpoConfig::max_grad_norm`. Set to 0.0 to disable.

## Implementation

Two-pass approach using the inner (non-autodiff) model:

1. **ModuleVisitor**: Extract each gradient from `GradientsParams`, compute its L2 norm squared, accumulate the global norm.
2. Compute `clip_coef = min(1.0, max_norm / (global_norm + 1e-6))`.
3. **ModuleMapper**: Scale each gradient by `clip_coef` and re-register it in a new `GradientsParams`.

The visitor/mapper operate on `B::InnerBackend` because Burn stores gradients on the inner backend, not the autodiff wrapper.
