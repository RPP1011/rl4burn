# Working with Burn's Autodiff

We discovered several Burn 0.20 behaviors that affect RL implementations. This chapter documents them so you don't hit the same issues.

## 1. Custom parameter initialization must use `from_data` + `load_record`

**Problem:** `Param::initialized(id, tensor)` creates parameters that are invisible to Burn's autodiff. The optimizer will silently produce zero updates — the model trains but weights never change.

**Cause:** Tensors created via `Tensor::from_data(TensorData::new(...), device)` are leaf nodes without gradient tracking. Wrapping them in `Param::initialized` doesn't register them.

**Fix:** Use Burn's record system:

```rust,ignore
// WRONG: gradients won't flow
let weight = Tensor::from_data(my_data, &device);
let param = Param::initialized(old_param.id.clone(), weight);

// RIGHT: use Param::from_data + load_record
use burn::nn::LinearRecord;

let record = LinearRecord {
    weight: Param::from_data(weight_data, &device),
    bias: Some(Param::from_data(bias_data, &device)),
};
let linear = LinearConfig::new(d_in, d_out).init(&device).load_record(record);
```

`orthogonal_linear` handles this correctly. If you implement custom initialization, follow the same pattern.

**Alternative:** If you must use `Param::initialized`, call `.set_require_grad(true)` on the result. But `from_data` + `load_record` is the canonical approach.

## 2. Gradient clipping is per-parameter, not global

**Problem:** `GradientClippingConfig::Norm(0.5)` on the optimizer clips each parameter tensor's gradient independently. PyTorch's `clip_grad_norm_` clips the global norm across all parameters.

**Impact:** With per-parameter clipping, the actor's small gradients are unaffected while the critic's large gradients are clipped. With global clipping, all gradients are scaled by the same factor, which is the standard behavior for PPO.

**Fix:** Use `clip::clip_grad_norm` instead of Burn's optimizer clipping:

```rust,ignore
use rl4burn::clip::clip_grad_norm;

// Don't configure clipping on the optimizer
let mut optim = AdamConfig::new().init();

// Clip manually between backward and step
let grads = loss.backward();
let mut grads = GradientsParams::from_grads(grads, &model);
grads = clip_grad_norm(&model, grads, 0.5);
model = optim.step(lr, model, grads);
```

PPO's `ppo_update` does this automatically when `max_grad_norm > 0`.

## 3. `mask_where` may not propagate gradients through the source argument

**Problem:** `tensor.mask_where(mask, source)` selects from `source` where mask is true, otherwise keeps `self`. Burn's autodiff may not propagate gradients through the `source` argument.

**Impact:** If you use `mask_where` to compute `max(a, b)` by selecting the larger value, and the mask happens to select from `source` for all elements, the gradient can be zero.

**Fix:** Use arithmetic alternatives:

```rust,ignore
use burn::tensor::activation::relu;

// Instead of mask_where for max:
// max(a, b) = a + relu(b - a)
let max_val = a.clone() + relu(b - a);

// Instead of mask_where for min:
// min(a, b) = b - relu(b - a)
let min_val = b.clone() - relu(b.clone() - a);
```

These have correct gradients everywhere (except at the exact boundary where `a == b`, which has measure zero in practice).

## General advice

- **Always test gradient flow.** After implementing a custom model or loss, verify that `optim.step` actually changes the model's weights. A simple test:

```rust,ignore
let before: Vec<f32> = model.weight.val().into_data().to_vec().unwrap();
// ... forward, loss, backward, step ...
let after: Vec<f32> = model.weight.val().into_data().to_vec().unwrap();
assert!(before != after, "weights should change");
```

- **Use `model.valid()` for inference.** This strips the autodiff layer, avoiding unnecessary computation graph construction during rollout collection.

- **Extract tensor data to break the computation graph.** When you need to use a value as a constant (e.g., target Q-values), call `.into_data().to_vec()` and create a fresh tensor from the result. This is equivalent to PyTorch's `.detach()`.
