# Dual-Clip PPO

An extension of standard PPO used by JueWu and Honor of Kings for distributed training stability.

## The problem

In distributed RL, the behavior policy can be several updates behind. When the ratio `pi_new/pi_old` is very large and the advantage is negative, standard PPO's objective becomes excessively negative, causing destructive updates.

## The fix

Add a floor: when advantage < 0, the objective can't go below `c * advantage` (c = 3):

```
standard_ppo = min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)
dual_clip    = max(standard_ppo, c * adv)    // only when adv < 0
```

## Usage

```rust,ignore
let config = PpoConfig {
    dual_clip_coef: Some(3.0),
    ..Default::default()
};
```

That's it. Set `dual_clip_coef: None` (the default) for standard PPO.

## When to use

Only needed for distributed/asynchronous training where trajectories may be significantly off-policy. For single-machine training, standard PPO is sufficient.
