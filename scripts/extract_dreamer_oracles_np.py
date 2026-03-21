#!/usr/bin/env python3
"""Generate test oracle fixtures using only numpy (no torch dependency).

Reproduces the same computations as extract_dreamer_oracles.py but uses
numpy with a fixed seed equivalent to torch.manual_seed(42).

Usage:
    python scripts/extract_dreamer_oracles_np.py
"""

import json
import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    return x - np.log(np.exp(x - x.max(axis=axis, keepdims=True)).sum(axis=axis, keepdims=True)) - x.max(axis=axis, keepdims=True) + x.max(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m
    return x - lse


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1.0)


# Use a fixed RNG for reproducibility
rng = np.random.default_rng(42)


# ---- Barlow Twins loss ----
def dump_barlow_twins():
    B, D = 8, 16
    z_a = rng.standard_normal((B, D)).astype(np.float64)
    z_b = rng.standard_normal((B, D)).astype(np.float64)

    # Batch normalize (zero-mean, unit-std along dim=0)
    # Using biased std (ddof=0) to match PyTorch std(unbiased=False)
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0, ddof=0) + 1e-6)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0, ddof=0) + 1e-6)

    # Cross-correlation matrix
    C = (z_a_norm.T @ z_b_norm) / B

    # Loss components
    on_diag = ((1 - np.diag(C)) ** 2).sum()
    mask = ~np.eye(D, dtype=bool)
    off_diag = (C[mask] ** 2).sum()

    alpha = 0.0051
    loss = on_diag + alpha * off_diag

    return {
        "z_a": z_a.tolist(),
        "z_b": z_b.tolist(),
        "z_a_norm": z_a_norm.tolist(),
        "z_b_norm": z_b_norm.tolist(),
        "C": C.tolist(),
        "on_diag": float(on_diag),
        "off_diag": float(off_diag),
        "alpha": alpha,
        "loss": float(loss),
    }


# ---- Barlow Twins gradients ----
def dump_barlow_twins_gradients():
    """Compute BT gradient w.r.t. z_a numerically via finite differences."""
    B, D = 8, 16
    z_a = rng.standard_normal((B, D)).astype(np.float64)
    z_b = rng.standard_normal((B, D)).astype(np.float64)

    def bt_loss(za, zb):
        za_n = (za - za.mean(0)) / (za.std(0, ddof=0) + 1e-6)
        zb_n = (zb - zb.mean(0)) / (zb.std(0, ddof=0) + 1e-6)
        c = (za_n.T @ zb_n) / B
        on = ((1 - np.diag(c)) ** 2).sum()
        mask = ~np.eye(D, dtype=bool)
        off = (c[mask] ** 2).sum()
        return on + 0.0051 * off

    loss = bt_loss(z_a, z_b)

    # Numerical gradient
    eps = 1e-5
    grad_z_a = np.zeros_like(z_a)
    for i in range(B):
        for j in range(D):
            z_a_p = z_a.copy()
            z_a_p[i, j] += eps
            z_a_m = z_a.copy()
            z_a_m[i, j] -= eps
            grad_z_a[i, j] = (bt_loss(z_a_p, z_b) - bt_loss(z_a_m, z_b)) / (2 * eps)

    return {
        "z_a": z_a.tolist(),
        "z_b": z_b.tolist(),
        "loss": float(loss),
        "grad_z_a": grad_z_a.tolist(),
        "z_b_has_grad": False,
    }


# ---- Symlog / symexp ----
def dump_symlog():
    x = np.array([-100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0, 1e6])
    sl = symlog(x)
    rt = symexp(sl)
    return {
        "input": x.tolist(),
        "symlog": sl.tolist(),
        "roundtrip": rt.tolist(),
    }


# ---- Twohot encode/decode ----
def dump_twohot():
    n_bins = 255
    low, high = -20.0, 20.0
    bins = np.linspace(low, high, n_bins)

    values = np.array([0.0, 1.0, -1.0, 10.0, -10.0, 100.0])

    transformed = symlog(values)
    clamped = np.clip(transformed, low, high)
    bin_width = (high - low) / (n_bins - 1)
    normalized = (clamped - low) / bin_width
    lower_idx = np.clip(np.floor(normalized).astype(int), 0, n_bins - 1)
    upper_idx = np.clip(lower_idx + 1, 0, n_bins - 1)
    upper_weight = normalized - lower_idx.astype(float)
    lower_weight = 1.0 - upper_weight

    dist = np.zeros((len(values), n_bins))
    for i in range(len(values)):
        dist[i, lower_idx[i]] = lower_weight[i]
        dist[i, upper_idx[i]] += upper_weight[i]

    decoded_symlog = (dist * bins[np.newaxis, :]).sum(1)
    decoded = symexp(decoded_symlog)

    return {
        "values": values.tolist(),
        "distribution": dist.tolist(),
        "row_sums": dist.sum(1).tolist(),
        "decoded": decoded.tolist(),
    }


# ---- KL balancing ----
def dump_kl_balance():
    posterior_logits = rng.standard_normal((4, 32)).astype(np.float64)
    prior_logits = rng.standard_normal((4, 32)).astype(np.float64)
    free_bits = 1.0
    dyn_weight = 0.5
    rep_weight = 0.1

    def cat_kl(p_logits, q_logits):
        p = softmax(p_logits, axis=-1)
        log_p = log_softmax(p_logits, axis=-1)
        log_q = log_softmax(q_logits, axis=-1)
        return (p * (log_p - log_q)).sum(-1)

    dyn_kl = cat_kl(posterior_logits, prior_logits)
    rep_kl = cat_kl(posterior_logits, prior_logits)

    dyn_loss = np.clip(dyn_kl, free_bits, None).mean()
    rep_loss = np.clip(rep_kl, free_bits, None).mean()
    total = dyn_weight * dyn_loss + rep_weight * rep_loss

    return {
        "posterior_logits": posterior_logits.tolist(),
        "prior_logits": prior_logits.tolist(),
        "dyn_kl_per_sample": dyn_kl.tolist(),
        "rep_kl_per_sample": rep_kl.tolist(),
        "dyn_loss_clamped": float(dyn_loss),
        "rep_loss_clamped": float(rep_loss),
        "total": float(total),
    }


# ---- BT signal pair verification ----
def dump_bt_signal_pair():
    batch, seq_len = 2, 4
    embed_dim = 64
    det_size = 128
    stoch_size = 32 * 32
    state_dim = det_size + stoch_size
    proj_dim = embed_dim

    return {
        "bt_input_a_source": "projector(concat(h_t, z_t))",
        "bt_input_a_shape": [batch * seq_len, proj_dim],
        "bt_input_a_requires_grad": True,
        "bt_input_b_source": "encoder(obs_t).detach()",
        "bt_input_b_shape": [batch * seq_len, embed_dim],
        "bt_input_b_requires_grad": False,
        "bt_computed_over": "B*T samples jointly, not per-timestep",
        "projector_type": "linear (no activation, no norm)",
        "embed_dim": embed_dim,
        "state_dim": state_dim,
        "proj_dim": proj_dim,
        "input_b_dim_equals_encoder_output_dim": True,
    }


# ---- Lambda returns ----
def dump_lambda_returns():
    rewards = [1.0, 2.0, 3.0]
    values = [10.0, 20.0, 30.0, 40.0]
    continues = [1.0, 1.0, 0.5]
    gamma = 0.99
    lam = 0.95
    horizon = 3

    returns = [0.0] * horizon
    returns[-1] = rewards[-1] + gamma * continues[-1] * (
        (1 - lam) * values[-1] + lam * values[horizon]
    )
    for t in range(horizon - 2, -1, -1):
        target = (1 - lam) * values[t + 1] + lam * returns[t + 1]
        returns[t] = rewards[t] + gamma * continues[t] * target

    return {
        "rewards": rewards,
        "values": values,
        "continues": continues,
        "gamma": gamma,
        "lambda": lam,
        "returns": returns,
    }


# ---- Assemble and write ----
oracles = {
    "barlow_twins": dump_barlow_twins(),
    "barlow_twins_gradients": dump_barlow_twins_gradients(),
    "symlog": dump_symlog(),
    "twohot": dump_twohot(),
    "kl_balance": dump_kl_balance(),
    "bt_signal_pair": dump_bt_signal_pair(),
    "lambda_returns": dump_lambda_returns(),
}

with open("crates/rl4burn-algo/test_fixtures/dreamer_oracles.json", "w") as f:
    json.dump(oracles, f, indent=2)

print(f"Wrote {len(oracles)} oracle groups")
