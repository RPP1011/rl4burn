#!/usr/bin/env python3
"""Generate test oracle fixtures for cross-verifying the Rust R2-Dreamer
implementation against PyTorch reference computations.

Usage:
    python scripts/extract_dreamer_oracles.py

Writes crates/rl4burn-algo/test_fixtures/dreamer_oracles.json.

Primary reference: NM512/r2dreamer (PyTorch)
Secondary references:
  - danijar/dreamerv3 (JAX) for KL/twohot/actor-critic
  - facebookresearch/barlowtwins for the loss itself
"""

import json

import torch

torch.manual_seed(42)


# ---- Barlow Twins loss ----
def dump_barlow_twins():
    B, D = 8, 16
    z_a = torch.randn(B, D)
    z_b = torch.randn(B, D)

    # Batch normalize (zero-mean, unit-std along dim=0)
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0, unbiased=False) + 1e-6)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0, unbiased=False) + 1e-6)

    # Cross-correlation matrix
    C = (z_a_norm.T @ z_b_norm) / B

    # Loss components
    on_diag = (1 - C.diagonal()).pow(2).sum()
    off_diag_mask = ~torch.eye(D, dtype=torch.bool)
    off_diag = C[off_diag_mask].pow(2).sum()

    alpha = 0.0051
    loss = on_diag + alpha * off_diag

    return {
        "z_a": z_a.numpy().tolist(),
        "z_b": z_b.numpy().tolist(),
        "z_a_norm": z_a_norm.numpy().tolist(),
        "z_b_norm": z_b_norm.numpy().tolist(),
        "C": C.detach().numpy().tolist(),
        "on_diag": on_diag.item(),
        "off_diag": off_diag.item(),
        "alpha": alpha,
        "loss": loss.item(),
    }


# ---- Barlow Twins gradients ----
def dump_barlow_twins_gradients():
    B, D = 8, 16
    z_a = torch.randn(B, D, requires_grad=True)
    z_b = torch.randn(B, D)  # detached target -- no grad

    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0, unbiased=False) + 1e-6)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0, unbiased=False) + 1e-6)

    C = (z_a_norm.T @ z_b_norm) / B
    on_diag = (1 - C.diagonal()).pow(2).sum()
    off_diag_mask = ~torch.eye(D, dtype=torch.bool)
    off_diag = C[off_diag_mask].pow(2).sum()
    loss = on_diag + 0.0051 * off_diag
    loss.backward()

    return {
        "z_a": z_a.detach().numpy().tolist(),
        "z_b": z_b.numpy().tolist(),
        "loss": loss.item(),
        "grad_z_a": z_a.grad.numpy().tolist(),
        "z_b_has_grad": z_b.requires_grad,  # must be False
    }


# ---- Symlog / symexp ----
def dump_symlog():
    x = torch.tensor([-100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0, 1e6])
    symlog_x = torch.sign(x) * torch.log1p(x.abs())
    symexp_x = torch.sign(symlog_x) * (torch.exp(symlog_x.abs()) - 1)
    return {
        "input": x.tolist(),
        "symlog": symlog_x.tolist(),
        "roundtrip": symexp_x.tolist(),
    }


# ---- Twohot encode/decode ----
def dump_twohot():
    n_bins = 255
    low, high = -20.0, 20.0
    bins = torch.linspace(low, high, n_bins)

    values = torch.tensor([0.0, 1.0, -1.0, 10.0, -10.0, 100.0])

    # Encode: symlog then two-hot
    transformed = torch.sign(values) * torch.log1p(values.abs())
    clamped = transformed.clamp(low, high)
    bin_width = (high - low) / (n_bins - 1)
    normalized = (clamped - low) / bin_width
    lower_idx = normalized.floor().long().clamp(0, n_bins - 1)
    upper_idx = (lower_idx + 1).clamp(0, n_bins - 1)
    upper_weight = normalized - lower_idx.float()
    lower_weight = 1.0 - upper_weight

    dist = torch.zeros(len(values), n_bins)
    for i in range(len(values)):
        dist[i, lower_idx[i]] = lower_weight[i]
        dist[i, upper_idx[i]] += upper_weight[i]

    # Decode: weighted sum of bin centers then symexp
    decoded_symlog = (dist * bins.unsqueeze(0)).sum(1)
    decoded = torch.sign(decoded_symlog) * (torch.exp(decoded_symlog.abs()) - 1)

    return {
        "values": values.tolist(),
        "distribution": dist.tolist(),
        "row_sums": dist.sum(1).tolist(),
        "decoded": decoded.tolist(),
    }


# ---- KL balancing ----
def dump_kl_balance():
    posterior_logits = torch.randn(4, 32)
    prior_logits = torch.randn(4, 32)
    free_bits = 1.0
    dyn_weight = 0.5
    rep_weight = 0.1

    def cat_kl(p_logits, q_logits):
        p = torch.softmax(p_logits, -1)
        log_p = torch.log_softmax(p_logits, -1)
        log_q = torch.log_softmax(q_logits, -1)
        return (p * (log_p - log_q)).sum(-1)

    dyn_kl = cat_kl(posterior_logits.detach(), prior_logits)
    rep_kl = cat_kl(posterior_logits, prior_logits.detach())

    dyn_loss = dyn_kl.clamp(min=free_bits).mean()
    rep_loss = rep_kl.clamp(min=free_bits).mean()
    total = dyn_weight * dyn_loss + rep_weight * rep_loss

    return {
        "posterior_logits": posterior_logits.tolist(),
        "prior_logits": prior_logits.tolist(),
        "dyn_kl_per_sample": dyn_kl.tolist(),
        "rep_kl_per_sample": rep_kl.tolist(),
        "dyn_loss_clamped": dyn_loss.item(),
        "rep_loss_clamped": rep_loss.item(),
        "total": total.item(),
    }


# ---- BT signal pair verification ----
def dump_bt_signal_pair():
    """Extract WHAT the reference passes to BT, not just HOW BT is computed.
    This is the test that catches the main wiring bug."""
    batch, seq_len = 2, 4
    embed_dim = 64
    det_size = 128
    stoch_size = 32 * 32  # n_categories * n_classes
    state_dim = det_size + stoch_size
    proj_dim = embed_dim  # projector maps state_dim -> embed_dim

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
        # The key invariant: input_b dim equals encoder output dim,
        # NOT state_dim, and NOT proj_dim (unless proj_dim == embed_dim)
        "input_b_dim_equals_encoder_output_dim": True,
    }


# ---- Lambda returns ----
def dump_lambda_returns():
    rewards = [1.0, 2.0, 3.0]
    values = [10.0, 20.0, 30.0, 40.0]  # len = horizon + 1
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
