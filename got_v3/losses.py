"""
All loss functions for GOT-v3.

Design notes
────────────
• supervised_contrastive_loss normalises z to the unit sphere (SimCLR-style)
  before computing similarities — prevents unbounded scale growth.
• neighborhood_preservation_loss also works on normalised z for the same reason.
• z_reg_loss (L2 on z) is always included as a soft anchor against scale drift.
• var_reg_loss (-var) encourages all k dimensions to be active — prevents
  dimension collapse (TwoNN < k target).
• operator_consistency_loss uses sample_operator (5 CF operators) instead of
  pure Gaussian noise, making the latent space operator-geometric.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from operators import sample_operator


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss functions
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_loss(coeff_hat: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(coeff_hat, coeff)


def delta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def conv_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def plateau_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target)


def mixture_loss(logits: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, component)


def z_reg_loss(z: torch.Tensor) -> torch.Tensor:
    """L2 penalty on z — keeps scale bounded (weight controlled by WEIGHT_PRESETS)."""
    return z.pow(2).mean()


def var_reg_loss(z: torch.Tensor) -> torch.Tensor:
    """
    VICReg-style anti-collapse regulariser.

    Penalises each z-dimension when its per-batch std < 1.
    Loss = mean( relu(1 - std_d) ), bounded in [0, 1].

    Unlike -z.var(0).mean(), the gradient is exactly ZERO once std ≥ 1,
    so z magnitude cannot grow unboundedly.
    Works together with z_reg (L2) which keeps magnitudes stable.
    """
    return F.relu(1.0 - z.std(0)).mean()


def neighborhood_preservation_loss(
    z:       torch.Tensor,
    coeff:   torch.Tensor,
    n_pairs: int = 2048,
) -> torch.Tensor:
    """
    Soft constraint: coeff-space distance ≈ latent-space distance.

    Works on unit-sphere-normalised z to prevent scale drift.
    """
    B = z.shape[0]
    if B < 4:
        return z.new_zeros(())

    with torch.no_grad():
        idx_i = torch.randint(0, B, (n_pairs,), device=z.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=z.device)
        d_c   = torch.norm(coeff[idx_i] - coeff[idx_j], dim=1)
        d_c   = d_c / (d_c.mean() + 1e-8)

    z_n = F.normalize(z, dim=1)
    d_z = torch.norm(z_n[idx_i] - z_n[idx_j], dim=1)
    d_z = d_z / (d_z.mean() + 1e-8)

    return F.smooth_l1_loss(d_z, d_c.detach())


def supervised_contrastive_loss(
    z:           torch.Tensor,
    component:   torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    SimCLR-style supervised contrastive loss (Khosla et al., 2020).

    z is projected to the unit sphere before similarity computation,
    which bounds distances to [0, 2] and prevents scale explosion.
    """
    B = z.shape[0]
    if B < 4:
        return z.new_zeros(())

    z_norm   = F.normalize(z, dim=1)
    sim      = torch.matmul(z_norm, z_norm.T) / temperature   # (B, B)
    eye      = torch.eye(B, device=z.device)
    mask_pos = (component.view(-1, 1) == component.view(1, -1)).float() * (1.0 - eye)

    exp_sim  = torch.exp(sim) * (1.0 - eye)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_cnt  = mask_pos.sum(dim=1).clamp(min=1.0)

    return (-(mask_pos * log_prob).sum(dim=1) / pos_cnt).mean()


def operator_consistency_loss(
    model: Any,
    an:    torch.Tensor,
    bn:    torch.Tensor,
    z:     torch.Tensor,
) -> torch.Tensor:
    """
    Enforces:  E(x + δ_op) − E(x)  ≈  F_op(z, δ_op)

    Uses sample_operator (5 CF operators) to generate algebraically
    meaningful perturbations rather than pure Gaussian noise.
    This turns z into an operator-geometric coordinate system.
    """
    an_new, bn_new, delta_coeff = sample_operator(an, bn)

    with torch.no_grad():
        _, z_target = model.encode(an_new, bn_new)

    # Normalise to unit sphere before computing the delta.
    # This caps |z_delta_true| ≤ 2 regardless of z magnitude,
    # preventing MSE explosion when OOD operators (e.g. shift) produce
    # large z_target values.  Consistent with nbr/contrast normalisation.
    z_n      = F.normalize(z, dim=1)
    zt_n     = F.normalize(z_target, dim=1)
    z_delta_true = (zt_n - z_n).detach()
    z_delta_pred = model.predict_op_vector(z_n.detach(), delta_coeff)

    return F.mse_loss(z_delta_pred, z_delta_true)


# ─────────────────────────────────────────────────────────────────────────────
# Composite loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_losses(
    model:   Any,
    batch:   Dict[str, torch.Tensor],
    out:     Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted composite loss.

    All individual losses are always evaluated (for logging); only those
    with non-zero weight contribute to the gradient.
    """
    an, bn, coeff = batch["an"], batch["bn"], batch["coeff"]
    z             = out["z"]

    raw: Dict[str, torch.Tensor] = {
        "recon":    reconstruction_loss(out["coeff_hat"], coeff),
        "nbr":      neighborhood_preservation_loss(z, coeff),
        "op":       operator_consistency_loss(model, an, bn, z),
        "contrast": supervised_contrastive_loss(z, batch["component"]),
        "mix":      mixture_loss(out["component_logits"], batch["component"]),
        "delta":    delta_loss(out["delta_pred"], batch["delta"]),
        "conv":     conv_loss(out["conv_pred"], batch["conv_rate"]),
        "plateau":  plateau_loss(out["plateau_logit"], batch["plateau"]),
        "z_reg":    z_reg_loss(z),
        "var_reg":  var_reg_loss(z),
    }

    total = sum(weights.get(k, 0.0) * v for k, v in raw.items())
    logs  = {k: float(v.detach().cpu()) for k, v in raw.items()}
    logs["total"] = float(total.detach().cpu())

    return total, logs
