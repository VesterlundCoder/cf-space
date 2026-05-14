"""
got_v3/cf_algebra.py
────────────────────
High-level algebra operations on CF latent vectors.

Three primitive operations
──────────────────────────
1. encode(an, bn) → z ∈ ℝ^k
   Position of a CF in the geometric latent space.

2. op_vector(model, z, an, bn, operator) → Δz ∈ ℝ^k
   Predicted movement in z-space caused by applying algebraic operator T.
   This is a *vector field*: its direction and magnitude depend on the
   current position z (i.e., operators are nonlinear in general, but the
   model learns a local linearisation).

3. cosine_sim(z1, z2) → scalar ∈ [-1, 1]
   Angular similarity between two CF positions.
   Uses the same unit-sphere normalisation as the contrastive loss.

Derived operations
──────────────────
• step(z, Δz)            — move one operator step:  z + Δz
• interpolate(z1, z2, t) — slerp between two CFs
• direction(z1, z2)      — unit vector from CF1 toward CF2
• orbit(model, an, bn, T, n_steps) — trace n-step operator trajectory
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from operators import (
    apery_perturb,
    random_perturb,
    scale_coeffs,
    shift_poly_coeffs,
    sign_flip_b,
    sample_operator,
)


def _wrap(fn_an=None, fn_bn=None):
    """Build (an, bn) → (an_new, bn_new, delta_coeff) wrapper for a per-array op."""
    def _op(an: torch.Tensor, bn: torch.Tensor):
        an_new = fn_an(an) if fn_an is not None else an
        bn_new = fn_bn(bn) if fn_bn is not None else bn
        return an_new, bn_new, torch.cat([an_new - an, bn_new - bn], dim=1)
    return _op


_OPERATOR_MAP: dict[str, Callable] = {
    "shift":  _wrap(shift_poly_coeffs, shift_poly_coeffs),
    "scale":  _wrap(fn_bn=scale_coeffs),
    "apery":  _wrap(fn_an=apery_perturb),
    "sign":   _wrap(fn_bn=sign_flip_b),
    "random": _wrap(random_perturb, random_perturb),
    "sample": sample_operator,   # random operator chosen each call
}


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode(model, an: torch.Tensor, bn: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of CFs to geometric latent vectors z ∈ ℝ^k.

    Args:
        model: GOTv3 instance (eval mode recommended)
        an:    (B, n_a) normalised a-coefficients
        bn:    (B, n_b) normalised b-coefficients
    Returns:
        z:     (B, k)
    """
    model.eval()
    _, z = model.encode(an.to(next(model.parameters()).device),
                        bn.to(next(model.parameters()).device))
    return z


@torch.no_grad()
def op_vector(
    model,
    z:        torch.Tensor,
    an:       torch.Tensor,
    bn:       torch.Tensor,
    operator: str = "shift",
) -> torch.Tensor:
    """
    Compute Δz — the predicted movement in latent space caused by applying
    algebraic operator T to (an, bn).

    The op_field is a learned vector field:
        F_T : ℝ^k × ℝ^n_tokens → ℝ^k
        Δz = F_T(z, Δcoeff)

    Args:
        model:    GOTv3 instance
        z:        (B, k)  current latent positions
        an:       (B, n_a)
        bn:       (B, n_b)
        operator: one of "shift", "scale", "apery", "sign", "random"
    Returns:
        dz: (B, k) predicted displacement vectors
    """
    fn = _OPERATOR_MAP[operator]
    an_new, bn_new, delta_coeff = fn(an, bn)
    device = next(model.parameters()).device
    delta_coeff = delta_coeff.to(device)
    z = z.to(device)
    return model.predict_op_vector(z, delta_coeff)


def cosine_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between CF latent vectors (unit-sphere normalised).

    Args:
        z1: (B, k) or (k,)
        z2: (B, k) or (k,)
    Returns:
        sim: (B,) or scalar in [-1, 1]
    """
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    if z1n.dim() == 1:
        return (z1n * z2n).sum()
    return (z1n * z2n).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Derived operations
# ──────────────────────────────────────────────────────────────────────────────

def step(z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """Move one operator step: z_new = z + Δz."""
    return z + dz


def direction(z_from: torch.Tensor, z_to: torch.Tensor) -> torch.Tensor:
    """Unit vector pointing from z_from toward z_to."""
    d = z_to - z_from
    return F.normalize(d, dim=-1)


def distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between two latent positions."""
    return (z1 - z2).norm(dim=-1)


def slerp(
    z1: torch.Tensor,
    z2: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """
    Spherical linear interpolation between two z vectors.

    t=0 → z1, t=1 → z2.
    Preserves the norm (interpolates on the z-magnitude sphere).
    """
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    cos_theta = (z1n * z2n).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    r1 = torch.sin((1 - t) * theta) / sin_theta
    r2 = torch.sin(t * theta) / sin_theta
    norm = (1 - t) * z1.norm(dim=-1, keepdim=True) + t * z2.norm(dim=-1, keepdim=True)
    return (r1 * z1 + r2 * z2) * norm / (r1 * z1 + r2 * z2).norm(dim=-1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def orbit(
    model,
    an:       torch.Tensor,
    bn:       torch.Tensor,
    operator: str = "shift",
    n_steps:  int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trace n operator steps from (an, bn) in latent space.

    Each step:  z_{t+1} = z_t + F_T(z_t, Δcoeff_t)

    Returns:
        zs:    (n_steps+1, k) trajectory of latent positions
        steps: (n_steps, k)  displacement vectors at each step
    """
    model.eval()
    device = next(model.parameters()).device
    an = an.to(device); bn = bn.to(device)

    z = encode(model, an, bn)   # (1, k)
    zs     = [z]
    steps  = []

    for _ in range(n_steps):
        dz = op_vector(model, z, an, bn, operator)
        z  = step(z, dz)
        zs.append(z)
        steps.append(dz)

        # apply operator to coefficients for next step
        fn = _OPERATOR_MAP[operator]
        an, bn, _ = fn(an, bn)

    return torch.cat(zs, dim=0), torch.cat(steps, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo (run as script)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import argparse
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/got_v3_ae_pretrain_k3.pt")
    ap.add_argument("--data", default="data/cf_large")
    args = ap.parse_args()

    # ── load model ──────────────────────────────────────────────────────────
    from models import GOTv3
    state = torch.load(args.ckpt, map_location="cpu")
    a_cfg = state.get("args", {})
    model = GOTv3(
        n_a=5, n_b=5,
        k=a_cfg.get("k", 3),
        d_model=a_cfg.get("d_model", 128),
        n_heads=a_cfg.get("heads", 8),
        n_layers=a_cfg.get("layers", 4),
        n_components=a_cfg.get("n_components", 20),
    )
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()

    # ── load two CFs ────────────────────────────────────────────────────────
    data_dir = Path(args.data)
    an_raw = np.load(data_dir / "an_coeffs.npy")[:, :5].astype(np.float32)
    bn_raw = np.load(data_dir / "bn_coeffs.npy")[:, :5].astype(np.float32)

    # normalise
    an_mu, an_sd = an_raw.mean(0), an_raw.std(0) + 1e-6
    bn_mu, bn_sd = bn_raw.mean(0), bn_raw.std(0) + 1e-6
    an_n = (an_raw - an_mu) / an_sd
    bn_n = (bn_raw - bn_mu) / bn_sd

    cf_A = torch.from_numpy(an_n[[0]]), torch.from_numpy(bn_n[[0]])
    cf_B = torch.from_numpy(an_n[[1]]), torch.from_numpy(bn_n[[1]])

    zA = encode(model, *cf_A)
    zB = encode(model, *cf_B)

    print("\n── CF positions ─────────────────────────────────────────────")
    print(f"  z_A = {zA[0].numpy().round(4)}")
    print(f"  z_B = {zB[0].numpy().round(4)}")

    print("\n── Cosine similarity ────────────────────────────────────────")
    sim = cosine_sim(zA, zB)
    print(f"  cos(z_A, z_B) = {float(sim):.4f}")

    print("\n── Direction vector A → B ───────────────────────────────────")
    d = direction(zA, zB)
    print(f"  dir(A→B) = {d[0].numpy().round(4)}  (unit vector)")

    print("\n── Operator vectors at z_A ──────────────────────────────────")
    for op_name in ["shift", "scale", "apery", "sign"]:
        dz = op_vector(model, zA, *cf_A, operator=op_name)
        print(f"  Δz({op_name:6s}) = {dz[0].numpy().round(4)}"
              f"  |Δz|={dz.norm().item():.4f}  "
              f"  cos(Δz, dir_AB)={float(cosine_sim(dz, d)):.3f}")

    print("\n── Orbit (5 shift steps from A) ─────────────────────────────")
    zs, steps = orbit(model, *cf_A, operator="shift", n_steps=5)
    for i, (z_i, dz_i) in enumerate(zip(zs, steps)):
        print(f"  t={i}  z={z_i.numpy().round(3)}  |step|={dz_i.norm().item():.4f}")
    print(f"  t=5  z={zs[-1].numpy().round(3)}  (final)")
