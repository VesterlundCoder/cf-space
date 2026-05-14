"""
CF coefficient operators for operator-consistency loss.

Five operator types are implemented:
    0  random_perturb   — small Gaussian noise (tangent approximation)
    1  shift_poly       — exact p(n) → p(n+1) via binomial theorem
    2  scale_b          — b(n) → c·b(n), c ∼ Uniform(0.8, 1.2)
    3  apery_perturb    — perturb a₂ coefficient (Apéry-step direction)
    4  sign_flip_b      — b(n) → b(-n) via sign-flip of odd-degree coefficients

The key function is sample_operator(an, bn) which draws one type at random
and returns (an_new, bn_new, delta_coeff) where
    delta_coeff = [an_new − an | bn_new − bn]  ∈ ℝ^(n_a + n_b)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Individual operators
# ─────────────────────────────────────────────────────────────────────────────

def shift_poly_coeffs(coeff: torch.Tensor) -> torch.Tensor:
    """
    Exact polynomial shift: p(n) → p(n+1).

    If coeff[:, i] is the coefficient of n^i then
        p(n+1) = Σ_i c_i (n+1)^i = Σ_i c_i Σ_{j≤i} C(i,j) n^j

    This is the most mathematically meaningful CF operator.
    """
    B, D = coeff.shape
    out  = torch.zeros_like(coeff)
    for i in range(D):
        for j in range(i + 1):
            out[:, j] = out[:, j] + coeff[:, i] * math.comb(i, j)
    return out


def scale_coeffs(coeff: torch.Tensor, lo: float = 0.8, hi: float = 1.2) -> torch.Tensor:
    """Scale all coefficients by an independent uniform random scalar per sample."""
    c = coeff.new_empty(coeff.shape[0], 1).uniform_(lo, hi)
    return coeff * c


def apery_perturb(an: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Perturb the a₂ coefficient — tangent direction toward Apéry-like CFs."""
    an_new = an.clone()
    if an.shape[1] > 2:
        eps = torch.randn(an.shape[0], device=an.device) * sigma
        an_new[:, 2] = an_new[:, 2] + eps
    return an_new


def random_perturb(coeff: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Add small isotropic Gaussian noise — approximates a random tangent direction."""
    return coeff + torch.randn_like(coeff) * sigma


def sign_flip_b(coeff: torch.Tensor) -> torch.Tensor:
    """
    Sign-flip odd-degree coefficients: p(n) → p(-n).
    Corresponds to the symmetry b(n) → b(-n) in the CF recurrence.
    """
    out = coeff.clone()
    for i in range(1, coeff.shape[1], 2):
        out[:, i] = -out[:, i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Operator sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_operator(
    an: torch.Tensor,
    bn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw one of 5 CF operators at random and apply it to (an, bn).

    Returns:
        an_new       (B, n_a)         perturbed numerator coefficients
        bn_new       (B, n_b)         perturbed denominator coefficients
        delta_coeff  (B, n_a + n_b)   coefficient displacement [an_new−an | bn_new−bn]
    """
    op_type = int(torch.randint(0, 5, (1,)).item())

    if op_type == 0:               # random Gaussian perturbation
        an_new = random_perturb(an)
        bn_new = random_perturb(bn)

    elif op_type == 1:             # exact polynomial shift  n → n+1
        an_new = shift_poly_coeffs(an)
        bn_new = shift_poly_coeffs(bn)

    elif op_type == 2:             # scale b(n) by random scalar
        an_new = an
        bn_new = scale_coeffs(bn)

    elif op_type == 3:             # Apéry-step: perturb a₂
        an_new = apery_perturb(an)
        bn_new = bn

    else:                          # sign-flip b(n) → b(-n)
        an_new = an
        bn_new = sign_flip_b(bn)

    delta_coeff = torch.cat([an_new - an, bn_new - bn], dim=1)
    return an_new, bn_new, delta_coeff
