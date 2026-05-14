# Dataset — Polynomial Continued Fractions (cf_large)

## Overview

The corpus contains **286,310** polynomial continued fraction (PCF) generators
sampled from a structured family of degree-4 polynomials.

## Format

Each generator is a pair `(a(n), b(n))` of polynomials with integer or
rational coefficients, defining the PCF:

```
CF(a,b) = a(0) + b(1) / (a(1) + b(2) / (a(2) + b(3) / ...))
```

Evaluated to depth 400 using mpmath (dps=50).

## Files

| File | Shape | Description |
|---|---|---|
| `an_coeffs.npy` | (286310, 10) | Coefficients of a(n), degrees 0–9 |
| `bn_coeffs.npy` | (286310, 10) | Coefficients of b(n), degrees 0–9 |
| `features.npy` | (286310, 20) | Derived numerical features |
| `labels.npy` | (286310,) | Integer cluster label (k-means, K=20) |
| `conv_exponent.npy` | (286310,) | Convergence exponent (log-rate) |
| `cf_quality.npy` | (286310,) | Quality score in [0, 1] |

## Generation

The corpus was generated with `cf_corpus_large.py` from the companion
repository. Generators are filtered by:
- Convergence at depth 400 (|CF_400 - CF_200| < 1e-6)
- Non-trivial b(n) (not identically zero)
- Coefficient magnitude bounded: all |c_i| ≤ 10

## Sampling

Coefficients drawn from:
- `a_i ~ Uniform({-3,...,3})`, degree randomly 1–4
- `b_i ~ Uniform({-3,...,3})`, degree randomly 1–4

## Access

The full corpus (~120 MB) is available as a Zenodo release accompanying the
preprint. See `paper/main.tex` for the DOI reference.

To regenerate from scratch:
```bash
python3 cf_corpus_large.py --n 300000 --out data/cf_large/
```
