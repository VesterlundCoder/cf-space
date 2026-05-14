# cf-space — Empirical Operator Algebra on a Latent Manifold of Polynomial Continued Fractions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data companion for the preprint:

> **An Empirical Operator Algebra on a Latent Manifold of Polynomial Continued Fractions**  
> David Svensson, 2026  
> [`paper/main.pdf`](paper/main.pdf)

---

## Overview

We construct a finite-dimensional coefficient space of polynomial continued-fraction (PCF)
generators, train a **Geometric Operator Autoencoder (GOT v3)** to embed them into a
low-dimensional latent manifold z ∈ ℝ³, and then empirically test algebraic laws for a
set of symbolic operators on that manifold.

Ten operators are tested: `identity`, `shift`, `backshift`, `scale_up`, `scale_down`,
`apery_pos`, `apery_neg`, `sign_flip_b`, `sign_flip_a`, `random_sm`.

Thirty-five algebraic laws are evaluated across three levels:
- **[SYM]** Symbolic / coefficient space
- **[LAT]** Latent space (encoded z)
- **[ARI]** Arithmetic (delta-head predictions)

## Repository Structure

```
cf-space/
├── paper/
│   ├── main.tex            — LaTeX source for the preprint
│   └── references.bib      — Bibliography
├── got_v3/
│   ├── models.py           — GOT v3 architecture (Transformer + GeometricBottleneck)
│   ├── operators.py        — 10 CF operators
│   ├── losses.py           — All loss functions (recon, nbr, op, contrast, var_reg…)
│   ├── dataset.py          — CFDataset loader
│   ├── train_got_v3.py     — 3-phase training loop
│   ├── eval_latent.py      — TwoNN, kNN overlap, trustworthiness, PCA plots
│   ├── cf_algebra.py       — High-level algebra utilities (op_vector, cosine_sim, orbit)
│   └── configs/
│       ├── ae_pretrain.yaml
│       ├── target_train.yaml
│       └── joint.yaml
├── latent_operator_algebra.py  — Main algebra test suite (35 laws, 4 levels)
├── cfspacealgebra.md           — Full results report with all tables
├── results/
│   └── operator_algebra/
│       └── algebra_report.json — Machine-readable results
├── data/
│   └── README_DATA.md          — Dataset description and generation instructions
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate or download the dataset (see data/README_DATA.md)
python3 cf_corpus_large.py --n 300000 --out data/cf_large/

# 3. Train Phase 1 (geometry)
python3 got_v3/train_got_v3.py \
    --data data/cf_large \
    --config got_v3/configs/ae_pretrain.yaml

# 4. Train Phase 2 (arithmetic heads, frozen encoder)
python3 got_v3/train_got_v3.py \
    --data data/cf_large \
    --config got_v3/configs/target_train.yaml \
    --load ckpt/got_v3_ae_pretrain_k3.pt

# 5. Train Phase 3 (joint fine-tuning)
python3 got_v3/train_got_v3.py \
    --data data/cf_large \
    --config got_v3/configs/joint.yaml \
    --load ckpt/got_v3_target_train_k3.pt

# 6. Run the full operator algebra test suite
python3 latent_operator_algebra.py \
    --ckpt ckpt/got_v3_joint_k3.pt \
    --data data/cf_large \
    --n 2000 \
    --n_manifold 8000 \
    --out results/operator_algebra/
```

## Key Results

| Finding | Value |
|---|---|
| Algebraic structure | Non-commutative monoid + approximate Lie subalgebra |
| Smoothest inverse pair | apery_pos ↔ apery_neg, cos = −0.998 |
| Only operator toward ζ(3) | `scale_down` (alignment = +0.265) |
| Lie-like closure | {scale, apery_pos, apery_neg}, cos > 0.97 |
| Disruptive generator | `shift` (Lipschitz max = 11.76, breaks manifold closure) |
| Phase 1 mix_acc | 0.986 (k=3, 64-epoch pretrain) |

Full tables and analysis: [`cfspacealgebra.md`](cfspacealgebra.md)

## Model Architecture

GOT v3 separates geometric and predictive representations:

- **Encoder**: TransformerEncoder (d=128, L=4, h=8) → CLS token h ∈ ℝ¹²⁸
- **Bottleneck**: Linear projection → z ∈ ℝ³ (geometric manifold)
- **Decoder**: MLP(z) → reconstructed coefficients
- **Heads**: delta, conv, plateau — use (h ‖ z)
- **Op-field**: MLP(z_norm, δ_coeff) → Δz (predicts latent displacement)

## Citation

```bibtex
@techreport{svensson2026cfspacealgebra,
  title   = {An Empirical Operator Algebra on a Latent Manifold of
             Polynomial Continued Fractions},
  author  = {Svensson, David},
  year    = {2026},
  type    = {Technical Report / Research Note},
  url     = {https://github.com/VesterlundCoder/cf-space}
}
```

## License

Code: MIT. See [LICENSE](LICENSE).  
Paper: CC BY 4.0.
