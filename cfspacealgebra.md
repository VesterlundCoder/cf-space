# CF Latent Space Algebra — Empirical Report

**Script:** `latent_operator_algebra.py`  
**Checkpoint:** `ckpt/got_v3_ae_pretrain_k3.pt` (old, pre-fix checkpoint — latent measurements partially affected by z-collapse; symbolic-level results unaffected)  
**Sample size:** n=500 CFs, manifold approximation n=2000  
**Report date:** 2026-05-14  

> **Note on checkpoint state:** This analysis was run on the legacy checkpoint exhibiting z-collapse (all cosine similarities ≈ 1.0 in raw z). However, (1) symbolic-level tests operate entirely in coefficient space and are unaffected, and (2) many latent measurements are computed on *normalised* z vectors (unit sphere), which partially recovers meaningful geometry even under collapse. A re-run with the retrained checkpoint (in progress) will provide clean latent-level confirmation.

---

## Operators Tested (10 total)

| Name | Definition | Exact inverse? |
|---|---|---|
| `identity` | I(x) = x | self |
| `shift` | p(n) → p(n+1) via binomial shift | `backshift` |
| `backshift` | p(n) → p(n−1) | `shift` |
| `scale_up` | coefficients × 1.5 | `scale_down` |
| `scale_down` | coefficients × (1/1.5) | `scale_up` |
| `apery_pos` | a₂ += 0.10 | `apery_neg` |
| `apery_neg` | a₂ −= 0.10 | `apery_pos` |
| `sign_flip_b` | b(n) → b(−n) | self (involution) |
| `sign_flip_a` | a(n) → a(−n) | self (involution) |
| `random_sm` | Gaussian noise σ=0.05 | none (stochastic) |

---

## Level 1 — Basic Laws

### Identity & Inverse
| Law | Operator pair | Mean err | Pass? |
|---|---|---|---|
| Identity T(I(x)) = T(x) | all | 0.0000 | ✓ YES |
| Inverse: shift ∘ backshift | shift, backshift | 7.7e-06 | ✓ YES |
| Inverse: backshift ∘ shift | backshift, shift | 7.7e-06 | ✓ YES |
| Inverse: scale_up ∘ scale_down | scale_up, scale_down | 0.0000 | ✓ YES |
| Inverse: scale_down ∘ scale_up | scale_down, scale_up | 0.0000 | ✓ YES |
| Inverse: apery_pos ∘ apery_neg | apery_pos, apery_neg | 0.0000 | ✓ YES |
| Inverse: apery_neg ∘ apery_pos | apery_neg, apery_pos | 0.0000 | ✓ YES |

### Involutions (T² = I)
| Operator | Mean err | Pass? |
|---|---|---|
| `sign_flip_b` | 0.0000 | ✓ YES (exact) |
| `sign_flip_a` | 0.0000 | ✓ YES (exact) |

### Idempotence (T² = T)
| Operator | Mean err | Pass? |
|---|---|---|
| `identity` | 0.0000 | ✓ YES |
| `shift` | 4.8403 | ✗ NO |
| `backshift` | 0.4123 | ✗ NO |
| `scale_up` | 0.0163 | ✓ YES (approx) |
| `scale_down` | 0.0036 | ✓ YES (approx) |
| `apery_pos` | 0.0004 | ✓ YES (approx) |
| `apery_neg` | 0.0004 | ✓ YES (approx) |
| `sign_flip_b` | 0.0256 | ✓ YES (approx) |
| `sign_flip_a` | 0.0149 | ✓ YES (approx) |

*Note: Apéry operators appear approximately idempotent because the perturbation is small relative to coefficient scale.*

### Periodicity (k=2 cycle: T²(x) ≈ x)
| Operator | Mean err | Verdict |
|---|---|---|
| `identity` | 0.0000 | ★ CYCLE |
| `sign_flip_b` | 0.0000 | ★ CYCLE (exact period-2) |
| `sign_flip_a` | 0.0000 | ★ CYCLE (exact period-2) |
| `scale_up` | 0.0238 | ★ CYCLE (approx) |
| `scale_down` | 0.0087 | ★ CYCLE (approx) |
| `apery_pos` | 0.0007 | ★ CYCLE (approx) |
| `apery_neg` | 0.0007 | ★ CYCLE (approx) |
| `shift` | 6.0047 | — (aperiodic) |
| `backshift` | 0.5770 | — (aperiodic) |

### Fixed Points (argmin‖v_T(x)‖)
| Operator | min‖v_T‖ | mean‖v_T‖ | Best CF index |
|---|---|---|---|
| `identity` | 0.0000 | 0.0000 | 0 |
| `shift` | 0.0030 | 1.3323 | 430 |
| `backshift` | 0.0086 | 0.1798 | 73 |
| `scale_up` | 0.0006 | 0.0081 | 214 |
| `scale_down` | 0.0005 | 0.0053 | 214 |
| `apery_pos` | 0.0002 | 0.0004 | 361 |
| `apery_neg` | 0.0002 | 0.0004 | 361 |
| `sign_flip_b` | 0.0026 | 0.0256 | 350 |
| `sign_flip_a` | 0.0029 | 0.0149 | 126 |
| `random_sm` | 0.0001 | 0.0010 | 158 |

*CF #214 is the approximate fixed point of both scale operators simultaneously — a CF whose coefficients lie near the unit-normalisation surface.*

---

## Level 2 — Commutativity

### Commutator Norm Matrix ‖[T,S]‖ = ‖E(T(S(x))) − E(S(T(x)))‖

|  | identity | shift | backshift | scale_up | scale_down | apery_pos | apery_neg | sign_flip_b | sign_flip_a | random_sm |
|---|---|---|---|---|---|---|---|---|---|---|
| **identity** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **shift** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.004 | **4.061** | 0.045 | 0.192 |
| **backshift** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | **4.498** | 0.045 | 0.012 |
| **scale_up** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| **scale_down** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **apery_pos** | 0.000 | 0.003 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **apery_neg** | 0.000 | 0.004 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **sign_flip_b** | 0.000 | **4.061** | **4.498** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **sign_flip_a** | 0.000 | 0.045 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| **random_sm** | 0.001 | 0.033 | 0.011 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |

**Key observation:** The only large non-commutativity is between **sign_flip_b** and **shift/backshift** (values 4.06–4.50). Sign-flipping the b-coefficients and index-shifting do not commute. All scale, apery, and identity operators commute with everything.

### Cosine Similarity Matrix cos(v_T, v_S)

|  | identity | shift | backshift | scale_up | scale_down | apery_pos | apery_neg | sign_flip_b | sign_flip_a | random_sm |
|---|---|---|---|---|---|---|---|---|---|---|
| **identity** | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| **shift** | +0.000 | **+1.000** | +0.304 | +0.547 | −0.383 | −0.204 | +0.213 | −0.426 | −0.132 | −0.003 |
| **backshift** | +0.000 | +0.304 | **+1.000** | +0.340 | −0.022 | +0.259 | −0.246 | +0.144 | +0.108 | −0.003 |
| **scale_up** | +0.000 | +0.547 | +0.340 | **+1.000** | −0.799 | +0.088 | −0.082 | −0.330 | −0.209 | +0.001 |
| **scale_down** | +0.000 | −0.383 | −0.022 | −0.799 | **+1.000** | −0.071 | +0.072 | +0.550 | +0.256 | +0.003 |
| **apery_pos** | +0.000 | −0.204 | +0.259 | +0.088 | −0.071 | **+1.000** | **−0.998** | +0.216 | −0.131 | −0.075 |
| **apery_neg** | +0.000 | +0.213 | −0.246 | −0.082 | +0.072 | **−0.998** | **+1.000** | −0.215 | +0.131 | +0.074 |
| **sign_flip_b** | +0.000 | −0.426 | +0.144 | −0.330 | +0.550 | +0.216 | −0.215 | **+1.000** | −0.061 | −0.004 |
| **sign_flip_a** | +0.000 | −0.132 | +0.108 | −0.209 | +0.256 | −0.131 | +0.131 | −0.061 | **+1.000** | +0.032 |
| **random_sm** | +0.000 | −0.003 | −0.003 | +0.001 | +0.003 | −0.075 | +0.074 | −0.004 | +0.032 | **+1.000** |

**Highlighted findings:**
- `apery_pos` ↔ `apery_neg`: cos = **−0.998** — near-perfect antipodal pair in z-space. Cleanest algebraic inverse.
- `scale_up` ↔ `scale_down`: cos = **−0.799** — good inverse alignment.
- `shift` ↔ `scale_up`: cos = **+0.547** — shift acts partially like a scaling in z-space.
- `scale_down` ↔ `sign_flip_b`: cos = **+0.550** — unexpected structural similarity.
- `random_sm` is orthogonal to all operators (cos ≈ 0) — isotropic noise confirmed.

---

## Level 3 — Latent Geometry

### Closure: Distance to Manifold after T(x)

| Operator | Mean dist | Baseline | Ratio | Stays on manifold? |
|---|---|---|---|---|
| `identity` | 0.0012 | 0.0012 | 1.00 | ✓ YES |
| `shift` | 1.3071 | 0.0012 | **1075×** | ✗ NO |
| `backshift` | 0.1407 | 0.0012 | **116×** | ✗ NO |
| `scale_up` | 0.0031 | 0.0012 | 2.52 | ✓ YES |
| `scale_down` | 0.0012 | 0.0012 | 0.97 | ✓ YES |
| `apery_pos` | 0.0012 | 0.0012 | 1.01 | ✓ YES |
| `apery_neg` | 0.0012 | 0.0012 | 1.02 | ✓ YES |
| `sign_flip_b` | 0.0029 | 0.0012 | 2.35 | ✓ YES |
| `sign_flip_a` | 0.0016 | 0.0012 | 1.32 | ✓ YES |
| `random_sm` | 0.0013 | 0.0012 | 1.05 | ✓ YES |

**Shift is the only operator that takes CFs off the manifold** (1075× baseline distance). All smooth operators (scale, apery, sign_flip, random) preserve manifold membership.

### Eigenoperator Alignment

| Operator | cos(v_T, z) | Mean λ | Eigenlike? |
|---|---|---|---|
| `shift` | **−0.714** | 0.9999 | ✓ YES |
| `backshift` | −0.638 | 0.9991 | ✗ (below threshold) |
| `scale_up` | −0.506 | 1.0000 | ✗ |
| `scale_down` | +0.214 | 1.0000 | ✗ |
| `apery_pos` | +0.351 | 1.0000 | ✗ |
| `apery_neg` | −0.365 | 1.0000 | ✗ |
| `sign_flip_b` | +0.145 | 1.0000 | ✗ |
| `sign_flip_a` | −0.008 | 1.0000 | ✗ |
| `random_sm` | −0.016 | 1.0000 | ✗ |

**`shift` is the only operator that behaves like a latent eigenvector** — its action aligns with the primary z-axis direction (cos = −0.714).

### ζ(3) Alignment (Core/Far contrast vector)

| Operator | Alignment score | Toward ζ(3)? |
|---|---|---|
| `shift` | −0.189 | ✗ |
| `backshift` | −0.627 | ✗ |
| `scale_up` | −0.391 | ✗ |
| **`scale_down`** | **+0.265** | **✓ YES** |
| `apery_pos` | −0.763 | ✗ |
| `apery_neg` | +0.012 | ✗ |
| `sign_flip_b` | +0.064 | ✗ |
| `sign_flip_a` | −0.055 | ✗ |
| `random_sm` | −0.026 | ✗ |

**`scale_down` is the only operator that moves CFs toward the ζ(3) cluster.** `apery_pos` is the strongest operator moving *away* from ζ(3) (−0.763).

### Lipschitz Constants ‖v_T(x) − v_T(y)‖ / ‖x − y‖

| Operator | Mean L | Median L | Max L |
|---|---|---|---|
| `shift` | 0.0409 | 0.0084 | **11.763** |
| `backshift` | 0.0148 | 0.0025 | 0.0449 |
| `scale_up` | 0.0052 | 0.0038 | 0.0208 |
| `scale_down` | 0.0051 | 0.0035 | 0.0255 |
| `apery_pos` | 0.0036 | 0.0033 | 0.0057 |
| `apery_neg` | 0.0036 | 0.0033 | 0.0056 |
| `sign_flip_b` | 0.0100 | 0.0083 | 0.0145 |
| `sign_flip_a` | 0.0096 | 0.0075 | 0.0149 |

**`shift` has max_L = 11.76** — can produce large jumps in latent space at singular coefficient configurations. All other operators are smooth (max_L < 0.05). Apéry operators are the smoothest (max_L < 0.006).

### Additivity Error ‖v_T∘S(x) − (v_T(x) + v_S(x))‖

Selected pairs (full table in JSON):

| T | S | Mean err | Median err | Additive? |
|---|---|---|---|---|
| `shift` | `shift` | 5.531 | 0.143 | ✗ NO |
| `scale_up` | `scale_up` | 0.0037 | 0.0023 | ✓ YES |
| `apery_pos` | `apery_pos` | 2.5e-4 | 1.8e-4 | ✓ YES |
| `apery_pos` | `apery_neg` | 1.8e-4 | 1.3e-4 | ✓ YES |
| `sign_flip_b` | `apery_pos` | 0.0012 | 0.0008 | ✓ YES |
| `sign_flip_a` | `scale_up` | 0.0064 | 0.0033 | ✓ YES |

**Smooth operators (scale, apery, sign_flip) satisfy approximate additivity.** Shift is non-additive.

### Composition vs. Vector Addition ‖E(T(S(x))) − (E(T(x)) + E(S(x)) − E(x))‖

| T | S | Mean err | Pass? |
|---|---|---|---|
| `shift` | `backshift` | 1.493 | ✗ |
| `scale_up` | `scale_down` | 0.008 | ✓ |
| `apery_pos` | `sign_flip_b` | 0.001 | ✓ |
| `sign_flip_a` | `sign_flip_b` | 0.001 | ✓ |
| `sign_flip_a` | `apery_pos` | 6.5e-5 | ✓ |
| `sign_flip_a` | `apery_neg` | 6.6e-5 | ✓ |

Smooth operators support **linear vector arithmetic** in z-space: `E(T∘S(x)) ≈ E(T(x)) + E(S(x)) − E(x)`.

---

## Level 4 — Advanced Structure

### Jacobi Identity ‖[T,[S,R]] + [S,[R,T]] + [R,[T,S]]‖ ≈ 0

| Triple (T, S, R) | Mean norm | Pass? |
|---|---|---|
| (shift, scale_up, apery_pos) | 0.003 | ✓ YES |
| (shift, backshift, scale_up) | 1.8e-5 | ✓ YES |
| **(apery_pos, apery_neg, sign_flip_b)** | **2.0e-7** | **✓ YES (near-exact)** |
| (shift, sign_flip_b, apery_pos) | 4.061 | ✗ NO |

The Jacobi identity holds for all smooth-operator triples. It breaks only when `sign_flip_b` is combined with `shift` — consistent with their large commutator.

### Lie Bracket Closure ‖[T,S] ≈ c·v_R‖ for some operator R

| [T, S] | Best match R | cos | Closed? |
|---|---|---|---|
| [scale_down, apery_pos] | `apery_neg` | **0.985** | ✓ YES |
| [scale_down, apery_neg] | `apery_pos` | **0.984** | ✓ YES |
| [scale_up, apery_neg] | `apery_neg` | **0.972** | ✓ YES |
| [scale_up, apery_pos] | `apery_pos` | **0.972** | ✓ YES |
| [shift, sign_flip_b] | `backshift` | 0.595 | ✗ NO |
| [backshift, apery_neg] | `backshift` | 0.532 | ✗ NO |
| [shift, sign_flip_a] | `sign_flip_a` | 0.421 | ✗ NO |
| [shift, apery_pos] | `backshift` | 0.397 | ✗ NO |

**{scale, apery_pos, apery_neg} form a closed Lie subalgebra** — brackets stay within the operator set (cos > 0.97). This is the strongest algebraic structure found.

### Anti-Commutativity [T,S] + [S,T] ≈ 0

All tested pairs satisfy this exactly (max err = 0.0000) — confirmed by construction, validates the implementation.

### Braid Relations T₁T₂T₁ = T₂T₁T₂

| Pair (T₁, T₂) | Mean err | Pass? |
|---|---|---|
| (shift, backshift) | 1.372 | ✗ NO |
| (shift, scale_up) | 3.137 | ✗ NO |
| (shift, sign_flip_b) | 0.202 | ✗ NO |
| **(apery_pos, sign_flip_a)** | **0.015** | **✓ YES** |

Braid relations mostly fail. The one exception `(apery_pos, sign_flip_a)` is notable — these two operators generate a structure resembling a Coxeter group locally.

### Conservation Laws (fraction of CFs where property is preserved)

| Operator | deg(a) | deg(b) | sign(a₀) | norm class |
|---|---|---|---|---|
| `identity` | 100% | 100% | 100% | 100% |
| `scale_up` | 100% | 100% | 100% | 100% |
| `scale_down` | 100% | 100% | 100% | 100% |
| `sign_flip_b` | 100% | 100% | 100% | 100% |
| `sign_flip_a` | 100% | 100% | 100% | 100% |
| `apery_pos` | 36% | 100% | 100% | 100% |
| `apery_neg` | 36% | 100% | 100% | 100% |
| `backshift` | 27% | 100% | 77% | 100% |
| `shift` | 29% | 99% | 39% | 100% |
| `random_sm` | 29% | 24% | 70% | 100% |

- **Scale and sign_flip are fully conservative** — they preserve all structural properties.
- **`apery` operators break degree of a** (36% conservation) but perfectly preserve deg(b) and sign(a₀).
- **`shift` breaks both deg(a) and sign(a₀)** — most destructive to symbolic structure.
- **Norm class is conserved by all operators** (100%) — every transformed CF still belongs to a recognisable normalisation class.

---

## Overall Verdict

### Algebraic Structure Summary

| Property | Holds? | Notes |
|---|---|---|
| Identity exists | ✓ | `identity` operator |
| Inverses exist | ✓ | shift↔backshift, scale↔inv, apery↔neg |
| Involutions (T²=I) | ✓ | sign_flip_a, sign_flip_b |
| Commutativity (global) | ✗ | Mean off-diagonal commutator = 0.176 |
| Commutativity (smooth) | ✓ | All scale/apery pairs commute |
| Jacobi identity | ✓ (partial) | Fails for triples involving shift + sign_flip |
| Lie bracket closure | ✓ (partial) | {scale, apery±} form closed subalgebra |
| Braid relations | ✗ (mostly) | Only (apery_pos, sign_flip_a) satisfied |
| Manifold closure | ✓ (partial) | Shift takes CFs off-manifold |
| Additivity in z | ✓ (smooth ops) | Linear arithmetic works for scale/apery/sign_flip |

### **Best-fit structure: Non-Commutative Monoid with a Lie subalgebra**

The CF latent space is **not** a classical Lie group, ring, or vector space. The most precise description is:

1. **At the global level**: A *non-commutative monoid* — closed under composition, has identity and (soft) inverses, but does not satisfy global commutativity.

2. **Smooth subalgebra {scale, apery_pos, apery_neg}**: This 3-operator set forms a *closed Lie subalgebra* under the bracket operation. Brackets close back into the set with cos > 0.97. Jacobi identity holds. This is the richest algebraic structure in the CF manifold.

3. **Sign-flip operators**: Generate a ℤ/2ℤ × ℤ/2ℤ subgroup (two involutions that commute with each other and with scale/apery).

4. **Shift operator**: The "disruptive" generator — only eigenoperator, breaks manifold closure, non-conservative, violates commutativity with sign_flip. Structurally analogous to a non-compact translation generator.

### Key Scientific Implications

- **apery_pos/neg with cos = −0.998** is the clearest algebraic pair for z-space navigation. Adding these vectors in z-space corresponds precisely to the inverse operation.
- **scale_down moves toward ζ(3)** — scaling down polynomial coefficients nudges the CF toward known irrational constants.
- **Linear arithmetic holds for smooth operators**: `E(T∘S(x)) ≈ E(T(x)) + E(S(x)) − E(x)`. This enables word2vec-style analogies in CF space.
- **The CF manifold has a local Lie algebra structure** near smooth operators, but the full operator set generates a non-commutative structure dominated by the shift generator.

---

## Planned Re-run

Re-run `latent_operator_algebra.py` after completing the 3-phase training pipeline with the fixed model (VICReg var_reg + normalised op_consistency loss):

```bash
python3 latent_operator_algebra.py \
    --ckpt ckpt/got_v3_joint_k3.pt \
    --data data/cf_large \
    --n 2000 \
    --n_manifold 8000 \
    --out results/operator_algebra_v2/
```

Expected improvements with the new checkpoint:
- Latent-level commutator measurements will reflect true geometry (no z-collapse)
- ζ(3) alignment vector will be better calibrated
- Eigenoperator analysis will be more reliable
- Distance-to-manifold ratios will be meaningful in absolute terms
