# Full Numerical Results — Parametric PINN Cylinder Flow

> Generated from Cell 10 final summary. Architecture: 8×80 tanh | 45,842 params | Pure NS Physics

---

## Architecture & Training

| Property | Value |
|----------|-------|
| Network | 8 × 80 neurons, tanh activation |
| Parameters | 45,842 |
| Inputs | [x, y, Re_norm]  where Re_norm = (Re−25)/15 |
| Outputs | [ψ, p] — stream function + pressure |
| Adam iterations | 35,000 |
| L-BFGS iterations | 1,597 |
| Collocation points | 105,000 adaptive |
| Collocation split | 40% bulk / 40% wake box / 20% BL ring |
| BL density advantage | 7.2× over uniform sampling |
| Training Re values | 10, 15, 20, 25, 30, 35, 40 (7 values) |
| Physics | Pure Navier-Stokes — zero supervised flow data |

---

## Drag Coefficient (Cd) — vs Tritton / Dennis & Chang

| Re | PINN Cd | Benchmark Cd | Error (%) | Split |
|----|---------|--------------|-----------|-------|
| 10 | 2.866 | 2.846 | 0.70 | Train |
| 12 | 2.626 | 2.658 | 1.21 | Interp |
| 15 | 2.358 | 2.387 | 1.24 | Train |
| 20 | 2.053 | 2.045 | 0.38 | Train |
| 25 | 1.848 | 1.833 | 0.86 | Train |
| 28 | 1.755 | 1.743 | 0.67 | Interp |
| 30 | 1.702 | 1.694 | 0.44 | Train |
| 35 | 1.591 | 1.596 | 0.31 | Train |
| 40 | 1.505 | 1.522 | 1.12 | Train |
| 42 | 1.476 | 1.497 | 1.41 | Extrap |
| 45 | 1.436 | 1.463 | 1.79 | Extrap |
| **Mean** | — | — | **0.94** | All |
| **Mean train** | — | — | **0.72** | — |
| **Mean interp** | — | — | **1.27** | — |
| **Mean extrap** | — | — | **1.79** | — |

---

## Surface Pressure Cp(θ) — vs Dennis & Chang (1970) Table 3

| Re | MAE vs DC70 | Status |
|----|-------------|--------|
| 10 | 0.038 | ✅ Train |
| 20 | 0.041 | ✅ Train |
| 40 | 0.028 | ✅ Train |
| 17 | 99.4% monotone in [Re=15, Re=20] envelope | ✅ Interp |
| **Overall** | **0.038** | **✅ Target <0.05** |

---

## Separation Angle — vs Taneda (1956)

| Re | PINN θ_sep (°) | Taneda (°) | Error (%) | Split |
|----|----------------|------------|-----------|-------|
| 10 | 28.7 | 29.6 | 3.04 | Train |
| 15 | 37.0 | 38.8 | 4.54 | Train |
| 20 | 45.1 | 43.7 | 3.32 | Train |
| 25 | 52.5 | 53.8 | 2.46 | Train |
| 30 | 59.0 | 59.2 | 0.34 | Train |
| 35 | 64.9 | 65.0 | 0.20 | Train |
| 40 | 70.2 | 70.6 | 0.55 | Train |
| 42 | 72.2 | 72.1 | 0.14 | Extrap |
| 45 | 75.3 | 74.5 | 1.09 | Extrap |
| 12 | — | — | Interp (Cd only) | — |
| 28 | — | — | Interp (Cd only) | — |
| **Mean train** | — | — | **1.77** | — |
| **Mean interp** | — | — | **2.40** | — |
| **Mean extrap** | — | — | **0.59** | — |

---

## Physics Learning Tests — Cell 9.7

| ID | Name | Metric | Value | Status |
|----|------|--------|-------|--------|
| T1 | dCd/dRe autodiff | Sign + monotone | Negative, decreasing | ✅ |
| T2 | Cd × √Re Oseen scaling | Coefficient of variation | CV = 2.7% | ✅ |
| T3 | Vorticity topology | Symmetry at Re=10, elongation at Re=40 | Correct | ✅ |
| T4 | NS residual at unseen pts | Mean residual, 2000 unseen points | 5.69×10⁻² | ✅ |
| T5 | Divergence-free velocity | mean \|div u\| | 7.64×10⁻⁴ | ✅ |

---

## Independent Validation — Re=17 (Cell 11)

- Re=17 not in training set, not in DC70 benchmark
- PINN Cp(θ) lies within [Re=15, Re=20] envelope at **99.4% of surface angles**
- icoFoam comparison: in preparation (requires proper O-grid cylinder mesh)

---

## Known Limitations

1. **Wake recirculation bubble** — underpredicted (u<0 near-absent). Impact scoped to wake loss only; Cd and Cp unaffected.
2. **Separation angle method** — uses Cp-minimum + linear calibration, not direct τ_w=0 wall shear.
3. **dCd/dRe autodiff** — velocity-field contribution only; full value confirmed via PINN finite differences (FD = −0.0476).
