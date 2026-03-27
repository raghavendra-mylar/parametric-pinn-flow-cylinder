# Parametric PINN — Steady Laminar Cylinder Flow

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Cd Error](https://img.shields.io/badge/Cd%20Error-<2%25-brightgreen?style=flat-square)
![Physics](https://img.shields.io/badge/Physics-Pure%20NS%20Only-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Paper%20Ready-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76B900?style=flat-square&logo=nvidia)

**A single parametric PINN that predicts drag, surface pressure, and separation angle\nacross Re ∈ [10, 45] — trained on Navier-Stokes equations alone, zero CFD solution data.**

*M.Tech Research — Indian Institute of Space Science and Technology (IIST)*

</div>

---

## Final Results Summary

![Final Summary](results/figures/10_final_summary.png)

---

## What This Is

A **parametric Physics-Informed Neural Network** trained on pure Navier-Stokes physics — no CFD velocity or pressure fields used as training data. The model takes `(x, y, Re)` as input and simultaneously predicts:

- **Drag coefficient Cd** — validated against Tritton (1959) / Dennis & Chang (1970)
- **Surface pressure Cp(θ)** — validated against Dennis & Chang (1970) Table 3
- **Separation angle θ_sep** — validated against Taneda (1956) benchmark
- **∂Cd/∂Re via autodiff** — parametric sensitivity without re-training

All four outputs emerge from a **single forward pass** of a 45,842-parameter network.

---

## Validation Results

### Drag Coefficient — All 11 Reynolds Numbers

![Cd Validation](results/figures/08_Cd_validation%20(2).png)

| Metric | Train | Interp | Extrap | Target | Status |
|--------|-------|--------|--------|--------|--------|
| **Cd error** | 0.72% | 1.27% | 1.79% | <5% | ✅ |
| **Cp(θ) MAE** | 0.038 | monotone ✅ | monotone ✅ | <0.05 | ✅ |
| **Sep. angle** | 1.77% | 2.40% | 0.59% | <10% | ✅ |

### Surface Pressure Cp(θ) — Novelty N1

![Cp Theta](results/figures/08_Cp_theta.png)

> First Cp(θ) benchmark comparison in parametric PINN literature for cylinder flow. MAE=0.038 vs Dennis & Chang (1970) Table 3.

### Cp(θ) Across All Reynolds Numbers

![Cp All Re](results/figures/09.5_Cp_all_Re.png)

### Separation Angle vs Taneda (1956)

![Separation Angle](results/figures/09_separation_angle%20(1).png)

### Physics Learning — Oseen Scaling & dCd/dRe

![Oseen Scaling](results/figures/09.7_Cd_scaling.png)

### Independent Validation at Re=17 (Never in Training)

![Re17 Validation](results/figures/11_Cp_Re17_validation.png)

> 99.4% of surface angles lie within [Re=15, Re=20] envelope — physically consistent parametric interpolation confirmed.

---

## Architecture

```
Input:   [x, y, Re_norm]   Re_norm = (Re − 25) / 15
Network: 8 × 80  tanh      45,842 parameters
Output:  [ψ, p]            stream function + pressure
Derive:  u = ∂ψ/∂y        v = −∂ψ/∂x   (automatic differentiation)
```

| Property | Value |
|----------|-------|
| Architecture | 8 hidden layers × 80 neurons, tanh |
| Parameters | 45,842 |
| Inputs | x, y, Re (normalized) |
| Outputs | ψ (stream function), p (pressure) |
| Adam | 35,000 iterations |
| L-BFGS | 1,597 iterations |
| Collocation points | 105,000 adaptive (40% bulk / 40% wake / 20% BL ring) |
| Physics | Pure Navier-Stokes — no supervised flow data |
| Re range | 10–40 training, 42–45 extrapolation |

---

## Physics Learning Proof — 5/5 Tests Passed

| Test | What It Checks | Result |
|------|---------------|--------|
| T1: dCd/dRe autodiff | Sign negative, monotone — parametric gradient correct | ✅ |
| T2: Cd × √Re Oseen scaling | CV = 2.7% — classical scaling discovered without supervision | ✅ |
| T3: Vorticity topology | Symmetric at Re=10, elongating at Re=40 | ✅ |
| T4: NS residual at unseen pts | 5.69×10⁻² at 2,000 unseen pts | ✅ |
| T5: Divergence-free velocity | mean |div u| = 7.64×10⁻⁴ | ✅ |

---

## Novelty Contributions

**N1 — Cp(θ) surface pressure benchmark**  
MAE = 0.038 vs DC70 Table 3. First Cp(θ) benchmark in any parametric PINN paper for cylinder flow.

**N2 — Adaptive collocation (40/40/20 split)**  
7.2× higher point density in boundary layer vs uniform sampling.

**N3 — ∂Cd/∂Re via automatic differentiation**  
Differentiable parametric simulator — sensitivity without re-training.

**N4 — Simultaneous multi-quantity validation**  
Cd + Cp(θ) + θ_sep from one forward pass — pure NS physics only.

---

## Publication Targets

| Role | Venue | Note |
|------|-------|------|
| **Primary** | MDPI Fluids / Applied Sciences | Open access, fast review |
| **Secondary** | Computers & Fluids (Elsevier) | Needs OpenFOAM ground truth |
| **Conference** | WCCM-ECCOMAS 2026 (Munich, Jul) | PINN minisymposium |
| **Preprint** | arXiv — cs.CE or physics.flu-dyn | Ready now |

---

## Repository Structure

```
parametric-pinn-flow-cylinder/
├── src/                          # Model, training, physics, postprocessing
├── notebooks/                    # Full Kaggle notebook (Cells 1–11)
├── results/
│   ├── figures/                  # All 22 output PNG figures
│   └── checkpoints/              # Trained weights (available on request)
├── data/                         # Benchmark data — DC70, Tritton, Taneda
├── RESULTS.md                    # Full numerical tables
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Known Limitations

- Wake recirculation bubble underpredicted — Cd and Cp unaffected
- Separation angle via Cp-minimum calibration, not direct τ_w=0
- dCd/dRe autodiff captures velocity-field contribution only

---

## References

- **Dennis & Chang (1970)** — *J. Fluid Mechanics*, 42(3), 471–489
- **Tritton (1959)** — *J. Fluid Mechanics*, 6(4), 547–567
- **Taneda (1956)** — *J. Physical Society of Japan*, 11(3), 302–307
- **Raissi et al. (2019)** — *J. Computational Physics*, 378, 686–707

---

## Citation

```bibtex
@misc{raghavendra2026parametricpinn,
  author    = {Raghavendra M},
  title     = {Parametric Physics-Informed Neural Networks for Steady Laminar
               Flow Over a Cylinder: Simultaneous Drag, Surface Pressure, and
               Separation Angle Prediction across Reynolds Number},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/raghavendra-mylar/parametric-pinn-flow-cylinder},
  note      = {M.Tech Research, IIST — Target: MDPI Fluids / arXiv preprint}
}
```

---

## Author

**Raghavendra M**  
M.Tech Aerospace Engineering (Thermal & Propulsion) @ IIST  
📧 ragharit586@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/raghavendra-mylar-b00b95240/)  
🐙 [GitHub](https://github.com/raghavendra-mylar)

---

<div align="center">
<sub>Pure Navier-Stokes · No CFD Solution Data · 45,842 Parameters · IIST 2026</sub>
</div>
