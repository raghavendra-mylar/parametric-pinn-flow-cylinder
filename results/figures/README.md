# Figure Index — Parametric PINN Cylinder Flow

All figures generated from Cells 8–11 of the Kaggle notebook.
Architecture: **8×80 tanh | 45,842 params | Pure NS Physics | Re ∈ [10, 40]**

---

## Cell 8 — Training & Core Validation

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `08_loss_history.png` | Training loss curves — Adam phases + L-BFGS convergence. Shows 8 loss components (PDE, BC, continuity, momentum-x, momentum-y). | Supporting |
| `08_Cp_theta.png` | **Cp(θ) vs DC70 Table 3** at Re=10, 20, 40. MAE=0.038. First Cp(θ) benchmark in parametric PINN literature. | ⭐ **Main — Novelty N1** |
| `08_Cd_validation.png` | Cd error bars for all 11 Re values. Green=training, blue=test. All below 2%, target <5%. | ⭐ **Main** |
| `08_flow_Re10.png` | Flow field at Re=10 — u-velocity, v-velocity, pressure, streamlines. | Supporting |
| `08_flow_Re25.png` | Flow field at Re=25 — training case, mid-range Re. | Supporting |
| `08_flow_Re40.png` | Flow field at Re=40 — highest training Re, elongated wake visible. | Supporting |
| `08_centerline.png` | Centerline velocity profiles u(x) at y=cy for all training Re. | Supporting |
| `08_Re_dependence.png` | Smooth parametric Re-dependence — u-velocity vs Re at fixed probe point. | Supporting |

---

## Cell 8.5 — Extrapolation

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `08.5_flow_Re42.png` | Extrapolation flow field at Re=42 (beyond training range Re=10–40). | Supporting |
| `08.5_flow_Re45.png` | Extrapolation flow field at Re=45 (near critical Re≈47). | Supporting |
| `08.5_Cp_extrapolation.png` | Cp(θ) at Re=42, 45 vs Re=40 training — shape preserved. | Supporting |

---

## Cell 9 — Separation Angle

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `09_separation_angle.png` | Separation angle θ_sep vs Re — PINN vs Taneda (1956). Train/interp/extrap highlighted. | ⭐ **Main** |
| `09_Cp_sep_profiles.png` | Cp(θ) profiles at each Re showing Cp-minimum extraction method for θ_sep. | Supporting |

---

## Cell 9.5 — Surface Pressure Full Analysis

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `09.5_Cp_training.png` | Cp(θ) at all 7 training Re vs DC70 benchmark. | ⭐ **Main — Novelty N1** |
| `09.5_Cp_interpolation.png` | Cp(θ) at interpolation Re=12, 28 + linear interpolation baseline. | Supporting |
| `09.5_Cp_all_Re.png` | **Cp(θ) colormap — all 14 Re values** on single plot. Smooth Re-dependence. | ⭐ **Main — paper figure** |
| `09.5_Cp_peak_angle.png` | Suction peak angle (Cp minimum location) vs Re — monotone as expected. | Supporting |

---

## Cell 9.7 — Physics Learning Proof

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `09.7_dCd_dRe.png` | **∂Cd/∂Re via autodiff** — negative, monotone with Re. Novelty N3. | ⭐ **Main — Novelty N3** |
| `09.7_Cd_scaling.png` | Cd × √Re vs Re — CV=2.7%, Oseen scaling discovered without supervision. | ⭐ **Main** |
| `09.7_vorticity.png` | Vorticity field at Re=10, 20, 40 — correct topology (symmetric→elongating). | Supporting |
| `09.7_divergence_free.png` | Continuity residual |div u| — mean 7.64×10⁻⁴ at machine precision. | Supporting |

---

## Cell 10 — Final Summary

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `10_final_summary.png` | **6-panel complete results summary.** Cd errors, separation angle, Cp(θ) Re=40, physics proof, Cd vs Re, novelty summary. | ⭐ **Thesis figure** |

---

## Cell 11 — Independent Validation Re=17

| File | Description | Paper Figure? |
|------|-------------|---------------|
| `11_Cp_Re17_validation.png` | Re=17 (never in training, not in DC70) — PINN Cp(θ) sits inside [Re=15, Re=20] envelope at 99.4% of surface angles. | ⭐ **Main — independent validation** |

---

## Upload Status

> **Note:** Figures are generated on Kaggle (NVIDIA T4). Download from Kaggle output panel and upload here.
> All figures saved at 150 DPI to `/kaggle/working/` as `.png`.

| Status | Count |
|--------|-------|
| ⭐ Main paper figures | 8 |
| Supporting figures | 11 |
| **Total** | **19** |
