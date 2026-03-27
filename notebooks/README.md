# Notebooks

## `parametric_pinn_cylinder.ipynb`

Full training and validation pipeline for the parametric PINN — run on Kaggle with NVIDIA T4 GPU.

**Cell structure:**

| Cell | Name | Description |
|------|------|-------------|
| 1–5 | Setup & Architecture | PINN class (8×80 tanh), domain geometry, benchmark data |
| 6 | Collocation Sampling | Adaptive 40/40/20 split — bulk / wake box / BL ring |
| 7 | Loss Functions | NS continuity + momentum-x/y PDE residuals, boundary conditions |
| 8 | Training | Adam 35k + L-BFGS 1,597 iter. Core validation figures. |
| 8.5 | Extrapolation | Flow fields at Re=42, 45. Cp(θ) extrapolation check. |
| 9 | Separation Angle | Cp-minimum method, Taneda benchmark comparison. |
| 9.5 | Cp(θ) Full Analysis | All-Re colormap, training vs DC70, interpolation check. |
| 9.7 | Physics Proofs | T1–T5: dCd/dRe autodiff, Oseen scaling, vorticity, NS residual, div-free. |
| 10 | Final Summary | Results table, paper claims, publication targets, 6-panel summary figure. |
| 11 | Re=17 Validation | Independent Cp(θ) check — 99.4% monotonicity in [Re=15, Re=20] envelope. |

---

> ⚠️ **Note:** The full notebook is not publicly shared as this work is currently unpublished and under preparation for journal submission.
>
> 📧 **Available on request** — contact ragharit586@gmail.com with subject `[PINN Notebook] parametric-cylinder`
