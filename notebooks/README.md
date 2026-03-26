# Notebooks

## Main Notebook

### `parametric_pinn_cylinder.ipynb`

Full training and validation pipeline — run on Kaggle with NVIDIA T4 GPU.

**Cell structure:**

| Cell | Name | Description |
|------|------|-------------|
| 1–5 | Setup & Architecture | Imports, PINN class (8×80 tanh), domain geometry, benchmark data |
| 6 | Collocation Sampling | Adaptive 40/40/20 split — bulk / wake box / BL ring |
| 7 | Loss Functions | PDE residuals (NS continuity + momentum-x/y), BCs, boundary terms |
| 8 | Training | Adam 35k + L-BFGS 1,597 iter. Saves weights + loss history. Core validation figures. |
| 8.5 | Extrapolation | Flow fields at Re=42, 45. Cp(θ) extrapolation check. |
| 9 | Separation Angle | Cp-minimum method, linear calibration, Taneda benchmark comparison. |
| 9.5 | Cp(θ) Full Analysis | All-Re colormap, training vs DC70, interpolation check. |
| 9.7 | Physics Proofs | T1–T5: dCd/dRe autodiff, Oseen scaling, vorticity, NS residual, div-free. |
| 10 | Final Summary | Results table, paper claims, publication targets, 6-panel summary figure. |
| 11 | Re=17 Validation | Independent Cp(θ) check — 99.4% monotonicity in [Re=15, Re=20] envelope. |

**To run:**
1. Open on Kaggle: [kaggle.com/raghavendra-mylar](https://www.kaggle.com)
2. Enable GPU (NVIDIA T4)
3. Run All Cells (~45 min)
4. Download outputs from `/kaggle/working/`

> The `.ipynb` file is large (~3 MB). Download from Kaggle and upload here, or available on request.
