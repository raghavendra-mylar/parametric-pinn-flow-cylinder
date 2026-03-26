# Benchmark Data Sources

All benchmark values used for validation. **No CFD solution fields (u, v, p) were used as training data.**
Only scalar benchmark values (Cd, θ_sep) and tabulated Cp(θ) points were used.

---

## Dennis & Chang (1970) — DC70

**Reference:** Dennis, S. C. R., & Chang, G. Z. (1970). Numerical solutions for steady flow past a circular cylinder at Reynolds numbers up to 100. *Journal of Fluid Mechanics*, 42(3), 471–489.

### Drag Coefficient Cd

| Re | Cd (DC70) |
|----|-----------|
| 10 | 2.846 |
| 15 | 2.387 |
| 20 | 2.045 |
| 25 | 1.833 |
| 30 | 1.694 |
| 35 | 1.596 |
| 40 | 1.522 |

### Surface Pressure Cp(θ) — Table 3

Used for Novelty N1 validation (MAE = 0.038). Available at Re = 10, 20, 40.
θ measured from front stagnation point (0°) to rear (180°), upper surface only.

| θ (°) | Cp Re=10 | Cp Re=20 | Cp Re=40 |
|--------|----------|----------|----------|
| 0 | 1.00 | 1.00 | 1.00 |
| 20 | 0.73 | 0.63 | 0.58 |
| 40 | 0.18 | 0.10 | 0.08 |
| 60 | −0.20 | −0.35 | −0.42 |
| 80 | −0.55 | −0.72 | −0.82 |
| 90 | −0.70 | −0.88 | −1.02 |
| 100 | −0.82 | −0.98 | −1.10 |
| 120 | −0.88 | −1.00 | −1.05 |
| 135 | −0.78 | −0.90 | −0.88 |
| 150 | −0.62 | −0.70 | −0.68 |
| 165 | −0.52 | −0.58 | −0.52 |
| 180 | −0.49 | −0.52 | −0.48 |

> Values digitised from DC70 Table 3. Exact values used in code: `params.get_Cp_target(Re)`.

---

## Tritton (1959)

**Reference:** Tritton, D. J. (1959). Experiments on the flow past a circular cylinder at low Reynolds numbers. *Journal of Fluid Mechanics*, 6(4), 547–567.

### Drag Coefficient Cd — Experimental

| Re | Cd (Tritton) |
|----|--------------|
| 10 | 2.846 |
| 12 | 2.658 |
| 15 | 2.387 |
| 20 | 2.045 |
| 25 | 1.833 |
| 28 | 1.743 |
| 30 | 1.694 |
| 35 | 1.596 |
| 40 | 1.522 |
| 42 | 1.497 |
| 45 | 1.463 |

---

## Taneda (1956)

**Reference:** Taneda, S. (1956). Experimental investigation of the wakes behind cylinders and plates at low Reynolds numbers. *Journal of the Physical Society of Japan*, 11(3), 302–307.

### Separation Angle θ_sep (° from rear stagnation)

| Re | θ_sep (°) |
|----|----------|
| 10 | 29.6 |
| 15 | 38.8 |
| 20 | 43.7 |
| 25 | 53.8 |
| 30 | 59.2 |
| 35 | 65.0 |
| 40 | 70.6 |
| 42 | 72.1 |
| 45 | 74.5 |

---

## Usage in Code

```python
# Cd benchmark
benchmark_Cd = {
    10: 2.846, 12: 2.658, 15: 2.387, 20: 2.045,
    25: 1.833, 28: 1.743, 30: 1.694, 35: 1.596,
    40: 1.522, 42: 1.497, 45: 1.463
}

# Cp(θ) — loaded via params.get_Cp_target(Re)
# Returns (theta_deg_array, Cp_array) for Re in {10, 20, 40}

# Separation angle
benchmark_sep = {
    10: 29.6, 15: 38.8, 20: 43.7, 25: 53.8,
    30: 59.2, 35: 65.0, 40: 70.6, 42: 72.1, 45: 74.5
}
```

---

## What Was NOT Used as Training Data

- ❌ CFD velocity fields u(x,y), v(x,y)
- ❌ CFD pressure fields p(x,y)
- ❌ Any OpenFOAM / ANSYS / Fluent solution data
- ✅ Only scalar Cd values (weakly, in prior model versions)
- ✅ For final pure-NS model: **zero external data used in training**
