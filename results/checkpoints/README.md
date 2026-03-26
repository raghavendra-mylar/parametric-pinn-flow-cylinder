# Model Checkpoints

## Final Trained Weights

| File | Size | Description |
|------|------|-------------|
| `pinn_lbfgs.weights.h5` | ~3.5 MB | Final weights after Adam + L-BFGS. **Available on request.** |
| `adam_history.npz` | ~500 KB | Adam training history — 8 loss keys across all phases. |
| `lbfgs_history.npz` | ~100 KB | L-BFGS training history — 8 loss keys, 1,597 iterations. |

## How to Request

Email **ragharit586@gmail.com** with subject `[PINN Weights] parametric-cylinder` or open a GitHub Issue.

## How to Load Weights

```python
import tensorflow as tf

# Rebuild architecture (must match exactly)
model = build_pinn_8x80()  # see src/model.py

# Load weights
model.load_weights('pinn_lbfgs.weights.h5')
print('Weights loaded — 45,842 parameters')
```

## Training Configuration

```
Adam:    35,000 iterations   lr=1e-3 → 1e-4 decay
L-BFGS:  1,597 iterations    max_iterations=5000, tolerance=1e-9
Hardware: NVIDIA T4 GPU (Kaggle)
Runtime:  ~45 min total
```

## Reproducibility

The full training notebook is at `notebooks/parametric_pinn_cylinder.ipynb`.
All random seeds are fixed. Re-training from scratch should reproduce results within ±0.2% Cd error.
