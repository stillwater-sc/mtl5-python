#!/usr/bin/env python3
"""Quantized Neural Network Inference — Posit/Fixpnt vs Float32

Demonstrates the ML quantization story that motivates posit and fixed-point
hardware: a small feed-forward network's forward pass executed at multiple
precisions, comparing inference accuracy on a synthetic regression problem.

The reference network is trained in float64 (using a closed-form solution
since the problem is linear). Inference is then performed at:

    float32  — IEEE single precision baseline
    posit16  — 16-bit posit, very competitive on inference
    posit8   — 8-bit posit, aggressive quantization
    fixpnt8  — 8-bit fixed point with saturation

For each precision we report the inference RMSE on a held-out test set.
This is the kind of analysis that drives quantization decisions for
edge inference and KPU deployment.

Usage:
    python quantized_inference.py [--samples 200] [--features 8]
"""

import argparse

import numpy as np

import mtl5


def make_regression_problem(n_samples: int, n_features: int, rng: np.random.Generator):
    """Generate a synthetic linear regression problem y = X @ w + noise."""
    X = rng.standard_normal((n_samples, n_features))
    w_true = rng.standard_normal(n_features)
    noise = 0.05 * rng.standard_normal(n_samples)
    y = X @ w_true + noise
    return X, y, w_true


def train_linear_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form ridge solution: w = (XᵀX + λI)⁻¹ Xᵀy."""
    n_features = X.shape[1]
    lam = 1e-3
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    return mtl5.solve(A, b)


def predict_native(X: np.ndarray, w: np.ndarray, dtype: type) -> np.ndarray:
    """Forward pass at a NumPy native dtype (float32 or float64)."""
    Xq = X.astype(dtype)
    wq = w.astype(dtype)
    return (Xq @ wq).astype(np.float64)


def predict_universal(X: np.ndarray, w: np.ndarray, vector_factory, matrix_factory) -> np.ndarray:
    """Forward pass with weights and inputs quantized to a Universal type.

    Uses MTL5 matvec at the target precision so that all multiplies and
    accumulates happen in the chosen number system.
    """
    Xq = matrix_factory(X)
    wq = vector_factory(w)
    yq = mtl5.matvec(Xq, wq)
    return np.array(yq.to_list())


def evaluate_precision(
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_fn,
) -> tuple[float, float]:
    """Run a forward pass at the given precision and return (rmse, max_err)."""
    y_pred = predict_fn()
    err = y_pred - y_test
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))
    return rmse, max_err


def main():
    parser = argparse.ArgumentParser(
        description="Quantized inference comparison across number types"
    )
    parser.add_argument("--samples", type=int, default=200, help="Total samples")
    parser.add_argument("--features", type=int, default=8, help="Feature dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — Quantized Inference Showcase")
    print("  Available types: f32, f64, posit8/16/32, fixpnt8/16, lns16/32")
    print()

    rng = np.random.default_rng(args.seed)

    # Generate problem and split train/test
    X, y, w_true = make_regression_problem(args.samples, args.features, rng)
    n_train = int(0.7 * args.samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Training set: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples")

    # Train in float64 via mtl5.solve (closed-form ridge regression)
    w = train_linear_model(X_train, y_train)
    print(f"Trained model: ||w||₂ = {np.linalg.norm(w):.4f}")
    print(f"  ||w_true - w||₂ = {np.linalg.norm(w_true - w):.4f}")
    print()

    # Build the comparison table
    print("Inference Accuracy by Precision (RMSE on test set)")
    print("=" * 70)
    print(
        f"  {'precision':<12s}  {'bits':>4s}  {'RMSE':>14s}"
        f"  {'max abs err':>14s}  {'rel to f64':>10s}"
    )
    print(
        f"  {'---------':<12s}  {'----':>4s}  {'----':>14s}"
        f"  {'-----------':>14s}  {'----------':>10s}"
    )

    results = []

    # f32
    rmse, mxe = evaluate_precision(
        "f32",
        X_test,
        y_test,
        lambda: predict_native(X_test, w, np.float32),
    )
    results.append(("f32", 32, rmse, mxe))

    # f64
    rmse, mxe = evaluate_precision(
        "f64",
        X_test,
        y_test,
        lambda: predict_native(X_test, w, np.float64),
    )
    results.append(("f64", 64, rmse, mxe))
    rmse_f64 = rmse  # baseline

    # posit32
    rmse, mxe = evaluate_precision(
        "posit32",
        X_test,
        y_test,
        lambda: predict_universal(X_test, w, mtl5.vector_posit32, mtl5.matrix_posit32),
    )
    results.append(("posit32", 32, rmse, mxe))

    # posit16
    rmse, mxe = evaluate_precision(
        "posit16",
        X_test,
        y_test,
        lambda: predict_universal(X_test, w, mtl5.vector_posit16, mtl5.matrix_posit16),
    )
    results.append(("posit16", 16, rmse, mxe))

    # posit8 (very aggressive quantization)
    rmse, mxe = evaluate_precision(
        "posit8",
        X_test,
        y_test,
        lambda: predict_universal(X_test, w, mtl5.vector_posit8, mtl5.matrix_posit8),
    )
    results.append(("posit8", 8, rmse, mxe))

    # fixpnt16 (range [-128, 128) with 8 fractional bits)
    rmse, mxe = evaluate_precision(
        "fixpnt16",
        X_test,
        y_test,
        lambda: predict_universal(X_test, w, mtl5.vector_fixpnt16, mtl5.matrix_fixpnt16),
    )
    results.append(("fixpnt16", 16, rmse, mxe))

    # fixpnt8 (range [-8, 8) with 4 fractional bits — very tight)
    rmse, mxe = evaluate_precision(
        "fixpnt8",
        X_test,
        y_test,
        lambda: predict_universal(X_test, w, mtl5.vector_fixpnt8, mtl5.matrix_fixpnt8),
    )
    results.append(("fixpnt8", 8, rmse, mxe))

    for name, bits, rmse, mxe in results:
        rel = rmse / max(rmse_f64, 1e-30)
        marker = "  ←baseline" if name == "f64" else ""
        print(f"  {name:<12s}  {bits:4d}  {rmse:14.6e}  {mxe:14.6e}  {rel:10.2f}x{marker}")

    # Storage savings calculation
    print()
    print("Storage Footprint")
    print("=" * 70)
    n_weights = args.features
    n_inputs = X_test.size
    print(f"  {'precision':<12s}  {'weight bytes':>14s}  {'input bytes':>14s}  {'total':>10s}")
    print(f"  {'---------':<12s}  {'------------':>14s}  {'-----------':>14s}  {'-----':>10s}")
    for name, bits, _, _ in results:
        wb = n_weights * bits // 8
        ib = n_inputs * bits // 8
        total = wb + ib
        print(f"  {name:<12s}  {wb:14d}  {ib:14d}  {total:10d}")

    print()
    print("Observations")
    print("=" * 70)
    print("- posit16 typically achieves near-f32 accuracy at half the storage,")
    print("  making it attractive for edge inference and KPU deployment.")
    print("- posit8 trades accuracy for 4x storage reduction vs f32.")
    print("- fixpnt8 saturates aggressively if values exceed its small range.")
    print("- All forward passes run through MTL5's matvec — on KPU hardware,")
    print("  these operations execute natively without software emulation.")


if __name__ == "__main__":
    main()
