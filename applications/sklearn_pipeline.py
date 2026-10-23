#!/usr/bin/env python3
"""scikit-learn on MTL5 storage: does reduced-precision feature storage cost accuracy?

A regression problem where the feature matrix lives in MTL5 containers: built
with an MTL5 product, then stored at float64, float32, posit16 and fp16. Each
container goes straight into a scikit-learn pipeline (StandardScaler + Ridge),
because MTL5's dense containers speak the NumPy array protocol. For a native
container ``np.asarray`` is a view of its storage; for a Universal one it is a
float64 copy of the rounded values.

What the comparison measures is the cost of the STORAGE format. scikit-learn
computes in float64 (or float32) whatever it is given, so the arithmetic is the
same for every row of the table; only the stored features differ.

Two things scikit-learn does not take directly, and the way round each:

  - MTL5 sparse matrices: pass ``mtl5.sparse.to_scipy(S)``.
  - float32 kept as float32: an estimator that accepts either float type sees
    the container's dtype as a name, not a NumPy dtype, and picks float64.
    ``np.asarray(M)`` first is zero-copy and keeps float32.

Requires scikit-learn. Run with:

    pip install 'mtl5[sklearn]'
    python sklearn_pipeline.py [--samples 2000] [--features 8] [--noise 0.1]
"""

import argparse

import numpy as np

import mtl5

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit(
        "This demo requires scikit-learn. Install with:\n    pip install 'mtl5[sklearn]'"
    )

STORAGE = ["f64", "f32", "posit16", "fp16"]


def make_problem(n_samples: int, n_features: int, noise: float, rng: np.random.Generator):
    """Correlated features (latent factors times a mixing matrix, an MTL5
    product) and a linear target with noise."""
    latent = mtl5.matrix(rng.standard_normal((n_samples, n_features)))
    mixing = mtl5.matrix(np.eye(n_features) + 0.4 * rng.standard_normal((n_features, n_features)))
    X = (latent @ mixing).to_numpy()
    w = rng.uniform(-2.0, 2.0, n_features)
    y = X @ w + 1.0 + noise * rng.standard_normal(n_samples)
    return X, y


def store(X: np.ndarray, kind: str):
    """The feature matrix as an MTL5 container of the given element type."""
    if kind == "f64":
        return mtl5.matrix(X)
    if kind == "f32":
        return mtl5.matrix(X.astype(np.float32))
    return mtl5.convert(X, kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = make_problem(args.samples, args.features, args.noise, rng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed
    )

    print(
        f"Ridge regression, {args.samples} samples x {args.features} features, noise {args.noise}"
    )
    print(
        f"{'storage':>8} {'bytes/feature':>14} {'max |rounding|':>15} {'test MSE':>12} {'R^2':>10}"
    )
    print("-" * 63)
    for kind in STORAGE:
        A_train, A_test = store(X_train, kind), store(X_test, kind)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
        model.fit(A_train, mtl5.vector(y_train))
        pred = model.predict(A_test)
        rounding = np.abs(np.asarray(A_train, dtype=np.float64) - X_train).max()
        nbytes = {"f64": 8, "f32": 4, "posit16": 2, "fp16": 2}[kind]
        print(
            f"{kind:>8} {nbytes:>14} {rounding:>15.2e} "
            f"{mean_squared_error(y_test, pred):>12.4e} {r2_score(y_test, pred):>10.7f}"
        )

    print(
        "\nThe noise floor is MSE = noise^2 =",
        f"{args.noise**2:.4e}.",
        "A 16-bit format whose rounding",
    )
    print("stays well under the noise costs nothing measurable; one whose rounding")
    print("approaches it shows up in the MSE first.")


if __name__ == "__main__":
    main()
