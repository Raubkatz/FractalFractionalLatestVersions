"""
Compare every activation in fractal_activation_functions_keras3.py with the
original fractal_activation_functions.py: forward values and gradients, in
eager mode and inside tf.function, on 20,001 points in [-5, 5] plus a random
2-D batch. The random-depth functions are compared with the same TF seed.

    python test_activations_equivalence.py <legacy_dir> <keras3_dir>
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

LEGACY_DIR, KERAS3_DIR = sys.argv[1], sys.argv[2]
sys.path.insert(0, KERAS3_DIR)
sys.path.insert(1, LEGACY_DIR)

import fractal_activation_functions as old  # noqa: E402
import fractal_activation_functions_keras3 as new  # noqa: E402

grid = np.linspace(-5.0, 5.0, 20001).astype(np.float32)
batch = np.random.default_rng(3).normal(scale=2.0, size=(64, 32)).astype(np.float32)

rows = []
all_ok = True
for name, new_fn in new.ALL_FRACTAL_ACTIVATIONS.items():
    old_fn = getattr(old, name)
    worst_val, worst_grad, exact = 0.0, 0.0, True
    for mode in ("eager", "graph"):
        for data in (grid, batch):
            results = []
            for fn in (old_fn, new_fn):
                def value_and_grad(x, fn=fn):
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        y = fn(x)
                    return y, tape.gradient(y, x)
                f = tf.function(value_and_grad) if mode == "graph" else value_and_grad
                tf.random.set_seed(7)
                y, g = f(tf.constant(data))
                results.append((y.numpy(), None if g is None else g.numpy()))
            (y0, g0), (y1, g1) = results
            assert y0.dtype == y1.dtype == np.float32, (name, y0.dtype, y1.dtype)
            same_nan = np.array_equal(np.isnan(y0), np.isnan(y1))
            dv = float(np.nanmax(np.abs(y0.astype(np.float64) - y1))) if y0.size else 0.0
            exact &= bool(np.array_equal(y0, y1, equal_nan=True)) and same_nan
            worst_val = max(worst_val, dv)
            if g0 is None or g1 is None:
                exact &= (g0 is None) == (g1 is None)
            else:
                finite = np.isfinite(g0) & np.isfinite(g1)
                dg = float(np.max(np.abs(g0[finite].astype(np.float64) - g1[finite]))) if finite.any() else 0.0
                exact &= bool(np.array_equal(g0, g1, equal_nan=True))
                worst_grad = max(worst_grad, dg)
    all_ok &= exact
    rows.append((name, worst_val, worst_grad, exact))

w = max(len(r[0]) for r in rows)
print(f"{'activation':{w}s} {'max|dy|':>10s} {'max|dgrad|':>11s}  bit-identical (values+gradients)")
for name, dv, dg, exact in rows:
    print(f"{name:{w}s} {dv:10.3e} {dg:11.3e}  {'yes' if exact else 'NO'}")
print(f"\n{sum(r[3] for r in rows)}/{len(rows)} activations bit-identical")
sys.exit(0 if all_ok else 1)
