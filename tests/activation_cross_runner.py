"""
activation_cross_runner.py  (called by test_activations_torch_vs_keras3.py)

Evaluates every fractal activation of the Keras 3 OR the PyTorch module on
fixed inputs and stores values and gradients.

    python activation_cross_runner.py keras3 <keras3_dir>  <out.npz>
    python activation_cross_runner.py torch  <pytorch_dir> <out.npz>

Inputs: 20,001 points in [-5, 5] and a random 64 x 32 batch, each in float32
and float64. The random-depth functions are evaluated with the sampled depth
fixed to its minimum, middle and maximum value (the random draw itself is
framework-specific and is tested separately).
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

FLAVOUR, CODE_DIR, OUT_FILE = sys.argv[1:4]
sys.path.insert(0, CODE_DIR)

grid = np.linspace(-5.0, 5.0, 20001)
batch = np.random.default_rng(3).normal(scale=2.0, size=(64, 32))
DATA = {"grid": grid, "batch": batch}
FRACTIONS = (0.0, 0.5, 1.0)

if FLAVOUR == "keras3":
    import tensorflow as tf
    import fractal_activation_functions_keras3 as faf

    def set_depth(fraction):
        faf._sample_random_depth = lambda min_terms=10, max_terms=50: tf.constant(
            int(round(min_terms + fraction * (max_terms - min_terms))), tf.int32)

    def value_and_grad(fn, x_np, dtype):
        x = tf.constant(x_np.astype(dtype))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = fn(x)
        g = tape.gradient(y, x)
        return y.numpy(), None if g is None else g.numpy()
else:
    import torch
    import fractal_activation_functions_torch as faf

    def set_depth(fraction):
        faf._sample_random_depth = lambda min_terms=10, max_terms=50, device=None: torch.tensor(
            int(round(min_terms + fraction * (max_terms - min_terms))), device=device)

    def value_and_grad(fn, x_np, dtype):
        x = torch.tensor(x_np.astype(dtype), requires_grad=True)
        y = fn(x)
        if not y.requires_grad:
            return y.detach().numpy(), None
        y.sum().backward()
        return y.detach().numpy(), x.grad.numpy()

results = {}
for name, fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
    fractions = FRACTIONS if name.startswith("random_depth") else (None,)
    for fraction in fractions:
        if fraction is not None:
            set_depth(fraction)
        for dname, data in DATA.items():
            for dtype in ("float32", "float64"):
                key = f"{name}|{dname}|{dtype}|{fraction}"
                y, g = value_and_grad(fn, data, dtype)
                results[key + "|y"] = y
                results[key + "|has_grad"] = np.array(g is not None)
                if g is not None:
                    results[key + "|g"] = g
np.savez(OUT_FILE, **results)
print(f"{FLAVOUR}: {len(faf.ALL_FRACTAL_ACTIVATIONS)} activations written to {OUT_FILE}")
