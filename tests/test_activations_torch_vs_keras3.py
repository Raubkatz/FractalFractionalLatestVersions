"""
Cross-framework check: every PyTorch fractal activation against its Keras 3
twin, values and gradients.

    python tests/test_activations_torch_vs_keras3.py <keras3_dir> <pytorch_dir>

Each framework runs in its own process (activation_cross_runner.py) on the
same inputs: 20,001 points in [-5, 5] and a random 64 x 32 batch, in
float32 and float64. The random-depth functions are compared with the depth
fixed to its minimum, middle and maximum value.

Pass criteria (relative to the largest |value| or |gradient| of the Keras result):
- dtype of the output, presence of a gradient path, and the positions of
  NaN/inf in values and gradients must be identical;
- float64 inputs: values and gradients <= 1e-12;
- float32 inputs, functions that compute internally in float64 (the
  Weierstrass family): <= 1e-6 (i.e. the same float32 result up to rounding);
- float32 inputs, functions that compute in float32 (blancmange, decaying
  cosine, Cantor, and the random-depth blancmange / decaying cosine, which
  compute in float32 even for float64 input): <= 2e-5 for values and
  gradients, because sin/cos/tanh/sigmoid of the two libraries differ in
  the last bit;
- exception: random_depth_cantor_function_tf in float32 at its maximum
  depth (15 ternary levels multiply the last-bit difference of sigmoid by
  3^15, so a few points near the ternary boundaries fall on the other side):
  values <= 1e-4 and at most 0.5% of the gradients may differ by more than
  1e-3 (the same function in float64 must still pass at 1e-12).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KERAS3_DIR, TORCH_DIR = sys.argv[1], sys.argv[2]
out_dir = tempfile.mkdtemp(prefix="activations_torch_vs_keras3_")
files = {}
for flavour, code_dir in (("keras3", KERAS3_DIR), ("torch", TORCH_DIR)):
    files[flavour] = os.path.join(out_dir, f"{flavour}.npz")
    r = subprocess.run([sys.executable, os.path.join(HERE, "activation_cross_runner.py"),
                        flavour, code_dir, files[flavour]], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(f"{flavour} runner failed")
    print("  " + r.stdout.strip().splitlines()[-1])
K, T = np.load(files["keras3"]), np.load(files["torch"])

FLOAT32_INTERNAL = {"modulated_blancmange_curve", "decaying_cosine_function_tf",
                    "cantor_function_tf", "cantor_set_activation_tf",
                    "random_depth_cantor_function_tf"}
ALWAYS_FLOAT32 = {"random_depth_modulated_blancmange_curve",
                  "random_depth_decaying_cosine_function_tf"}


def tolerance(name, dtype):
    if name in ALWAYS_FLOAT32 or (dtype == "float32" and name in FLOAT32_INTERNAL):
        return 2e-5
    return 1e-12 if dtype == "float64" else 1e-6


def rel(a, b, mask):
    if not mask.any():
        return 0.0
    a64, b64 = a[mask].astype(np.float64), b[mask].astype(np.float64)
    return float(np.max(np.abs(a64 - b64)) / max(np.max(np.abs(a64)), 1e-300))


cases = sorted({k.rsplit("|", 1)[0] for k in K.files})
summary = {}
for case in cases:
    name, data, dtype, fraction = case.split("|")
    yk, yt = K[case + "|y"], T[case + "|y"]
    hk, ht = bool(K[case + "|has_grad"]), bool(T[case + "|has_grad"])
    ok = yk.dtype == yt.dtype and hk == ht
    ok &= np.array_equal(np.isnan(yk), np.isnan(yt)) and np.array_equal(np.isinf(yk), np.isinf(yt))
    dv = rel(yk, yt, np.isfinite(yk))
    dg, outlier_share = 0.0, 0.0
    if hk and ht:
        gk, gt = K[case + "|g"], T[case + "|g"]
        ok &= np.array_equal(np.isnan(gk), np.isnan(gt)) and np.array_equal(np.isinf(gk), np.isinf(gt))
        fin = np.isfinite(gk)
        dg = rel(gk, gt, fin)
        scale = max(np.max(np.abs(gk[fin])), 1e-300) if fin.any() else 1.0
        outlier_share = float(np.mean(np.abs(gk[fin].astype(np.float64) - gt[fin]) > 1e-3 * scale))
    tol = tolerance(name, dtype)
    if name == "random_depth_cantor_function_tf" and dtype == "float32" and fraction == "1.0":
        ok &= dv <= 1e-4 and outlier_share <= 0.005
    else:
        ok &= dv <= tol and dg <= tol
    s = summary.setdefault((name, dtype), dict(dv=0.0, dg=0.0, out=0.0, ok=True, tol=tol))
    s["dv"], s["dg"] = max(s["dv"], dv), max(s["dg"], dg)
    s["out"], s["ok"] = max(s["out"], outlier_share), s["ok"] and bool(ok)

print(f"\n{'activation':55s} {'input':8s} {'values':>9s} {'grads':>9s} {'tol':>7s} {'grad>1e-3':>9s}")
for (name, dtype), s in summary.items():
    print(f"{name:55s} {dtype:8s} {s['dv']:9.1e} {s['dg']:9.1e} {s['tol']:7.0e} {100 * s['out']:8.2f}%  "
          f"{'PASS' if s['ok'] else 'FAIL'}")
n_ok = sum(s["ok"] for s in summary.values())
print(f"\n{n_ok}/{len(summary)} activation x dtype combinations PASS "
      f"({len(cases)} cases: 18 activations x 2 inputs x 2 dtypes, random depth at 3 depths)")
sys.exit(0 if n_ok == len(summary) else 1)
