"""
Cross-framework check: every PyTorch optimizer against its Keras 3 twin.

    python tests/test_torch_vs_keras3.py <keras3_dir> <pytorch_dir> [float64|float32]

All 167 configurations of equivalence_configs.py (every class, the
classical limits, momentum / Nesterov / centered / AMSGrad variants, the
loss-feedback runs with set_current_loss() after every step, and the
Bernoulli p = 0.5 runs of FCSGD_GL / FCAdam_GL) are run for 60 steps in a
Keras 3 process and in a PyTorch process (cross_runner.py) and compared.

Two experiments:

1) OPEN LOOP. Both optimizers receive the same recorded sequence of 60
   gradients and losses (from plain SGD on the toy problem), so the inputs
   are identical and only the optimizer arithmetic is compared.
2) CLOSED LOOP. Each optimizer trains the 4-8-3 tanh MLP on its own weights
   (loss and gradients from one shared NumPy function), i.e. normal training.

float64 (strict): pass if max |w_torch - w_keras| / max |w_keras| and the
   same for the final lambda_t / alpha_t values are <= 1e-12 (open loop) and
   <= 1e-10 (closed loop, also for the loss curve). The two math libraries
   differ only in the last bits of pow/exp/sqrt/lgamma.
   Note: tf.cast(0.9, tf.float64) rounds through float32, so in float64 the
   Keras code would use float32-rounded hyperparameters; the Keras process
   of this test converts Python numbers straight to float64 (see
   cross_runner.py). This does not touch the optimizer code and makes no
   difference in float32.
float32 (normal training precision): float32 round-off is amplified by some
   updates (most of all by AOFGD's (|theta| + eps)^(1 - alpha) for weights
   near zero), so the difference PyTorch-vs-Keras is compared with the
   float32 error of the Keras version itself (Keras float32 vs Keras
   float64). Pass if, for weights, lambda/alpha and the loss curve,
       |torch32 - keras32| <= max(10 * |keras32 - keras64|, 1e-5)  (relative).

For the p = 0.5 runs tf.random.uniform and torch.rand are replaced by the
same seeded NumPy stream, and the test checks that both frameworks draw one
fresh mask per variable and step. Keras keeps its learning rate in a
float32 variable, so the PyTorch optimizers get the float32-rounded rate.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KERAS3_DIR, TORCH_DIR = sys.argv[1], sys.argv[2]
DTYPE = sys.argv[3] if len(sys.argv) > 3 else "float64"
TOL = {"open": 1e-12, "closed": 1e-10}        # float64
FLOOR32, FACTOR32 = 1e-5, 10.0                   # float32
N_STEPS, N_VARS = 60, 4
sys.path.insert(0, HERE)
from equivalence_configs import CONFIGS  # noqa: E402

t0 = time.time()
out_dir = tempfile.mkdtemp(prefix="torch_vs_keras3_")
runner = os.path.join(HERE, "cross_runner.py")


def run(args):
    r = subprocess.run([sys.executable, runner] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(f"runner failed: {args}")
    print("  " + r.stdout.strip().splitlines()[-1])


grads_file = os.path.join(out_dir, "grads.npz")
run(["record", "-", grads_file, "float64", "open"])
files = {}
for mode in ("open", "closed"):
    for flavour, code_dir in (("keras3", KERAS3_DIR), ("torch", TORCH_DIR)):
        files[flavour, mode] = os.path.join(out_dir, f"{flavour}_{DTYPE}_{mode}.npz")
        run([flavour, code_dir, files[flavour, mode], DTYPE, mode]
            + ([grads_file] if mode == "open" else []))
    if DTYPE == "float32":       # float64 Keras reference for the float32 criterion
        files["keras64", mode] = os.path.join(out_dir, f"keras3_float64_{mode}.npz")
        run(["keras3", KERAS3_DIR, files["keras64", mode], "float64", mode]
            + ([grads_file] if mode == "open" else []))


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


def errors(A, B, cid):
    e = {"w": max(rel(A[f"{cid}::w{i}"], B[f"{cid}::w{i}"]) for i in range(N_VARS)), "d": 0.0}
    for key in ("mix", "alpha"):
        name = f"{cid}::{key}"
        if name in A.files or name in B.files:
            assert name in A.files and name in B.files, name
            e["d"] = max(e["d"], rel(A[name], B[name]))
    la, lb = A[f"{cid}::loss"], B[f"{cid}::loss"]
    e["loss"] = float(np.max(np.abs(la - lb) / np.abs(la)))
    return e


def compare(mode):
    K, T = np.load(files["keras3", mode]), np.load(files["torch", mode])
    R = np.load(files["keras64", mode]) if DTYPE == "float32" else None
    rows = []
    for cfg in CONFIGS:
        cid = cfg["id"]
        e = errors(K, T, cid)
        k_calls, t_calls = int(K[f"{cid}::random_calls"]), int(T[f"{cid}::random_calls"])
        random_ok = k_calls == t_calls
        if cfg["kwargs"].get("perturbation_prob", 1.0) not in (0.0, 1.0):
            random_ok &= k_calls == N_STEPS * N_VARS     # one fresh mask per variable and step
        finite = all(np.isfinite(T[f"{cid}::w{i}"]).all() for i in range(N_VARS))
        if DTYPE == "float64":
            ok = all(e[k] <= TOL[mode] for k in ("w", "d", "loss"))
            ref = None
        else:
            ref = errors(R, K, cid)          # float32 error of the Keras version itself
            ok = all(e[k] <= max(FACTOR32 * ref[k], FLOOR32) for k in ("w", "d", "loss"))
        ok = ok and random_ok and finite
        rows.append(dict(module=cfg["module"], id=cid, ok=ok, calls=(k_calls, t_calls), ref=ref, **e))
    return rows


all_ok = True
for mode in ("open", "closed"):
    rows = compare(mode)
    crit = (f"torch-vs-keras <= {TOL[mode]:g} relative" if DTYPE == "float64"
            else f"torch-vs-keras <= max({FACTOR32:g} x keras32-vs-keras64, {FLOOR32:g})")
    print(f"\n=== {mode.upper()} LOOP, {DTYPE}, {N_STEPS} steps, pass if {crit} ===")
    head = f"{'family':6s} {'configuration':70s} {'weights':>8s} {'lam/alp':>8s} {'loss':>8s}"
    print(head + ("" if DTYPE == "float64" else f"   {'k32-k64 w':>9s} {'loss':>8s}"))
    for r in rows:
        line = f"{r['module']:6s} {r['id'][:70]:70s} {r['w']:8.1e} {r['d']:8.1e} {r['loss']:8.1e}"
        if r["ref"] is not None:
            line += f"   {r['ref']['w']:9.1e} {r['ref']['loss']:8.1e}"
        print(line + f"  {'PASS' if r['ok'] else 'FAIL'}")
    print()
    for fam in "BCMD":
        fr = [r for r in rows if r["module"] == fam]
        print(f"  family {fam}: {sum(r['ok'] for r in fr)}/{len(fr)} PASS, "
              f"worst weights {max(r['w'] for r in fr):.1e}, "
              f"worst lambda/alpha {max(r['d'] for r in fr):.1e}, "
              f"worst loss {max(r['loss'] for r in fr):.1e}")
    n_ok = sum(r["ok"] for r in rows)
    print(f"  {mode} loop: {n_ok}/{len(rows)} PASS")
    for r in rows:
        if not r["ok"]:
            print("  FAIL", r)
    all_ok &= n_ok == len(rows)

print(f"\nfinished in {time.time() - t0:.0f} s: {'ALL PASS' if all_ok else 'FAILURES'}")
sys.exit(0 if all_ok else 1)
