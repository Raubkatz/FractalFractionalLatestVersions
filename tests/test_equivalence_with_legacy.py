"""
test_equivalence_with_legacy.py

Runs every configuration in equivalence_configs.py twice, once with the
legacy optimizer_v2 files (through tf_keras) and once with the Keras 3 files,
on the same closed-loop toy problem, in graph mode (tf.function) and in eager
mode, and compares the final weights (plus lambda_t / alpha_t diagnostics).

    pip install tf-keras==2.21.0            # only needed for this test
    python tests/test_equivalence_with_legacy.py <legacy_dir> <keras3_dir> [out_dir]

Environment variables:
    FAMILIES=BCMD        which files to test (B=class_b, C=class_c,
                         M=class_minimal, D=class_d); default: all
    DISABLE_GRAPPLER=1   switch off TensorFlow's graph optimizer (Grappler).
                         Grappler may fuse/reorder float ops differently for
                         the two (differently built) graphs, which produces
                         last-bit rounding differences in graph mode; with it
                         off, graph mode is bit-identical as well.
"""
import collections
import os
import subprocess
import sys
import tempfile

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR, KERAS3_DIR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="equivalence_")
py = sys.executable
summary = []
for mode in ("graph", "eager"):
    files = {}
    for flavour, d in (("legacy", LEGACY_DIR), ("keras3", KERAS3_DIR)):
        f = os.path.join(OUT, f"eq_{flavour}_{mode}.npz")
        r = subprocess.run([py, os.path.join(HERE, "equivalence_runner.py"), flavour, d, f, mode],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=HERE))
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-6000:]); sys.exit(1)
        print(r.stdout.strip().splitlines()[-1])
        files[flavour] = np.load(f)
    a, b = files["legacy"], files["keras3"]
    assert set(a.files) == set(b.files), set(a.files) ^ set(b.files)
    per_cfg = collections.defaultdict(lambda: [0.0, 0.0, True])
    for key in a.files:
        cid, what = key.split("::")
        if what == "loss":
            continue
        x, y = a[key].astype(np.float64), b[key].astype(np.float64)
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        rel = d / max(float(np.max(np.abs(x))), 1e-30) if x.size else 0.0
        per_cfg[cid][0] = max(per_cfg[cid][0], d)
        per_cfg[cid][1] = max(per_cfg[cid][1], rel)
        per_cfg[cid][2] &= bool(np.array_equal(a[key], b[key]))
    for cid, (d, rel, exact) in sorted(per_cfg.items()):
        summary.append((mode, cid, d, rel, exact))
w = max(len(s[1]) for s in summary)
n_exact = sum(s[4] for s in summary)
print(f"\n{'mode':6s} {'configuration':{w}s} {'max|diff|':>11s} {'max rel':>10s}  bit-identical")
for mode, cid, d, rel, exact in summary:
    print(f"{mode:6s} {cid:{w}s} {d:11.3e} {rel:10.3e}  {'yes' if exact else 'NO'}")
print(f"\n{n_exact}/{len(summary)} runs bit-identical; worst max|diff| = {max(s[2] for s in summary):.3e}; worst rel = {max(s[3] for s in summary):.3e}")

# Pass criterion: every eager run must be bit-identical. Graph-mode runs are
# reported; they are bit-identical with DISABLE_GRAPPLER=1.
eager_failures = [s for s in summary if s[0] == "eager" and not s[4]]
graph_failures = [s for s in summary if s[0] == "graph" and not s[4]]
if os.environ.get("DISABLE_GRAPPLER"):
    eager_failures += graph_failures
if eager_failures:
    print(f"FAILED: {len(eager_failures)} runs that must be bit-identical are not.")
    sys.exit(1)
print("PASSED.")
