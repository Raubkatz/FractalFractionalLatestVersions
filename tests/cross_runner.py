"""
cross_runner.py  (called by test_torch_vs_keras3.py)

Runs every configuration of equivalence_configs.py with the Keras 3 OR the
PyTorch optimizers and stores the results.

    python cross_runner.py keras3 <keras3_dir>  <out.npz> <float64|float32> closed
    python cross_runner.py torch  <pytorch_dir> <out.npz> <float64|float32> closed
    python cross_runner.py ...                                            open <grads.npz>
    python cross_runner.py record - <grads.npz> float64 open      (writes grads.npz)

closed: each optimizer trains the 4-8-3 tanh MLP on its own weights
        (loss and gradients from one shared NumPy function).
open:   each optimizer receives the same pre-recorded gradient and loss
        sequence (grads.npz), so the inputs are identical in both frameworks
        and only the optimizer arithmetic is compared.

TensorFlow and PyTorch run in separate processes (see the README note on
importing both in one process).
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

FLAVOUR, CODE_DIR, OUT_FILE, DTYPE, MODE = sys.argv[1:6]
assert FLAVOUR in ("keras3", "torch", "record"), FLAVOUR
assert DTYPE in ("float64", "float32") and MODE in ("open", "closed")
RECORDED = np.load(sys.argv[6]) if MODE == "open" and FLAVOUR != "record" else None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, CODE_DIR)

from equivalence_configs import CONFIGS, MODULES  # noqa: E402

TORCH_MODULES = {
    "B": "class_b_fractional_optimizers_OHA_torch",
    "C": "class_c_fractional_optimizers_SR_torch",
    "M": "class_minimal_fractional_adaptive_optimizers_torch",
    "D": "class_d_related_work_optimizers_torch",
}
if FLAVOUR == "keras3":
    import tensorflow as tf
    import keras
    if DTYPE == "float64":
        # tf.cast(0.9, tf.float64) first makes a float32 tensor, so every Python
        # hyperparameter that the Keras code casts with tf.cast arrives in
        # float64 as its float32 rounding (0.8999999761581421). That is
        # harmless in float32 training (the normal case) but would hide the
        # comparison in float64, so in this process Python numbers are
        # converted straight to the target dtype. The optimizer code is unchanged.
        _tf_cast = tf.cast

        def _exact_cast(x, dtype, name=None):
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return tf.constant(x, dtype=dtype, name=name)
            return _tf_cast(x, dtype, name=name)

        tf.cast = _exact_cast
    mods = {k: __import__(v[1]) for k, v in MODULES.items()}
elif FLAVOUR == "torch":
    import torch
    mods = {k: __import__(v) for k, v in TORCH_MODULES.items()}

# ---------------------------------------------------------------------------
# Toy problem (same data and initial weights as equivalence_runner.py)
# ---------------------------------------------------------------------------
np_dtype = np.float64 if DTYPE == "float64" else np.float32
rng = np.random.default_rng(0)
X = rng.normal(size=(64, 4)).astype(np.float32)
Y = rng.integers(0, 3, size=(64,)).astype(np.int32)
INIT = [
    (rng.normal(size=(4, 8)) * 0.7).astype(np.float32),
    (rng.normal(size=(8,)) * 0.1).astype(np.float32),
    (rng.normal(size=(8, 3)) * 0.7).astype(np.float32),
    (rng.normal(size=(3,)) * 0.1).astype(np.float32),
]
N_STEPS = 60
BATCHES = [(X[i:i + 16].astype(np_dtype), Y[i:i + 16]) for i in range(0, 64, 16)]


def loss_and_grads(ws, xb, yb):
    """Softmax cross-entropy of a 4-8-3 tanh MLP and its gradients (NumPy)."""
    w0, b0, w2, b2 = ws
    h = np.tanh(xb @ w0 + b0)
    logits = h @ w2 + b2
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    n = len(yb)
    loss = float(-np.mean(np.log(p[np.arange(n), yb])))
    d = p.copy()
    d[np.arange(n), yb] -= 1.0
    d /= n
    dh = (d @ w2.T) * (1.0 - h * h)
    grads = [xb.T @ dh, dh.sum(axis=0), h.T @ d, d.sum(axis=0)]
    return loss, [g.astype(np_dtype) for g in grads]


def next_inputs(t, ws, xb, yb):
    """closed: loss/grads at the optimizer's own weights; open: recorded sequence."""
    if MODE == "closed":
        return loss_and_grads(ws, xb, yb)
    loss = float(RECORDED["loss"][t])
    grads = [RECORDED[f"g{i}"][t].astype(np_dtype) for i in range(len(INIT))]
    return loss, grads


def record_sequence(path, lr=0.05):
    """Gradient/loss sequence of plain float64 SGD on the toy problem (for open mode)."""
    global np_dtype
    saved, np_dtype = np_dtype, np.float64
    ws = [w.astype(np.float64) for w in INIT]
    losses, grads_seq = [], [[] for _ in INIT]
    for t in range(N_STEPS):
        xb, yb = BATCHES[t % len(BATCHES)]
        loss, grads = loss_and_grads(ws, xb.astype(np.float64), yb)
        losses.append(loss)
        for i, g in enumerate(grads):
            grads_seq[i].append(g)
        ws = [w - lr * g for w, g in zip(ws, grads)]
    np_dtype = saved
    np.savez(path, loss=np.array(losses), **{f"g{i}": np.stack(g) for i, g in enumerate(grads_seq)})


# ---------------------------------------------------------------------------
# Shared Bernoulli stream for the p = 0.5 runs
# ---------------------------------------------------------------------------
class SharedUniform:
    def __init__(self):
        self.rng = np.random.default_rng(2024)
        self.calls = 0

    def draw(self, n):
        self.calls += 1
        return self.rng.random(n).astype(np.float32)


def run_keras(cfg):
    kwargs = dict(cfg["kwargs"])
    lr = kwargs.pop("lr")
    variables = [keras.Variable(w.astype(np_dtype), dtype=DTYPE) for w in INIT]
    opt = getattr(mods[cfg["module"]], cfg["cls"])(learning_rate=lr, **kwargs)
    shared, original = SharedUniform(), tf.random.uniform
    tf.random.uniform = lambda shape, dtype=tf.float32, **kw: tf.constant(shared.draw(shape[0]), dtype)
    try:
        losses = []
        for t in range(N_STEPS):
            xb, yb = BATCHES[t % len(BATCHES)]
            loss, grads = next_inputs(t, [v.numpy() for v in variables], xb, yb)
            opt.apply_gradients(zip([tf.constant(g) for g in grads], variables))
            if cfg.get("loss_feedback"):
                opt.set_current_loss(loss)
            losses.append(loss)
    finally:
        tf.random.uniform = original
    out = {"w": [v.numpy().astype(np.float64) for v in variables], "loss": np.array(losses)}
    if hasattr(opt, "current_mix_values"):
        out["mix"] = np.array([float(x) for x in opt.current_mix_values()])
    if hasattr(opt, "current_alpha_values"):
        out["alpha"] = np.array([float(x) for x in opt.current_alpha_values()])
    return out, shared.calls


def run_torch(cfg):
    kwargs = dict(cfg["kwargs"])
    lr = float(np.float32(kwargs.pop("lr")))          # Keras lr variable is float32
    tdtype = torch.float64 if DTYPE == "float64" else torch.float32
    params = [torch.nn.Parameter(torch.tensor(w.astype(np_dtype), dtype=tdtype)) for w in INIT]
    opt = getattr(mods[cfg["module"]], cfg["cls"])(params, lr=lr, **kwargs)
    shared, original = SharedUniform(), torch.rand
    torch.rand = lambda n, **kw: torch.from_numpy(shared.draw(n))
    try:
        losses = []
        for t in range(N_STEPS):
            xb, yb = BATCHES[t % len(BATCHES)]
            loss, grads = next_inputs(t, [p.detach().numpy() for p in params], xb, yb)
            for p, g in zip(params, grads):
                p.grad = torch.from_numpy(g.copy())
            opt.step()
            if cfg.get("loss_feedback"):
                opt.set_current_loss(loss)
            losses.append(loss)
    finally:
        torch.rand = original
    out = {"w": [p.detach().numpy().astype(np.float64) for p in params], "loss": np.array(losses)}
    if hasattr(opt, "current_mix_values"):
        out["mix"] = np.array([float(x) for x in opt.current_mix_values()])
    if hasattr(opt, "current_alpha_values"):
        out["alpha"] = np.array([float(x) for x in opt.current_alpha_values()])
    return out, shared.calls


if FLAVOUR == "record":
    record_sequence(OUT_FILE)
    print(f"recorded {N_STEPS} gradient steps to {OUT_FILE}")
    sys.exit(0)

results = {}
for cfg in CONFIGS:
    out, calls = (run_keras if FLAVOUR == "keras3" else run_torch)(cfg)
    for i, w in enumerate(out["w"]):
        results[f"{cfg['id']}::w{i}"] = w
    for key in ("loss", "mix", "alpha"):
        if key in out:
            results[f"{cfg['id']}::{key}"] = out[key]
    results[f"{cfg['id']}::random_calls"] = np.array(calls)
np.savez(OUT_FILE, **results)
print(f"{FLAVOUR}/{DTYPE}/{MODE}: {len(CONFIGS)} configurations written to {OUT_FILE}")
