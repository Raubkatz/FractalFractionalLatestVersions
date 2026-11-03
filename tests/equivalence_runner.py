"""
equivalence_runner.py  (called by test_equivalence_with_legacy.py)

Runs one optimizer family on a fixed, closed-loop toy problem and stores the
final parameters (and optimizer diagnostics) in an .npz file.

    python equivalence_runner.py legacy  <legacy_dir> <out.npz> <mode>
    python equivalence_runner.py keras3  <keras3_dir> <out.npz> <mode>

mode = "graph" (train step wrapped in tf.function) or "eager".

"legacy" runs the original optimizer_v2 files through tf_keras
(TF_USE_LEGACY_KERAS=1); "keras3" runs the new Keras 3 files. Both use the
same initial weights, the same mini-batches and the same TensorFlow ops for
the loss and gradients, so any difference comes from the optimizers.
"""

import os
import sys

FLAVOUR, CODE_DIR, OUT_FILE, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

if FLAVOUR == "legacy":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

if FLAVOUR == "legacy":
    # class_b / class_d subclass tf.keras.optimizers.Optimizer directly and
    # expect the optimizer_v2 API, which tf_keras exposes as .legacy.Optimizer.
    tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Optimizer

if os.environ.get("DISABLE_GRAPPLER"):
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

sys.path.insert(0, CODE_DIR)

from equivalence_configs import CONFIGS, MODULES  # noqa: E402

FAMILIES = os.environ.get("FAMILIES", "".join(MODULES)).strip()
modules = {}
for key, (legacy_name, keras3_name) in MODULES.items():
    if key in FAMILIES:
        modules[key] = __import__(legacy_name if FLAVOUR == "legacy" else keras3_name)

if FLAVOUR == "keras3":
    import keras

# ---------------------------------------------------------------------------
# Fixed toy problem: 4-8-3 tanh MLP, softmax cross-entropy, 64 samples,
# mini-batches of 16 cycled in a fixed order.
# ---------------------------------------------------------------------------
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
BATCHES = [(X[i:i + 16], Y[i:i + 16]) for i in range(0, 64, 16)]


def make_vars():
    if FLAVOUR == "legacy":
        return [tf.Variable(w) for w in INIT]
    return [keras.Variable(w) for w in INIT]


def loss_and_grads(variables, xb, yb):
    ws = [tf.convert_to_tensor(v) for v in variables]
    with tf.GradientTape() as tape:
        tape.watch(ws)
        h = tf.tanh(tf.matmul(xb, ws[0]) + ws[1])
        logits = tf.matmul(h, ws[2]) + ws[3]
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yb, logits=logits)
        )
    return loss, tape.gradient(loss, ws)


results = {}
for cfg in CONFIGS:
    if MODE not in cfg.get("modes", ("graph", "eager")) or cfg["module"] not in modules:
        continue
    tf.random.set_seed(1234)
    variables = make_vars()
    kwargs = dict(cfg["kwargs"])
    lr_value = kwargs.pop("lr")
    if FLAVOUR == "legacy":
        opt = getattr(modules[cfg["module"]], cfg["cls"])(lr=lr_value, **kwargs)
    else:
        opt = getattr(modules[cfg["module"]], cfg["cls"])(learning_rate=lr_value, **kwargs)

    loss_feedback = cfg.get("loss_feedback", False)

    def step(xb, yb):
        loss, grads = loss_and_grads(variables, xb, yb)
        opt.apply_gradients(zip(grads, variables))
        return loss

    step_fn = tf.function(step) if MODE == "graph" else step

    losses = []
    for t in range(N_STEPS):
        xb, yb = BATCHES[t % len(BATCHES)]
        loss = step_fn(tf.constant(xb), tf.constant(yb))
        if loss_feedback:
            opt.set_current_loss(float(loss))
        losses.append(float(loss))

    out = {"loss": np.array(losses, dtype=np.float64)}
    for i, v in enumerate(variables):
        out[f"w{i}"] = np.array(v.numpy())
    if hasattr(opt, "current_mix_values"):
        out["mix"] = np.array([float(x) for x in opt.current_mix_values()])
    if hasattr(opt, "current_alpha_values"):
        out["alpha"] = np.array([float(x) for x in opt.current_alpha_values()])
    for k, v in out.items():
        results[f"{cfg['id']}::{k}"] = v

np.savez(OUT_FILE, **results)
print(f"{FLAVOUR}/{MODE}: {len({k.split('::')[0] for k in results})} configurations written to {OUT_FILE}")
