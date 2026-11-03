"""
Keras 3 integration test for every optimizer and the activation helpers:

  1. model.fit on a small Iris-shaped problem (BatchNorm + Dropout model, as
     in the paper), with and without XLA (jit_compile=True), loss finite and
     decreasing;
  2. get_config() -> from_config() round trip;
  3. model.save(".keras") -> keras.models.load_model(): predictions equal,
     optimizer class/config restored, optimizer variables restored, training
     continues;
  4. learning-rate schedule + clipnorm accepted;
  5. FractalActivation layer and Dense(activation=fractal fn) survive saving.

    python test_keras3_features.py <keras3_dir>
"""

import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras
import tensorflow as tf

sys.path.insert(0, sys.argv[1])

import class_b_tf_fractional_optimizers_OHA_keras3 as B  # noqa: E402
import class_c_tf_fractional_optimizers_SR_keras3 as C  # noqa: E402
import class_minimal_fractional_adaptive_optimizers_keras3 as M  # noqa: E402
import class_d_related_work_optimizers_keras3 as D  # noqa: E402
import fractal_activation_functions_keras3 as faf  # noqa: E402

OPTIMIZERS = [
    ("FSGD", lambda: B.FSGD(learning_rate=1e-2, momentum=0.9, vderiv=0.75)),
    ("FAdam", lambda: B.FAdam(learning_rate=1e-2, vderiv=1.25)),
    ("FAdagrad", lambda: B.FAdagrad(learning_rate=5e-2, vderiv=0.75)),
    ("FAdadelta", lambda: B.FAdadelta(learning_rate=1.0, vderiv=1.25)),
    ("FRMSprop", lambda: B.FRMSprop(learning_rate=1e-2, vderiv=0.75)),
    ("FAdamW", lambda: B.FAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("MemoryFSGD", lambda: C.MemoryFSGD(learning_rate=1e-2, momentum=0.9, vderiv=0.75, history_size=6)),
    ("MemoryFRMSprop", lambda: C.MemoryFRMSprop(learning_rate=1e-2, vderiv=1.25, history_size=6)),
    ("MemoryFAdam", lambda: C.MemoryFAdam(learning_rate=1e-2, vderiv=1.5, history_size=6)),
    ("MemoryFAdadelta", lambda: C.MemoryFAdadelta(learning_rate=1.0, vderiv=0.75, history_size=6)),
    ("MemoryFAdagrad", lambda: C.MemoryFAdagrad(learning_rate=5e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamW", lambda: C.MemoryFAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75, history_size=6)),
    ("MemoryFNadam", lambda: C.MemoryFNadam(learning_rate=1e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamax", lambda: C.MemoryFAdamax(learning_rate=1e-2, vderiv=0.75, history_size=6)),
    ("AdaptiveMemoryFSGD", lambda: M.AdaptiveMemoryFSGD(learning_rate=1e-2, momentum=0.9, vderiv=1.25)),
    ("AdaptiveMemoryFRMSprop", lambda: M.AdaptiveMemoryFRMSprop(learning_rate=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdam", lambda: M.AdaptiveMemoryFAdam(learning_rate=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdadelta", lambda: M.AdaptiveMemoryFAdadelta(learning_rate=1.0, vderiv=0.75)),
    ("AdaptiveMemoryFAdagrad", lambda: M.AdaptiveMemoryFAdagrad(learning_rate=5e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamW", lambda: M.AdaptiveMemoryFAdamW(learning_rate=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("AdaptiveMemoryFNadam", lambda: M.AdaptiveMemoryFNadam(learning_rate=1e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamax", lambda: M.AdaptiveMemoryFAdamax(learning_rate=1e-2, vderiv=0.75)),
    ("AdaGL", lambda: D.AdaGL(learning_rate=1e-2, alpha=0.75, history_size=10)),
    ("FCSGD_GL", lambda: D.FCSGD_GL(learning_rate=1e-2, momentum=0.9, alpha=0.75, history_size=11)),
    ("FCAdam_GL", lambda: D.FCAdam_GL(learning_rate=1e-2, alpha=0.75, history_size=11)),
    ("AOFGD_SGD", lambda: D.AOFGD_SGD(learning_rate=1e-2, momentum=0.9, alpha_init=0.75, alpha_max=1.5)),
    ("AOFGD_Adam", lambda: D.AOFGD_Adam(learning_rate=1e-2, alpha_init=1.25, alpha_max=1.5)),
]

rng = np.random.default_rng(0)
X = rng.uniform(size=(150, 4)).astype(np.float32)
Y = (X[:, 0] * 2 + X[:, 2]).astype(np.int32) % 3


def build_model(activation="tanh"):
    init = keras.initializers.GlorotUniform(seed=42)
    return keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(16, activation=activation, kernel_initializer=init),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation=activation, kernel_initializer=init),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax", kernel_initializer=init),
        ]
    )


failures = []
tmp = tempfile.mkdtemp()
print(f"{'optimizer':24s} {'fit':>14s} {'fit+XLA':>14s}  config  save/load  schedule+clip")
for name, make in OPTIMIZERS:
    status = {}
    try:
        # 1) fit, plain and XLA
        losses = {}
        for jit in (False, True):
            keras.utils.set_random_seed(1)
            model = build_model()
            model.compile(optimizer=make(), loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"], jit_compile=jit)
            h = model.fit(X, Y, epochs=8, batch_size=16, verbose=0)
            losses[jit] = h.history["loss"]
            assert np.all(np.isfinite(losses[jit])), f"non-finite loss (jit={jit})"
            # noisy tiny problem: compare the mean of the last 3 epochs with epoch 1
            assert np.mean(losses[jit][-3:]) < losses[jit][0], f"loss did not decrease (jit={jit})"
        # 2) config round trip
        opt = make()
        cfg = opt.get_config()
        opt2 = type(opt).from_config(cfg)
        assert opt2.get_config() == cfg, "config round trip changed values"
        # 3) save / load with optimizer state (model from the XLA run above)
        path = os.path.join(tmp, f"{name}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0))
        assert type(loaded.optimizer) is type(model.optimizer)
        assert loaded.optimizer.get_config() == model.optimizer.get_config()
        assert len(loaded.optimizer.variables) == len(model.optimizer.variables)
        for a, b in zip(model.optimizer.variables, loaded.optimizer.variables):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        loaded.fit(X, Y, epochs=1, batch_size=16, verbose=0)
        # 4) schedule + clipnorm via Keras 3 base arguments
        sched = keras.optimizers.schedules.CosineDecay(1e-2, decay_steps=50)
        base_cfg = {k: v for k, v in cfg.items()
                    if k not in ("learning_rate", "clipnorm", "name")}
        opt3 = type(opt)(learning_rate=sched, clipnorm=1.0, **base_cfg)
        keras.utils.set_random_seed(1)
        m3 = build_model()
        m3.compile(optimizer=opt3, loss="sparse_categorical_crossentropy")
        h3 = m3.fit(X, Y, epochs=3, batch_size=16, verbose=0)
        assert np.all(np.isfinite(h3.history["loss"]))
        print(f"{name:24s} {losses[False][0]:.3f}->{losses[False][-1]:.3f} "
              f"{losses[True][0]:.3f}->{losses[True][-1]:.3f}  ok      ok         ok")
    except Exception as e:  # noqa: BLE001
        failures.append((name, repr(e)[:300]))
        print(f"{name:24s} FAILED: {repr(e)[:300]}")

# 5) activations: Dense(activation=fn) and FractalActivation layer, save/load
for label, layer_act in [
    ("Dense(activation=modified_weierstrass_function_tanh)", faf.modified_weierstrass_function_tanh),
    ("Dense(activation=weierstrass_mandelbrot_function_xpsin)", faf.weierstrass_mandelbrot_function_xpsin),
]:
    try:
        keras.utils.set_random_seed(2)
        model = build_model(activation=layer_act)
        model.compile(optimizer=M.AdaptiveMemoryFAdam(learning_rate=1e-2, vderiv=1.25),
                      loss="sparse_categorical_crossentropy")
        model.fit(X, Y, epochs=2, batch_size=16, verbose=0)
        path = os.path.join(tmp, "act.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0))
        print(f"{label}: save/load ok")
    except Exception as e:  # noqa: BLE001
        failures.append((label, repr(e)[:300]))
        print(f"{label}: FAILED {repr(e)[:300]}")

try:
    keras.utils.set_random_seed(3)
    inp = keras.Input(shape=(4,))
    h = keras.layers.Dense(16)(inp)
    h = faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)(h)
    out = keras.layers.Dense(3, activation="softmax")(h)
    model = keras.Model(inp, out)
    model.compile(optimizer=C.MemoryFAdam(learning_rate=1e-2, vderiv=0.75, history_size=6),
                  loss="sparse_categorical_crossentropy")
    model.fit(X, Y, epochs=2, batch_size=16, verbose=0)
    ref = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(16)])
    ref.layers[0].set_weights(model.layers[1].get_weights())
    expected = faf.modified_weierstrass_function_tanh(ref(X), n_terms=50)
    got = keras.Model(inp, model.layers[2].output)(X)
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(got))
    path = os.path.join(tmp, "layer.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_array_equal(model.predict(X, verbose=0), loaded.predict(X, verbose=0))
    assert loaded.layers[2].function_kwargs == {"n_terms": 50}
    print("FractalActivation layer (n_terms=50): matches function, save/load ok")
except Exception as e:  # noqa: BLE001
    failures.append(("FractalActivation", repr(e)[:300]))
    print(f"FractalActivation: FAILED {repr(e)[:300]}")

print()
if failures:
    print(f"{len(failures)} FAILURES")
    for f in failures:
        print("  ", f)
    sys.exit(1)
print(f"All {len(OPTIMIZERS)} optimizers and the activation helpers passed.")
