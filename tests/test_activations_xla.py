"""Every fractal activation inside model.fit with jit_compile=True (XLA)."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, keras
sys.path.insert(0, sys.argv[1])
import fractal_activation_functions_keras3 as faf
import class_minimal_fractional_adaptive_optimizers_keras3 as M
rng = np.random.default_rng(0)
X = rng.uniform(size=(64, 4)).astype(np.float32); Y = rng.integers(0, 3, 64).astype(np.int32)
bad = []
for name, fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(8, activation=fn),
                          keras.layers.Dense(3, activation="softmax")])
    m.compile(optimizer=M.AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=1.25),
              loss="sparse_categorical_crossentropy", jit_compile=True)
    try:
        h = m.fit(X, Y, epochs=2, batch_size=16, verbose=0)
        ok = np.all(np.isfinite(h.history["loss"]))
        print(f"{name:55s} XLA fit ok, loss {h.history['loss'][-1]:.4f}" + ("" if ok else "  NON-FINITE"))
        if not ok: bad.append(name)
    except Exception as e:
        bad.append(name); print(f"{name:55s} FAILED {repr(e)[:200]}")
print("failures:", bad); sys.exit(1 if bad else 0)
