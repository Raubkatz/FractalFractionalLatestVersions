"""
Memory check: train several models one after another in the same process
(as the hello-world tours do) and make sure the resident memory does not
grow by the size of a training graph per model.

    python test_memory_flat.py <keras3_dir>

Prints the resident set size after each model. With the Keras built-in Adam
the growth is roughly 130-150 MB per model with this activation (TensorFlow
keeps every traced graph alive, see README); with the fractional optimizers
it should stay roughly flat.
"""

import gc
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

sys.path.insert(0, sys.argv[1])

import fractal_activation_functions_keras3 as faf  # noqa: E402
from class_b_tf_fractional_optimizers_OHA_keras3 import FAdam  # noqa: E402
from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam  # noqa: E402
from class_minimal_fractional_adaptive_optimizers_keras3 import AdaptiveMemoryFAdam  # noqa: E402
from class_d_related_work_optimizers_keras3 import AdaGL, AOFGD_Adam  # noqa: E402


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


X = np.random.default_rng(0).random((105, 4)).astype(np.float32)
Y = np.random.default_rng(1).integers(0, 3, 105).astype(np.int32)
makers = [
    lambda: AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75),
    lambda: MemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=6),
    lambda: FAdam(learning_rate=1e-3, vderiv=0.75),
    lambda: AdaGL(learning_rate=1e-3, alpha=0.75),
    lambda: AOFGD_Adam(learning_rate=1e-3, alpha_init=0.75, alpha_max=1.5),
    lambda: AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75),
]
readings = []
for i, make in enumerate(makers):
    keras.backend.clear_session()
    keras.utils.set_random_seed(1)
    model = keras.Sequential([
        keras.Input(shape=(4,)),
        keras.layers.Dense(32, activation=faf.decaying_cosine_function_tf),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation=faf.decaying_cosine_function_tf),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=make(), loss="sparse_categorical_crossentropy")
    model.fit(X, Y, epochs=2, batch_size=16, verbose=0)
    del model
    gc.collect()
    readings.append(rss_mb())
    print(f"model {i + 1}: RSS {readings[-1]:.0f} MB", flush=True)

growth_per_model = (readings[-1] - readings[1]) / (len(readings) - 2)
print(f"average growth per model after the second one: {growth_per_model:.1f} MB")
sys.exit(0 if growth_per_model < 40 else 1)
