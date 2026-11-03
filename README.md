# Fractional Optimizers × Fractal Activations — Keras 3 edition

Keras 3 / TensorFlow 2.21 implementations of the fractional optimizers and fractal activation functions used in

> S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger.
> *Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks.*
> Fractal and Fractional, 2026 (accepted). Preprint: [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · Experiment code: [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026)

The fractal activation functions were introduced in

> S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger.
> *Fractals in neural networks: Introducing a new class of activation functions.*
> Communications in Nonlinear Science and Numerical Simulation **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · Preprint: [doi:10.20944/preprints202508.0478.v1](https://doi.org/10.20944/preprints202508.0478.v1) · Code: [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions)

> **This repository collects work by many authors.** Its only purpose is to provide **one platform on which all of these optimizers can be used, side by side, with Keras 3**.
>
> **The following methods are not our work, and we do not claim them:**
> - the **Herrera-style fractional optimizers** (`FSGD`, `FAdam`, `FRMSprop`, `FAdadelta`, `FAdagrad`, `FAdamW`), developed by **Oscar Herrera-Alcántara** and co-authors [[H1]](#ref-h1), [[H2]](#ref-h2);
> - the **related-work optimizers**: **AdaGL** by Chen, Zhang & Mu [[R1]](#ref-r1); **FCSGD_GL / FCAdam_GL** by Zhou, Zhao & Huang [[R2]](#ref-r2); **AOFGD** by Xiang, Zhao, Su, Zhou, Ye & Huang [[R3]](#ref-r3);
> - the **Cantor-based activations**, by Chipera, Du & Tsapara [[C1]](#ref-c1);
> - the **classical base optimizers** (SGD, momentum, Adam, RMSprop, …) [[B1]–[B10]](#base-optimizers). The baselines used here are Keras' own implementations.
>
> The ideas, the mathematics and the credit belong to these authors. Our part is limited to re-implementing their published update rules in Keras 3, from the papers, and testing them side by side. These classes are **not the original authors' code**, and any error in them is ours.
>
> **If you use any of these optimizers or activations, cite the original publications** listed under [Attribution and how to cite](#attribution-and-how-to-cite), with the links given there. You may additionally cite this repository for the Keras 3 implementation. Only the **GL-memory** and **adaptive-memory** optimizers come from our article [[O1]](#ref-o1), and the fractal activations from our earlier article [[O2]](#ref-o2).

The original classes were written for the old `optimizer_v2` API (`_set_hyper`, `_create_slots`, `_resource_apply_dense`, `lr=`). That API no longer exists in Keras 3, which has been the default `tf.keras` since TensorFlow 2.16. Under a current TensorFlow, the original files fail on import or when an optimizer is created.

The files in this folder implement **the same update rules, operation for operation**, on the Keras 3 optimizer API (`build`, `update_step`). They have been checked against the original files: the same weights come out after the same training steps, bit for bit (see [Verification](#verification)).

- 27 optimizers in four families, plus the Keras built-in baselines
- 18 fractal activation functions, plus a `FractalActivation` layer
- one linear, step-by-step [hello-world script](#hello-world-iris-in-one-run) on Iris

---

## Attribution and how to cite

Whose work is what:

| What | Classes in this repository | Original work, please cite |
|---|---|---|
| **Herrera-style fractional optimizers** (Caputo-type gradient scaling) | `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` | **Oscar Herrera-Alcántara** and co-authors: Herrera-Alcántara, *Appl. Sci.* 2022 [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264) [[H1]](#ref-h1); Herrera-Alcántara & Castelán-Aguilar, *Fractal Fract.* 2023 [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500) [[H2]](#ref-h2) |
| **AdaGL** | `AdaGL` | **Chen, Zhang & Mu**, *Neural Process. Lett.* 2024 [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7) [[R1]](#ref-r1) |
| **FCSGD_GL / FCAdam_GL** (Grünwald–Letnikov optimizers with stochastic perturbation) | `FCSGD_GL`, `FCAdam_GL` | **Zhou, Zhao & Huang**, *Mathematics* 2023 [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316) [[R2]](#ref-r2) |
| **AOFGD** (adaptive-order fractional gradient descent) | `AOFGD_SGD`, `AOFGD_Adam` | **Xiang, Zhao, Su, Zhou, Ye & Huang**, SSRN preprint 2025 [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) [[R3]](#ref-r3) |
| **Classical base optimizers** (the Keras baselines, and the base update rules of every fractional variant) | `keras.optimizers.*`; the SGD/Adam/RMSprop/Adadelta/Adagrad/AdamW/Nadam/Adamax parts of all classes | the original papers [[B1]–[B10]](#base-optimizers) |
| Grünwald–Letnikov fractional derivative and short-memory principle (background of all GL-based classes) | `MemoryF*`, `AdaptiveMemoryF*`, `AdaGL`, `FC*_GL` | Podlubny (1999) [[G1]](#ref-g1); Wei et al. (2017) [[G2]](#ref-g2) |
| **Adaptive memory-fractional optimizers** (contribution of our paper) | `AdaptiveMemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | Raubitzek et al. (2026, Fractal and Fractional) [[O1]](#ref-o1) |
| **GL-memory optimizers** (our implementation of the finite-history GL gradient substitution, with norm matching and descent safeguard) | `MemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | Raubitzek et al. (2026, Fractal and Fractional) [[O1]](#ref-o1), plus the GL background [[G1]](#ref-g1), [[G2]](#ref-g2); see also Zhou et al. (2023) [[R2]](#ref-r2) |
| **Fractal activation functions** (Blancmange, decaying cosine, modified Weierstrass, Weierstrass–Mandelbrot variants) | the nine fixed-depth functions in `fractal_activation_functions_keras3.py` | Raubitzek et al., *CNSNS* 2026 [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) [[O2]](#ref-o2); mathematical origins [[F1]–[F5]](#fractal-functions-mathematical-origins) |
| Random-depth variants | `random_depth_*` | extensions in this code base, not part of the published experiments; cite [[O2]](#ref-o2) for the underlying functions |
| **Cantor-based activations** | `cantor_function_tf`, `cantor_set_activation_tf`, `random_depth_cantor_function_tf` | **Chipera, Du & Tsapara**, [arXiv:2512.14675](https://arxiv.org/abs/2512.14675) (2025) [[C1]](#ref-c1); Cantor function/set [[F5]](#ref-f5) |
| Software and data used by the scripts | — | Keras [[S1]](#ref-s1), TensorFlow [[S2]](#ref-s2), scikit-learn [[S3]](#ref-s3); Iris data [[D1]](#ref-d1) |

**Original code.** Oscar Herrera-Alcántara's own TensorFlow-Keras implementations of his optimizers are available at [ia.azc.uam.mx](http://ia.azc.uam.mx/), as stated in the data availability section of [[H1]](#ref-h1). For AdaGL, FCSGD_GL/FCAdam_GL and AOFGD we did not find code published by the authors; the AdaGL article states that code "could be made available at a later stage". If the authors publish their code, it is the authoritative reference.

The Herrera-style and related-work classes are implemented from the published descriptions; the update rules used are written out in each class docstring. For example, AOFGD's convergence-evaluation factor is computed here from consecutive gradient norms, and FCSGD_GL/FCAdam_GL use 10 historical terms. They may differ in details from the original authors' own implementations, so if a result must match the original method exactly, check it against the original publication or code.

The full reference list and ready-to-use BibTeX entries are under [References](#references) at the end of this file.

---

## Contents

| New file (Keras 3) | Replaces (legacy) | Contents |
|---|---|---|
| `class_b_tf_fractional_optimizers_OHA_keras3.py` | `class_b_tf_fractional_optimizers_OHA.py` | Herrera-style: `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` |
| `class_c_tf_fractional_optimizers_SR_keras3.py` | `class_c_tf_fractional_optimizers_SR.py` | Grünwald–Letnikov memory: `MemoryFSGD`, `MemoryFRMSprop`, `MemoryFAdam`, `MemoryFAdadelta`, `MemoryFAdagrad`, `MemoryFAdamW`, `MemoryFNadam`, `MemoryFAdamax` |
| `class_minimal_fractional_adaptive_optimizers_keras3.py` | `class_minimal_fractional_adaptive_optimizers.py` | Adaptive memory: `AdaptiveMemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax`, plus the `AdaptiveLossFeedback` callback |
| `class_d_related_work_optimizers_keras3.py` | `class_d_related_work_optimizers.py` | Related work: `AdaGL`, `FCSGD_GL`, `FCAdam_GL`, `AOFGD_SGD`, `AOFGD_Adam` |
| `fractal_activation_functions_keras3.py` | `fractal_activation_functions.py` | All 18 fractal activations, `PAPER_FRACTAL_ACTIVATIONS`, `ALL_FRACTAL_ACTIVATIONS`, `get_fractal_activation()`, `FractalActivation` layer |
| `hello_world_iris.py` | — | One run on Iris that uses every optimizer and every activation, with full classification metrics |
| `tests/` | — | Equivalence tests against the legacy files and Keras 3 integration tests |
| `requirements.txt` | — | The pinned versions below, for plain `pip` |

Class names, constructor arguments and defaults are unchanged. In existing code, changing the import line is usually enough.

---

## Installation

Create the environment with:

```
conda create -n fractal-fractional-keras3 python=3.12
conda activate fractal-fractional-keras3

pip install \
    numpy==2.5.4 \
    scipy==1.18.1 \
    pandas==3.1.0 \
    scikit-learn==1.9.1 \
    tensorflow==2.21.0 \
    keras==3.15.1 \
    matplotlib==3.11.2 \
    seaborn==0.13.2
```

Check the installation:

```
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__, keras.backend.backend())"
# 2.21.0 3.15.1 tensorflow
```

Notes:

- The environment has a new name, so it can sit next to the old `fractal-fractional` environment (Python 3.8, TensorFlow 2.10) that the paper's experiments ran in.
- TensorFlow 2.21 provides wheels for Python 3.10–3.13 on Linux (x86-64 and ARM64), macOS on Apple silicon (macOS 12+), and Windows (CPU). There are no wheels for Intel Macs.
- **GPU (Linux, NVIDIA):** replace `tensorflow==2.21.0` with `"tensorflow[and-cuda]==2.21.0"`. Windows users who want a GPU need WSL2.
- Install TensorFlow with `pip` inside the conda environment, as shown above, not with `conda install tensorflow`. The official TensorFlow builds are published on PyPI; the conda builds lag behind.
- `seaborn` is not needed by the scripts here. It is kept so the environment matches the analysis environment of the paper.
- This exact list was installed into a fresh Python 3.12.13 virtual environment, and `hello_world_iris.py` was run in it. The output shown below comes from that run.

---

## Hello world: Iris in one run

```
python hello_world_iris.py
```

The script is deliberately linear: no `main()`, no factories, no class hierarchy. It reads top to bottom like a recipe, with every step numbered and explained in the code.

| Step | What it shows |
|---|---|
| 0 | All settings at the top (seed, paper protocol, fractional order, learning rates, which parts to run) |
| 1–2 | Imports, version printout, one seed for Python/NumPy/TensorFlow/Keras (`keras.utils.set_random_seed`) |
| 3–4 | Load Iris [[D1]](#ref-d1); 70/30 split with `random_state = seed`; Min-Max scaling fitted on the training part (paper protocol) |
| 5 | Call the fractal activations directly; optional plot of all 18 |
| 6 | Paper's Iris network: Dense(32) → BN → Dropout(0.2) → Dense(64) → BN → Dropout(0.2) → softmax, Glorot(seed 42) |
| 7 | One fractional optimizer (`AdaptiveMemoryFAdam`), every argument explained |
| 8 | `model.compile` + `model.fit` (30 epochs, batch 16) |
| 9 | Accuracy, macro precision/recall/F1, confusion matrix, classification report |
| 10 | Inside the optimizer: GL coefficients, DC gain, λ_t per variable, Herrera factor f_ν(w) |
| 11 | `model.save("….keras")` and `keras.models.load_model(...)`, with optimizer state restored |
| 12 | **Optimizer tour:** all 35 optimizers (8 Keras baselines + 27 fractional), same model/seed/data |
| 13 | **Activation tour:** relu, tanh and all 18 fractal activations, same optimizer |
| 14 | Custom `GradientTape` loop with `set_current_loss()` (loss feedback for the adaptive family) |
| 15 | Results to `hello_world_iris_results/` (JSON, CSV, `.keras` model, PNG plots) |

The main example uses **AdaptiveMemoryFAdam + decaying cosine, ν = 0.75**. That is rank 2 of the paper's Iris table (mean accuracy 0.937 ± 0.037 over 40 runs) and the fastest of the top three configurations.

**Runtime.** The full script trains 57 small networks and took 25 minutes on a 2-core cloud CPU. Most of that is TensorFlow tracing the 75- to 100-term activation graphs once per model. For a run of about 1–2 minutes, set `RUN_OPTIMIZER_TOUR = False` and `RUN_ACTIVATION_TOUR = False` at the top.

### Output of the reference run

Reference run: Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.1, CPU only, seed 238974, 25 minutes in total.

**Main example (STEPS 6–11):** `AdaptiveMemoryFAdam` (ν = 0.75) + `decaying_cosine_function_tf`, 30 epochs.

```
[STEP 9] test-set results
         test_loss        0.1855
         accuracy         0.9111
         precision_macro  0.9365
         recall_macro     0.9111
         f1_macro         0.9136

         confusion matrix (rows = true class, columns = predicted class):
            setosa  versicolor  virginica
setosa          13           0          0
versicolor       0          11          4
virginica        0           0         17
```

The model saved to `.keras` reloads with the same optimizer class and the same λ_t values, and gives identical predictions.

**Optimizer tour (STEP 12):** decaying cosine activation, ν = 0.75 for every fractional optimizer, one run each.

| Family | Optimizer | Accuracy | Macro F1 |
|---|---|---:|---:|
| Keras baseline | `SGD` | 0.6444 | 0.5878 |
| Keras baseline | `Adam` | 0.9111 | 0.9153 |
| Keras baseline | `RMSprop` | 0.9556 | 0.9577 |
| Keras baseline | `Adadelta` | 0.9333 | 0.9359 |
| Keras baseline | `Adagrad` | 0.9556 | 0.9577 |
| Keras baseline | `AdamW` | 0.9111 | 0.9163 |
| Keras baseline | `Nadam` | 0.9778 | 0.9790 |
| Keras baseline | `Adamax` | 0.9111 | 0.9153 |
| Herrera | `FSGD` | 0.4889 | 0.4172 |
| Herrera | `FAdam` | 0.9556 | 0.9582 |
| Herrera | `FRMSprop` | 0.9556 | 0.9577 |
| Herrera | `FAdadelta` | 0.9111 | 0.9153 |
| Herrera | `FAdagrad` | 0.9556 | 0.9577 |
| Herrera | `FAdamW` | 0.9556 | 0.9577 |
| GL memory | `MemoryFSGD` | 0.8444 | 0.8529 |
| GL memory | `MemoryFAdam` | 0.8000 | 0.7975 |
| GL memory | `MemoryFRMSprop` | 0.8444 | 0.8362 |
| GL memory | `MemoryFAdadelta` | 0.8000 | 0.7906 |
| GL memory | `MemoryFAdagrad` | 0.9333 | 0.9359 |
| GL memory | `MemoryFAdamW` | 0.8667 | 0.8667 |
| GL memory | `MemoryFNadam` | 0.8444 | 0.8416 |
| GL memory | `MemoryFAdamax` | 0.7556 | 0.7582 |
| Adaptive memory | `AdaptiveMemoryFSGD` | 0.4889 | 0.4172 |
| Adaptive memory | `AdaptiveMemoryFAdam` | 0.9111 | 0.9136 |
| Adaptive memory | `AdaptiveMemoryFRMSprop` | 0.9556 | 0.9577 |
| Adaptive memory | `AdaptiveMemoryFAdadelta` | 0.9111 | 0.9163 |
| Adaptive memory | `AdaptiveMemoryFAdagrad` | 0.9556 | 0.9582 |
| Adaptive memory | `AdaptiveMemoryFAdamW` | 0.9778 | 0.9790 |
| Adaptive memory | `AdaptiveMemoryFNadam` | 0.9333 | 0.9369 |
| Adaptive memory | `AdaptiveMemoryFAdamax` | 0.9111 | 0.9153 |
| Related work | `AdaGL` | 0.7111 | 0.7119 |
| Related work | `FCSGD_GL` | 0.8889 | 0.8949 |
| Related work | `FCAdam_GL` | 0.9333 | 0.9369 |
| Related work | `AOFGD_SGD` | 0.7111 | 0.6458 |
| Related work | `AOFGD_Adam` | 0.9333 | 0.9359 |

**Activation tour (STEP 13):** `AdaptiveMemoryFAdam`, ν = 0.75, one run each.

| Group | Activation | Accuracy | Macro F1 | Note |
|---|---|---:|---:|---|
| classical | `relu` | 0.8000 | 0.8072 |  |
| classical | `tanh` | 0.6222 | 0.5461 |  |
| fractal (paper) | `modulated_blancmange_curve` | 0.9111 | 0.9167 |  |
| fractal (paper) | `decaying_cosine_function_tf` | 0.9111 | 0.9136 |  |
| fractal (paper) | `modified_weierstrass_function_tanh` | 0.6889 | 0.6606 |  |
| fractal | `modified_weierstrass_function_relu` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_xsinsquared` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal (paper) | `weierstrass_mandelbrot_function_xpsin` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_relupsin` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_tanhpsin` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_function_tf` | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `random_depth_modified_weierstrass_function_tanh` | 0.6667 | 0.6257 |  |
| fractal | `random_depth_decaying_cosine_function_tf` | 0.7556 | 0.7593 |  |
| fractal | `random_depth_modulated_blancmange_curve` | 0.8444 | 0.8505 |  |
| fractal | `random_depth_weierstrass_mandelbrot_function_xpsin` | 0.8889 | 0.8932 |  |
| fractal | `random_depth_weierstrass_mandelbrot_function_tanhpsin` | 0.8667 | 0.8704 |  |
| fractal | `random_depth_weierstrass_function_tf` | 0.3778 | 0.3643 |  |
| fractal | `cantor_function_tf` | 0.2889 | 0.1494 | no gradient (piecewise constant) |
| fractal | `cantor_set_activation_tf` | 0.2889 | 0.1494 | no gradient (piecewise constant) |
| fractal | `random_depth_cantor_function_tf` | 0.2889 | 0.1494 | no gradient (piecewise constant) |

**Custom loop with loss feedback (STEP 14):** test accuracy 0.8889, macro F1 0.8932.

How to read these numbers:

- **One seed is not a benchmark.** On Iris the paper's run-to-run standard deviation is 0.03–0.06 over 40 runs, and single runs can be much further off. Example: in an earlier development run of this script, `FSGD` scored 0.91 instead of 0.49. That version differed from the final one only in how the graph was laid out (the all-reduce change described under Migration), which shifts graph-mode results by about one float32 rounding step. With momentum SGD on 105 training samples, that was enough to send training somewhere else. Use the paper's protocol (40 seeds) for comparisons. Your numbers can also differ slightly on other CPUs, on GPUs, or with oneDNN switched off.
- **The broad picture matches the paper's Iris results.** Decaying cosine and Blancmange are the strongest activations, the GL-memory optimizers trail the other fractional families, and the SGD-type optimizers at lr = 1e-2 struggle with this activation in 30 epochs.
- **λ_t ends at 0 here.** With batches of 16, the relative gradient change ("stability") stays around 2, far above `stability_threshold = 0.25`. The adaptive optimizers therefore fade the memory term out and behave like their base optimizer (λ_t starts at 0.10; see the STEP 14 log). That is the intended safe fallback.
- **Divergence and no-gradient cases are properties of the activation functions.** The Weierstrass–Mandelbrot variants and `weierstrass_function_tf` diverge with 100 terms, and the Cantor functions have no gradient. See [Using the fractal activations](#using-the-fractal-activations).

---

## Using the optimizers

Every optimizer is a regular `keras.optimizers.Optimizer`:

```python
import keras
from class_minimal_fractional_adaptive_optimizers_keras3 import AdaptiveMemoryFAdam

model.compile(
    optimizer=AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=30, batch_size=16)
```

What every class supports:

- **`learning_rate=`** is the Keras 3 argument. `lr=` still works as an alias, so old experiment scripts keep running.
- **Learning-rate schedules:** `learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=1000)`.
- **Keras 3 base arguments:** `clipnorm`, `clipvalue`, `global_clipnorm`, `use_ema`, `ema_momentum`, `loss_scale_factor`, `gradient_accumulation_steps`. Also `weight_decay` (Keras-style decoupled decay applied *before* the step; the `…AdamW` classes keep their own post-step decay, see below).
- **Saving:** every class is registered with Keras. `model.save("model.keras")` stores the optimizer configuration and state (moments, gradient history, λ_t, α_t). `keras.models.load_model("model.keras")` restores them, provided the module has been imported.
- **XLA:** `model.compile(..., jit_compile=True)` works for every optimizer and activation.
- **`fit`, custom loops and `tf.distribute`:** all supported. The optimizers implement `update_step`, so they run inside `model.fit`, inside custom `tf.GradientTape` loops, and under `tf.distribute` strategies (gradients are all-reduced whenever there is more than one replica).

### Family 1 — Herrera-style (`class_b_…_keras3.py`)

> **Original work by Oscar Herrera-Alcántara and co-authors. Please cite** [[H1]](#ref-h1) Herrera-Alcántara, *Applied Sciences* 12, 9264 (2022), [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264), and [[H2]](#ref-h2) Herrera-Alcántara & Castelán-Aguilar, *Fractal and Fractional* 7, 500 (2023), [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500).

The gradient is scaled element-wise by f_ν(w) = (|w| + ε)^(1−ν) / Γ(2−ν). There is no memory, and ν = 1 gives back the base optimizer.

```python
from class_b_tf_fractional_optimizers_OHA_keras3 import FSGD, FAdam, FRMSprop, FAdadelta, FAdagrad, FAdamW

FSGD(learning_rate=1e-2, momentum=0.9, nesterov=False, vderiv=0.75, frac_epsilon=1e-6)
FAdam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, vderiv=1.25)
FRMSprop(learning_rate=1e-3, rho=0.9, momentum=0.0, centered=False, vderiv=1.25)
FAdadelta(learning_rate=1.0, rho=0.95, vderiv=1.25)
FAdagrad(learning_rate=1e-2, initial_accumulator_value=0.1, vderiv=1.25)
FAdamW(learning_rate=1e-3, weight_decay=0.004, vderiv=1.25)
```

### Family 2 — Grünwald–Letnikov memory (`class_c_…_keras3.py`)

> Implementation from our paper [[O1]](#ref-o1), based on the Grünwald–Letnikov derivative [[G1]](#ref-g1) and the short-memory principle [[G2]](#ref-g2); the base update rules are those of [[B1]–[B10]](#base-optimizers).

The direction is a GL sum of the last K raw gradients, g^(ν) = Σ c_k g_{t−k}, with c_0 = 1 and c_k = (1 − (ν+1)/k)·c_{k−1}. Three defaults control how it is used:

- `norm_match=True` rescales the sum to ‖g_t‖.
- `descent_check=True` falls back to g_t when ⟨g^(ν), g_t⟩ ≤ 0. This matters for ν > 1, where the coefficient sum is negative.
- `use_raw_gradient_for_second_moment=True` feeds the raw gradient to all squared-gradient accumulators.

The history is a circular buffer with one row written per step.

```python
from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam

opt = MemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=6,
                  normalize_coefficients=False, norm_match=True,
                  descent_check=True, use_raw_gradient_for_second_moment=True)
opt.gl_coefficients        # [1.0, -0.75, -0.0938, -0.0391, -0.022, -0.0143]
opt.gl_coefficient_sum()   # +0.0809, the DC gain of the truncated kernel (negative for nu > 1)
```

### Family 3 — Adaptive memory (`class_minimal_…_keras3.py`)

> New in our paper [[O1]](#ref-o1). Please cite it when you use these optimizers; the base update rules are those of [[B1]–[B10]](#base-optimizers).

The update uses g_eff = (1 − λ_t)·g_t + λ_t·ĝ^(ν). Only the mixing weight λ_t is adapted: it grows while the relative gradient change stays below `stability_threshold`, shrinks otherwise, and is bounded by `[min_mix, max_mix]`. With λ_t = 0, or with ν = 1, the update is exactly the base optimizer.

```python
from class_minimal_fractional_adaptive_optimizers_keras3 import AdaptiveMemoryFAdam, AdaptiveLossFeedback

opt = AdaptiveMemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=4,
                          normalize_coefficients=True, initial_mix=0.10, min_mix=0.0,
                          max_mix=0.30, adapt_step=0.01, mix_ema_gamma=0.90,
                          stability_ema_gamma=0.95, stability_threshold=0.25, warmup_steps=0)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")
model.fit(x, y, epochs=30)                                  # paper setting
model.fit(x, y, epochs=30, callbacks=[AdaptiveLossFeedback()])  # optional loss feedback
opt.current_mix_values()        # λ_t, one scalar per model variable
opt.current_stability_values()  # smoothed stability score per variable
```

In a custom loop, call `opt.set_current_loss(loss)` after each step (STEP 14 of the hello world).

### Family 4 — Related work (`class_d_…_keras3.py`)

> **Not our work. Please cite the original authors:**
> - `AdaGL`: Chen, Zhang & Mu, *Neural Processing Letters* 56, 106 (2024), [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7) [[R1]](#ref-r1)
> - `FCSGD_GL`, `FCAdam_GL`: Zhou, Zhao & Huang, *Mathematics* 11, 316 (2023), [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316) [[R2]](#ref-r2)
> - `AOFGD_SGD`, `AOFGD_Adam`: Xiang, Zhao, Su, Zhou, Ye & Huang, SSRN preprint (2025), [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) [[R3]](#ref-r3)

```python
from class_d_related_work_optimizers_keras3 import AdaGL, FCSGD_GL, FCAdam_GL, AOFGD_SGD, AOFGD_Adam

AdaGL(learning_rate=1e-3, alpha=0.75, history_size=10)                      # Chen, Zhang & Mu 2024
FCSGD_GL(learning_rate=1e-2, momentum=0.9, alpha=0.75, history_size=11,
         perturbation_prob=0.5)                                             # Zhou, Zhao & Huang 2023
FCAdam_GL(learning_rate=1e-3, alpha=0.75, history_size=11, perturbation_prob=0.5)
AOFGD_SGD(learning_rate=1e-2, momentum=0.9, alpha_init=0.75, alpha_min=0.5,
          alpha_max=1.5, delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95)  # Xiang et al. 2025
opt = AOFGD_Adam(learning_rate=1e-3, alpha_init=0.75, alpha_min=0.5, alpha_max=1.5)
opt.current_alpha_values()      # after training: current smoothed order per variable
```

### Keras baselines

Use the Keras 3 classes with `learning_rate=` (Keras 3 rejects `lr=`). Original papers: SGD [[B1]](#ref-b1), momentum [[B2]](#ref-b2), Nesterov [[B3]](#ref-b3), Adam/Adamax [[B4]](#ref-b4), AMSGrad [[B5]](#ref-b5), RMSprop [[B6]](#ref-b6), Adadelta [[B7]](#ref-b7), Adagrad [[B8]](#ref-b8), AdamW [[B9]](#ref-b9), Nadam [[B10]](#ref-b10).

```python
keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
keras.optimizers.Adam(learning_rate=1e-3)
keras.optimizers.RMSprop(learning_rate=1e-3)
keras.optimizers.Adadelta(learning_rate=1.0)
```

### Settings used in the paper's runner

| | value |
|---|---|
| learning rates | SGD-type 1e-2 (momentum 0.9), Adam-type 1e-3, RMSprop-type 1e-3, Adadelta-type 1.0 |
| fractional orders | ν ∈ {0.75, 1.25, 1.5} (plus 1.75 in additional runs) |
| Herrera | `frac_epsilon=1e-6` |
| GL memory | `history_size=6`, `normalize_coefficients=False` |
| Adaptive memory | `history_size=4`, `normalize_coefficients=True`, `initial_mix=0.10`, `max_mix=0.30`, `adapt_step=0.01`, `mix_ema_gamma=0.90`, `stability_ema_gamma=0.95`, `stability_threshold=0.25` |
| AdaGL | `history_size=10` |
| FCSGD_GL / FCAdam_GL | `history_size=11`, `perturbation_prob=0.5` |
| AOFGD | `alpha_min=0.5`, `alpha_max=max(1.5, ν)`, `delta_alpha=0.02`, `cef_threshold=0.05`, `ema_gamma=0.95` |

---

## Using the fractal activations

> The fractal activation functions were introduced in Raubitzek et al., *Communications in Nonlinear Science and Numerical Simulation* 161, 110064 (2026), [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) [[O2]](#ref-o2). The Cantor-based activations follow Chipera, Du & Tsapara, [arXiv:2512.14675](https://arxiv.org/abs/2512.14675) (2025) [[C1]](#ref-c1).

```python
import keras
import fractal_activation_functions_keras3 as faf

# (a) as the activation of a layer, with default parameters (as in the paper)
keras.layers.Dense(32, activation=faf.decaying_cosine_function_tf)

# (b) as a separate layer, with your own parameters; the model can still be saved
x = keras.layers.Dense(32)(x)
x = faf.FractalActivation("weierstrass_mandelbrot_function_xpsin", num_terms=10)(x)

faf.PAPER_FRACTAL_ACTIVATIONS   # the four used in the article
faf.ALL_FRACTAL_ACTIVATIONS     # all 18
faf.get_fractal_activation("modified_weierstrass_function_tanh")
```

Activations used in the article: `modulated_blancmange_curve`, `decaying_cosine_function_tf`, `modified_weierstrass_function_tanh`, `weierstrass_mandelbrot_function_xpsin`.

Behaviour to know about. All of it is unchanged from the original functions:

- The Weierstrass–Mandelbrot variants and `weierstrass_function_tf` sum terms 2^(−k/2)·sin(2π·2^k·x) for k up to 99. Their derivatives grow like 2π·2^(k/2), which is about 10^15 for the last terms, so with the default 100 terms training can overflow to NaN. Fewer terms (`num_terms=…` via `FractalActivation`) or `clipnorm` help.
- `cantor_function_tf` and `cantor_set_activation_tf` are piecewise constant, so no gradient flows through them.
- `modulated_blancmange_curve` contains √|x|, whose derivative is unbounded at 0.
- The `random_depth_*` functions draw a new number of series terms on every call (`tf.random`). Use `keras.utils.set_random_seed` for repeatable runs.

---

## Migrating from the legacy files

| Legacy (TF ≤ 2.15, Keras 2) | Keras 3 (TF ≥ 2.16) |
|---|---|
| `from class_c_tf_fractional_optimizers_SR import MemoryFAdam` | `from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam` |
| `MemoryFAdam(lr=1e-3, ...)` | `MemoryFAdam(learning_rate=1e-3, ...)` (`lr=` still accepted) |
| `tf.keras.optimizers.SGD(lr=0.01)` | `keras.optimizers.SGD(learning_rate=0.01)` |
| `Dense(n, input_dim=d)` as first layer | `keras.Input(shape=(d,))`, then `Dense(n)` |
| `tf.keras.backend.clear_session()` | `keras.backend.clear_session()` |
| `import fractal_activation_functions as faf` | `import fractal_activation_functions_keras3 as faf` |

Differences, all deliberate:

1. **Optimizer checkpoints** written by the legacy classes cannot be restored into the Keras 3 classes; the variable layout differs. Retrain, or keep the legacy environment for old checkpoints.
2. **`set_current_loss()`** (adaptive family) now actually reaches compiled training steps. The "loss got worse" flag is an optimizer variable; in the legacy file it was a Python float that `tf.function` froze at trace time. When `set_current_loss()` is never called (the paper setting), results are identical.
3. **`FAdamW`, `MemoryFAdamW`, `AdaptiveMemoryFAdamW`** keep the legacy order: Adam step first, then decoupled decay on the updated weights. Keras' own pre-step decay is switched off for these three classes. For every other class, the Keras 3 `weight_decay=` argument is available as an extra; it is off by default.
4. **Fractal activations** that compute in float64 now return the dtype of their input instead of always float32. For float32 models the output is the same; under mixed precision it now matches the layer.
5. **Graph-mode rounding.** In graph mode (`model.fit`, `tf.function`), TensorFlow's graph optimizer (Grappler) rewrites and fuses floating-point operations depending on how a graph is built. In the tests below, the Keras 3 graphs give bit-identical results to the legacy graphs. For other models or TensorFlow versions, differences in the last float32 bit are possible in principle; they are rounding, not a change of the update rule.
6. **No memory growth when training many models in one process.** With Keras 3 on TensorFlow 2.21, every `model.fit` leaves its traced training graph in memory, even after `keras.backend.clear_session()`. The cause is the optimizer's gradient all-reduce: `tf.distribute` registers a `tf.custom_gradient` in a global registry on every trace and never removes it. With the fractal activations these graphs are large, about 130–150 MB per model, so a loop over many configurations (such as the hello-world tours, or the paper's 40-run loops) eventually runs out of memory. Our optimizers skip the all-reduce when there is only one replica, where it is the identity anyway, so their memory stays flat. The Keras built-in optimizers are still affected. If you loop over many models with them, run each configuration in its own process.

### Running the original files unchanged on a current TensorFlow

To re-run the paper's code exactly as published on TF ≥ 2.16:

1. Install the legacy Keras package with the same version number as TensorFlow: `pip install tf-keras==2.21.0`.
2. Set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow.
3. Add `tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Optimizer` before importing `class_b_…` and `class_d_…`.

This is how the tests below run the legacy files.

---

## Verification

All checks were run with TensorFlow 2.21.0 and Keras 3.15.1 on CPU (Python 3.13, plus the final hello-world run in Python 3.12). They are in `tests/`.

**1. Same numbers as the legacy implementation** (`tests/test_equivalence_with_legacy.py`). The test covers 167 optimizer configurations (320 runs): every class; ν < 1, ν = 1 and ν > 1; several history lengths; momentum, Nesterov, centered and AMSGrad variants; norm matching, descent check and second-moment switches; warm-up; loss feedback; and Bernoulli masks. Each configuration runs with the legacy file (via `tf_keras`) and with the Keras 3 file on the same 4-8-3 network, the same initial weights and the same mini-batches, for 60 steps, in eager mode and inside `tf.function`. Then the final weights and the λ_t / α_t states are compared.

Result: **320 of 320 runs bit-identical** (all 167 configurations, in eager mode and in graph mode, with TensorFlow's default graph optimizer). They are also bit-identical with the graph optimizer switched off (`DISABLE_GRAPPLER=1`).

| Family | Configurations | Eager | Graph (`tf.function`) |
|---|---:|---:|---:|
| Herrera (`class_b`) | 22 | 22/22 identical | 22/22 identical |
| GL memory (`class_c`) | 61 | 61/61 identical | 61/61 identical |
| Adaptive memory (`class_minimal`) | 52 | 52/52 identical | 44/44 identical |
| Related work (`class_d`) | 32 | 32/32 identical | 26/26 identical |

Fourteen configurations run only in eager mode. They can only be compared there: the loss-feedback runs, because the legacy flag is frozen at trace time, and the FCGL runs with p = 0.5, because the random stream depends on how the graph is built.

```
pip install tf-keras==2.21.0
python tests/test_equivalence_with_legacy.py <folder with the legacy files> .
DISABLE_GRAPPLER=1 python tests/test_equivalence_with_legacy.py <folder with the legacy files> .
```

**2. Fractal activations** (`tests/test_activations_equivalence.py`). All 18 functions are compared with the originals on 20,001 points in [−5, 5] plus a random batch, in eager and graph mode. Values and gradients are bit-identical for all 18. The random-depth functions were compared with the same seed.

**3. Keras 3 integration** (`tests/test_keras3_features.py`, `tests/test_activations_xla.py`). For all 27 optimizers:

- `model.fit` with the paper's BatchNorm/Dropout architecture, with and without XLA (`jit_compile=True`);
- `get_config()` / `from_config()` round trip;
- `.keras` save and reload: identical predictions, identical optimizer variables, and training continues;
- learning-rate schedule plus `clipnorm`.

All 18 activations train under XLA. `Dense(activation=…)` and `FractalActivation(…)` models survive save and reload.

**4. Memory** (`tests/test_memory_flat.py`). Six decaying-cosine models are trained one after another in one process, with fractional optimizers from all four families. Resident memory stays flat: between 0.8 and 1.0 GB after every model, with no upward trend. With the Keras built-in `Adam`, the same loop grows by about 130 MB per model.

```
python tests/test_activations_equivalence.py <folder with the legacy files> .
python tests/test_keras3_features.py .
python tests/test_activations_xla.py .
python tests/test_memory_flat.py .
```

---

## References

<a id="ref-h1"></a>**[H1]** O. Herrera-Alcántara. Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition. *Applied Sciences* **12**(18), 9264 (2022). [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264)

<a id="ref-h2"></a>**[H2]** O. Herrera-Alcántara, J. R. Castelán-Aguilar. Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT. *Fractal and Fractional* **7**(7), 500 (2023). [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500)

<a id="ref-r1"></a>**[R1]** S. Chen, C. Zhang, H. Mu. An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on G–L Fractional-Order Derivative. *Neural Processing Letters* **56**, 106 (2024). [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7)

<a id="ref-r2"></a>**[R2]** X. Zhou, C. Zhao, Y. Huang. A Deep Learning Optimizer Based on Grünwald–Letnikov Fractional Order Definition. *Mathematics* **11**(2), 316 (2023). [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316)

<a id="ref-r3"></a>**[R3]** K. Xiang, C. Zhao, Q. Su, X. Zhou, J. Ye, Y. Huang. AOFGD: Adaptive Order Fractional Gradient Descent Method. SSRN preprint, not peer-reviewed (2025). [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) · [ssrn.com/abstract=5717167](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5717167)

<a id="ref-c1"></a>**[C1]** R. Chipera, J. Du, I. Tsapara. Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property. arXiv preprint (2025). [arXiv:2512.14675](https://arxiv.org/abs/2512.14675)

<a id="ref-o1"></a>**[O1]** S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger. Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks. *Fractal and Fractional* (2026, accepted). [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · [code](https://github.com/Raubkatz/FractalAndFractional2026)

<a id="ref-o2"></a>**[O2]** S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger. Fractals in neural networks: Introducing a new class of activation functions. *Communications in Nonlinear Science and Numerical Simulation* **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · [code](https://github.com/Raubkatz/Fractal_Activation_Functions)

**Grünwald–Letnikov background**

<a id="ref-g1"></a>**[G1]** I. Podlubny. *Fractional Differential Equations.* Mathematics in Science and Engineering, vol. 198. Academic Press, San Diego (1999).

<a id="ref-g2"></a>**[G2]** Y. Wei, Y. Chen, S. Cheng, Y. Wang. A Note on Short Memory Principle of Fractional Calculus. *Fractional Calculus and Applied Analysis* **20**, 1382–1404 (2017). [doi:10.1515/fca-2017-0073](https://doi.org/10.1515/fca-2017-0073)

<a id="base-optimizers"></a>**Base optimizers**

<a id="ref-b1"></a>**[B1]** SGD: H. Robbins, S. Monro. A Stochastic Approximation Method. *Annals of Mathematical Statistics* **22**, 400–407 (1951). [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)

<a id="ref-b2"></a>**[B2]** Momentum: B. T. Polyak. Some Methods of Speeding up the Convergence of Iteration Methods. *USSR Computational Mathematics and Mathematical Physics* **4**, 1–17 (1964). [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5)

<a id="ref-b3"></a>**[B3]** Nesterov momentum: Y. Nesterov. A method for solving the convex programming problem with convergence rate O(1/k²). *Soviet Mathematics Doklady* **27**, 372–376 (1983); in deep learning: I. Sutskever, J. Martens, G. Dahl, G. Hinton. On the importance of initialization and momentum in deep learning. *Proc. ICML*, PMLR **28**(3), 1139–1147 (2013). [proceedings.mlr.press/v28/sutskever13](https://proceedings.mlr.press/v28/sutskever13.html)

<a id="ref-b4"></a>**[B4]** Adam and Adamax: D. P. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. *ICLR* (2015). [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

<a id="ref-b5"></a>**[B5]** AMSGrad: S. J. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. *ICLR* (2018). [openreview.net/forum?id=ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ)

<a id="ref-b6"></a>**[B6]** RMSprop: T. Tieleman, G. Hinton. Lecture 6.5 — RMSProp: Divide the gradient by a running average of its recent magnitude. *Neural Networks for Machine Learning*, Coursera (2012). [lecture slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

<a id="ref-b7"></a>**[B7]** Adadelta: M. D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. arXiv (2012). [arXiv:1212.5701](https://arxiv.org/abs/1212.5701)

<a id="ref-b8"></a>**[B8]** Adagrad: J. Duchi, E. Hazan, Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *Journal of Machine Learning Research* **12**, 2121–2159 (2011). [jmlr.org (PDF)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

<a id="ref-b9"></a>**[B9]** AdamW: I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. *ICLR* (2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

<a id="ref-b10"></a>**[B10]** Nadam: T. Dozat. Incorporating Nesterov Momentum into Adam. Stanford CS229 project report (2015); ICLR 2016 Workshop track. [report (PDF)](http://cs229.stanford.edu/proj2015/054_report.pdf) · [ICLR workshop paper (PDF)](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf)

<a id="fractal-functions-mathematical-origins"></a>**Fractal functions: mathematical origins**

<a id="ref-f1"></a>**[F1]** Weierstrass function: K. Weierstraß. Über continuirliche Functionen eines reellen Arguments, die für keinen Werth des letzteren einen bestimmten Differentialquotienten besitzen (lecture, 1872). Reprinted in R. Siegmund-Schultze (ed.), *Ausgewählte Kapitel aus der Funktionenlehre*, Teubner-Archiv zur Mathematik 9, Springer, Vienna (1988), pp. 190–193.

<a id="ref-f2"></a>**[F2]** Blancmange (Takagi) function: T. Takagi. A Simple Example of the Continuous Function without Derivative. *Proc. Phys.-Math. Soc. Japan Ser. II* (Tokyo Sugaku-Butsurigakkwai Hokoku) **1**, F176–F177 (1903). [doi:10.11429/subutsuhokoku1901.1.F176](https://doi.org/10.11429/subutsuhokoku1901.1.F176)

<a id="ref-f3"></a>**[F3]** Weierstrass–Mandelbrot function: M. V. Berry, Z. V. Lewis. On the Weierstrass–Mandelbrot Fractal Function. *Proc. R. Soc. Lond. A* **370**, 459–484 (1980). [doi:10.1098/rspa.1980.0044](https://doi.org/10.1098/rspa.1980.0044)

<a id="ref-f4"></a>**[F4]** B. B. Mandelbrot. *The Fractal Geometry of Nature.* W. H. Freeman, San Francisco (1982).

<a id="ref-f5"></a>**[F5]** Cantor set and Cantor function: G. Cantor. De la puissance des ensembles parfaits de points. *Acta Mathematica* **4**, 381–392 (1884).

**Software**

<a id="ref-s1"></a>**[S1]** F. Chollet et al. Keras (2015–). [keras.io](https://keras.io) · [github.com/keras-team/keras](https://github.com/keras-team/keras)

<a id="ref-s2"></a>**[S2]** M. Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. arXiv (2016). [arXiv:1603.04467](https://arxiv.org/abs/1603.04467) · [tensorflow.org](https://www.tensorflow.org)

<a id="ref-s3"></a>**[S3]** F. Pedregosa et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* **12**, 2825–2830 (2011). [jmlr.org/papers/v12/pedregosa11a](https://jmlr.org/papers/v12/pedregosa11a.html)

**Data**

<a id="ref-d1"></a>**[D1]** Iris data: R. A. Fisher. The use of multiple measurements in taxonomic problems. *Annals of Eugenics* **7**(2), 179–188 (1936). [doi:10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) · OpenML dataset 61: [openml.org/d/61](https://www.openml.org/d/61)

### BibTeX

```bibtex
@article{herrera2022fractional,
  author  = {Herrera-Alc{\'a}ntara, Oscar},
  title   = {Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition},
  journal = {Applied Sciences},
  volume  = {12},
  number  = {18},
  pages   = {9264},
  year    = {2022},
  doi     = {10.3390/app12189264}
}

@article{herrera2023pytorch,
  author  = {Herrera-Alc{\'a}ntara, Oscar and Castel{\'a}n-Aguilar, Josu{\'e} R.},
  title   = {Fractional Gradient Optimizers for {PyTorch}: Enhancing {GAN} and {BERT}},
  journal = {Fractal and Fractional},
  volume  = {7},
  number  = {7},
  pages   = {500},
  year    = {2023},
  doi     = {10.3390/fractalfract7070500}
}

@article{chen2024adagl,
  author  = {Chen, Shuang and Zhang, Changlun and Mu, Haibing},
  title   = {An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on {G--L} Fractional-Order Derivative},
  journal = {Neural Processing Letters},
  volume  = {56},
  number  = {2},
  pages   = {106},
  year    = {2024},
  doi     = {10.1007/s11063-024-11571-7}
}

@article{zhou2023gl,
  author  = {Zhou, Xiaojun and Zhao, Chunna and Huang, Yaqun},
  title   = {A Deep Learning Optimizer Based on {Gr{\"u}nwald--Letnikov} Fractional Order Definition},
  journal = {Mathematics},
  volume  = {11},
  number  = {2},
  pages   = {316},
  year    = {2023},
  doi     = {10.3390/math11020316}
}

@misc{xiang2025aofgd,
  author       = {Xiang, Kemeng and Zhao, Chunna and Su, Qian and Zhou, Xiaojun and Ye, Junjie and Huang, Yaqun},
  title        = {{AOFGD}: Adaptive Order Fractional Gradient Descent Method},
  howpublished = {SSRN preprint, abstract ID 5717167},
  year         = {2025},
  doi          = {10.2139/ssrn.5717167}
}

@misc{chipera2025fractal,
  author        = {Chipera, Rae and Du, Jenny and Tsapara, Irene},
  title         = {Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property},
  year          = {2025},
  eprint        = {2512.14675},
  archivePrefix = {arXiv}
}

@article{raubitzek2026fractional,
  author  = {Raubitzek, Sebastian and Goldenits, Georg and Schrittwieser, Sebastian and K{\"o}nig, Philip and Mallinger, Kevin},
  title   = {Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks},
  journal = {Fractal and Fractional},
  year    = {2026},
  note    = {Accepted. Preprint: arXiv:2608.14636. Code: https://github.com/Raubkatz/FractalAndFractional2026}
}

@article{raubitzek2026fractals,
  author  = {Raubitzek, Sebastian and Kietreiber, Tobias and Eresheim, Sebastian and Buchelt, Alexander and Mallinger, Kevin},
  title   = {Fractals in neural networks: Introducing a new class of activation functions},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  volume  = {161},
  pages   = {110064},
  year    = {2026},
  doi     = {10.1016/j.cnsns.2026.110064}
}
```

---

## Citation

If you use this repository, please cite

- **the original publication of every optimizer and activation you use** (see [Attribution and how to cite](#attribution-and-how-to-cite)). In particular: Herrera-Alcántara [[H1]](#ref-h1), [[H2]](#ref-h2) for the Herrera-style optimizers; Chen, Zhang & Mu [[R1]](#ref-r1) for AdaGL; Zhou, Zhao & Huang [[R2]](#ref-r2) for FCSGD_GL/FCAdam_GL; Xiang et al. [[R3]](#ref-r3) for AOFGD; Chipera, Du & Tsapara [[C1]](#ref-c1) for the Cantor activations;
- our article [[O1]](#ref-o1) for the adaptive-memory and GL-memory optimizers, the experimental comparison, and this Keras 3 platform;
- Raubitzek et al. [[O2]](#ref-o2) for the fractal activation functions.

The full experiment code of the article (all ten datasets, 40 runs per configuration, analytic test surfaces) is at [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026). The code of the fractal activation paper is at [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions).
