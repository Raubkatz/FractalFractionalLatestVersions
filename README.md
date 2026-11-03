# Fractional Optimizers × Fractal Activations — Keras 3 and PyTorch

Keras 3 / TensorFlow 2.21 **and** PyTorch 2.15 implementations of the fractional optimizers and fractal activation functions used in

> S. Raubitzek, G. Goldenits, S. Schrittwieser, P. König, K. Mallinger.
> *Fractional Optimizers Meet Fractal Activation Functions: An Empirical Study of Multi-Scale Optimization in Neural Networks.*
> Fractal and Fractional, 2026 (accepted). Preprint: [arXiv:2608.14636](https://arxiv.org/abs/2608.14636) · Experiment code: [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026)

The fractal activation functions were introduced in

> S. Raubitzek, T. Kietreiber, S. Eresheim, A. Buchelt, K. Mallinger.
> *Fractals in neural networks: Introducing a new class of activation functions.*
> Communications in Nonlinear Science and Numerical Simulation **161**, 110064 (2026). [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) · Preprint: [doi:10.20944/preprints202508.0478.v1](https://doi.org/10.20944/preprints202508.0478.v1) · Code: [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions)

> **This repository collects work by many authors.** Its only purpose is to provide **one platform on which all of these optimizers can be used, side by side, with Keras 3 and with PyTorch**.
>
> **The following methods are not our work, and we do not claim them:**
> - the **Herrera-style fractional optimizers** (`FSGD`, `FAdam`, `FRMSprop`, `FAdadelta`, `FAdagrad`, `FAdamW`), developed by **Oscar Herrera-Alcántara** and co-authors [[H1]](#ref-h1), [[H2]](#ref-h2);
> - the **related-work optimizers**: **AdaGL** by Chen, Zhang & Mu [[R1]](#ref-r1); **FCSGD_GL / FCAdam_GL** by Zhou, Zhao & Huang [[R2]](#ref-r2); **AOFGD** by Xiang, Zhao, Su, Zhou, Ye & Huang [[R3]](#ref-r3);
> - the **Cantor-based activations**, by Chipera, Du & Tsapara [[C1]](#ref-c1);
> - the **classical base optimizers** (SGD, momentum, Adam, RMSprop, …) [[B1]–[B10]](#base-optimizers). The baselines used here are Keras' and PyTorch's own implementations.
>
> The ideas, the mathematics and the credit belong to these authors. Our part is limited to re-implementing their published update rules in Keras 3 and in PyTorch, from the papers, and testing them side by side. These classes are **not the original authors' code**, and any error in them is ours.
>
> **If you use any of these optimizers or activations, cite the original publications** listed under [Attribution and how to cite](#attribution-and-how-to-cite), with the links given there. You may additionally cite this repository for the Keras 3 / PyTorch implementation. Only the **GL-memory** and **adaptive-memory** optimizers come from our article [[O1]](#ref-o1), and the fractal activations from our earlier article [[O2]](#ref-o2).

**Two frameworks, one set of update rules.**

- **`keras3/`** — The original classes were written for the old `optimizer_v2` API (`_set_hyper`, `_create_slots`, `_resource_apply_dense`, `lr=`). That API no longer exists in Keras 3, which has been the default `tf.keras` since TensorFlow 2.16, so under a current TensorFlow the original files fail on import or when an optimizer is created. The files in `keras3/` implement **the same update rules, operation for operation**, on the Keras 3 optimizer API (`build`, `update_step`). They were checked against the original files: the same weights come out after the same training steps, bit for bit.
- **`pytorch/`** — The same 27 optimizers as `torch.optim.Optimizer` subclasses and the same 18 activations as PyTorch functions, with the same class names, argument names and defaults. They were checked against the Keras 3 files: in float64 the weights after 60 training steps agree to ≤ 2·10⁻¹² (relative) for all 167 test configurations, and the activations agree to ≤ 10⁻¹³ (values and gradients).

Both are described in [Verification](#verification).

- 27 optimizers in four families, in both frameworks, plus the built-in baselines
- 18 fractal activation functions, plus a `FractalActivation` layer (Keras) / module (PyTorch)
- two linear, step-by-step [hello-world scripts](#hello-world-iris-in-one-run) on Iris, one per framework

---

## Attribution and how to cite

Whose work is what (the class names are the same in `keras3/` and `pytorch/`):

| What | Classes in this repository | Original work, please cite |
|---|---|---|
| **Herrera-style fractional optimizers** (Caputo-type gradient scaling) | `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` | **Oscar Herrera-Alcántara** and co-authors: Herrera-Alcántara, *Appl. Sci.* 2022 [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264) [[H1]](#ref-h1); Herrera-Alcántara & Castelán-Aguilar, *Fractal Fract.* 2023 [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500) [[H2]](#ref-h2) |
| **AdaGL** | `AdaGL` | **Chen, Zhang & Mu**, *Neural Process. Lett.* 2024 [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7) [[R1]](#ref-r1) |
| **FCSGD_GL / FCAdam_GL** (Grünwald–Letnikov optimizers with stochastic perturbation) | `FCSGD_GL`, `FCAdam_GL` | **Zhou, Zhao & Huang**, *Mathematics* 2023 [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316) [[R2]](#ref-r2) |
| **AOFGD** (adaptive-order fractional gradient descent) | `AOFGD_SGD`, `AOFGD_Adam` | **Xiang, Zhao, Su, Zhou, Ye & Huang**, SSRN preprint 2025 [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) [[R3]](#ref-r3) |
| **Classical base optimizers** (the built-in baselines, and the base update rules of every fractional variant) | `keras.optimizers.*`, `torch.optim.*`; the SGD/Adam/RMSprop/Adadelta/Adagrad/AdamW/Nadam/Adamax parts of all classes | the original papers [[B1]–[B10]](#base-optimizers) |
| Grünwald–Letnikov fractional derivative and short-memory principle (background of all GL-based classes) | `MemoryF*`, `AdaptiveMemoryF*`, `AdaGL`, `FC*_GL` | Podlubny (1999) [[G1]](#ref-g1); Wei et al. (2017) [[G2]](#ref-g2) |
| **Adaptive memory-fractional optimizers** (contribution of our paper) | `AdaptiveMemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | Raubitzek et al. (2026, Fractal and Fractional) [[O1]](#ref-o1) |
| **GL-memory optimizers** (our implementation of the finite-history GL gradient substitution, with norm matching and descent safeguard) | `MemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` | Raubitzek et al. (2026, Fractal and Fractional) [[O1]](#ref-o1), plus the GL background [[G1]](#ref-g1), [[G2]](#ref-g2); see also Zhou et al. (2023) [[R2]](#ref-r2) |
| **Fractal activation functions** (Blancmange, decaying cosine, modified Weierstrass, Weierstrass–Mandelbrot variants) | the nine fixed-depth functions in `fractal_activation_functions_keras3.py` / `fractal_activation_functions_torch.py` | Raubitzek et al., *CNSNS* 2026 [doi:10.1016/j.cnsns.2026.110064](https://doi.org/10.1016/j.cnsns.2026.110064) [[O2]](#ref-o2); mathematical origins [[F1]–[F5]](#fractal-functions-mathematical-origins) |
| Random-depth variants | `random_depth_*` | extensions in this code base, not part of the published experiments; cite [[O2]](#ref-o2) for the underlying functions |
| **Cantor-based activations** | `cantor_function_tf`, `cantor_set_activation_tf`, `random_depth_cantor_function_tf` | **Chipera, Du & Tsapara**, [arXiv:2512.14675](https://arxiv.org/abs/2512.14675) (2025) [[C1]](#ref-c1); Cantor function/set [[F5]](#ref-f5) |
| Software and data used by the scripts | — | Keras [[S1]](#ref-s1), TensorFlow [[S2]](#ref-s2), PyTorch [[S4]](#ref-s4), scikit-learn [[S3]](#ref-s3); Iris data [[D1]](#ref-d1) |

**Original code.** Oscar Herrera-Alcántara's own TensorFlow-Keras implementations of his optimizers are available at [ia.azc.uam.mx](http://ia.azc.uam.mx/), as stated in the data availability section of [[H1]](#ref-h1). Herrera-Alcántara & Castelán-Aguilar [[H2]](#ref-h2) describe their own PyTorch implementations of the fractional optimizers; the PyTorch classes in this repository are an independent re-implementation that follows our Keras version, not their code. For AdaGL, FCSGD_GL/FCAdam_GL and AOFGD we did not find code published by the authors; the AdaGL article states that code "could be made available at a later stage". If the authors publish their code, it is the authoritative reference.

The Herrera-style and related-work classes are implemented from the published descriptions; the update rules used are written out in each class docstring. For example, AOFGD's convergence-evaluation factor is computed here from consecutive gradient norms, and FCSGD_GL/FCAdam_GL use 10 historical terms. They may differ in details from the original authors' own implementations, so if a result must match the original method exactly, check it against the original publication or code.

The full reference list and ready-to-use BibTeX entries are under [References](#references) at the end of this file.

---

## Contents

```
keras3/                                              Keras 3 / TensorFlow 2.21
    class_b_tf_fractional_optimizers_OHA_keras3.py
    class_c_tf_fractional_optimizers_SR_keras3.py
    class_minimal_fractional_adaptive_optimizers_keras3.py
    class_d_related_work_optimizers_keras3.py
    fractal_activation_functions_keras3.py
    hello_world_iris_keras3.py
pytorch/                                             PyTorch 2.15
    class_b_fractional_optimizers_OHA_torch.py
    class_c_fractional_optimizers_SR_torch.py
    class_minimal_fractional_adaptive_optimizers_torch.py
    class_d_related_work_optimizers_torch.py
    fractal_activation_functions_torch.py
    hello_world_iris_torch.py
tests/                                               all checks (see Verification)
requirements.txt                                     the pinned versions below, for plain pip
```

| Keras 3 file (`keras3/`) | PyTorch file (`pytorch/`) | Replaces (legacy) | Contents |
|---|---|---|---|
| `class_b_tf_fractional_optimizers_OHA_keras3.py` | `class_b_fractional_optimizers_OHA_torch.py` | `class_b_tf_fractional_optimizers_OHA.py` | Herrera-style: `FSGD`, `FAdam`, `FAdagrad`, `FAdadelta`, `FRMSprop`, `FAdamW` |
| `class_c_tf_fractional_optimizers_SR_keras3.py` | `class_c_fractional_optimizers_SR_torch.py` | `class_c_tf_fractional_optimizers_SR.py` | Grünwald–Letnikov memory: `MemoryFSGD`, `MemoryFRMSprop`, `MemoryFAdam`, `MemoryFAdadelta`, `MemoryFAdagrad`, `MemoryFAdamW`, `MemoryFNadam`, `MemoryFAdamax` |
| `class_minimal_fractional_adaptive_optimizers_keras3.py` | `class_minimal_fractional_adaptive_optimizers_torch.py` | `class_minimal_fractional_adaptive_optimizers.py` | Adaptive memory: `AdaptiveMemoryFSGD`, `…FRMSprop`, `…FAdam`, `…FAdadelta`, `…FAdagrad`, `…FAdamW`, `…FNadam`, `…FAdamax` (Keras: plus the `AdaptiveLossFeedback` callback) |
| `class_d_related_work_optimizers_keras3.py` | `class_d_related_work_optimizers_torch.py` | `class_d_related_work_optimizers.py` | Related work: `AdaGL`, `FCSGD_GL`, `FCAdam_GL`, `AOFGD_SGD`, `AOFGD_Adam` |
| `fractal_activation_functions_keras3.py` | `fractal_activation_functions_torch.py` | `fractal_activation_functions.py` | All 18 fractal activations, `PAPER_FRACTAL_ACTIVATIONS`, `ALL_FRACTAL_ACTIVATIONS`, `get_fractal_activation()`, `FractalActivation` layer / module |
| `hello_world_iris_keras3.py` | `hello_world_iris_torch.py` | — | One run on Iris that uses every optimizer and every activation, with full classification metrics |

Class names, constructor arguments and defaults are the same as in the legacy files and the same in both frameworks; in PyTorch the parameters come first and the learning rate is called `lr` (`learning_rate=` also works). In existing Keras code, changing the import line is usually enough.

---

## Installation

Create the environment with:

```
conda create -n fractal-fractional-keras3-torch python=3.12
conda activate fractal-fractional-keras3-torch

pip install \
    numpy==2.5.4 \
    scipy==1.18.1 \
    pandas==3.1.0 \
    scikit-learn==1.9.1 \
    tensorflow==2.21.0 \
    keras==3.15.1 \
    torch==2.15.0 \
    matplotlib==3.11.2 \
    seaborn==0.13.2
```

Check the installation:

```
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__, keras.backend.backend())"
# 2.21.0 3.15.1 tensorflow
python -c "import torch; print(torch.__version__)"
# 2.15.0+cu132   (Linux; on macOS and Windows: 2.15.0)
```

Notes:

- **Only one framework?** Leave out `tensorflow==2.21.0 keras==3.15.1` (PyTorch only) or `torch==2.15.0` (Keras only). Everything in `keras3/` needs only TensorFlow/Keras, everything in `pytorch/` needs only PyTorch; only the cross-framework tests need both.
- **PyTorch on Linux:** the default wheel from PyPI is built for CUDA and pulls in the NVIDIA libraries (several GB, so check your free disk space). On a machine without an NVIDIA GPU you can install the much smaller CPU build instead: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`. On macOS the default wheel uses the CPU (and Apple's `mps` device).
- **TensorFlow:** 2.21 provides wheels for Python 3.10–3.13 on Linux (x86-64 and ARM64), macOS on Apple silicon (macOS 12+), and Windows (CPU). There are no wheels for Intel Macs. **GPU (Linux, NVIDIA):** replace `tensorflow==2.21.0` with `"tensorflow[and-cuda]==2.21.0"`. Windows users who want a TensorFlow GPU need WSL2.
- Install TensorFlow and PyTorch with `pip` inside the conda environment, as shown above, not with `conda install`. The official builds are published on PyPI (and on download.pytorch.org); the conda builds lag behind.
- **TensorFlow and PyTorch in the same Python process:** with the default Linux PyTorch wheel, which ships Triton, importing `tensorflow` first and then running any PyTorch optimizer step (built-in or ours) can abort the process with `LLVM ERROR: inconsistency in registered CommandLine options` (TensorFlow and Triton both bundle LLVM). Import `torch` and `torch._dynamo` *before* `tensorflow`, or keep the two frameworks in separate processes, as the tests in this repository do. Using one framework per script avoids the issue entirely.
- The environment has a new name, so it can sit next to the old `fractal-fractional` environment (Python 3.8, TensorFlow 2.10) that the paper's experiments ran in. `seaborn` is not needed by the scripts here; it is kept so the environment matches the analysis environment of the paper.
- This exact list was installed into a fresh Python 3.12.13 virtual environment. The full PyTorch hello world, the Keras hello world without the two tours, the PyTorch tests, the cross-framework tests and the Keras 3 feature test were run in it (see [Verification](#verification)); both hello worlds reproduced the reference numbers below exactly.

---

## Hello world: Iris in one run

```
cd keras3  && python hello_world_iris_keras3.py
cd pytorch && python hello_world_iris_torch.py
```

Both scripts are deliberately linear: no `main()`, no factories, no class hierarchy. They read top to bottom like a recipe, with every step numbered and explained in the code, and the two scripts have the same steps, the same settings and the same network, so you can read them side by side.

| Step | What it shows | Keras 3 | PyTorch |
|---|---|---|---|
| 0 | All settings at the top (seed, paper protocol, fractional order, learning rates, which parts to run) | | `DEVICE = "auto"` (GPU if available) |
| 1–2 | Imports, version printout, one seed for everything | `keras.utils.set_random_seed` | `random.seed`, `np.random.seed`, `torch.manual_seed` |
| 3–4 | Load Iris [[D1]](#ref-d1); 70/30 split with `random_state = seed`; Min-Max scaling fitted on the training part (paper protocol) | | |
| 5 | Call the fractal activations directly; optional plot of all 18 | | |
| 6 | Paper's Iris network: 32 units → BN → Dropout(0.2) → 64 units → BN → Dropout(0.2) → 3 classes, Glorot-uniform (seed 42) | `Dense(activation=fn)` | `nn.Linear` + `FractalActivation`, `BatchNorm1d(momentum=0.01, eps=1e-3)` (= Keras' BN defaults) |
| 7 | One fractional optimizer (`AdaptiveMemoryFAdam`), every argument explained | | |
| 8 | Training, 30 epochs, batch 16 | `model.compile` + `model.fit` | the standard `zero_grad / backward / step` loop |
| 9 | Accuracy, macro precision/recall/F1, confusion matrix, classification report | | |
| 10 | Inside the optimizer: GL coefficients, DC gain, λ_t per variable, Herrera factor f_ν(w) | | |
| 11 | Save and reload, with optimizer state | `model.save("….keras")`, `keras.models.load_model` | `torch.save(state_dicts)`, `torch.load(weights_only=True)` |
| 12 | **Optimizer tour:** all 35 optimizers (8 built-in baselines + 27 fractional), same model/seed/data | | |
| 13 | **Activation tour:** relu, tanh and all 18 fractal activations, same optimizer | | |
| 14 | Training loop with `set_current_loss()` (loss feedback for the adaptive family) | `GradientTape` loop | plus gradient clipping and a cosine LR schedule |
| 15 | Results to `hello_world_iris_keras3_results/` or `hello_world_iris_torch_results/` (JSON, CSV, saved model, PNG plots) | | |

The main example uses **AdaptiveMemoryFAdam + decaying cosine, ν = 0.75**. That is rank 2 of the paper's Iris table (mean accuracy 0.937 ± 0.037 over 40 runs) and the fastest of the top three configurations.

**Runtime** on a 2-core cloud CPU: the full Keras script trains 57 small networks in about 25 minutes (most of it is TensorFlow tracing the 75- to 100-term activation graphs once per model); the full PyTorch script takes about 9 minutes. For a run of 1–2 minutes, set `RUN_OPTIMIZER_TOUR = False` and `RUN_ACTIVATION_TOUR = False` at the top.

### Output of the reference runs

Reference runs: Python 3.12.13, CPU only, seed 238974. Keras 3: TensorFlow 2.21.0, Keras 3.15.1. PyTorch: torch 2.15.0.

**Main example (STEPS 6–11):** `AdaptiveMemoryFAdam` (ν = 0.75) + `decaying_cosine_function_tf`, 30 epochs.

```
Keras 3                                         PyTorch
[STEP 9] test-set results                       [STEP 9] test-set results
         test_loss        0.1855                         test_loss        0.2480
         accuracy         0.9111                         accuracy         0.9111
         precision_macro  0.9365                         precision_macro  0.9217
         recall_macro     0.9111                         recall_macro     0.9137
         f1_macro         0.9136                         f1_macro         0.9153

confusion matrix (rows = true, columns = predicted)
            setosa  versicolor  virginica                   setosa  versicolor  virginica
setosa          13           0          0       setosa          13           0          0
versicolor       0          11          4       versicolor       0          12          3
virginica        0           0         17       virginica        0           1         16
```

In both frameworks the saved model reloads with the same optimizer class and the same λ_t values, and gives identical predictions.

**Optimizer tour (STEP 12):** decaying cosine activation, ν = 0.75 for every fractional optimizer, one run each. Baselines: `keras.optimizers.*` and `torch.optim.*` (the PyTorch baselines are set to the Keras defaults: `eps=1e-7`, RMSprop `alpha=0.9`, Adadelta `rho=0.95`, Adagrad `initial_accumulator_value=0.1`, AdamW `weight_decay=0.004`).

| Family | Optimizer | Keras 3 accuracy | Keras 3 macro F1 | PyTorch accuracy | PyTorch macro F1 |
|---|---|---:|---:|---:|---:|
| baseline | `SGD` | 0.6444 | 0.5878 | 0.8889 | 0.8949 |
| baseline | `Adam` | 0.9111 | 0.9153 | 0.9111 | 0.9153 |
| baseline | `RMSprop` | 0.9556 | 0.9577 | 0.9111 | 0.9153 |
| baseline | `Adadelta` | 0.9333 | 0.9359 | 0.9111 | 0.9153 |
| baseline | `Adagrad` | 0.9556 | 0.9577 | 0.9556 | 0.9582 |
| baseline | `AdamW` | 0.9111 | 0.9163 | 0.9556 | 0.9582 |
| baseline | `Nadam` / `NAdam` | 0.9778 | 0.9790 | 0.9111 | 0.9153 |
| baseline | `Adamax` | 0.9111 | 0.9153 | 0.8000 | 0.7874 |
| Herrera | `FSGD` | 0.4889 | 0.4172 | 0.6222 | 0.5461 |
| Herrera | `FAdam` | 0.9556 | 0.9582 | 0.8667 | 0.8730 |
| Herrera | `FRMSprop` | 0.9556 | 0.9577 | 0.9111 | 0.9153 |
| Herrera | `FAdadelta` | 0.9111 | 0.9153 | 0.9111 | 0.9120 |
| Herrera | `FAdagrad` | 0.9556 | 0.9577 | 0.9333 | 0.9321 |
| Herrera | `FAdamW` | 0.9556 | 0.9577 | 0.8889 | 0.8932 |
| GL memory | `MemoryFSGD` | 0.8444 | 0.8529 | 0.8667 | 0.8750 |
| GL memory | `MemoryFAdam` | 0.8000 | 0.7975 | 0.7333 | 0.7407 |
| GL memory | `MemoryFRMSprop` | 0.8444 | 0.8362 | 0.6222 | 0.6427 |
| GL memory | `MemoryFAdadelta` | 0.8000 | 0.7906 | 0.5111 | 0.5302 |
| GL memory | `MemoryFAdagrad` | 0.9333 | 0.9359 | 0.7333 | 0.7270 |
| GL memory | `MemoryFAdamW` | 0.8667 | 0.8667 | 0.7111 | 0.6628 |
| GL memory | `MemoryFNadam` | 0.8444 | 0.8416 | 0.6222 | 0.5979 |
| GL memory | `MemoryFAdamax` | 0.7556 | 0.7582 | 0.3111 | 0.3207 |
| Adaptive memory | `AdaptiveMemoryFSGD` | 0.4889 | 0.4172 | 0.6222 | 0.5461 |
| Adaptive memory | `AdaptiveMemoryFAdam` | 0.9111 | 0.9136 | 0.9111 | 0.9153 |
| Adaptive memory | `AdaptiveMemoryFRMSprop` | 0.9556 | 0.9577 | 0.9333 | 0.9359 |
| Adaptive memory | `AdaptiveMemoryFAdadelta` | 0.9111 | 0.9163 | 0.9111 | 0.9153 |
| Adaptive memory | `AdaptiveMemoryFAdagrad` | 0.9556 | 0.9582 | 0.8667 | 0.8704 |
| Adaptive memory | `AdaptiveMemoryFAdamW` | 0.9778 | 0.9790 | 0.8667 | 0.8704 |
| Adaptive memory | `AdaptiveMemoryFNadam` | 0.9333 | 0.9369 | 0.8889 | 0.8906 |
| Adaptive memory | `AdaptiveMemoryFAdamax` | 0.9111 | 0.9153 | 0.7778 | 0.7691 |
| Related work | `AdaGL` | 0.7111 | 0.7119 | 0.3111 | 0.3194 |
| Related work | `FCSGD_GL` | 0.8889 | 0.8949 | 0.8444 | 0.8505 |
| Related work | `FCAdam_GL` | 0.9333 | 0.9369 | 0.8667 | 0.8667 |
| Related work | `AOFGD_SGD` | 0.7111 | 0.6458 | 0.6222 | 0.5461 |
| Related work | `AOFGD_Adam` | 0.9333 | 0.9359 | 0.8667 | 0.8704 |

**Activation tour (STEP 13):** `AdaptiveMemoryFAdam`, ν = 0.75, one run each.

| Group | Activation | Keras 3 accuracy | Keras 3 macro F1 | PyTorch accuracy | PyTorch macro F1 | Note |
|---|---|---:|---:|---:|---:|---|
| classical | `relu` | 0.8000 | 0.8072 | 0.3333 | 0.1667 | PyTorch: BatchNorm statistics not settled, see below |
| classical | `tanh` | 0.6222 | 0.5461 | 0.6222 | 0.5461 |  |
| fractal (paper) | `modulated_blancmange_curve` | 0.9111 | 0.9167 | 0.6889 | 0.6768 |  |
| fractal (paper) | `decaying_cosine_function_tf` | 0.9111 | 0.9136 | 0.9111 | 0.9153 |  |
| fractal (paper) | `modified_weierstrass_function_tanh` | 0.6889 | 0.6606 | 0.8889 | 0.8949 |  |
| fractal | `modified_weierstrass_function_relu` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_xsinsquared` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal (paper) | `weierstrass_mandelbrot_function_xpsin` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_relupsin` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_mandelbrot_function_tanhpsin` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `weierstrass_function_tf` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | diverged (NaN loss) |
| fractal | `random_depth_modified_weierstrass_function_tanh` | 0.6667 | 0.6257 | 0.9333 | 0.9374 |  |
| fractal | `random_depth_decaying_cosine_function_tf` | 0.7556 | 0.7593 | 0.7556 | 0.7593 |  |
| fractal | `random_depth_modulated_blancmange_curve` | 0.8444 | 0.8505 | 0.7556 | 0.7511 |  |
| fractal | `random_depth_weierstrass_mandelbrot_function_xpsin` | 0.8889 | 0.8932 | 0.9111 | 0.9163 |  |
| fractal | `random_depth_weierstrass_mandelbrot_function_tanhpsin` | 0.8667 | 0.8704 | 0.8889 | 0.8937 |  |
| fractal | `random_depth_weierstrass_function_tf` | 0.3778 | 0.3643 | 0.3556 | 0.3526 |  |
| fractal | `cantor_function_tf` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | no gradient (piecewise constant) |
| fractal | `cantor_set_activation_tf` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | no gradient (piecewise constant) |
| fractal | `random_depth_cantor_function_tf` | 0.2889 | 0.1494 | 0.2889 | 0.1494 | no gradient (piecewise constant) |

**Loop with loss feedback (STEP 14):** Keras 3 test accuracy 0.8889, macro F1 0.8932; PyTorch (with gradient clipping and a cosine LR schedule) 0.7778, macro F1 0.7691.

How to read these numbers:

- **One seed is not a benchmark, and the two columns are two different runs.** Keras and PyTorch draw different initial weights, Dropout masks and mini-batch orders from the same seed, so the Keras and PyTorch numbers of one row are two independent single runs, not a comparison of the frameworks. (Whether the two implementations compute the same thing is tested separately, with identical inputs, under [Verification](#verification).) On Iris the paper's run-to-run standard deviation is 0.03–0.06 over 40 runs, and single runs can be much further off. Example: in an earlier development run of the Keras script, `FSGD` scored 0.91 instead of 0.49. That version differed from the final one only in how the graph was laid out (the all-reduce change described under Migration), which shifts graph-mode results by about one float32 rounding step. With momentum SGD on 105 training samples, that was enough to send training somewhere else. Use the paper's protocol (40 seeds) for comparisons. Your numbers can also differ slightly on other CPUs, on GPUs, or with oneDNN switched off.
- **The PyTorch `relu` run (0.33) is a BatchNorm effect, not a training failure.** The network fits the training data (final training loss 0.19), but with the paper's BatchNorm momentum (Keras 0.99 = PyTorch 0.01) the running statistics are still far from the batch statistics after 210 steps, and in evaluation mode every flower ends up in one class. In a quick check with the same seed and optimizer, using batch statistics at test time gave 0.93, and PyTorch's default BatchNorm momentum (0.1) gave 0.98. The Keras run, with its own random draws, happened to land at 0.80. Comparing the training accuracy in inference mode (printed in STEP 9) with the per-epoch training accuracy reveals such cases.
- **The broad picture matches the paper's Iris results.** Decaying cosine is among the strongest activations in both frameworks, the GL-memory optimizers trail the other fractional families, and the SGD-type optimizers at lr = 1e-2 struggle with this activation in 30 epochs.
- **λ_t ends at 0 here.** With batches of 16, the relative gradient change ("stability") stays around 1.5–2, far above `stability_threshold = 0.25`. The adaptive optimizers therefore fade the memory term out and behave like their base optimizer (λ_t starts at 0.10; see the STEP 14 log). That is the intended safe fallback.
- **Divergence and no-gradient cases are properties of the activation functions**, identical in both frameworks. The Weierstrass–Mandelbrot variants and `weierstrass_function_tf` diverge with 100 terms, and the Cantor functions have no gradient. See [Using the fractal activations](#using-the-fractal-activations).

---

## Using the optimizers

**Keras 3** — every optimizer is a regular `keras.optimizers.Optimizer`:

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

**PyTorch** — every optimizer is a regular `torch.optim.Optimizer`:

```python
import torch
from class_minimal_fractional_adaptive_optimizers_torch import AdaptiveMemoryFAdam

optimizer = AdaptiveMemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.75, history_size=4)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(30):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
```

What every Keras 3 class supports:

- **`learning_rate=`** is the Keras 3 argument. `lr=` still works as an alias, so old experiment scripts keep running.
- **Learning-rate schedules:** `learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=1000)`.
- **Keras 3 base arguments:** `clipnorm`, `clipvalue`, `global_clipnorm`, `use_ema`, `ema_momentum`, `loss_scale_factor`, `gradient_accumulation_steps`. Also `weight_decay` (Keras-style decoupled decay applied *before* the step; the `…AdamW` classes keep their own post-step decay, see below).
- **Saving:** every class is registered with Keras. `model.save("model.keras")` stores the optimizer configuration and state (moments, gradient history, λ_t, α_t). `keras.models.load_model("model.keras")` restores them, provided the module has been imported.
- **XLA:** `model.compile(..., jit_compile=True)` works for every optimizer and activation.
- **`fit`, custom loops and `tf.distribute`:** all supported. The optimizers implement `update_step`, so they run inside `model.fit`, inside custom `tf.GradientTape` loops, and under `tf.distribute` strategies (gradients are all-reduced whenever there is more than one replica).

What every PyTorch class supports:

- **`lr=`** is the PyTorch argument; `learning_rate=` works as an alias. **All other argument names are the Keras names** (`beta_1`, `beta_2`, `epsilon`, `rho`, `momentum`, `vderiv`, `history_size`, …), so a configuration can be copied from one framework to the other unchanged. They differ from PyTorch's own `betas=` / `eps=` on purpose.
- **Parameter groups:** every hyperparameter can be set per group, e.g. `FAdam([{"params": body.parameters()}, {"params": head.parameters(), "lr": 1e-2, "vderiv": 1.25}], lr=1e-3, vderiv=0.75)`. Groups are validated when they are added.
- **`torch.optim.lr_scheduler.*`**, gradient clipping with `torch.nn.utils.clip_grad_norm_` (before `step()`), **automatic mixed precision** (`torch.autocast` + `torch.amp.GradScaler`), models wrapped in **`torch.compile`**, and float32, float64 and bfloat16 parameters. All state lives on the parameter's device, so the classes run on CUDA and `mps` as well (the tests here ran on CPU).
- **Checkpoints:** `optimizer.state_dict()` / `load_state_dict()` store and restore every hyperparameter and the whole state (moments, gradient history, λ_t, α_t, the loss-feedback values). The state dicts contain only tensors and numbers, so the safe `torch.load(path, weights_only=True)` works. Resuming from a checkpoint gives bit-identical results to an uninterrupted run.
- **`step(closure)`**, **sparse gradients** (e.g. `nn.Embedding(sparse=True)`; densified, as in the Keras version) and parameters without a gradient (frozen layers are skipped).
- Keras-only base arguments have the usual PyTorch counterparts: `clipnorm` → `clip_grad_norm_`, `use_ema` → `torch.optim.swa_utils.AveragedModel`, `gradient_accumulation_steps` → call `backward()` several times before `step()`.
- **Conventions kept from Keras:** the SGD-type classes use the Keras momentum form (velocity v ← μv − lr·g, w ← w + v), which equals `torch.optim.SGD` for a constant learning rate; the step counter is kept per parameter (`state["step"]`); the `…AdamW` classes apply the decoupled weight decay after the Adam step, as in the Keras and legacy files.

### Family 1 — Herrera-style (`class_b_…`)

> **Original work by Oscar Herrera-Alcántara and co-authors. Please cite** [[H1]](#ref-h1) Herrera-Alcántara, *Applied Sciences* 12, 9264 (2022), [doi:10.3390/app12189264](https://doi.org/10.3390/app12189264), and [[H2]](#ref-h2) Herrera-Alcántara & Castelán-Aguilar, *Fractal and Fractional* 7, 500 (2023), [doi:10.3390/fractalfract7070500](https://doi.org/10.3390/fractalfract7070500).

The gradient is scaled element-wise by f_ν(w) = (|w| + ε)^(1−ν) / Γ(2−ν). There is no memory, and ν = 1 gives back the base optimizer.

```python
# Keras 3
from class_b_tf_fractional_optimizers_OHA_keras3 import FSGD, FAdam, FRMSprop, FAdadelta, FAdagrad, FAdamW

FSGD(learning_rate=1e-2, momentum=0.9, nesterov=False, vderiv=0.75, frac_epsilon=1e-6)
FAdam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, vderiv=1.25)
FRMSprop(learning_rate=1e-3, rho=0.9, momentum=0.0, centered=False, vderiv=1.25)
FAdadelta(learning_rate=1.0, rho=0.95, vderiv=1.25)
FAdagrad(learning_rate=1e-2, initial_accumulator_value=0.1, vderiv=1.25)
FAdamW(learning_rate=1e-3, weight_decay=0.004, vderiv=1.25)
```

```python
# PyTorch
from class_b_fractional_optimizers_OHA_torch import FSGD, FAdam, FRMSprop, FAdadelta, FAdagrad, FAdamW

FSGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=False, vderiv=0.75, frac_epsilon=1e-6)
FAdam(model.parameters(), lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, vderiv=1.25)
FRMSprop(model.parameters(), lr=1e-3, rho=0.9, momentum=0.0, centered=False, vderiv=1.25)
FAdadelta(model.parameters(), lr=1.0, rho=0.95, vderiv=1.25)
FAdagrad(model.parameters(), lr=1e-2, initial_accumulator_value=0.1, vderiv=1.25)
FAdamW(model.parameters(), lr=1e-3, weight_decay=0.004, vderiv=1.25)
```

### Family 2 — Grünwald–Letnikov memory (`class_c_…`)

> Implementation from our paper [[O1]](#ref-o1), based on the Grünwald–Letnikov derivative [[G1]](#ref-g1) and the short-memory principle [[G2]](#ref-g2); the base update rules are those of [[B1]–[B10]](#base-optimizers).

The direction is a GL sum of the last K raw gradients, g^(ν) = Σ c_k g_{t−k}, with c_0 = 1 and c_k = (1 − (ν+1)/k)·c_{k−1}. Three defaults control how it is used:

- `norm_match=True` rescales the sum to ‖g_t‖.
- `descent_check=True` falls back to g_t when ⟨g^(ν), g_t⟩ ≤ 0. This matters for ν > 1, where the coefficient sum is negative.
- `use_raw_gradient_for_second_moment=True` feeds the raw gradient to all squared-gradient accumulators.

The history is a circular buffer with one row written per step.

```python
# Keras 3
from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam
opt = MemoryFAdam(learning_rate=1e-3, vderiv=0.75, history_size=6,
                  normalize_coefficients=False, norm_match=True,
                  descent_check=True, use_raw_gradient_for_second_moment=True)

# PyTorch
from class_c_fractional_optimizers_SR_torch import MemoryFAdam
opt = MemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.75, history_size=6,
                  normalize_coefficients=False, norm_match=True,
                  descent_check=True, use_raw_gradient_for_second_moment=True)

# both
opt.gl_coefficients        # [1.0, -0.75, -0.0938, -0.0391, -0.022, -0.0143]
opt.gl_coefficient_sum()   # +0.0809, the DC gain of the truncated kernel (negative for nu > 1)
```

### Family 3 — Adaptive memory (`class_minimal_…`)

> New in our paper [[O1]](#ref-o1). Please cite it when you use these optimizers; the base update rules are those of [[B1]–[B10]](#base-optimizers).

The update uses g_eff = (1 − λ_t)·g_t + λ_t·ĝ^(ν). Only the mixing weight λ_t is adapted: it grows while the relative gradient change stays below `stability_threshold`, shrinks otherwise, and is bounded by `[min_mix, max_mix]`. With λ_t = 0, or with ν = 1, the update is exactly the base optimizer.

```python
# Keras 3
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

```python
# PyTorch
from class_minimal_fractional_adaptive_optimizers_torch import AdaptiveMemoryFAdam

opt = AdaptiveMemoryFAdam(model.parameters(), lr=1e-3, vderiv=0.75, history_size=4,
                          normalize_coefficients=True, initial_mix=0.10, min_mix=0.0,
                          max_mix=0.30, adapt_step=0.01, mix_ema_gamma=0.90,
                          stability_ema_gamma=0.95, stability_threshold=0.25, warmup_steps=0)
for x, y in train_loader:
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.set_current_loss(loss)  # optional loss feedback (the Keras callback does this in fit)
opt.current_mix_values()        # λ_t, one scalar per parameter (order of model.parameters())
opt.current_stability_values()  # smoothed stability score per parameter
```

In a custom Keras loop, call `opt.set_current_loss(loss)` after each step as well (STEP 14 of the hello worlds).

### Family 4 — Related work (`class_d_…`)

> **Not our work. Please cite the original authors:**
> - `AdaGL`: Chen, Zhang & Mu, *Neural Processing Letters* 56, 106 (2024), [doi:10.1007/s11063-024-11571-7](https://doi.org/10.1007/s11063-024-11571-7) [[R1]](#ref-r1)
> - `FCSGD_GL`, `FCAdam_GL`: Zhou, Zhao & Huang, *Mathematics* 11, 316 (2023), [doi:10.3390/math11020316](https://doi.org/10.3390/math11020316) [[R2]](#ref-r2)
> - `AOFGD_SGD`, `AOFGD_Adam`: Xiang, Zhao, Su, Zhou, Ye & Huang, SSRN preprint (2025), [doi:10.2139/ssrn.5717167](https://doi.org/10.2139/ssrn.5717167) [[R3]](#ref-r3)

```python
# Keras 3
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

```python
# PyTorch
from class_d_related_work_optimizers_torch import AdaGL, FCSGD_GL, FCAdam_GL, AOFGD_SGD, AOFGD_Adam

AdaGL(model.parameters(), lr=1e-3, alpha=0.75, history_size=10)
FCSGD_GL(model.parameters(), lr=1e-2, momentum=0.9, alpha=0.75, history_size=11,
         perturbation_prob=0.5)
FCAdam_GL(model.parameters(), lr=1e-3, alpha=0.75, history_size=11, perturbation_prob=0.5,
          generator=torch.Generator().manual_seed(0))   # optional: own random stream for the masks
AOFGD_SGD(model.parameters(), lr=1e-2, momentum=0.9, alpha_init=0.75, alpha_min=0.5,
          alpha_max=1.5, delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95)
opt = AOFGD_Adam(model.parameters(), lr=1e-3, alpha_init=0.75, alpha_min=0.5, alpha_max=1.5)
opt.current_alpha_values()      # after training: current smoothed order per parameter
```

The Bernoulli masks of `FCSGD_GL` / `FCAdam_GL` are drawn with `tf.random` (Keras) or `torch.rand` (PyTorch), so they follow the framework's seed (`keras.utils.set_random_seed`, `torch.manual_seed`), or the `generator=` you pass in PyTorch.

### Built-in baselines

Original papers: SGD [[B1]](#ref-b1), momentum [[B2]](#ref-b2), Nesterov [[B3]](#ref-b3), Adam/Adamax [[B4]](#ref-b4), AMSGrad [[B5]](#ref-b5), RMSprop [[B6]](#ref-b6), Adadelta [[B7]](#ref-b7), Adagrad [[B8]](#ref-b8), AdamW [[B9]](#ref-b9), Nadam [[B10]](#ref-b10).

```python
# Keras 3: use learning_rate= (Keras 3 rejects lr=)
keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
keras.optimizers.Adam(learning_rate=1e-3)
keras.optimizers.RMSprop(learning_rate=1e-3)
keras.optimizers.Adadelta(learning_rate=1.0)

# PyTorch, set to the Keras defaults (PyTorch's own defaults differ)
torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-7)
torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-7)
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
# Keras 3
import keras
import fractal_activation_functions_keras3 as faf

# (a) as the activation of a layer, with default parameters (as in the paper)
keras.layers.Dense(32, activation=faf.decaying_cosine_function_tf)

# (b) as a separate layer, with your own parameters; the model can still be saved
x = keras.layers.Dense(32)(x)
x = faf.FractalActivation("weierstrass_mandelbrot_function_xpsin", num_terms=10)(x)
```

```python
# PyTorch
import torch
import fractal_activation_functions_torch as faf

# (a) as a function inside forward()
h = faf.decaying_cosine_function_tf(self.fc1(x))

# (b) as a module, e.g. in nn.Sequential, optionally with your own parameters
torch.nn.Sequential(
    torch.nn.Linear(4, 32),
    faf.FractalActivation("weierstrass_mandelbrot_function_xpsin", num_terms=10),
)
```

```python
# both
faf.PAPER_FRACTAL_ACTIVATIONS   # the four used in the article
faf.ALL_FRACTAL_ACTIVATIONS     # all 18
faf.get_fractal_activation("modified_weierstrass_function_tanh")
```

The function names keep their historical `_tf` suffixes in both frameworks, so the same names work everywhere. Activations used in the article: `modulated_blancmange_curve`, `decaying_cosine_function_tf`, `modified_weierstrass_function_tanh`, `weierstrass_mandelbrot_function_xpsin`.

Behaviour to know about. All of it is unchanged from the original functions and the same in both frameworks:

- The Weierstrass–Mandelbrot variants and `weierstrass_function_tf` sum terms 2^(−k/2)·sin(2π·2^k·x) for k up to 99. Their derivatives grow like 2π·2^(k/2), which is about 10^15 for the last terms, so with the default 100 terms training can overflow to NaN. Fewer terms (`num_terms=…` via `FractalActivation`) or gradient clipping (`clipnorm` in Keras, `clip_grad_norm_` in PyTorch) help.
- `cantor_function_tf` and `cantor_set_activation_tf` are piecewise constant, so no gradient flows through them. (In PyTorch, `cantor_set_activation_tf` returns a tensor without a gradient path; layers before it get `grad = None` and are skipped by the optimizers, as Keras skips variables without gradients.)
- `modulated_blancmange_curve` contains √|x|, whose derivative is unbounded at 0. At exactly x = 0 its gradient is 0 in both frameworks (the PyTorch file uses TensorFlow's rule for the gradient of the square root there; plain `torch.sqrt` would give NaN).
- The Weierstrass-type functions compute in float64 internally and return the dtype of their input; the blancmange, decaying-cosine and Cantor functions compute in the input dtype; the random-depth blancmange and decaying-cosine variants compute in float32, as in the original.
- The `random_depth_*` functions draw a new number of series terms on every call (`tf.random` / `torch.randint`). Use `keras.utils.set_random_seed` or `torch.manual_seed` for repeatable runs.

---

## Migrating from the legacy files

| Legacy (TF ≤ 2.15, Keras 2) | Keras 3 (TF ≥ 2.16) | PyTorch |
|---|---|---|
| `from class_c_tf_fractional_optimizers_SR import MemoryFAdam` | `from class_c_tf_fractional_optimizers_SR_keras3 import MemoryFAdam` | `from class_c_fractional_optimizers_SR_torch import MemoryFAdam` |
| `MemoryFAdam(lr=1e-3, ...)` | `MemoryFAdam(learning_rate=1e-3, ...)` (`lr=` still accepted) | `MemoryFAdam(model.parameters(), lr=1e-3, ...)` |
| `tf.keras.optimizers.SGD(lr=0.01)` | `keras.optimizers.SGD(learning_rate=0.01)` | `torch.optim.SGD(model.parameters(), lr=0.01)` |
| `Dense(n, input_dim=d)` as first layer | `keras.Input(shape=(d,))`, then `Dense(n)` | `torch.nn.Linear(d, n)` |
| `tf.keras.backend.clear_session()` | `keras.backend.clear_session()` | — |
| `import fractal_activation_functions as faf` | `import fractal_activation_functions_keras3 as faf` | `import fractal_activation_functions_torch as faf` |

Differences of the Keras 3 files from the legacy files, all deliberate:

1. **Optimizer checkpoints** written by the legacy classes cannot be restored into the Keras 3 classes; the variable layout differs. Retrain, or keep the legacy environment for old checkpoints.
2. **`set_current_loss()`** (adaptive family) now actually reaches compiled training steps. The "loss got worse" flag is an optimizer variable; in the legacy file it was a Python float that `tf.function` froze at trace time. When `set_current_loss()` is never called (the paper setting), results are identical.
3. **`FAdamW`, `MemoryFAdamW`, `AdaptiveMemoryFAdamW`** keep the legacy order: Adam step first, then decoupled decay on the updated weights. Keras' own pre-step decay is switched off for these three classes. For every other class, the Keras 3 `weight_decay=` argument is available as an extra; it is off by default.
4. **Fractal activations** that compute in float64 now return the dtype of their input instead of always float32. For float32 models the output is the same; under mixed precision it now matches the layer.
5. **Graph-mode rounding.** In graph mode (`model.fit`, `tf.function`), TensorFlow's graph optimizer (Grappler) rewrites and fuses floating-point operations depending on how a graph is built. In the tests below, the Keras 3 graphs give bit-identical results to the legacy graphs. For other models or TensorFlow versions, differences in the last float32 bit are possible in principle; they are rounding, not a change of the update rule.
6. **No memory growth when training many models in one process.** With Keras 3 on TensorFlow 2.21, every `model.fit` leaves its traced training graph in memory, even after `keras.backend.clear_session()`. The cause is the optimizer's gradient all-reduce: `tf.distribute` registers a `tf.custom_gradient` in a global registry on every trace and never removes it. With the fractal activations these graphs are large, about 130–150 MB per model, so a loop over many configurations (such as the hello-world tours, or the paper's 40-run loops) eventually runs out of memory. Our optimizers skip the all-reduce when there is only one replica, where it is the identity anyway, so their memory stays flat. The Keras built-in optimizers are still affected. If you loop over many models with them, run each configuration in its own process. (PyTorch has no traced graphs to keep, so the PyTorch versions are not affected; see the memory check under Verification.)

The PyTorch files follow the Keras 3 files. Where PyTorch has its own conventions (parameters as the first argument, `lr=`, per-parameter step counter, gradient clipping outside the optimizer, loss feedback by calling `set_current_loss()` in the loop), the list under [Using the optimizers](#using-the-optimizers) says so.

### Running the original files unchanged on a current TensorFlow

To re-run the paper's code exactly as published on TF ≥ 2.16:

1. Install the legacy Keras package with the same version number as TensorFlow: `pip install tf-keras==2.21.0`.
2. Set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow.
3. Add `tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Optimizer` before importing `class_b_…` and `class_d_…`.

This is how the tests below run the legacy files.

---

## Verification

All checks are in `tests/` and ran on CPU. Checks 1–4 (Keras 3 against the legacy files) were run with TensorFlow 2.21.0 and Keras 3.15.1 under Python 3.13 before the PyTorch port; the Keras 3 files have not changed since. Checks 5–7 (PyTorch) and the Keras 3 integration test were run in a fresh Python 3.12.13 environment with exactly the versions under [Installation](#installation).

### Keras 3 against the legacy files

**1. Same numbers as the legacy implementation** (`tests/test_equivalence_with_legacy.py`). The test covers 167 optimizer configurations (320 runs): every class; ν < 1, ν = 1 and ν > 1; several history lengths; momentum, Nesterov, centered and AMSGrad variants; norm matching, descent check and second-moment switches; warm-up; loss feedback; and Bernoulli masks. Each configuration runs with the legacy file (via `tf_keras`) and with the Keras 3 file on the same 4-8-3 network, the same initial weights and the same mini-batches, for 60 steps, in eager mode and inside `tf.function`. Then the final weights and the λ_t / α_t states are compared.

Result: **320 of 320 runs bit-identical** (all 167 configurations, in eager mode and in graph mode, with TensorFlow's default graph optimizer). They are also bit-identical with the graph optimizer switched off (`DISABLE_GRAPPLER=1`).

| Family | Configurations | Eager | Graph (`tf.function`) |
|---|---:|---:|---:|
| Herrera (`class_b`) | 22 | 22/22 identical | 22/22 identical |
| GL memory (`class_c`) | 61 | 61/61 identical | 61/61 identical |
| Adaptive memory (`class_minimal`) | 52 | 52/52 identical | 44/44 identical |
| Related work (`class_d`) | 32 | 32/32 identical | 26/26 identical |

Fourteen configurations run only in eager mode. They can only be compared there: the loss-feedback runs, because the legacy flag is frozen at trace time, and the FCGL runs with p = 0.5, because the random stream depends on how the graph is built.

**2. Fractal activations** (`tests/test_activations_equivalence.py`). All 18 functions are compared with the originals on 20,001 points in [−5, 5] plus a random batch, in eager and graph mode. Values and gradients are bit-identical for all 18. The random-depth functions were compared with the same seed.

**3. Keras 3 integration** (`tests/test_keras3_features.py`, `tests/test_activations_xla.py`). For all 27 optimizers:

- `model.fit` with the paper's BatchNorm/Dropout architecture, with and without XLA (`jit_compile=True`);
- `get_config()` / `from_config()` round trip;
- `.keras` save and reload: identical predictions, identical optimizer variables, and training continues;
- learning-rate schedule plus `clipnorm`.

All 18 activations train under XLA. `Dense(activation=…)` and `FractalActivation(…)` models survive save and reload.

**4. Memory** (`tests/test_memory_flat.py`). Six decaying-cosine models are trained one after another in one process, with fractional optimizers from all four families. Resident memory stays flat: between 0.8 and 1.0 GB after every model, with no upward trend. With the Keras built-in `Adam`, the same loop grows by about 130 MB per model.

```
pip install tf-keras==2.21.0
python tests/test_equivalence_with_legacy.py <folder with the legacy files> keras3
DISABLE_GRAPPLER=1 python tests/test_equivalence_with_legacy.py <folder with the legacy files> keras3
python tests/test_activations_equivalence.py <folder with the legacy files> keras3
python tests/test_keras3_features.py keras3
python tests/test_activations_xla.py keras3
python tests/test_memory_flat.py keras3
```

### PyTorch against Keras 3

TensorFlow and PyTorch use different math libraries, whose `pow`, `exp`, `sqrt`, `tanh`, `sin` or `lgamma` can differ in the last bit (we measured 1–4 units in the last place), so bit-identical results across the two frameworks are not possible. The tests therefore run each framework in its own process on identical inputs and compare with tolerances that are set by this last-bit noise.

**5. Optimizers** (`tests/test_torch_vs_keras3.py`, with `tests/cross_runner.py`). All 167 configurations of check 1 — every class and option, the loss-feedback runs (`set_current_loss()` after every step) and the Bernoulli p = 0.5 runs — are run for 60 steps with the Keras 3 and with the PyTorch class, from the same initial weights, with loss and gradients computed by one shared NumPy function. Two experiments:

- **open loop:** both optimizers receive the same recorded sequence of 60 gradients and losses, so the inputs are identical and only the optimizer arithmetic is compared;
- **closed loop:** each optimizer trains the 4-8-3 network on its own weights, i.e. normal training.

For the p = 0.5 runs, `tf.random.uniform` and `torch.rand` are replaced by the same seeded NumPy stream, and the test checks that both draw one fresh mask per variable and step. Keras keeps its learning rate in a float32 variable, so the PyTorch optimizers get the float32-rounded rate.

*float64 (strict):* relative difference of the final weights and λ_t / α_t, max |w_torch − w_keras| / max |w_keras|. Pass: ≤ 10⁻¹² (open loop) and ≤ 10⁻¹⁰ (closed loop, also for the loss curve). In float64, `tf.cast(0.9, tf.float64)` rounds through float32, so the Keras code would run with float32-rounded hyperparameters; the Keras process of this test therefore converts Python numbers straight to float64. This does not touch the optimizer code and makes no difference in float32.

| Family | Configurations | Open loop | worst difference | Closed loop | worst difference |
|---|---:|---:|---:|---:|---:|
| Herrera (`class_b`) | 22 | 22/22 | 7.7·10⁻¹⁵ | 22/22 | 1.1·10⁻¹³ |
| GL memory (`class_c`) | 61 | 61/61 | 6.6·10⁻¹⁶ | 61/61 | 1.5·10⁻¹⁵ |
| Adaptive memory (`class_minimal`) | 52 | 52/52 | 1.3·10⁻¹⁵ | 52/52 | 1.5·10⁻¹⁵ |
| Related work (`class_d`) | 32 | 32/32 | 5.4·10⁻¹³ | 32/32 | 1.7·10⁻¹² |

*float32 (normal training precision):* float32 round-off is amplified by some updates, most of all by AOFGD's (|θ| + ε)^(1−α) for weights near zero, so the PyTorch-vs-Keras difference is compared with the float32 error of the Keras version itself (Keras float32 vs Keras float64). Pass: |torch32 − keras32| ≤ max(10 × |keras32 − keras64|, 10⁻⁵) for weights, λ_t / α_t and the loss curve. Result: 167/167 in the open loop and 167/167 in the closed loop. The largest differences (AOFGD, up to 3·10⁻³) are the same size as the Keras version's own float32 error (up to 5·10⁻³); for all other classes they are ≤ 6·10⁻⁵.

**6. Fractal activations** (`tests/test_activations_torch_vs_keras3.py`, with `tests/activation_cross_runner.py`). All 18 functions on 20,001 points in [−5, 5] plus a random 64 × 32 batch, in float32 and float64; the random-depth functions with the depth fixed to its minimum, middle and maximum. Output dtype, gradient path and the positions of NaN/inf (in values and gradients) must agree exactly. Result: **36/36 activation × dtype combinations pass.**

- float64 input: values and gradients agree to ≤ 7·10⁻¹⁴ (relative) for every function (tolerance 10⁻¹²).
- float32 input, Weierstrass family (computed in float64): values and gradients bit-identical.
- float32 input, functions computed in float32 (blancmange, decaying cosine, Cantor): ≤ 7·10⁻⁶ (tolerance 2·10⁻⁵), from the last-bit differences of sin/cos/tanh/sigmoid.
- One documented exception: `random_depth_cantor_function_tf` in float32 at its maximum depth of 15 ternary levels multiplies the last-bit difference of the sigmoid by 3¹⁵, so 0.15 % of the points fall on the other side of a ternary boundary and get a different (piecewise) gradient; the values still agree to 3·10⁻⁵, and in float64 the function agrees to 10⁻¹³.

**7. PyTorch integration** (`tests/test_torch_features.py`). For all 27 optimizers:

- training of the paper's BatchNorm/Dropout network with tanh and with a fractal activation: loss finite and decreasing;
- checkpoint: `torch.save` of the model and optimizer `state_dict()`, `torch.load(weights_only=True)` into a fresh model and optimizer, training continues **bit-identically** to an uninterrupted run (including the loss feedback and the random masks);
- `LambdaLR` (a scheduled half learning rate gives exactly the same result as half the learning rate from the start; lr = 0 leaves the parameters unchanged) and `ReduceLROnPlateau`;
- parameter groups with their own learning rate and fractional order, frozen parameters (no state is created), invalid groups rejected;
- `step(closure)`, determinism, float64 and bfloat16 parameters, `torch.autocast` + `torch.amp.GradScaler`;
- sparse gradients from `nn.Embedding(sparse=True)` give exactly the same update as dense gradients;

plus: 14 invalid settings raise `ValueError`/`TypeError`; the Bernoulli masks hit their probability and are reproducible with `generator=`; the random depth covers its whole range; `gl_coefficients`, `current_mix_values()`, `current_alpha_values()` and the loss feedback work (λ_t is pushed down when the loss gets worse); `FractalActivation` equals the function, has finite gradients at x = 0, survives `torch.save` of the whole model and adds no `state_dict` entries; a model with a fractal activation under `torch.compile` matches eager mode; and memory stays flat over 20 consecutive models.

```
python tests/test_torch_vs_keras3.py keras3 pytorch float64
python tests/test_torch_vs_keras3.py keras3 pytorch float32
python tests/test_activations_torch_vs_keras3.py keras3 pytorch
python tests/test_torch_features.py pytorch
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

<a id="ref-s4"></a>**[S4]** A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems* **32** (NeurIPS 2019), 8024–8035. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) · [pytorch.org](https://pytorch.org)

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

@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems 32},
  pages     = {8024--8035},
  year      = {2019}
}
```

---

## Citation

If you use this repository, please cite

- **the original publication of every optimizer and activation you use** (see [Attribution and how to cite](#attribution-and-how-to-cite)). In particular: Herrera-Alcántara [[H1]](#ref-h1), [[H2]](#ref-h2) for the Herrera-style optimizers; Chen, Zhang & Mu [[R1]](#ref-r1) for AdaGL; Zhou, Zhao & Huang [[R2]](#ref-r2) for FCSGD_GL/FCAdam_GL; Xiang et al. [[R3]](#ref-r3) for AOFGD; Chipera, Du & Tsapara [[C1]](#ref-c1) for the Cantor activations;
- our article [[O1]](#ref-o1) for the adaptive-memory and GL-memory optimizers, the experimental comparison, and this Keras 3 / PyTorch platform;
- Raubitzek et al. [[O2]](#ref-o2) for the fractal activation functions.

The full experiment code of the article (all ten datasets, 40 runs per configuration, analytic test surfaces) is at [github.com/Raubkatz/FractalAndFractional2026](https://github.com/Raubkatz/FractalAndFractional2026). The code of the fractal activation paper is at [github.com/Raubkatz/Fractal_Activation_Functions](https://github.com/Raubkatz/Fractal_Activation_Functions).
