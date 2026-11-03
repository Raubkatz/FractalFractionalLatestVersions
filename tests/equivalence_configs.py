"""
Optimizer configurations compared by test_equivalence_with_legacy.py.

Every configuration is run with the legacy file (tf_keras, optimizer_v2) and
with the Keras 3 file on the same toy problem; the final weights must agree.
"""

MODULES = {
    "B": ("class_b_tf_fractional_optimizers_OHA", "class_b_tf_fractional_optimizers_OHA_keras3"),
    "C": ("class_c_tf_fractional_optimizers_SR", "class_c_tf_fractional_optimizers_SR_keras3"),
    "M": ("class_minimal_fractional_adaptive_optimizers", "class_minimal_fractional_adaptive_optimizers_keras3"),
    "D": ("class_d_related_work_optimizers", "class_d_related_work_optimizers_keras3"),
}

CONFIGS = []


def add(module, cls, tag="", modes=("graph", "eager"), loss_feedback=False, **kwargs):
    cid = f"{cls}{('[' + tag + ']') if tag else ''}"
    CONFIGS.append(dict(id=cid, module=module, cls=cls, kwargs=kwargs,
                        modes=modes, loss_feedback=loss_feedback))


# --------------------------------------------------------------------------
# class_b: Herrera-style
# --------------------------------------------------------------------------
for nu in (0.75, 1.25):
    add("B", "FSGD", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FSGD", f"nu={nu},mom", lr=1e-2, momentum=0.9, vderiv=nu)
    add("B", "FSGD", f"nu={nu},nesterov", lr=1e-2, momentum=0.9, nesterov=True, vderiv=nu)
    add("B", "FAdam", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FAdam", f"nu={nu},amsgrad", lr=1e-2, amsgrad=True, vderiv=nu)
    add("B", "FAdagrad", f"nu={nu}", lr=1e-1, vderiv=nu)
    add("B", "FAdadelta", f"nu={nu}", lr=1.0, vderiv=nu)
    add("B", "FRMSprop", f"nu={nu}", lr=1e-2, vderiv=nu)
    add("B", "FRMSprop", f"nu={nu},mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=nu)
    add("B", "FAdamW", f"nu={nu}", lr=1e-2, weight_decay=0.01, vderiv=nu)
    add("B", "FAdamW", f"nu={nu},amsgrad", lr=1e-2, weight_decay=0.01, amsgrad=True, vderiv=nu)

# --------------------------------------------------------------------------
# class_c: explicit memory (GL) family
# --------------------------------------------------------------------------
C_CLASSES = ["MemoryFSGD", "MemoryFRMSprop", "MemoryFAdam", "MemoryFAdadelta",
             "MemoryFAdagrad", "MemoryFAdamW", "MemoryFNadam", "MemoryFAdamax"]
C_LR = {"MemoryFSGD": 5e-2, "MemoryFRMSprop": 1e-2, "MemoryFAdam": 1e-2,
        "MemoryFAdadelta": 1.0, "MemoryFAdagrad": 1e-1, "MemoryFAdamW": 1e-2,
        "MemoryFNadam": 1e-2, "MemoryFAdamax": 1e-2}
for cls in C_CLASSES:
    for nu in (0.75, 1.5):
        for k in (4, 6):
            add("C", cls, f"nu={nu},K={k}", lr=C_LR[cls], vderiv=nu, history_size=k)
    add("C", cls, "nu=1(classical)", lr=C_LR[cls], vderiv=1.0, history_size=6)
    add("C", cls, "nu=0.9,K=6,no-norm-match,no-descent-check,frac-2nd-moment",
        lr=C_LR[cls], vderiv=0.9, history_size=6, norm_match=False,
        descent_check=False, use_raw_gradient_for_second_moment=False)
    add("C", cls, "nu=1.25,K=6,normalized", lr=C_LR[cls], vderiv=1.25, history_size=6,
        normalize_coefficients=True)
add("C", "MemoryFSGD", "nu=0.75,K=6,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, vderiv=0.75, history_size=6)
add("C", "MemoryFSGD", "nu=0.75,K=6,mom", lr=5e-2, momentum=0.9, vderiv=0.75, history_size=6)
add("C", "MemoryFRMSprop", "nu=0.75,K=6,mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=0.75, history_size=6)
add("C", "MemoryFAdam", "nu=0.75,K=6,amsgrad", lr=1e-2, amsgrad=True, vderiv=0.75, history_size=6)
add("C", "MemoryFAdamW", "nu=1.25,K=6,wd=0.02", lr=1e-2, weight_decay=0.02, vderiv=1.25, history_size=6)

# --------------------------------------------------------------------------
# class_minimal: adaptive memory family (defaults of the paper runner)
# --------------------------------------------------------------------------
M_CLASSES = ["AdaptiveMemoryFSGD", "AdaptiveMemoryFRMSprop", "AdaptiveMemoryFAdam",
             "AdaptiveMemoryFAdadelta", "AdaptiveMemoryFAdagrad", "AdaptiveMemoryFAdamW",
             "AdaptiveMemoryFNadam", "AdaptiveMemoryFAdamax"]
M_LR = {"AdaptiveMemoryFSGD": 5e-2, "AdaptiveMemoryFRMSprop": 1e-2, "AdaptiveMemoryFAdam": 1e-2,
        "AdaptiveMemoryFAdadelta": 1.0, "AdaptiveMemoryFAdagrad": 1e-1, "AdaptiveMemoryFAdamW": 1e-2,
        "AdaptiveMemoryFNadam": 1e-2, "AdaptiveMemoryFAdamax": 1e-2}
PAPER_ADAPTIVE = dict(history_size=4, normalize_coefficients=True, initial_mix=0.10,
                      min_mix=0.0, max_mix=0.30, adapt_step=0.01, mix_ema_gamma=0.90,
                      stability_ema_gamma=0.95, stability_threshold=0.25, stability_clip=10.0,
                      norm_match_fractional_gradient=True, use_raw_gradient_for_second_moment=True,
                      loss_ema_beta=0.95, loss_worsen_tolerance=1e-4, loss_mix_decrease=0.02,
                      warmup_steps=0)
for cls in M_CLASSES:
    for nu in (0.75, 1.25, 1.5):
        add("M", cls, f"paper,nu={nu}", lr=M_LR[cls], vderiv=nu, **PAPER_ADAPTIVE)
    add("M", cls, "nu=1(classical)", lr=M_LR[cls], vderiv=1.0, history_size=4)
    add("M", cls, "nu=0.9,K=6,raw-off,normmatch-off,descent-off,warmup=10,threshold=1.0",
        lr=M_LR[cls], vderiv=0.9, history_size=6, normalize_coefficients=False,
        use_raw_gradient_for_second_moment=False, norm_match_fractional_gradient=False,
        descent_check=False, warmup_steps=10, stability_threshold=1.0, max_mix=0.5,
        adapt_step=0.05)
    # Loss feedback through set_current_loss(): only comparable eagerly,
    # because the legacy file stores the flag as a Python float that a
    # tf.function freezes at trace time.
    add("M", cls, "nu=0.75,loss-feedback", modes=("eager",), loss_feedback=True,
        lr=M_LR[cls], vderiv=0.75, **dict(PAPER_ADAPTIVE, loss_mix_decrease=0.05,
                                          loss_worsen_tolerance=0.0))
add("M", "AdaptiveMemoryFSGD", "paper,nu=1.25,mom", lr=5e-2, momentum=0.9, vderiv=1.25, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFSGD", "paper,nu=1.25,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, vderiv=1.25, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFRMSprop", "paper,nu=0.75,mom,centered", lr=1e-2, momentum=0.5, centered=True, vderiv=0.75, **PAPER_ADAPTIVE)
add("M", "AdaptiveMemoryFAdam", "paper,nu=0.75,amsgrad", lr=1e-2, amsgrad=True, vderiv=0.75, **PAPER_ADAPTIVE)

# --------------------------------------------------------------------------
# class_d: related-work optimizers
# --------------------------------------------------------------------------
for nu in (0.75, 1.25, 1.5):
    add("D", "AdaGL", f"alpha={nu},K=10", lr=1e-2, alpha=nu, history_size=10)
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=1", lr=5e-2, alpha=nu, history_size=11, perturbation_prob=1.0)
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=1,mom", lr=5e-2, momentum=0.9, alpha=nu, history_size=11, perturbation_prob=1.0)
    add("D", "FCAdam_GL", f"alpha={nu},K=11,p=1", lr=1e-2, alpha=nu, history_size=11, perturbation_prob=1.0)
    # Bernoulli mask p=0.5: same random stream only in eager mode.
    add("D", "FCSGD_GL", f"alpha={nu},K=11,p=0.5", modes=("eager",), lr=5e-2, alpha=nu, history_size=11, perturbation_prob=0.5)
    add("D", "FCAdam_GL", f"alpha={nu},K=11,p=0.5", modes=("eager",), lr=1e-2, alpha=nu, history_size=11, perturbation_prob=0.5)
    amax = max(1.5, nu)
    add("D", "AOFGD_SGD", f"alpha_init={nu}", lr=5e-2, alpha_init=nu, alpha_min=0.5, alpha_max=amax,
        delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95, frac_epsilon=1e-6, eps_cef=1e-8)
    add("D", "AOFGD_SGD", f"alpha_init={nu},mom", lr=5e-2, momentum=0.9, alpha_init=nu, alpha_min=0.5,
        alpha_max=amax)
    add("D", "AOFGD_Adam", f"alpha_init={nu}", lr=1e-2, alpha_init=nu, alpha_min=0.5, alpha_max=amax,
        delta_alpha=0.02, cef_threshold=0.05, ema_gamma=0.95, frac_epsilon=1e-6, eps_cef=1e-8)
add("D", "AdaGL", "alpha=0.9,K=10,amsgrad", lr=1e-2, alpha=0.9, history_size=10, amsgrad=True)
add("D", "AdaGL", "alpha=1(classical)", lr=1e-2, alpha=1.0, history_size=10)
add("D", "FCSGD_GL", "alpha=0.9,K=11,p=1,mom,nesterov", lr=5e-2, momentum=0.9, nesterov=True, alpha=0.9, history_size=11, perturbation_prob=1.0)
add("D", "FCAdam_GL", "alpha=0.9,K=11,p=1,amsgrad", lr=1e-2, amsgrad=True, alpha=0.9, history_size=11, perturbation_prob=1.0)
add("D", "AOFGD_Adam", "alpha_init=0.9,amsgrad", lr=1e-2, amsgrad=True, alpha_init=0.9, alpha_min=0.5, alpha_max=1.9)
