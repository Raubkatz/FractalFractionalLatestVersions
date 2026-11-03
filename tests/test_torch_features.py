"""
PyTorch integration test for every optimizer and the activation helpers.

    python tests/test_torch_features.py <pytorch_dir>

 1. Training: every optimizer trains the paper-style model (Linear, act,
    BatchNorm1d, Dropout, x2) with tanh and with a fractal activation; the
    loss must stay finite and decrease (mean of the last 3 epochs < epoch 1).
 2. Checkpointing: train, torch.save(model/optimizer state_dict) and
    torch.load(weights_only=True) into a fresh model/optimizer, continue;
    the result must be bit-identical to an uninterrupted run (including the
    adaptive loss feedback and the RNG of the Bernoulli masks).
 3. LR schedulers (LambdaLR to 0 => no update, ReduceLROnPlateau runs).
 4. Parameter groups with their own lr / fractional order; frozen
    parameters (grad None) get no state; invalid groups are rejected.
 5. Argument validation (ValueError / TypeError).
 6. Sparse gradients (nn.Embedding(sparse=True)) give the same update as
    dense gradients.
 7. Determinism, closures, float64 / bfloat16 parameters.
 8. Bernoulli masks of FCSGD_GL / FCAdam_GL: probability, generator=
    reproducibility; random depth of the activations covers its range.
 9. Inspection helpers (gl_coefficients, current_mix_values, current_alpha_values)
    and the loss feedback (lambda_t is pushed down when the loss gets worse).
10. FractalActivation module (= function, repr, whole-model torch.save),
    gradients at x = 0, torch.compile of a model with a fractal activation,
    automatic mixed precision (torch.autocast + torch.amp.GradScaler).
11. Memory stays flat over 20 consecutive models.
"""

import gc
import io
import os
import resource
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])

import class_b_fractional_optimizers_OHA_torch as B  # noqa: E402
import class_c_fractional_optimizers_SR_torch as C  # noqa: E402
import class_minimal_fractional_adaptive_optimizers_torch as M  # noqa: E402
import class_d_related_work_optimizers_torch as D  # noqa: E402
import fractal_activation_functions_torch as faf  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

OPTIMIZERS = [
    ("FSGD", lambda p: B.FSGD(p, lr=1e-2, momentum=0.9, vderiv=0.75)),
    ("FAdam", lambda p: B.FAdam(p, lr=1e-2, vderiv=1.25)),
    ("FAdagrad", lambda p: B.FAdagrad(p, lr=5e-2, vderiv=0.75)),
    ("FAdadelta", lambda p: B.FAdadelta(p, lr=1.0, vderiv=1.25)),
    ("FRMSprop", lambda p: B.FRMSprop(p, lr=1e-2, vderiv=0.75)),
    ("FAdamW", lambda p: B.FAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("MemoryFSGD", lambda p: C.MemoryFSGD(p, lr=1e-2, momentum=0.9, vderiv=0.75, history_size=6)),
    ("MemoryFRMSprop", lambda p: C.MemoryFRMSprop(p, lr=1e-2, vderiv=1.25, history_size=6)),
    ("MemoryFAdam", lambda p: C.MemoryFAdam(p, lr=1e-2, vderiv=1.5, history_size=6)),
    ("MemoryFAdadelta", lambda p: C.MemoryFAdadelta(p, lr=1.0, vderiv=0.75, history_size=6)),
    ("MemoryFAdagrad", lambda p: C.MemoryFAdagrad(p, lr=5e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamW", lambda p: C.MemoryFAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75, history_size=6)),
    ("MemoryFNadam", lambda p: C.MemoryFNadam(p, lr=1e-2, vderiv=0.75, history_size=6)),
    ("MemoryFAdamax", lambda p: C.MemoryFAdamax(p, lr=1e-2, vderiv=0.75, history_size=6)),
    ("AdaptiveMemoryFSGD", lambda p: M.AdaptiveMemoryFSGD(p, lr=1e-2, momentum=0.9, vderiv=1.25)),
    ("AdaptiveMemoryFRMSprop", lambda p: M.AdaptiveMemoryFRMSprop(p, lr=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdam", lambda p: M.AdaptiveMemoryFAdam(p, lr=1e-2, vderiv=1.25)),
    ("AdaptiveMemoryFAdadelta", lambda p: M.AdaptiveMemoryFAdadelta(p, lr=1.0, vderiv=0.75)),
    ("AdaptiveMemoryFAdagrad", lambda p: M.AdaptiveMemoryFAdagrad(p, lr=5e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamW", lambda p: M.AdaptiveMemoryFAdamW(p, lr=1e-2, weight_decay=0.004, vderiv=0.75)),
    ("AdaptiveMemoryFNadam", lambda p: M.AdaptiveMemoryFNadam(p, lr=1e-2, vderiv=0.75)),
    ("AdaptiveMemoryFAdamax", lambda p: M.AdaptiveMemoryFAdamax(p, lr=1e-2, vderiv=0.75)),
    ("AdaGL", lambda p: D.AdaGL(p, lr=1e-2, alpha=0.75, history_size=10)),
    ("FCSGD_GL", lambda p: D.FCSGD_GL(p, lr=1e-2, momentum=0.9, alpha=0.75, history_size=11)),
    ("FCAdam_GL", lambda p: D.FCAdam_GL(p, lr=1e-2, alpha=0.75, history_size=11)),
    ("AOFGD_SGD", lambda p: D.AOFGD_SGD(p, lr=1e-2, momentum=0.9, alpha_init=0.75, alpha_max=1.5)),
    ("AOFGD_Adam", lambda p: D.AOFGD_Adam(p, lr=1e-2, alpha_init=1.25, alpha_max=1.5)),
]

g = torch.Generator().manual_seed(0)
X = torch.rand(150, 4, generator=g)
Y = ((X[:, 0] * 2 + X[:, 2]).long()) % 3

failures = []


def check(cond, label):
    if not cond:
        failures.append(label)
    return bool(cond)


def build_model(act="tanh", seed=0, dropout=0.2):
    torch.manual_seed(seed)
    act_mod = (lambda: torch.nn.Tanh()) if act == "tanh" else (lambda: faf.FractalActivation(act))
    return torch.nn.Sequential(
        torch.nn.Linear(4, 32), act_mod(), torch.nn.BatchNorm1d(32, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(32, 16), act_mod(), torch.nn.BatchNorm1d(16, momentum=0.01, eps=1e-3),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(16, 3),
    )


def is_adaptive(opt):
    return hasattr(opt, "set_current_loss")


def train(model, opt, epochs, seed=1, start_epoch=0):
    lf = torch.nn.CrossEntropyLoss()
    losses = []
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        perm = torch.randperm(150, generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.0
        for i in range(0, 150, 16):
            idx = perm[i:i + 16]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lf(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            if is_adaptive(opt):
                opt.set_current_loss(loss)
            total += float(loss.detach()) * len(idx)
        losses.append(total / 150)
    return losses


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()] +
                     [b.detach().reshape(-1).double() for b in model.buffers()])


t0 = time.time()

# ---------------------------------------------------------------------------
# 1) Training, 2) checkpointing, 3) schedulers, 7) determinism / closure
# ---------------------------------------------------------------------------
print(f"{'optimizer':24s} {'tanh':>14s} {'fractal':>14s}  ckpt  sched  determ  closure  f64  bf16")
for name, make in OPTIMIZERS:
    row = {}
    try:
        # 1) training with tanh and with a fractal activation
        for act in ("tanh", "modified_weierstrass_function_tanh"):
            model = build_model(act)
            losses = train(model, make(model.parameters()), epochs=12)
            ok = all(map(lambda v: v == v and abs(v) < 1e6, losses)) and \
                sum(losses[-3:]) / 3 < losses[0]
            check(ok, f"{name}: loss not decreasing with {act}: {losses}")
            row[act] = f"{losses[0]:.3f}->{losses[-1]:.3f}"

        # 2) checkpoint round trip (bit-identical continuation)
        torch.manual_seed(5)
        model = build_model("tanh", dropout=0.2)
        opt = make(model.parameters())
        train(model, opt, epochs=3)
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "rng": torch.get_rng_state()}, buf)
        train(model, opt, epochs=3, start_epoch=3)
        ref = flat(model)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=True)
        model2 = build_model("tanh", seed=123)
        opt2 = make(model2.parameters())
        model2.load_state_dict(ckpt["model"])
        opt2.load_state_dict(ckpt["opt"])
        torch.set_rng_state(ckpt["rng"])
        train(model2, opt2, epochs=3, start_epoch=3)
        ok_ckpt = check(torch.equal(ref, flat(model2)), f"{name}: checkpoint continuation differs")

        # 3) schedulers: the scheduled lr is the lr that is used
        #    (LambdaLR(0.5) == half the lr from the start), LambdaLR(0) before the
        #    first step => no change, ReduceLROnPlateau runs.
        runs = []
        for use_scheduler in (True, False):
            model = build_model("tanh", dropout=0.0)
            opt = make(model.parameters())
            if use_scheduler:
                torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.5)
            else:
                for gr in opt.param_groups:
                    gr["lr"] = gr["lr"] * 0.5
            train(model, opt, epochs=1)
            runs.append(flat(model))
        ok_sched = torch.equal(runs[0], runs[1])
        model = build_model("tanh", dropout=0.0).eval()
        opt = make(model.parameters())
        torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.0)
        before = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(X[:32]), Y[:32]).backward()
        opt.step()
        ok_sched &= all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        for gr in opt.param_groups:
            gr["lr"] = gr["initial_lr"]
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.5)
        for v in (1.0, 2.0, 3.0):
            plateau.step(v)
        ok_sched &= all(abs(gr["lr"] - 0.25 * gr["initial_lr"]) < 1e-15 for gr in opt.param_groups)
        check(ok_sched, f"{name}: scheduler check failed")

        # 7) determinism and closure
        runs = []
        for _ in range(2):
            torch.manual_seed(9)
            model = build_model("tanh")
            train(model, make(model.parameters()), epochs=2)
            runs.append(flat(model))
        ok_det = check(torch.equal(runs[0], runs[1]), f"{name}: not deterministic")
        model = build_model("tanh", dropout=0.0)
        opt = make(model.parameters())

        def closure(opt=opt, model=model):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X), Y)
            loss.backward()
            return loss
        ret = opt.step(closure)
        ok_clo = check(isinstance(ret, torch.Tensor) and ret.ndim == 0, f"{name}: closure return")

        # float64 and bfloat16 parameters
        ok_dt = {}
        for dt in (torch.float64, torch.bfloat16):
            model = build_model("tanh", dropout=0.0).to(dt)
            opt = make(model.parameters())
            for _ in range(5):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(X.to(dt)), Y).backward()
                opt.step()
            ok_dt[dt] = all(torch.isfinite(p).all() and p.dtype == dt for p in model.parameters()) and \
                all(v.dtype == dt for st in opt.state.values() for v in st.values()
                    if isinstance(v, torch.Tensor) and v.is_floating_point())
            check(ok_dt[dt], f"{name}: {dt} parameters/state")
        print(f"{name:24s} {row['tanh']:>14s} {row['modified_weierstrass_function_tanh']:>14s}  "
              f"{'ok' if ok_ckpt else 'FAIL':4s}  {'ok' if ok_sched else 'FAIL':5s}  "
              f"{'ok' if ok_det else 'FAIL':6s}  {'ok' if ok_clo else 'FAIL':7s}  "
              f"{'ok' if ok_dt[torch.float64] else 'FAIL':3s}  {'ok' if ok_dt[torch.bfloat16] else 'FAIL'}")
    except Exception as e:  # pragma: no cover
        failures.append(f"{name}: {e!r}")
        print(f"{name:24s} FAILED: {e!r}"[:300])

# ---------------------------------------------------------------------------
# 4) parameter groups, frozen parameters, invalid groups
# ---------------------------------------------------------------------------
GROUP_KEYS = {"B": "vderiv", "C": "vderiv", "M": "vderiv", "D": None}
ok_groups = True
for name, make in OPTIMIZERS:
    model = build_model("tanh", dropout=0.0)
    first, second, frozen = model[0], model[4], model[8]
    for p in frozen.parameters():
        p.requires_grad_(False)
    opt = make(first.parameters())
    base = dict(opt.param_groups[0])
    extra = {"lr": 0.0}
    if "vderiv" in base:
        extra["vderiv"] = 0.9
    if "alpha" in base:
        extra["alpha"] = 0.9
    opt.add_param_group({"params": list(second.parameters()), **extra})
    before_second = [p.detach().clone() for p in second.parameters()]
    before_first = [p.detach().clone() for p in first.parameters()]
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(X), Y).backward()
        opt.step()
    ok = all(torch.equal(a, b) for a, b in zip(before_second, second.parameters()))
    ok &= not all(torch.equal(a, b) for a, b in zip(before_first, first.parameters()))
    ok &= all(p not in opt.state for p in frozen.parameters())
    ok &= opt.param_groups[1].get("vderiv", 0.9) == 0.9 and opt.param_groups[1].get("alpha", 0.9) == 0.9
    bad_key = "vderiv" if "vderiv" in base else ("alpha" if "alpha" in base else "alpha_init")
    try:
        opt.add_param_group({"params": list(frozen.parameters()), bad_key: 5.0})
        ok = False
    except ValueError:
        pass
    ok_groups &= check(ok, f"{name}: parameter groups")
print(f"\nparameter groups (own lr / order, lr=0 group frozen, grad None skipped, invalid group rejected): "
      f"{'ok' if ok_groups else 'FAIL'}")

# ---------------------------------------------------------------------------
# 5) argument validation
# ---------------------------------------------------------------------------
ps = [torch.nn.Parameter(torch.zeros(3))]
bad_args = [
    (B.FSGD, dict(vderiv=2.5), ValueError), (B.FAdam, dict(vderiv=0.0), ValueError),
    (B.FAdam, dict(beta_1=1.0), ValueError), (B.FSGD, dict(momentum=-0.1), ValueError),
    (C.MemoryFAdam, dict(history_size=0), ValueError), (C.MemoryFAdam, dict(vderiv=2.0), ValueError),
    (C.MemoryFAdam, dict(histroy_size=4), TypeError),
    (M.AdaptiveMemoryFAdam, dict(min_mix=0.5, max_mix=0.2), ValueError),
    (M.AdaptiveMemoryFAdam, dict(warmup_steps=-1), ValueError),
    (M.AdaptiveMemoryFAdam, dict(inital_mix=0.1), TypeError),
    (D.AdaGL, dict(alpha=2.0), ValueError), (D.FCSGD_GL, dict(perturbation_prob=1.5), ValueError),
    (D.AOFGD_SGD, dict(alpha_init=0.9, alpha_max=0.8), ValueError),
    (D.AOFGD_Adam, dict(alpha_max=2.0), ValueError),
]
ok_args = True
for cls, kw, exc in bad_args:
    try:
        cls(ps, **kw)
        ok_args &= check(False, f"{cls.__name__}({kw}) accepted")
    except exc:
        pass
    except Exception as e:  # pragma: no cover
        ok_args &= check(False, f"{cls.__name__}({kw}) raised {e!r}, expected {exc.__name__}")
ok_args &= check(B.FAdam(ps, learning_rate=0.123).param_groups[0]["lr"] == 0.123, "learning_rate alias")
print(f"argument validation ({len(bad_args)} invalid settings rejected, learning_rate alias): "
      f"{'ok' if ok_args else 'FAIL'}")

# ---------------------------------------------------------------------------
# 6) sparse gradients
# ---------------------------------------------------------------------------
ok_sparse = True
idx = torch.tensor([3, 1, 7, 0, 5])            # unique rows -> identical dense/sparse sums
target = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
for name, make in OPTIMIZERS:
    weights = []
    for sparse in (False, True):
        torch.manual_seed(0)
        emb = torch.nn.Embedding(10, 4, sparse=sparse)
        opt = make(emb.parameters())
        for _ in range(4):
            opt.zero_grad()
            ((emb(idx) - target) ** 2).mean().backward()
            opt.step()
        weights.append(emb.weight.detach().clone())
    ok_sparse &= check(torch.equal(weights[0], weights[1]), f"{name}: sparse != dense")
print(f"sparse gradients (nn.Embedding(sparse=True) == dense): {'ok' if ok_sparse else 'FAIL'}")

# ---------------------------------------------------------------------------
# 8) Bernoulli masks and random depth
# ---------------------------------------------------------------------------
ok_rand = True
for cls in (D.FCSGD_GL, D.FCAdam_GL):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = cls([p], lr=0.0, alpha=0.9, history_size=11, perturbation_prob=0.3)
    # mask frequency, read from the optimizer's own torch.rand calls
    hits, n = 0, 0
    orig = torch.rand
    for _ in range(400):
        p.grad = torch.ones(1)
        captured = []
        torch.rand = lambda *a, **k: captured.append(orig(*a, **k)) or captured[-1]
        try:
            opt.step()
        finally:
            torch.rand = orig
        hits += int((captured[0] < 0.3).sum())
        n += captured[0].numel()
    ok_rand &= check(0.25 < hits / n < 0.35, f"{cls.__name__}: mask frequency {hits / n:.3f}")
    results = []
    for _ in range(2):
        p = torch.nn.Parameter(torch.zeros(4))
        opt = cls([p], lr=0.01, alpha=0.9, history_size=11, perturbation_prob=0.5,
                  generator=torch.Generator().manual_seed(42))
        for t in range(20):
            p.grad = torch.full((4,), float(t % 3) - 1.0)
            opt.step()
        results.append(p.detach().clone())
    ok_rand &= check(torch.equal(results[0], results[1]), f"{cls.__name__}: generator reproducibility")
depths = {int(faf._sample_random_depth(10, 50)) for _ in range(3000)}
ok_rand &= check(depths == set(range(10, 51)), f"random depth range {min(depths)}..{max(depths)}")
torch.manual_seed(0)
a = faf.random_depth_modified_weierstrass_function_tanh(torch.linspace(-2, 2, 101))
torch.manual_seed(0)
b = faf.random_depth_modified_weierstrass_function_tanh(torch.linspace(-2, 2, 101))
ok_rand &= check(torch.equal(a, b), "random depth not reproducible with torch.manual_seed")
print(f"Bernoulli masks (frequency, generator=) and random depth (range 10..50, seed): "
      f"{'ok' if ok_rand else 'FAIL'}")

# ---------------------------------------------------------------------------
# 9) inspection helpers and loss feedback
# ---------------------------------------------------------------------------
ok_insp = True
opt = C.MemoryFAdam([torch.nn.Parameter(torch.zeros(2))], vderiv=0.75, history_size=4)
coeffs = opt.gl_coefficients
ok_insp &= check(len(coeffs) == 4 and coeffs[0] == 1.0 and abs(coeffs[1] + 0.75) < 1e-15,
                 f"gl_coefficients {coeffs}")
ok_insp &= check(abs(opt.gl_coefficient_sum() - sum(coeffs)) < 1e-15, "gl_coefficient_sum")


def mix_after(losses):
    p = torch.nn.Parameter(torch.ones(3))
    opt = M.AdaptiveMemoryFAdam([p], lr=1e-3, vderiv=0.75, initial_mix=0.2, max_mix=0.3,
                                loss_mix_decrease=0.05, loss_worsen_tolerance=0.0)
    for t, lv in enumerate(losses):
        p.grad = torch.tensor([1.0, 0.5, -0.25]) * (1 + 0.01 * t)
        opt.step()
        if lv is not None:
            opt.set_current_loss(lv)
    return float(opt.current_mix_values()[0])


mix_none = mix_after([None] * 20)
mix_worse = mix_after([1.0 + 0.1 * t for t in range(20)])
ok_insp &= check(mix_worse < mix_none, f"loss feedback: lambda {mix_worse} !< {mix_none}")
ok_insp &= check(len(M.AdaptiveMemoryFAdam([torch.nn.Parameter(torch.zeros(1))]).current_mix_values()) == 0,
                 "current_mix_values before the first step")
p = torch.nn.Parameter(torch.ones(3))
opt = D.AOFGD_Adam([p], alpha_init=1.25, alpha_max=1.5)
for t in range(5):
    p.grad = torch.ones(3) * (0.5 ** t)        # shrinking gradient -> order goes up
    opt.step()
alpha = float(opt.current_alpha_values()[0])
ok_insp &= check(1.25 < alpha <= 1.5, f"AOFGD alpha {alpha}")
print(f"inspection helpers and loss feedback (lambda {mix_none:.4f} -> {mix_worse:.4f} when the loss worsens): "
      f"{'ok' if ok_insp else 'FAIL'}")

# ---------------------------------------------------------------------------
# 10) FractalActivation module, x = 0 gradients, torch.compile
# ---------------------------------------------------------------------------
ok_act = True
x = torch.linspace(-3, 3, 601)
for name, fn in faf.ALL_FRACTAL_ACTIVATIONS.items():
    torch.manual_seed(0)
    ref = fn(x)
    torch.manual_seed(0)
    ok_act &= check(torch.equal(faf.FractalActivation(name)(x), ref), f"FractalActivation({name})")
    x0 = torch.tensor([0.0, 1e-30, -1e-30, 0.5], requires_grad=True)
    y0 = fn(x0)
    if y0.requires_grad:
        y0.sum().backward()
        ok_act &= check(torch.isfinite(x0.grad[[0, 3]]).all(), f"{name}: non-finite gradient at x=0")
mod = faf.FractalActivation("modified_weierstrass_function_tanh", n_terms=50)
ok_act &= check(torch.equal(mod(x), faf.modified_weierstrass_function_tanh(x, n_terms=50)), "kwargs")
ok_act &= check("n_terms=50" in repr(mod), "repr")
try:
    faf.get_fractal_activation("nope")
    ok_act &= check(False, "unknown activation accepted")
except ValueError:
    pass
model = build_model("decaying_cosine_function_tf", dropout=0.0).eval()
buf = io.BytesIO()
torch.save(model, buf)
buf.seek(0)
loaded = torch.load(buf, weights_only=False)
ok_act &= check(torch.equal(loaded(X), model(X)), "whole-model torch.save/load")
ok_act &= check(set(model.state_dict()) == set(build_model("tanh").state_dict()),
                "FractalActivation adds no state_dict entries")
print(f"FractalActivation module, kwargs, repr, save/load, finite gradients at x = 0: "
      f"{'ok' if ok_act else 'FAIL'}")

try:
    t_c = time.time()
    model = build_model("weierstrass_mandelbrot_function_xpsin", dropout=0.0).eval()
    compiled = torch.compile(model)
    xin = X[:32].clone().requires_grad_(True)
    out_e = model(xin)
    out_e.sum().backward()
    g_e = xin.grad.clone()
    xin.grad = None
    out_c = compiled(xin)
    out_c.sum().backward()
    ok_c = torch.allclose(out_e, out_c, rtol=1e-4, atol=1e-5) and \
        torch.allclose(g_e, xin.grad, rtol=1e-3, atol=1e-4)
    check(ok_c, "torch.compile result differs from eager")
    print(f"torch.compile (inductor) forward/backward == eager: {'ok' if ok_c else 'FAIL'} "
          f"({time.time() - t_c:.0f} s)")
except Exception as e:  # pragma: no cover
    failures.append(f"torch.compile: {e!r}"[:300])
    print(f"torch.compile FAILED: {e!r}"[:300])

# ---------------------------------------------------------------------------
# 10b) automatic mixed precision: torch.autocast + torch.amp.GradScaler
# ---------------------------------------------------------------------------
ok_amp = True
for name, make in OPTIMIZERS:
    model = build_model("decaying_cosine_function_tf", dropout=0.0)
    opt = make(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 8)
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(X), Y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    ok_amp &= check(all(torch.isfinite(p).all() and p.dtype == torch.float32 for p in model.parameters())
                    and len(opt.state) > 0, f"{name}: autocast/GradScaler")
print(f"automatic mixed precision (torch.autocast bfloat16 + torch.amp.GradScaler), all optimizers: "
      f"{'ok' if ok_amp else 'FAIL'}")

# ---------------------------------------------------------------------------
# 11) memory
# ---------------------------------------------------------------------------
rss = []
for i in range(20):
    model = build_model("modified_weierstrass_function_tanh", seed=i)
    opt = M.AdaptiveMemoryFAdam(model.parameters(), lr=1e-2, vderiv=0.75)
    train(model, opt, epochs=1)
    del model, opt
    gc.collect()
    rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
growth = rss[-1] - rss[4]
ok_mem = check(growth < 50, f"memory grew by {growth:.0f} MB over 15 models")
print(f"memory over 20 models: {rss[0]:.0f} MB -> {rss[-1]:.0f} MB (growth after warm-up {growth:.0f} MB): "
      f"{'ok' if ok_mem else 'FAIL'}")

print(f"\nfinished in {time.time() - t0:.0f} s")
if failures:
    print(f"{len(failures)} FAILURES")
    for f in failures:
        print("  ", f)
    sys.exit(1)
print(f"All {len(OPTIMIZERS)} optimizers and the activation helpers passed.")
