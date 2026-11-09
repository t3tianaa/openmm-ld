"""Adaptive Landscape Flattening (ALF), ported from OpenMM-MSLD msld/alf.py.

Bare lambda-dynamics collapses to (or gets stuck near) one variant at a site, so the mean lambda is
not a real population. ALF learns lambda-biases that FLATTEN the landscape so every variant is
visited; the accumulated biases then approximate the free-energy differences (a Phase-3 estimator,
not built, turns them into DeltaDeltaG). See vault FEAT-012 ALF flattening heuristic.

  Phase 1 -- FIXED linear biases (flatten the ENDPOINTS). Each iteration, from each block's mean
             occupancy p_b, nudge its bias so under-sampled variants get more favorable:
                 b_b += rate * kT * ln(p_b / p_target),   p_target = 1 / blocks-at-site.
  Phase 2 -- pairwise COUPLING biases (flatten the midpoint BARRIER, so theta can shuttle between
             endpoints). The dwelling at the midpoint is measured by the overlap o_ij = <l_i*l_j>.
             The flat target is o_ref -- what <l_i*l_j> would be with NO physical energy (theta free
             -> uniform, so lambda follows the bare softmax); anything above that is excess midpoint
             trapping. Raise the coupling to push theta back to the endpoints:
                 k_ij += couple_rate * kT * (o_ij - o_ref) / o_ref.
             The coupling is 0 at the endpoints (l_i*l_j = 0 there), so it ONLY lifts the barrier --
             it never distorts which endpoint wins (that is Phase 1's job). This un-sticks the
             barrier-trapped 'lambda ~ 0.5' state.
  Phase 3 -- FREE ENERGY (`estimate_ddg`). At convergence the learned bias cancels the free energy
             (b_b ~= -G_b), so DeltaDeltaG(ref -> b) = G_b - G_ref = -(b_b - b_ref). The bias
             oscillates around its fixed point, so average the bias difference over the tail of the
             flatten history (SEM = its spread). The converged bias is the robust readout; endpoint-
             population reweighting is distorted by the softmax (deferred, see vault FEAT-012).

Usage:
      from openmm_ld.alf import add_flattening_biases, flatten, estimate_ddg
      model = bundle.to_model()
      add_flattening_biases(model)          # BEFORE scaling: tunable fixed biases + pair couplings
      model.scale_system(system)
      dyn = LambdaDynamics(system, model, positions, ...)
      rep = flatten(dyn, n_iters=30, steps_per_iter=20_000)   # learn biases (dyn.model carries them)
      ddg = estimate_ddg(rep, model)                          # {block: (DDG kJ/mol, SEM)} vs reference
"""
from __future__ import annotations

import math

import numpy as np

from .builder import set_biases

_REF_CACHE: dict = {}


def add_flattening_biases(model, *, couple: bool = True, coupling_type: int = 6):
    """Prepare a (finalized) model for ALF -- call BEFORE `scale_system`, so the tunable bias globals
    are built into the System.

      * fixed linear biases become tunable (`model.bias_flattening = True`) -- Phase 1.
      * couple=True adds a pairwise coupling bias (BLaDE type 6, `k*li*lj`, k=0) between every pair of
        blocks at each site -- the Phase-2 barrier-flattening knobs.
    """
    model.bias_flattening = True
    if couple:
        for s in range(1, model.n_sites):
            js = model.blocks_in_site(s)
            for a in range(len(js)):
                for b in range(a + 1, len(js)):
                    model.add_variable_bias(js[a], js[b], type=coupling_type, l0=0.0, k=0.0, n=0)
    return model


def ref_overlap(fnex: float, n_blocks: int, samples: int = 400000) -> float:
    """<l_i*l_j> for a pair at an n-block site under the BARE softmax (no physical energy -> theta
    free -> uniform). This is the flat-landscape target the Phase-2 coupling drives the observed
    overlap down to. Depends only on (fnex, n_blocks); cached."""
    key = (round(float(fnex), 6), int(n_blocks))
    if key not in _REF_CACHE:
        rng = np.random.default_rng(0)
        th = rng.uniform(-math.pi, math.pi, size=(samples, n_blocks))
        w = np.exp(fnex * np.sin(th))
        lam = w / w.sum(axis=1, keepdims=True)
        _REF_CACHE[key] = float(np.mean(lam[:, 0] * lam[:, 1]))   # all pairs equal by symmetry
    return _REF_CACHE[key]


class LambdaHistogram:
    """A `LambdaDynamics.run` reporter (`.every/.report/.close`) that accumulates, per iteration,
    each block's mean lambda (occupancy) and, per coupled pair, the mean product <l_i*l_j> (the
    midpoint 'overlap' Phase 2 flattens)."""

    def __init__(self, model, every: int = 100):
        self.model = model
        self.every = int(every)
        self.blocks = list(range(1, model.n_blocks))
        self.pairs = [(vb.i, vb.j) for vb in model.variable_biases]
        self._sum = {b: 0.0 for b in self.blocks}
        self._psum = {p: 0.0 for p in self.pairs}
        self._n = 0

    def report(self, dyn, step: int) -> None:
        lam = dyn.get_lambdas()
        for b in self.blocks:
            self._sum[b] += lam[b]
        for (i, j) in self.pairs:
            self._psum[(i, j)] += lam[i] * lam[j]
        self._n += 1

    def close(self) -> None:
        pass

    def occupancy(self) -> dict:
        n = max(self._n, 1)
        return {b: self._sum[b] / n for b in self.blocks}

    def overlap(self) -> dict:
        n = max(self._n, 1)
        return {p: self._psum[p] / n for p in self.pairs}


def flatten(dyn, *, couple: bool = True, n_iters: int = 20, steps_per_iter: int = 20000,
            sample_every: int = 100, rate: float = 1.0, couple_rate: float | None = None,
            max_step: float | None = None, floor: float = 1e-3, tol: float = 0.05,
            verbose: bool = True) -> dict:
    """Learn lambda-biases that flatten the landscape (Phases 1 + 2). Mutates `dyn.model` in place.

    dyn            : LambdaDynamics on a System scaled after `add_flattening_biases(model)`.
    couple         : also tune the pairwise coupling biases (Phase 2). Needs the couplings to exist.
    steps_per_iter : lambda-dynamics steps sampled per update.
    rate           : learning rate for the fixed biases; couple_rate for the couplings (default=rate).
    max_step       : optional cap on |db| / |dk| per iteration (kJ/mol).
    tol            : convergence -- stop when the worst |p_b - p_target| over all sites < tol.

    Returns {iterations, converged, history, bias_history, couplings (final vb.k), biases}.
    """
    model = dyn.model
    kT = dyn.kT
    couple_rate = rate if couple_rate is None else couple_rate
    history: list = []
    bias_history: list = []
    coupling_history: list = []
    converged = False
    for it in range(n_iters):
        hist = LambdaHistogram(model, every=sample_every)
        dyn.run(steps_per_iter, reporters=[hist])
        occ = hist.occupancy()
        overlap = hist.overlap()
        history.append(dict(occ))

        # Phase 1: fixed linear biases -> equalize endpoint populations
        worst = 0.0
        for s in range(1, model.n_sites):
            js = model.blocks_in_site(s)
            p_target = 1.0 / len(js)
            for b in js:
                worst = max(worst, abs(occ[b] - p_target))
                if model.blocks[b].fixed:
                    continue
                p = max(occ[b], floor)
                delta = rate * kT * math.log(p / p_target)
                if max_step is not None:
                    delta = max(-max_step, min(max_step, delta))
                model.blocks[b].lambda_bias += delta

        # Phase 2: pairwise couplings -> lift the midpoint barrier toward the bare-softmax reference
        if couple:
            for vb in model.variable_biases:
                n = model.blocks_per_site[model.blocks[vb.i].site]
                o_ref = ref_overlap(model.fnex, n)
                dk = couple_rate * kT * (overlap[(vb.i, vb.j)] - o_ref) / o_ref
                if max_step is not None:
                    dk = max(-max_step, min(max_step, dk))
                vb.k += dk

        set_biases(dyn.context, model)      # push updated coefficients into the live Context
        dyn.invalidate_forces()             # forces depend on the biases -> recompute next step
        bias_history.append({b: model.blocks[b].lambda_bias for b in range(1, model.n_blocks)})
        coupling_history.append([vb.k for vb in model.variable_biases])

        if verbose:
            occ_str = " ".join(f"{occ[b]:.2f}" for b in range(1, model.n_blocks))
            ov = " ".join(f"{overlap[p]:.2f}" for p in hist.pairs) if hist.pairs else "-"
            print(f"[ALF] iter {it + 1:2d}: occ={occ_str}  overlap={ov}  worst|dp|={worst:.3f}")
        if worst < tol:
            converged = True
            break

    return dict(iterations=len(history), converged=converged, history=history,
                bias_history=bias_history, coupling_history=coupling_history,
                couplings=[vb.k for vb in model.variable_biases],
                biases={b: model.blocks[b].lambda_bias for b in range(1, model.n_blocks)})


# ---------------------------------------------------------------------
# Phase 3 -- free-energy estimate (from the converged biases)
# ---------------------------------------------------------------------
class EndpointPopulation:
    """A `run` reporter counting, per site, how often each block is the COMMITTED endpoint (its
    lambda is the site's max AND above `cutoff`). A diagnostic ("which variant wins, how often") --
    NOT a free-energy estimator (endpoint counts are distorted by the softmax; use `estimate_ddg`)."""

    def __init__(self, model, every: int = 100, cutoff: float = 0.8):
        self.model = model
        self.every = int(every)
        self.cutoff = float(cutoff)
        self.blocks = list(range(1, model.n_blocks))
        self._on = {b: 0 for b in self.blocks}
        self._committed = {s: 0 for s in range(1, model.n_sites)}

    def report(self, dyn, step: int) -> None:
        lam = dyn.get_lambdas()
        for s in range(1, self.model.n_sites):
            js = self.model.blocks_in_site(s)
            bmax = max(js, key=lambda b: lam[b])
            if lam[bmax] > self.cutoff:
                self._on[bmax] += 1
                self._committed[s] += 1

    def close(self) -> None:
        pass

    def populations(self) -> dict:
        out = {}
        for s in range(1, self.model.n_sites):
            js = self.model.blocks_in_site(s)
            c = max(self._committed[s], 1)
            for b in js:
                out[b] = self._on[b] / c
        return out


def estimate_ddg(report, model, *, tail: int | None = None) -> dict:
    """DeltaDeltaG per variant from a converged `flatten` run, relative to each site's reference
    (first) variant, as {block: (mean_kJ_per_mol, sem_kJ_per_mol)}.

    ALF learns a bias b_b that flattens the landscape, so at convergence b_b ~= -G_b (the bias cancels
    the free energy). Hence DeltaDeltaG(ref -> b) = G_b - G_ref = -(b_b - b_ref). The bias oscillates
    around its fixed point, so we AVERAGE the bias difference over the tail of `report["bias_history"]`
    and take the SEM across those iterations as the statistical error. Positive DDG = that variant is
    LESS favorable than the reference.

    Why this and not endpoint-population reweighting: the softmax makes endpoint counts a nonlinear,
    biased function of the free energy (the same reference subtlety Phase 2 handles), so a naive
    -kT ln(p_b) - b_b is distorted away from the flat point. The converged bias is the robust readout;
    a proper reweighting/MBAR refinement is deferred (see vault FEAT-012)."""
    bh = report.get("bias_history") or []
    if not bh:
        raise ValueError("estimate_ddg needs a flatten() report with a non-empty bias_history")
    t = tail if tail is not None else max(len(bh) // 2, 1)
    tail_bh = bh[-t:]
    out = {}
    for s in range(1, model.n_sites):
        js = model.blocks_in_site(s)
        ref = js[0]
        for b in js:
            vals = np.array([-(it[b] - it[ref]) for it in tail_bh])   # DDG(ref->b) per iteration
            sem = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[b] = (float(vals.mean()), sem)
    return out
