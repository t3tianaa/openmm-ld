"""Published ALF (Hayes, Vilseck, Brooks; github.com/RyanLeeHayes/ALF, MIT licence) on openmm-ld.

This is the bias-flattening algorithm used with BLaDE and CHARMM, not a re-derivation:

  * the bias form is ALF's bcxs2018: per block a linear bias b, and per ordered pair of blocks at a site
    a quadratic c, an end-point x and a skew s term (BLaDE types 6, 10 and 8);
  * each cycle pools the lambda samples of this and the 4 previous cycles, reweights them with WHAM
    (`src/wham.c`, the CPU build of ALF's own solver, vendored unchanged except the G_imp path), builds
    1D, transition and 2D free-energy profiles, and solves the regularised linear model for the bias
    change that makes them flat (`_get_free_energy5`, a line-by-line port of ALF's GetFreeEnergy5.py);
  * the implicit-constraint profiles G_imp (fnex = 5.5) are ALF's own tables.

Units and signs. ALF works in kcal/mol with the bias energy E = -b.l - sum c l l - ...; the engine works in
kJ/mol with U = +lambda_bias * l + k * (term). So engine coefficient = -KCAL * ALF coefficient.

Production estimate (`estimate_dg`): ALF's postprocess. For each block, G = b - kT ln P(lambda > 0.99),
relative to the site's first block, with bootstrap over independent runs (or over trajectory blocks).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import sysconfig

import numpy as np

KCAL = 4.184
KB_KCAL = 0.001987            # ALF's kT = 0.001987 * T in GetFreeEnergy5 / GetVariance
X_L0 = -5.56                  # end-point (x) term: (1 - exp(-5.56 l_i)) l_j
S_L0 = 0.017                  # skew (s) term: l_i l_j / (l_i + 0.017)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------------------------------
# WHAM library (built once, cached)
# ---------------------------------------------------------------------------------------------------------
def wham_library() -> str:
    """Path to the compiled WHAM shared library; compiled from src/wham.c on first use."""
    cache = os.environ.get("OPENMM_LD_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "openmm_ld"))
    os.makedirs(cache, exist_ok=True)
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib = os.path.join(cache, "libalfwham" + ext)
    src = os.path.join(HERE, "src", "wham.c")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            raise RuntimeError("published ALF needs a C++ compiler to build its WHAM solver (set CXX)")
        tmp = lib + f".{os.getpid()}"
        subprocess.run([cxx, "-O2", "-fPIC", "-shared", "-x", "c++", src, "-o", tmp], check=True,
                       stderr=subprocess.PIPE, cwd=os.path.join(HERE, "src"))
        os.replace(tmp, lib)
    return lib


def run_wham(directory, n_sims, temperature, ms=0, msprof=0):
    """ALF's RunWham in `directory` (holding nsubs, Energy/, Lambda/, G_imp/). Writes multisite/C.dat and
    V.dat. Runs in a child process: the C code keeps global allocations and writes to the cwd."""
    os.makedirs(os.path.join(directory, "multisite"), exist_ok=True)
    code = ("import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); f=lib.wham; "
            "f.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_int]; "
            "f(int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))")
    with open(os.path.join(directory, "output"), "w") as out, open(os.path.join(directory, "error"), "w") as err:
        rc = subprocess.call([sys.executable, "-c", code, wham_library(), str(n_sims), str(temperature),
                              str(ms), str(msprof)], cwd=directory, stdout=out, stderr=err)
    if rc != 0 or not os.path.exists(os.path.join(directory, "multisite", "C.dat")):
        raise RuntimeError(f"ALF WHAM failed in {directory} (exit {rc}); see its output and error files")


# ---------------------------------------------------------------------------------------------------------
# Bias bookkeeping
# ---------------------------------------------------------------------------------------------------------
class Biases:
    """ALF bias parameters (kcal/mol, ALF sign): b (nblocks,), c, x, s (nblocks, nblocks)."""

    def __init__(self, nblocks, b=None, c=None, x=None, s=None):
        z = np.zeros((nblocks, nblocks))
        self.b = np.zeros(nblocks) if b is None else np.asarray(b, float).reshape(-1)
        self.c = z.copy() if c is None else np.asarray(c, float).reshape(nblocks, nblocks)
        self.x = z.copy() if x is None else np.asarray(x, float).reshape(nblocks, nblocks)
        self.s = z.copy() if s is None else np.asarray(s, float).reshape(nblocks, nblocks)

    def copy(self):
        return Biases(len(self.b), self.b.copy(), self.c.copy(), self.x.copy(), self.s.copy())

    def __add__(self, other):
        return Biases(len(self.b), self.b + other.b, self.c + other.c, self.x + other.x, self.s + other.s)

    def energy(self, lam):
        """ALF's bias energy (kcal/mol) of lambda samples lam (n, nblocks), as in GetEnergy.py."""
        lam = np.atleast_2d(lam)
        e = lam @ -self.b
        e += np.sum((lam @ -self.c) * lam, axis=1)
        e += np.sum(((1 - np.exp(X_L0 * lam)) @ -self.x) * lam, axis=1)
        e += np.sum(((lam / (lam + S_L0)) @ -self.s) * lam, axis=1)
        return e

    def save(self, directory, suffix):
        np.savetxt(os.path.join(directory, f"b_{suffix}.dat"), self.b.reshape(1, -1), fmt=" %.6f")
        for name in "cxs":
            np.savetxt(os.path.join(directory, f"{name}_{suffix}.dat"), getattr(self, name), fmt=" %.6f")

    @classmethod
    def load(cls, directory, suffix):
        b = np.loadtxt(os.path.join(directory, f"b_{suffix}.dat"))
        n = b.size
        return cls(n, b, *(np.loadtxt(os.path.join(directory, f"{k}_{suffix}.dat")).reshape(n, n) for k in "cxs"))

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in "bcxs"}


def add_alf_biases(model):
    """Give the (finalized) model ALF's bias terms; call BEFORE scale_system. Tunable linear biases on every
    block, and per ordered intra-site pair a type-10 x term and a type-8 s term, per unordered pair a type-6
    c term. Returns the term list used by `apply_biases`."""
    model.bias_flattening = True
    terms = []
    for site in range(1, model.n_sites):
        js = model.blocks_in_site(site)
        for a in range(len(js)):
            for b in range(len(js)):
                if a == b:
                    continue
                i, j = js[a], js[b]
                if a < b:
                    model.add_variable_bias(i, j, type=6, l0=0.0, k=0.0, n=0)
                    terms.append(("c", i, j))
                model.add_variable_bias(i, j, type=10, l0=X_L0, k=0.0, n=0)
                terms.append(("x", i, j))
                model.add_variable_bias(i, j, type=8, l0=S_L0, k=0.0, n=0)
                terms.append(("s", i, j))
    model._alf_terms = terms
    return terms


def apply_biases(model, biases: Biases, context=None):
    """Write ALF biases into the model (and the live Context): engine = -KCAL * ALF."""
    terms = getattr(model, "_alf_terms", None)
    if terms is None:
        raise RuntimeError("call add_alf_biases(model) before scaling the System")
    for b in range(1, model.n_blocks):
        model.blocks[b].lambda_bias = -KCAL * float(biases.b[b - 1])
    for vb, (kind, i, j) in zip(model.variable_biases, terms):
        vb.k = -KCAL * float(getattr(biases, kind)[i - 1, j - 1])
    if context is not None:
        from ..builder import set_biases
        set_biases(context, model)


def site_sizes(model):
    return [len(model.blocks_in_site(s)) for s in range(1, model.n_sites)]


# ---------------------------------------------------------------------------------------------------------
# GetFreeEnergy5 (bcxs2018), ported line by line from ALF
# ---------------------------------------------------------------------------------------------------------
def _get_free_energy5(directory, nsubs, prev: Biases, ms=0):
    krest = 1
    cutb, cutc, cutx, cuts = 2, 8, 2, 1
    cutc2, cutx2, cuts2 = 2, 0.5, 0.5
    nblocks = sum(nsubs)
    b = np.zeros((1, nblocks)); c = np.zeros((nblocks, nblocks))
    x = np.zeros((nblocks, nblocks)); s = np.zeros((nblocks, nblocks))
    nparm = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]; n2 = n1 * (n1 - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                nparm += n1 + 5 * n2
            elif ms == 1:
                nparm += 5 * n3
            elif ms == 2:
                nparm += n3
    cutlist = np.zeros(nparm); reglist = np.zeros(nparm)
    n0 = 0; iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        n1 = nsubs[isite]; n2 = n1 * (n1 - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                cutlist[n0:n0 + n1] = cutb; n0 += n1
                cutlist[n0:n0 + n2] = cutc; n0 += n2
                cutlist[n0:n0 + 2 * n2] = cutx; n0 += 2 * n2
                cutlist[n0:n0 + 2 * n2] = cuts; n0 += 2 * n2
            elif ms == 1:
                cutlist[n0:n0 + n3] = cutc2; n0 += n3
                ind = n0
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        reglist[ind] = -prev.x[iblock + i, jblock + j]; ind += 1
                        reglist[ind] = -prev.x[jblock + j, iblock + i]; ind += 1
                cutlist[n0:n0 + 2 * n3] = cutx2; n0 += 2 * n3
                ind = n0
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        reglist[ind] = -prev.s[iblock + i, jblock + j]; ind += 1
                        reglist[ind] = -prev.s[jblock + j, iblock + i]; ind += 1
                cutlist[n0:n0 + 2 * n3] = cuts2; n0 += 2 * n3
            elif ms == 2:
                cutlist[n0:n0 + n3] = cutc2; n0 += n3
            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    C = np.loadtxt(os.path.join(directory, "multisite", "C.dat"))
    C = np.atleast_2d(C)
    for i in range(n0):
        C[i, i] += krest * cutlist[i] ** -2
    for i in range(C.shape[0]):
        if C[i, i] == 0:
            C[i, i] = 1
    V = np.atleast_1d(np.loadtxt(os.path.join(directory, "multisite", "V.dat")))
    for i in range(n0):
        V[i] += (krest * cutlist[i] ** -2) * reglist[i]
    coeff = np.linalg.solve(C, V)
    scaling = min(1.0, 1.5 / np.max(np.abs(coeff[0:n0] / cutlist)))
    coeff = coeff * scaling

    ind = 0; iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite == jsite:
                for i in range(nsubs[isite]):
                    b[0, iblock + i] = coeff[ind]; ind += 1
                for i in range(nsubs[isite]):
                    for j in range(i + 1, nsubs[isite]):
                        c[iblock + i, jblock + j] = coeff[ind]; ind += 1
                for i in range(nsubs[isite]):
                    for j in range(nsubs[isite]):
                        if i != j:
                            x[iblock + i, jblock + j] = coeff[ind]; ind += 1
                for i in range(nsubs[isite]):
                    for j in range(nsubs[isite]):
                        if i != j:
                            s[iblock + i, jblock + j] = coeff[ind]; ind += 1
            elif ms == 1:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]; ind += 1
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        x[iblock + i, jblock + j] = coeff[ind]; ind += 1
                        x[jblock + j, iblock + i] = coeff[ind]; ind += 1
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        s[iblock + i, jblock + j] = coeff[ind]; ind += 1
                        s[jblock + j, iblock + i] = coeff[ind]; ind += 1
            elif ms == 2:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]; ind += 1
            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite != jsite:
                for i in range(nsubs[isite]):
                    b[0, iblock + i] += c[iblock + i, jblock]
                    c[iblock + i, jblock:jblock + nsubs[jsite]] -= c[iblock + i, jblock]
                for j in range(nsubs[jsite]):
                    b[0, jblock + j] += c[iblock, jblock + j]
                    c[iblock:iblock + nsubs[isite], jblock + j] -= c[iblock, jblock + j]
            jblock += nsubs[jsite]
        iblock += nsubs[isite]
    iblock = 0
    for isite in range(len(nsubs)):
        b[0, iblock:iblock + nsubs[isite]] -= b[0, iblock]
        iblock += nsubs[isite]
    # ALF writes the change with %7.2f and sums rounded values; keep the same resolution
    change = Biases(nblocks, np.round(b, 2), np.round(c, 2), np.round(x, 2), np.round(s, 2))
    return change, float(scaling)


# ---------------------------------------------------------------------------------------------------------
# One ALF analysis step (GetEnergy + RunWham + GetFreeEnergy5 + SetVars)
# ---------------------------------------------------------------------------------------------------------
def analyse_cycle(workdir, cycle, nsubs, temperature, first=1, ms=0, msprof=0):
    """Analyse cycle `cycle` in workdir/analysis<cycle>, pooling cycles max(cycle-4, first)..cycle.
    Each analysis<i> must hold data/Lambda.dat (lambda samples) and the biases it ran with (b_prev.dat ...).
    Writes b_sum.dat etc. (the biases for the next cycle) and returns (new Biases, scaling)."""
    adir = os.path.join(workdir, f"analysis{cycle}")
    i0 = max(cycle - 4, first)
    lambdas, biases = [], []
    for i in range(i0, cycle + 1):
        d = os.path.join(workdir, f"analysis{i}")
        lambdas.append(np.atleast_2d(np.loadtxt(os.path.join(d, "data", "Lambda.dat"))))
        biases.append(Biases.load(d, "prev"))
    os.makedirs(os.path.join(adir, "Energy"), exist_ok=True)
    os.makedirs(os.path.join(adir, "Lambda"), exist_ok=True)
    n = len(lambdas)
    # GetEnergy: ESim<i> has column 0 = energy of trajectory i under the current (last) biases, then one
    # column per simulation's biases (energies relative to nothing; WHAM takes the differences)
    for i in range(n):
        cols = [biases[-1].energy(lambdas[i])] + [biases[j].energy(lambdas[i]) for j in range(n)]
        np.savetxt(os.path.join(adir, "Energy", f"ESim{i + 1}.dat"), np.column_stack(cols), fmt="%12.5f")
        np.savetxt(os.path.join(adir, "Lambda", f"Lambda{i + 1}.dat"), lambdas[i], fmt="%10.6f")
    with open(os.path.join(adir, "nsubs"), "w") as fh:
        fh.write(" " + " ".join(str(k) for k in nsubs) + "\n")
    gimp = os.path.join(adir, "G_imp")
    if not os.path.exists(gimp):
        os.symlink(os.path.join(HERE, "G_imp"), gimp)
    run_wham(adir, n, temperature, ms, msprof)
    prev = biases[-1]
    change, scaling = _get_free_energy5(adir, nsubs, prev, ms)
    change.save(adir, "change")
    new = prev + change
    new.save(adir, "sum")
    return new, scaling


# ---------------------------------------------------------------------------------------------------------
# Driver on LambdaDynamics
# ---------------------------------------------------------------------------------------------------------
class _LambdaRecorder:
    def __init__(self, every, start_step=0):
        self.every = int(every)
        self.start = start_step
        self.rows = []
        self.t_theta = []

    def report(self, dyn, step):
        if step > self.start:
            self.rows.append(dyn.get_lambdas()[1:])
            self.t_theta.append(dyn.theta_temperature())

    def close(self):
        pass


def flatten(dyn, workdir, *, cycles=50, steps=25000, equil_steps=None, sample_every=10, start_cycle=1,
            biases: Biases | None = None, ms=0, msprof=0, log=print, checkpoint=None):
    """Run ALF cycles start_cycle..cycles on a LambdaDynamics whose model went through add_alf_biases.

    steps        : MD steps per cycle (sampled); equil_steps (default steps // 4, as ALF) are run first
                   after each bias change and discarded.
    sample_every : steps between lambda samples (ALF's nsavl, 10 in its scripts).
    Restart: cycles whose analysis<i>/b_sum.dat exists are skipped (biases are read from the last one), as in
    ALF's runflat. checkpoint(dyn, cycle) is called after each cycle (e.g. to save the State).
    Returns the final Biases."""
    model = dyn.model
    nsubs = site_sizes(model)
    nblocks = sum(nsubs)
    temperature = dyn.temperature
    equil_steps = steps // 4 if equil_steps is None else equil_steps
    os.makedirs(workdir, exist_ok=True)
    if biases is None:
        biases = Biases(nblocks)
    # resume
    first = start_cycle
    cycle = start_cycle
    while os.path.exists(os.path.join(workdir, f"analysis{cycle}", "b_sum.dat")):
        biases = Biases.load(os.path.join(workdir, f"analysis{cycle}"), "sum")
        cycle += 1
    for i in range(cycle, cycles + 1):
        import time
        t = time.time()
        adir = os.path.join(workdir, f"analysis{i}")
        os.makedirs(os.path.join(adir, "data"), exist_ok=True)
        biases.save(adir, "prev")
        apply_biases(model, biases, dyn.context)
        dyn.invalidate_forces()
        rec = _LambdaRecorder(sample_every, start_step=equil_steps)
        dyn.run(equil_steps + steps, reporters=[rec])
        lam = np.array(rec.rows)
        np.savetxt(os.path.join(adir, "data", "Lambda.dat"), lam, fmt="%10.6f")
        biases, scaling = analyse_cycle(workdir, i, nsubs, temperature, first=first, ms=ms, msprof=msprof)
        p_end = (lam > 0.99).mean(axis=0)
        log(f"[ALF] cycle {i}: {time.time() - t:.0f} s, scaling {scaling:.3f}, mean lambda "
            f"{np.round(lam.mean(axis=0), 3).tolist()}, frac(l>0.99) {np.round(p_end, 3).tolist()}, "
            f"b {np.round(biases.b, 2).tolist()}, T_theta {np.mean(rec.t_theta):.0f} K (cycle mean)")
        if checkpoint is not None:
            checkpoint(dyn, i)
    apply_biases(model, biases, dyn.context)
    dyn.invalidate_forces()
    return biases


# ---------------------------------------------------------------------------------------------------------
# Production estimate (ALF postprocess / GetVariance, single site or several, nreps = 1)
# ---------------------------------------------------------------------------------------------------------
def endpoint_free_energies(lambda_runs, biases: Biases, nsubs, temperature, cutoff=0.99, n_boot=1000, seed=2401):
    """ALF's GetVariance for independent production runs at fixed biases (no replica exchange).

    lambda_runs : list of (n, nblocks) lambda arrays, one per independent run (or per trajectory chunk).
    Returns (names-free) dict with, per substituent combination (one block per site), the free energy
    (kcal/mol) relative to the first combination, its bootstrap error, and the endpoint counts."""
    kT = KB_KCAL * temperature
    nblocks = sum(nsubs)
    starts = np.cumsum([0] + list(nsubs[:-1]))
    combos = [()]
    for s in range(len(nsubs)):
        combos = [c + (starts[s] + k,) for c in combos for k in range(nsubs[s])]
    nf = len(lambda_runs)
    G = np.zeros((nf, len(combos)))
    counts = np.zeros((nf, len(combos)), dtype=int)
    for j, combo in enumerate(combos):
        L = np.zeros(nblocks); L[list(combo)] = 1
        e_end = float(biases.energy(L)[0])
        for i, lam in enumerate(lambda_runs):
            P = int(np.sum(np.all(lam[:, list(combo)] > cutoff, axis=1)))
            counts[i, j] = P
            G[i, j] = -e_end - kT * np.log(P) if P > 0 else np.inf
    def combine(g):
        gmin = np.min(g, axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            v = gmin - kT * np.log(np.mean(np.exp(-(g - gmin) / kT), axis=0))
        return v - v[0]
    value = combine(G)
    rng = np.random.default_rng(seed)
    boot = np.array([combine(G[rng.integers(0, nf, nf)]) for _ in range(n_boot)]) if nf > 1 else None
    error = np.std(boot, axis=0) if boot is not None else np.full(len(combos), np.nan)
    return {"combos": [list(map(int, c)) for c in combos], "dG_kcal": value.tolist(),
            "err_kcal": error.tolist(), "counts": counts.tolist()}
