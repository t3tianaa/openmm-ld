"""Partial-lambda check of PME, cutoffs/reaction field and NBFIX against the Python engine.

The NumPy oracle (test_oracle_toy.py) only covers NoCutoff. Here the plugin is compared with the
OpenMM-MSLD Python engine (msld/forces.py + LambdaDynamics, PR #6 branch) on

  * the oracle toy (seed 1) in a 3.2 nm periodic box, cutoff 1.0 nm, switching from 0.85 nm:
    NoCutoff, CutoffNonPeriodic, CutoffPeriodic (reaction field) and PME;
  * the same toy with CHARMM-style tabulated LJ (acoef/bcoef, two NBFIX type pairs), NoCutoff and PME.

Both sides get identical bonded terms (stock custom forces), no biases and no restraints, so any
difference is nonbonded. The engine's dU/dlambda for its NonbondedForce part is a finite
difference (exact: that part is quadratic in lambda).

Reaction field: the engine's within-block correction (lambda - lambda^2) KE q q g(r) uses
g = erfc(alpha r)/r for CutoffPeriodic and g = 1/r for CutoffNonPeriodic, instead of the reaction
field form 1/r + k_rf r^2 - c_rf. That is an engine bug. For those methods the test checks that
plugin - engine equals exactly minus that error, summed over the within-block 1-5+ pairs.

Skipped unless MSLD_ENGINE_PATH points at an OpenMM-MSLD checkout (the directory with msld/).
"""
import math
import os
import sys

import numpy as np
import openmm as mm
import pytest

import msldplugin
from test_oracle_toy import _cases, _lam, build_system

ENGINE = os.environ.get("MSLD_ENGINE_PATH")
pytestmark = pytest.mark.skipif(not ENGINE, reason="MSLD_ENGINE_PATH not set")
if ENGINE:
    sys.path.insert(0, ENGINE)

BOX = 3.2
CUTOFF = 1.0
SWITCH = 0.85
RF_EPS = 78.3
EWALD_TOL = 1e-5
KE_ENGINE = 138.935456          # msld/forces.py _KE
REL_TOL = 1e-7
NBFIX_TYPES = 3
NBFIX = {(0, 1): (0.27, 0.9), (1, 2): (0.36, 0.12)}      # (type1, type2) -> (sigma, epsilon)

MSLD = msldplugin.MSLDNonbondedForce
NB = mm.NonbondedForce
METHODS = {"NoCutoff": (MSLD.NoCutoff, NB.NoCutoff),
           "CutoffNonPeriodic": (MSLD.CutoffNonPeriodic, NB.CutoffNonPeriodic),
           "CutoffPeriodic": (MSLD.CutoffPeriodic, NB.CutoffPeriodic),
           "PME": (MSLD.PME, NB.PME)}


def _toy():
    params = list(_cases())
    spec = dict(params[0].values[0])
    lams = [p.values[1]["lam"] for p in params if p.values[0] is params[0].values[0]]
    x = np.asarray(spec["positions"])
    spec["positions"] = (x - x.mean(axis=0) + 0.5 * BOX).tolist()      # centre in the box
    spec["fixed_bias"] = [0.0] * len(spec["fixed_bias"])
    spec["variable_biases"] = []
    spec["restraints"] = []
    return spec, lams


def _type_params(spec):
    """Per LJ type (sigma, epsilon): the mean over the atoms of that type."""
    n = len(spec["positions"])
    out = []
    for t in range(NBFIX_TYPES):
        atoms = [i for i in range(n) if i % NBFIX_TYPES == t]
        out.append((float(np.mean([spec["sigma"][i] for i in atoms])), float(np.mean([spec["epsilon"][i] for i in atoms]))))
    return out


def _pair_type_params(tp, t1, t2):
    key = (min(t1, t2), max(t1, t2))
    if key in NBFIX:
        return NBFIX[key]
    return 0.5 * (tp[t1][0] + tp[t2][0]), math.sqrt(tp[t1][1] * tp[t2][1])


# ---- plugin side ------------------------------------------------------------------------------

def plugin_system(spec, method, nbfix):
    tp = _type_params(spec) if nbfix else None

    def configure(nb):
        nb.setNonbondedMethod(METHODS[method][0])
        nb.setCutoffDistance(CUTOFF)
        nb.setUseSwitchingFunction(method != "NoCutoff")
        nb.setSwitchingDistance(SWITCH)
        nb.setReactionFieldDielectric(RF_EPS)
        nb.setEwaldErrorTolerance(EWALD_TOL)
        if nbfix:
            for i in range(nb.getNumParticles()):
                q, _, _, block = nb.getParticleParameters(i)
                t = i % NBFIX_TYPES
                nb.setParticleParameters(i, q, tp[t][0], tp[t][1], block)
                nb.setParticleLJType(i, t)
            for (t1, t2), (s, e) in NBFIX.items():
                nb.addNBFixPair(t1, t2, s, e)

    system = build_system(spec, configure)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    return system


# ---- engine side ------------------------------------------------------------------------------

def engine_system(spec, method, nbfix):
    from msld.builder import scale_msld_system
    from msld.model import MSLDModel

    n = len(spec["positions"])
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    for _ in range(n):
        system.addParticle(12.0)
    nbf = mm.NonbondedForce()
    nbf.setNonbondedMethod(METHODS[method][1])
    nbf.setCutoffDistance(CUTOFF)
    nbf.setUseSwitchingFunction(method != "NoCutoff")
    nbf.setSwitchingDistance(SWITCH)
    nbf.setReactionFieldDielectric(RF_EPS)
    nbf.setEwaldErrorTolerance(EWALD_TOL)
    nbf.setUseDispersionCorrection(False)
    for i in range(n):
        if nbfix:
            nbf.addParticle(spec["charge"][i], 1.0, 0.0)
        else:
            nbf.addParticle(spec["charge"][i], spec["sigma"][i], spec["epsilon"][i])
    pairs14 = {(min(i, j), max(i, j)): (qq, s, e) for i, j, qq, s, e in spec["pairs14"]}
    excluded = sorted({(min(i, j), max(i, j)) for i, j in spec["excluded"]})
    for i, j in excluded:
        if (i, j) in pairs14:
            nbf.addException(i, j, *pairs14[(i, j)])
        else:
            nbf.addException(i, j, 0.0, 0.1, 0.0)
    system.addForce(nbf)

    if nbfix:   # ParmEd-style tabulated LJ: U = (a/r^6)^2 - b/r^6, a^2 = 4 eps sigma^12, b = 4 eps sigma^6
        tp = _type_params(spec)
        acoef, bcoef = [0.0] * NBFIX_TYPES ** 2, [0.0] * NBFIX_TYPES ** 2
        for t1 in range(NBFIX_TYPES):
            for t2 in range(NBFIX_TYPES):
                s, e = _pair_type_params(tp, t1, t2)
                acoef[t1 + NBFIX_TYPES * t2] = math.sqrt(4 * e * s ** 12)
                bcoef[t1 + NBFIX_TYPES * t2] = 4 * e * s ** 6
        lj = mm.CustomNonbondedForce("(a/r6)^2-b/r6; r6=r^6; a=acoef(type1, type2); b=bcoef(type1, type2)")
        lj.addTabulatedFunction("acoef", mm.Discrete2DFunction(NBFIX_TYPES, NBFIX_TYPES, acoef))
        lj.addTabulatedFunction("bcoef", mm.Discrete2DFunction(NBFIX_TYPES, NBFIX_TYPES, bcoef))
        lj.addPerParticleParameter("type")
        for i in range(n):
            lj.addParticle([float(i % NBFIX_TYPES)])
        for i, j in excluded:
            lj.addExclusion(i, j)
        cn = mm.CustomNonbondedForce
        lj.setNonbondedMethod({"NoCutoff": cn.NoCutoff, "CutoffNonPeriodic": cn.CutoffNonPeriodic}.get(method, cn.CutoffPeriodic))
        lj.setCutoffDistance(CUTOFF)
        lj.setUseSwitchingFunction(method != "NoCutoff")
        lj.setSwitchingDistance(SWITCH)
        system.addForce(lj)

    # identical bonded terms to the plugin side: take them from the plugin-built System
    reference = build_system(spec)
    for k in range(reference.getNumForces()):
        f = reference.getForce(k)
        if not MSLD.isinstance(f):
            system.addForce(mm.XmlSerializer.clone(f))

    model = MSLDModel(n, fnex=spec["fnex"])
    model.softcore_alpha = spec["alpha"]
    model.softcore_sigma_q = spec["sigma_q"]
    for b in range(1, len(spec["block_site"])):
        atoms = [i for i in range(n) if spec["atom_block"][i] == b]
        model.add_block(site=spec["block_site"][b], atoms=atoms, lambda_bias=0.0)
    model.finalize()
    # bonded terms are already lambda-scaled custom forces; turn off the engine's own bonded scaling
    model.scale_bond = model.scale_angle = model.scale_torsion = False
    scale_msld_system(system, model)
    return system, model


# ---- evaluation -------------------------------------------------------------------------------

def plugin_eval(system, spec, lam):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(spec["positions"]))
    for b in range(1, len(lam)):
        ctx.setParameter(_lam(b), lam[b])
    st = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    d = st.getEnergyParameterDerivatives()
    return st.getPotentialEnergy()._value, np.array([0.0] + [d[_lam(b)] for b in range(1, len(lam))])


def engine_eval(system, model, spec, lam):
    from msld.builder import set_lambdas
    from msld.integrator import LambdaDynamics
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(spec["positions"]))
    dyn = LambdaDynamics(ctx, model, temperature=300.0, seed=0)
    set_lambdas(ctx, lam)
    d, _ = dyn.dUdlambda(lam=list(lam))
    return dyn.potential_energy(), np.array(d)


def engine_rf_error(spec, method, lam):
    """Energy and dU/dlambda error of the engine's within-block RF correction (see module doc)."""
    x = np.asarray(spec["positions"])
    q = spec["charge"]
    block = spec["atom_block"]
    excluded = {(min(i, j), max(i, j)) for i, j in spec["excluded"]}
    krf = (RF_EPS - 1) / ((2 * RF_EPS + 1) * CUTOFF ** 3)
    crf = 3 * RF_EPS / ((2 * RF_EPS + 1) * CUTOFF)
    alpha = math.sqrt(-math.log(2 * EWALD_TOL)) / CUTOFF       # what the engine's _ewald_alpha uses
    err_e, err_d = 0.0, np.zeros(len(lam))
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            b = block[i]
            if b == 0 or block[j] != b or (i, j) in excluded:
                continue
            d = x[j] - x[i]
            if method == "CutoffPeriodic":
                d -= BOX * np.round(d / BOX)
            r = float(np.linalg.norm(d))
            if r >= CUTOFF:
                continue
            g_true = 1 / r + krf * r * r - crf
            g_engine = math.erfc(alpha * r) / r if method == "CutoffPeriodic" else 1 / r
            w = KE_ENGINE * q[i] * q[j] * (g_engine - g_true)
            err_e += (lam[b] - lam[b] ** 2) * w
            err_d[b] += (1 - 2 * lam[b]) * w
    return err_e, err_d


@pytest.fixture(scope="module")
def toy():
    return _toy()


@pytest.mark.parametrize("nbfix", [False, True], ids=["LB", "NBFIX"])
@pytest.mark.parametrize("method", list(METHODS))
def test_plugin_matches_engine(toy, method, nbfix):
    if nbfix and method not in ("NoCutoff", "PME"):
        pytest.skip("NBFIX is checked with NoCutoff and PME")
    spec, lams = toy
    psys = plugin_system(spec, method, nbfix)
    esys, model = engine_system(spec, method, nbfix)
    for lam in lams:
        e_p, d_p = plugin_eval(psys, spec, lam)
        e_e, d_e = engine_eval(esys, model, spec, lam)
        if method in ("CutoffNonPeriodic", "CutoffPeriodic"):
            err_e, err_d = engine_rf_error(spec, method, lam)
            e_e -= err_e
            d_e = d_e - err_d
        scale = max(1.0, abs(e_e))
        assert abs(e_p - e_e) <= REL_TOL * scale, (method, lam, e_p, e_e)
        dscale = max(1.0, float(np.max(np.abs(d_e))))
        assert float(np.max(np.abs(d_p[1:] - d_e[1:]))) <= REL_TOL * dscale, (method, lam, d_p, d_e)


@pytest.mark.parametrize("method", ["CutoffNonPeriodic", "CutoffPeriodic"])
def test_engine_rf_bug_is_real(toy, method):
    """Without the correction above, plugin and engine disagree for reaction field."""
    spec, lams = toy
    lam = lams[0]
    e_p, _ = plugin_eval(plugin_system(spec, method, False), spec, lam)
    esys, model = engine_system(spec, method, False)
    e_e, _ = engine_eval(esys, model, spec, lam)
    err_e, _ = engine_rf_error(spec, method, lam)
    assert abs(err_e) > 1e-3
    assert abs((e_p - e_e) + err_e) <= REL_TOL * max(1.0, abs(e_e))
