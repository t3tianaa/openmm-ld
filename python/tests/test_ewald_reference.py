"""The plugin's PME at partial lambda against a plain Ewald sum (ewald_reference.py).

Unlike the Python-engine comparison, this reference shares no code or force construction with OpenMM.
The reference is first tested on its own: the NaCl Madelung constant, independence of the splitting
parameter alpha where the theory requires it, and convergence of its k-space sum. Direct space uses the
same alpha and the same cutoff on both sides, so what is left is the PME grid error, which must shrink
with the Ewald error tolerance.
"""
import json
import math
import os

import numpy as np
import openmm as mm
import pytest

import msldplugin
from ewald_reference import denergy_dlambda, energy, reciprocal

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 3.2
CUTOFF = 1.0
MADELUNG_NACL = 1.7475645946331822

_plugin = os.environ.get("MSLD_REFERENCE_PLUGIN")
if _plugin:
    msldplugin.loadReferencePlugin(_plugin)
MSLD = msldplugin.MSLDNonbondedForce
LAM = MSLD.getLambdaName


def _coulomb_constant():
    """OpenMM's 1/(4 pi eps0), from the energy of two unit charges 1 nm apart. The plugin uses this
    value; the spec rounds it to 138.935456."""
    system = mm.System()
    nb = mm.NonbondedForce()
    for _ in range(2):
        system.addParticle(1.0)
        nb.addParticle(1.0, 1.0, 0.0)
    system.addForce(nb)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [1, 0, 0]])
    return ctx.getState(getEnergy=True).getPotentialEnergy()._value


KE = _coulomb_constant()


def _toys():
    with open(os.path.join(HERE, "data", "oracle_toy.json")) as fh:
        systems = json.load(fh)["systems"]
    out = []
    for s in systems:
        spec = dict(s["spec"])
        x = np.asarray(spec["positions"])
        spec["positions"] = (x - x.mean(axis=0) + 0.5 * BOX).tolist()
        out.append((f"seed{s['seed']}{'-overlap' if s['overlap'] else ''}", spec, [c["lam"] for c in s["cases"]]))
    return out


def plugin_eval(spec, lam, tolerance):
    """Plugin energy and dU/dlambda with PME at an Ewald tolerance; also its alpha and grid."""
    n = len(spec["positions"])
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    f = MSLD()
    f.setNonbondedMethod(MSLD.PME)
    f.setCutoffDistance(CUTOFF)
    f.setEwaldErrorTolerance(tolerance)
    f.setSoftcoreAlpha(spec["alpha"])
    f.setSoftcoreSigmaQ(spec["sigma_q"])
    for b in range(1, len(spec["block_site"])):
        f.addBlock(spec["block_site"][b])
    for i in range(n):
        system.addParticle(1.0)
        f.addParticle(spec["charge"][i], spec["sigma"][i], spec["epsilon"][i], spec["atom_block"][i])
    pairs14 = {(min(i, j), max(i, j)): (qq, s, e) for i, j, qq, s, e in spec["pairs14"]}
    for i, j in spec["excluded"]:
        f.addException(i, j, *pairs14.get((min(i, j), max(i, j)), (0.0, 1.0, 0.0)))
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.asarray(spec["positions"]))
    for b in range(1, len(lam)):
        ctx.setParameter(LAM(b), lam[b])
    st = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    d = st.getEnergyParameterDerivatives()
    alpha, nx, ny, nz = f.getPMEParametersInContext(ctx)
    return (st.getPotentialEnergy()._value, np.array([0.0] + [d[LAM(b)] for b in range(1, len(lam))]),
            alpha, (nx, ny, nz))


# ---- the reference on its own -------------------------------------------------------------------

def test_reference_reproduces_the_nacl_madelung_constant():
    """8 ions of a NaCl cell, nearest-neighbour distance d: U = -(N/2) M KE q^2 / d."""
    box, d = 1.0, 0.5
    x, q = [], []
    for i, j, k in np.ndindex(2, 2, 2):
        x.append([i * d, j * d, k * d])
        q.append(1.0 if (i + j + k) % 2 == 0 else -1.0)
    spec = dict(positions=x, charge=q, sigma=[1.0] * 8, epsilon=[0.0] * 8, atom_block=[0] * 8, block_site=[0],
                alpha=0.0, sigma_q=0.3, pairs14=[], excluded=[])
    U = energy(spec, [1.0], 6.0, box, KE, None, lj=False, images=2)
    assert U == pytest.approx(-4.0 * MADELUNG_NACL * KE / d, rel=1e-12)


def test_reference_does_not_depend_on_alpha_when_it_must_not():
    """Where direct and reciprocal space scale alike (no soft-core, no within-block pairs), the
    converged sum at partial lambda cannot depend on the Ewald splitting."""
    _, spec, lams = _toys()[0]
    spec = dict(spec, alpha=0.0, atom_block=[0] * 20 + [1, 2, 3, 4], block_site=[0, 1, 1, 2, 2], pairs14=[],
                excluded=[])
    lam = [1.0, 0.3, 0.7, 0.6, 0.4]
    values = [energy(spec, lam, a, BOX, KE, None, lj=False, images=1) for a in (2.5, 3.5, 5.0)]
    assert values[1] == pytest.approx(values[0], rel=1e-11)
    assert values[2] == pytest.approx(values[0], rel=1e-11)


def test_reference_k_sum_is_converged():
    _, spec, lams = _toys()[0]
    x = np.asarray(spec["positions"])
    Q = np.array([lams[0][b] * q for b, q in zip(spec["atom_block"], spec["charge"])])
    a = reciprocal(x, Q, 3.3, BOX, KE, k_tol=1e-18)
    b = reciprocal(x, Q, 3.3, BOX, KE, k_tol=1e-24)
    assert abs(a - b) < 1e-11


# ---- the plugin against the reference -------------------------------------------------------------

# Ewald tolerance -> largest accepted |plugin - Ewald|, in kJ/mol, about 3x the worst value measured over
# both toys (24 atoms, box 3.2 nm, 4 lambda vectors each): 2.9e-3 and 1.3e-2 at 5e-4; 3.8e-5 and 1.4e-4 at
# 1e-5; 2.9e-6 and 2.5e-6 at 1e-6.
ENERGY_LIMIT = {5e-4: 4e-2, 1e-5: 5e-4, 1e-6: 1e-5}
DERIV_LIMIT = {5e-4: 4e-2, 1e-5: 3e-4, 1e-6: 2e-4}


@pytest.mark.parametrize("name,spec,lams", _toys(), ids=[t[0] for t in _toys()])
@pytest.mark.parametrize("tolerance", sorted(ENERGY_LIMIT, reverse=True))
def test_plugin_pme_matches_ewald(name, spec, lams, tolerance):
    for lam in lams:
        e_p, d_p, alpha, _ = plugin_eval(spec, lam, tolerance)
        e_r = float(np.real(energy(spec, lam, alpha, BOX, KE, CUTOFF)))
        d_r = denergy_dlambda(spec, lam, alpha, BOX, KE, CUTOFF)
        assert abs(e_p - e_r) < ENERGY_LIMIT[tolerance], (lam, e_p, e_r)
        assert np.max(np.abs(d_p[1:] - d_r[1:])) < DERIV_LIMIT[tolerance], (lam, d_p, d_r)


def test_pme_error_shrinks_with_the_tolerance():
    _, spec, lams = _toys()[0]
    errors = []
    for tolerance in (5e-4, 1e-5, 1e-6, 1e-7):
        e_p, _, alpha, _ = plugin_eval(spec, lams[1], tolerance)
        errors.append(abs(e_p - float(np.real(energy(spec, lams[1], alpha, BOX, KE, CUTOFF)))))
    assert errors[3] < errors[1] < errors[0]


def test_partial_lambda_energy_depends_on_alpha_with_within_block_pairs():
    """A property of the spec, not a defect of either code: a within-block pair carries lambda in direct
    space and lambda^2 in reciprocal space, so the split between them (alpha) changes the total at partial
    lambda. With no within-block pairs the sum does not depend on alpha (test above). Anything compared at
    partial lambda under PME has to use the same alpha."""
    _, spec, _ = _toys()[0]
    spec = dict(spec, alpha=0.0)
    lam = [1.0, 0.3, 0.1, 0.6, 0.5, 0.5]
    values = [float(np.real(energy(spec, lam, a, BOX, KE, CUTOFF, lj=False))) for a in (2.5, 4.0)]
    assert abs(values[1] - values[0]) > 0.1
