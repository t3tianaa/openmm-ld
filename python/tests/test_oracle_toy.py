"""MSLDNonbondedForce + stock custom forces vs the NumPy partial-lambda oracle (S1).

Rebuilds every toy system in data/oracle_toy.json (copied from OpenMM-MSLD PR #6,
tests/data/oracle_toy.json, commit 70f79de) as an OpenMM System:

  nonbonded   MSLDNonbondedForce (soft-core, NoCutoff, 1-2/1-3 exclusions, 1-4 exceptions)
  bonded      CustomBond/Angle/TorsionForce per lambda signature, energy lambda1*lambda3*U
              (the product of the distinct non-environment block lambdas, as msld/forces.py)
  biases      one-particle CustomCompoundBondForce whose energy depends only on the lambdas
  restraints  0.5 k sum_i |x_i - centroid|^2 = (k / 2n) sum_{i<j} |x_i - x_j|^2 (CustomBondForce)

and checks the energy and dU/dlambda_b against the oracle for every case. The relative tolerance is
5e-8 because the oracle uses KE = 138.935456 and OpenMM uses 1/(4 pi eps0) = 138.9354576...
"""
import json
import os

import numpy as np
import openmm as mm
import pytest

import msldplugin

HERE = os.path.dirname(os.path.abspath(__file__))
REL_TOL = 5e-8

_plugin = os.environ.get("MSLD_REFERENCE_PLUGIN")
if _plugin:
    msldplugin.loadReferencePlugin(_plugin)


def _lam(b):
    return msldplugin.MSLDNonbondedForce.getLambdaName(b)


def _signature(atom_block, atoms):
    return tuple(sorted({atom_block[a] for a in atoms if atom_block[a] != 0}))


def _prefix(sig):
    return "*".join(_lam(b) for b in sig) + "*" if sig else ""


def _register(force, blocks):
    for b in blocks:
        force.addGlobalParameter(_lam(b), 1.0)
        force.addEnergyParameterDerivative(_lam(b))


def build_system(spec):
    n = len(spec["positions"])
    atom_block = spec["atom_block"]
    block_site = spec["block_site"]
    num_blocks = len(block_site)
    system = mm.System()
    for _ in range(n):
        system.addParticle(1.0)

    # nonbonded
    nb = msldplugin.MSLDNonbondedForce()
    nb.setNonbondedMethod(msldplugin.MSLDNonbondedForce.NoCutoff)
    nb.setSoftcoreAlpha(spec["alpha"])
    nb.setSoftcoreSigmaQ(spec["sigma_q"])
    for b in range(1, num_blocks):
        assert nb.addBlock(block_site[b]) == b
    for i in range(n):
        nb.addParticle(spec["charge"][i], spec["sigma"][i], spec["epsilon"][i], atom_block[i])
    pairs14 = {(min(i, j), max(i, j)): (qq, s, e) for i, j, qq, s, e in spec["pairs14"]}
    for i, j in spec["excluded"]:
        key = (min(i, j), max(i, j))
        if key in pairs14:
            nb.addException(i, j, *pairs14[key])
        else:
            nb.addException(i, j, 0.0, 1.0, 0.0)
    assert set(pairs14) <= {(min(i, j), max(i, j)) for i, j in spec["excluded"]}
    system.addForce(nb)

    # bonded terms grouped by lambda signature
    def grouped(terms, natoms):
        groups = {}
        for t in terms:
            groups.setdefault(_signature(atom_block, t[:natoms]), []).append(t)
        return groups

    for sig, terms in grouped(spec["bonds"], 2).items():
        f = mm.CustomBondForce(_prefix(sig) + "0.5*k*(r-r0)^2")
        f.addPerBondParameter("r0")
        f.addPerBondParameter("k")
        _register(f, sig)
        for i, j, r0, k in terms:
            f.addBond(i, j, [r0, k])
        system.addForce(f)
    for sig, terms in grouped(spec["angles"], 3).items():
        f = mm.CustomAngleForce(_prefix(sig) + "0.5*k*(theta-theta0)^2")
        f.addPerAngleParameter("theta0")
        f.addPerAngleParameter("k")
        _register(f, sig)
        for i, j, l, t0, k in terms:
            f.addAngle(i, j, l, [t0, k])
        system.addForce(f)
    for sig, terms in grouped(spec["torsions"], 4).items():
        f = mm.CustomTorsionForce(_prefix(sig) + "k*(1+cos(per*theta-phase))")
        f.addPerTorsionParameter("per")
        f.addPerTorsionParameter("phase")
        f.addPerTorsionParameter("k")
        _register(f, sig)
        for i, j, l, m, per, phase, k in terms:
            f.addTorsion(i, j, l, m, [per, phase, k])
        system.addForce(f)

    # biases: energy of the lambdas only, on a one-particle CustomCompoundBondForce (as msld/forces.py)
    terms = []
    for b, c in enumerate(spec["fixed_bias"]):
        if b and c:
            terms.append(f"({c!r})*{_lam(b)}")
    for i, j, typ, l0, k in spec["variable_biases"]:
        li, lj = _lam(i), _lam(j)
        if typ == 6:
            terms.append(f"({k!r})*{li}*{lj}")
        elif typ == 8:
            terms.append(f"({k!r})*{li}*{lj}/({li}+({l0!r}))")
        elif typ == 10:
            terms.append(f"({k!r})*{lj}*(1-exp(({l0!r})*{li}))")
        else:
            raise ValueError(typ)
    if terms:
        f = mm.CustomCompoundBondForce(1, "+".join(terms))
        _register(f, range(1, num_blocks))
        f.addBond([0], [])
        system.addForce(f)

    # restraints (not lambda-scaled)
    for atoms, k in spec["restraints"]:
        f = mm.CustomBondForce(f"{k / (2 * len(atoms))!r}*r^2")
        for a in range(len(atoms)):
            for b in range(a + 1, len(atoms)):
                f.addBond(atoms[a], atoms[b], [])
        system.addForce(f)
    return system


def _cases():
    with open(os.path.join(HERE, "data", "oracle_toy.json")) as fh:
        data = json.load(fh)
    for s in data["systems"]:
        for c, case in enumerate(s["cases"]):
            yield pytest.param(s["spec"], case, id=f"seed{s['seed']}-{'overlap' if s['overlap'] else 'normal'}-case{c}")


@pytest.fixture(scope="module")
def platform():
    return mm.Platform.getPlatformByName("Reference")


@pytest.mark.parametrize("spec,case", list(_cases()))
def test_oracle_case(spec, case, platform):
    system = build_system(spec)
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    context.setPositions(np.asarray(spec["positions"]))
    lam = case["lam"]
    for b in range(1, len(lam)):
        context.setParameter(_lam(b), lam[b])
    state = context.getState(getEnergy=True, getParameterDerivatives=True)
    energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
    derivs = state.getEnergyParameterDerivatives()

    assert energy == pytest.approx(case["energy"], rel=REL_TOL, abs=1e-6)
    for b in range(1, len(lam)):
        assert derivs[_lam(b)] == pytest.approx(case["dU_dlambda"][b], rel=REL_TOL, abs=1e-6), _lam(b)


def test_cpu_matches_reference():
    """The same kernel serves the CPU platform."""
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if "CPU" not in names:
        pytest.skip("CPU platform not available")
    spec, case = next(iter(_cases())).values
    results = []
    for name in ("Reference", "CPU"):
        context = mm.Context(build_system(spec), mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(name))
        context.setPositions(np.asarray(spec["positions"]))
        for b in range(1, len(case["lam"])):
            context.setParameter(_lam(b), case["lam"][b])
        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        results.append((state.getPotentialEnergy()._value, dict(state.getEnergyParameterDerivatives())))
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-6)
    for name, value in results[0][1].items():
        assert results[1][1][name] == pytest.approx(value, rel=1e-6, abs=1e-6)
