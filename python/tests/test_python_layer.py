"""The Python layer (openmm_ld) against the NumPy oracle and against OpenMM-MSLD.

Without OpenMM-MSLD: energies and dU/dlambda against the oracle fixture, and a real ALF run. With it
(MSLD_ENGINE_PATH): the same plain System scaled by both, energies and dU/dlambda within 1e-10; ALF on a fixed set of
lambda trajectories, identical updates; the output writers, identical files; dynamics from one seed.
"""
import json
import math
import os
import sys

import numpy as np
import openmm as mm
import pytest

from openmm_ld import LambdaDynamics, MSLDModel, model_from_mapping, set_lambdas
from openmm_ld import alf as new_alf
from openmm_ld import io as new_io

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("MSLD_ENGINE_PATH")
needs_engine = pytest.mark.skipif(not ENGINE, reason="MSLD_ENGINE_PATH not set")
if ENGINE:
    sys.path.insert(0, ENGINE)
REFERENCE = mm.Platform.getPlatformByName("Reference")
BOX, CUTOFF = 3.2, 1.0
NB = mm.NonbondedForce

with open(os.path.join(HERE, "data", "oracle_toy.json")) as fh:
    SYSTEMS = json.load(fh)["systems"]
SPEC = SYSTEMS[0]["spec"]
N = len(SPEC["positions"])
NBFIX = {(0, 1): (0.27, 0.9), (1, 2): (0.36, 0.12)}


def plain_system(spec=SPEC, method=NB.NoCutoff, nbfix=False, improper=False):
    """The toy as an ordinary OpenMM System, before any lambda scaling."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    for _ in range(N):
        system.addParticle(12.0)
    nbf = NB()
    nbf.setNonbondedMethod(method)
    nbf.setCutoffDistance(CUTOFF)
    nbf.setEwaldErrorTolerance(1e-5)
    nbf.setUseDispersionCorrection(False)
    if method != NB.NoCutoff:
        nbf.setUseSwitchingFunction(True)
        nbf.setSwitchingDistance(0.85)
    for i in range(N):
        nbf.addParticle(spec["charge"][i], 1.0 if nbfix else spec["sigma"][i], 0.0 if nbfix else spec["epsilon"][i])
    pairs14 = {(min(i, j), max(i, j)): (qq, s, e) for i, j, qq, s, e in spec["pairs14"]}
    excluded = sorted({(min(i, j), max(i, j)) for i, j in spec["excluded"]})
    for i, j in excluded:
        nbf.addException(i, j, *pairs14.get((i, j), (0.0, 0.1, 0.0)))
    system.addForce(nbf)
    if nbfix:
        tp = [(0.30, 0.5), (0.32, 0.3), (0.28, 0.7)]
        a, b = [0.0] * 9, [0.0] * 9
        for t1 in range(3):
            for t2 in range(3):
                s, e = NBFIX.get((min(t1, t2), max(t1, t2)),
                                 (0.5 * (tp[t1][0] + tp[t2][0]), math.sqrt(tp[t1][1] * tp[t2][1])))
                a[t1 + 3 * t2], b[t1 + 3 * t2] = math.sqrt(4 * e * s ** 12), 4 * e * s ** 6
        lj = mm.CustomNonbondedForce("(a/r6)^2-b/r6; r6=r^6; a=acoef(type1, type2); b=bcoef(type1, type2)")
        lj.addTabulatedFunction("acoef", mm.Discrete2DFunction(3, 3, a))
        lj.addTabulatedFunction("bcoef", mm.Discrete2DFunction(3, 3, b))
        lj.addPerParticleParameter("type")
        for i in range(N):
            lj.addParticle([float(i % 3)])
        for i, j in excluded:
            lj.addExclusion(i, j)
        cn = mm.CustomNonbondedForce
        lj.setNonbondedMethod(cn.NoCutoff if method == NB.NoCutoff else cn.CutoffPeriodic)
        lj.setCutoffDistance(CUTOFF)
        system.addForce(lj)
    bonds = mm.HarmonicBondForce()
    for i, j, r0, k in spec["bonds"]:
        bonds.addBond(i, j, r0, k)
    system.addForce(bonds)
    angles = mm.HarmonicAngleForce()
    for i, j, l, t0, k in spec["angles"]:
        angles.addAngle(i, j, l, t0, k)
    system.addForce(angles)
    torsions = mm.PeriodicTorsionForce()
    for i, j, l, m, n, ph, k in spec["torsions"]:
        torsions.addTorsion(i, j, l, m, n, ph, k)
    system.addForce(torsions)
    if improper:
        f = mm.CustomTorsionForce("k*(theta-theta0)^2")
        f.addPerTorsionParameter("k")
        f.addPerTorsionParameter("theta0")
        for i, j, l, m, n, ph, k in spec["torsions"][::4]:
            f.addTorsion(i, j, l, m, [k, ph])
        system.addForce(f)
    return system


def fill_model(model, spec=SPEC, biases=True):
    for b in range(1, len(spec["block_site"])):
        model.add_block(site=spec["block_site"][b], atoms=[i for i in range(N) if spec["atom_block"][i] == b],
                        lambda_bias=spec["fixed_bias"][b] if biases else 0.0, theta_friction=0.0)
    if biases:
        for i, j, typ, l0, k in spec["variable_biases"]:
            model.add_variable_bias(i, j, type=typ, l0=l0, k=k, n=0)
        for atoms, _ in spec["restraints"]:
            model.add_atom_restraint(atoms)
        model.k_restraint = spec["restraints"][0][1]
    model.softcore_alpha, model.softcore_sigma_q = spec["alpha"], spec["sigma_q"]
    return model.finalize()


def positions(spec=SPEC, centred=False):
    x = np.asarray(spec["positions"])
    return x - x.mean(axis=0) + 0.5 * BOX if centred else x


def evaluate(system, x, lam, platform=REFERENCE):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(x)
    set_lambdas(ctx, lam)
    st = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    d = st.getEnergyParameterDerivatives()
    return (st.getPotentialEnergy()._value, np.array([d[f"lambda{b}"] for b in range(1, len(lam))]),
            np.array(st.getForces(asNumpy=True)._value))


@pytest.mark.parametrize("system", SYSTEMS, ids=[f"seed{s['seed']}" for s in SYSTEMS])
def test_against_the_numpy_oracle(system):
    spec = system["spec"]
    model = fill_model(MSLDModel(N, fnex=spec["fnex"]), spec)
    s = plain_system(spec)
    model.scale_system(s)
    for case in system["cases"]:
        e, d, _ = evaluate(s, positions(spec), case["lam"])
        assert e == pytest.approx(case["energy"], rel=5e-8)
        assert d == pytest.approx(case["dU_dlambda"][1:], rel=5e-8, abs=1e-6)


CONFIGS = {"NoCutoff": (NB.NoCutoff, False, False), "PME": (NB.PME, False, False),
           "RF": (NB.CutoffPeriodic, False, False), "NBFIX": (NB.PME, True, False),
           "impropers": (NB.NoCutoff, False, True)}


@needs_engine
@pytest.mark.parametrize("scale_bonded", [True, False], ids=["bonded-scaled", "bonded-unscaled"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_scaling_matches_openmm_msld(name, scale_bonded):
    from msld.model import MSLDModel as OldModel
    method, nbfix, improper = CONFIGS[name]
    results = []
    for cls, kwargs in ((MSLDModel, {}), (OldModel, {"use_plugin": True})):
        model = fill_model(cls(N, fnex=SPEC["fnex"]))
        model.scale_bond = model.scale_angle = scale_bonded
        s = plain_system(method=method, nbfix=nbfix, improper=improper)
        model.scale_system(s, **kwargs)
        results.append([evaluate(s, positions(centred=True), c["lam"]) for c in SYSTEMS[0]["cases"]])
    for (e0, d0, f0), (e1, d1, f1) in zip(*results):
        assert e0 == pytest.approx(e1, rel=1e-10, abs=1e-10)
        assert d0 == pytest.approx(d1, rel=1e-10, abs=1e-10)
        assert np.max(np.abs(f0 - f1)) <= 1e-10 * max(1.0, np.max(np.abs(f1)))


@needs_engine
def test_model_from_mapping_matches_openmm_msld():
    from msld.prep.mapping.core import HybridBlueprint
    data = {"n_atoms": N, "sites": [
        {"env_atoms": [0, 1, 2, 3, 4, 5], "blocks": {"A": [6, 7, 8, 9, 10], "B": [11, 12, 13, 14, 15], "C": [16, 17, 18]},
         "attachments": [], "align_xform": {}, "core_charges": {}, "buffer_atoms": [], "core_corresp": {}, "atom_names": {}},
        {"env_atoms": [0, 1, 2, 3, 4, 5], "blocks": {"D": [19, 20], "E": [21, 22, 23]},
         "attachments": [], "align_xform": {}, "core_charges": {}, "buffer_atoms": [], "core_corresp": {}, "atom_names": {}}]}
    new = model_from_mapping(data)
    old = HybridBlueprint.from_dict(data).to_model()
    assert new.atom_block == old.atom_block
    assert [b.site for b in new.blocks] == [b.site for b in old.blocks]
    theta = [0.0, 0.3, -0.2, 0.4, 0.1, -0.5]
    assert new.lambda_from_theta(theta) == old.lambda_from_theta(theta)


class ScriptedDynamics:
    """Stands in for LambdaDynamics: a fixed lambda trajectory instead of dynamics."""

    def __init__(self, model, lambdas, every):
        self.model, self.lambdas, self.every, self.kT = model, lambdas, every, 2.494
        self.context = self
        self.dt = 0.001
        self.cursor = 0
        self.parameters = {}

    def getParameters(self):
        return self.parameters

    def setParameter(self, name, value):
        self.parameters[name] = value

    def run(self, nsteps, reporters=()):
        for step in range(self.every, nsteps + 1, self.every):
            self.current = self.lambdas[self.cursor % len(self.lambdas)]
            self.cursor += 1
            for r in reporters:
                r.report(self, step)

    def get_lambdas(self):
        return self.current

    def invalidate_forces(self):
        pass

    def potential_energy(self):
        return -123.456789 + self.cursor

    def theta_temperature(self):
        return 300.0 + self.cursor


def _lambda_series(model, n=400):
    rng = np.random.default_rng(3)
    return [model.lambda_from_theta([0.0] + list(rng.uniform(-math.pi, math.pi, model.n_blocks - 1))) for _ in range(n)]


@needs_engine
def test_alf_updates_are_identical():
    from msld import alf as old_alf
    from msld.model import MSLDModel as OldModel
    reports = []
    for module, cls in ((new_alf, MSLDModel), (old_alf, OldModel)):
        model = fill_model(cls(N, fnex=SPEC["fnex"]), biases=False)
        module.add_flattening_biases(model)
        dyn = ScriptedDynamics(model, _lambda_series(model), every=100)
        report = module.flatten(dyn, n_iters=6, steps_per_iter=5000, sample_every=100, rate=0.7, couple_rate=0.3,
                                max_step=8.0, tol=0.0, verbose=False)
        reports.append((report, module.estimate_ddg(report, model), dyn.parameters))
    for key in ("bias_history", "coupling_history", "history"):
        assert reports[0][0][key] == reports[1][0][key]
    assert reports[0][1] == reports[1][1]
    assert reports[0][2] == reports[1][2]


@needs_engine
def test_metrics_files_are_identical(tmp_path):
    from msld import io as old_io
    from msld.model import MSLDModel as OldModel
    texts = []
    for module, cls in ((new_io, MSLDModel), (old_io, OldModel)):
        model = fill_model(cls(N, fnex=SPEC["fnex"]), biases=False)
        dyn = ScriptedDynamics(model, _lambda_series(model, 10), every=100)
        path = tmp_path / f"{module.__name__}.csv"
        writer = module.LambdaTrajectoryWriter(str(path), model, every=100)
        dyn.run(1000, reporters=[writer])
        writer.close()
        texts.append((path.read_text(), module.populations(str(path))))
    assert texts[0] == texts[1]


@needs_engine
def test_dynamics_match_openmm_msld_from_one_seed():
    from msld.integrator import LambdaDynamics as OldDynamics
    from msld.model import MSLDModel as OldModel
    new_model = fill_model(MSLDModel(N, fnex=SPEC["fnex"]))
    new_system = plain_system()
    new_model.scale_system(new_system)
    new = LambdaDynamics(new_system, new_model, positions(), atom_friction=0.0, platform=REFERENCE, seed=5)

    old_model = fill_model(OldModel(N, fnex=SPEC["fnex"]))
    old_system = plain_system()
    old_model.scale_system(old_system, use_plugin=True)
    ctx = mm.Context(old_system, mm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPositions(positions())
    old = OldDynamics(ctx, old_model, temperature=300.0, timestep=0.001, atom_friction=0.0, seed=5)

    theta = [0.0, 0.3, -0.2, 0.4, 0.1, -0.5]
    new.theta = theta
    new.project()
    old.theta = list(theta)
    old.project()
    assert np.array_equal(new.vx, old.vx)
    assert new.vtheta == pytest.approx(old.vtheta, rel=1e-14)
    assert new.potential_energy() == pytest.approx(old.potential_energy(), rel=1e-12)
    assert new.dUdtheta()[1:] == pytest.approx(old.dUdtheta()[1:], rel=1e-10)
    assert list(new.endpoint_energies().values()) == pytest.approx(list(old.endpoint_energies().values()), rel=1e-10)
    for _ in range(5):
        new.step(10)
        old.step(10)
        assert np.max(np.abs(new.x - old.x)) < 1e-9
        assert np.max(np.abs(np.array(new.theta) - np.array(old.theta))) < 1e-9
    assert new.atom_temperature() == pytest.approx(old.atom_temperature(), rel=1e-8)
    assert new.theta_temperature() == pytest.approx(old.theta_temperature(), rel=1e-8)


def test_alf_runs_on_real_dynamics_and_updates_the_context():
    model = fill_model(MSLDModel(N, fnex=SPEC["fnex"]), biases=False)
    new_alf.add_flattening_biases(model)
    system = plain_system()
    model.scale_system(system)
    dyn = LambdaDynamics(system, model, positions(), atom_friction=1.0, platform=REFERENCE, seed=1)
    report = new_alf.flatten(dyn, n_iters=2, steps_per_iter=500, sample_every=50, tol=0.0, verbose=False)
    assert report["iterations"] == 2
    assert dyn.context.getParameter("biascoef1") == pytest.approx(report["biases"][1])
    # the integrator sees the new coefficients: dU/dlambda inside it equals a fresh evaluation
    dyn.step(1)
    from_integrator = dyn.integrator.getGlobalVariableByName("dUdlambda1")
    fresh = dyn.dUdlambda(lam=dyn.get_lambdas())[0][1]
    assert from_integrator == pytest.approx(fresh, rel=1e-10)


def test_writers_on_real_dynamics(tmp_path):
    model = fill_model(MSLDModel(N, fnex=SPEC["fnex"]), biases=False)
    system = plain_system()
    model.scale_system(system)
    dyn = LambdaDynamics(system, model, positions(), platform=REFERENCE, seed=1)
    metrics = tmp_path / "m.csv"
    dyn.run(100, metrics=str(metrics), metrics_every=10)
    labels, columns = new_io.read_trajectory(str(metrics))
    assert len(columns["step"]) == 10
    assert len(labels) == model.n_blocks - 1
