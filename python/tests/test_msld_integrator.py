"""MSLDIntegrator (openmm_ld/integrator.py) on the oracle toy.

Against LambdaDynamics of OpenMM-MSLD with friction 0 (the deterministic parts, step by step; needs
MSLD_ENGINE_PATH), energy conservation with second-order convergence in dt, temperatures in NVT, theta
surviving getState/setState, a barostat, and a replica-exchange sampler.
"""
import json
import os
import sys

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

import msldplugin
from openmm_ld import Block, MSLDIntegrator
from test_oracle_toy import build_system

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("MSLD_ENGINE_PATH")
needs_engine = pytest.mark.skipif(not ENGINE, reason="MSLD_ENGINE_PATH not set")
if ENGINE:
    sys.path.insert(0, ENGINE)

REFERENCE = mm.Platform.getPlatformByName("Reference")
THETA0 = [0.3, -0.2, 0.4, 0.1, -0.5]
with open(os.path.join(HERE, "data", "oracle_toy.json")) as fh:
    SPEC = json.load(fh)["systems"][0]["spec"]
N = len(SPEC["positions"])
X0 = np.asarray(SPEC["positions"])


def toy_blocks(friction):
    return [Block(site=SPEC["block_site"][b], theta_friction=friction) for b in range(1, len(SPEC["block_site"]))]


def toy_system(constraints=False):
    system = build_system(SPEC)
    for f in system.getForces():
        f.setForceGroup(1)         # everything analytic, so LambdaDynamics has nothing to finite-difference
    if constraints:
        for i, j, r0, _ in SPEC["bonds"][::3]:
            system.addConstraint(i, j, r0)
    return system


def total_energy(context, integrator):
    st = context.getState(getEnergy=True)
    return st.getPotentialEnergy()._value + st.getKineticEnergy()._value + integrator.getThetaKineticEnergy()


def start(integrator, system, velocities, theta_velocities, tolerance=None):
    ctx = mm.Context(system, integrator, REFERENCE)
    ctx.setPositions(X0)
    if system.getNumConstraints():
        ctx.applyConstraints(tolerance or 1e-6)
    ctx.setVelocities(velocities)
    integrator.setTheta(THETA0)
    integrator.setThetaVelocities(theta_velocities)
    return ctx


def seeded_velocities(seed=0):
    rng = np.random.default_rng(seed)
    kT = 0.00831446261815324 * 300.0
    return rng.standard_normal((N, 3)) * np.sqrt(kT / 1.0), rng.standard_normal(5) * np.sqrt(kT / 0.05)


@needs_engine
@pytest.mark.parametrize("constraints", [False, True], ids=["free", "constraints"])
def test_matches_lambda_dynamics_without_friction(constraints):
    from msld.integrator import LambdaDynamics
    from msld.model import MSLDModel
    tolerance = 1e-10
    model = MSLDModel(N, fnex=SPEC["fnex"])
    for b in range(1, len(SPEC["block_site"])):
        model.add_block(site=SPEC["block_site"][b], atoms=[i for i in range(N) if SPEC["atom_block"][i] == b],
                        theta_friction=0.0)
    model.finalize()
    v0, vt0 = seeded_velocities()

    integrator = MSLDIntegrator(300.0, 0.0, 0.001, toy_blocks(0.0), constraint_tolerance=tolerance)
    ctx = start(integrator, toy_system(constraints), v0, vt0, tolerance)
    ctx.applyVelocityConstraints(tolerance)
    x_start = ctx.getState(getPositions=True).getPositions(asNumpy=True)._value
    v_start = ctx.getState(getVelocities=True).getVelocities(asNumpy=True)._value

    c1 = mm.Context(toy_system(constraints), mm.VerletIntegrator(0.001), REFERENCE)
    c1.setPositions(x_start)
    dyn = LambdaDynamics(c1, model, temperature=300.0, timestep=0.001, atom_friction=0.0, seed=0)
    assert dyn._uses_plugin and not dyn._has_fd
    dyn.constraint_tol = tolerance
    dyn.theta = [0.0] + THETA0
    dyn.vtheta = [0.0] + list(vt0)
    dyn.vx = np.array(v_start)
    dyn.invalidate_forces()
    dyn.project()

    limit = 2e-5 if constraints else 1e-10
    for _ in range(100):
        dyn.step(1)
        integrator.step(1)
        st = ctx.getState(getPositions=True, getVelocities=True)
        assert np.max(np.abs(st.getPositions(asNumpy=True)._value - dyn.x)) < limit
        assert np.max(np.abs(st.getVelocities(asNumpy=True)._value - dyn.vx)) < 100 * limit
        assert np.max(np.abs(np.array(integrator.getTheta()) - np.array(dyn.theta[1:]))) < limit
        assert np.max(np.abs(np.array(integrator.getThetaVelocities()) - np.array(dyn.vtheta[1:]))) < 100 * limit


def energy_spread(dt):
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 0.0, dt, toy_blocks(0.0))
    ctx = start(integrator, toy_system(), v0, vt0)
    energies = []
    for _ in range(int(round(0.1 / dt / 5))):
        integrator.step(5)
        energies.append(total_energy(ctx, integrator))
    return float(np.std(energies))


def test_energy_conservation_is_second_order_in_dt():
    spreads = [energy_spread(dt) for dt in (0.001, 0.0005, 0.00025)]
    assert spreads[0] < 3.0                       # out of about 930 kJ/mol
    assert spreads[1] < spreads[0] / 2.5
    assert spreads[2] < spreads[1] / 2.5


def test_one_force_evaluation_per_step():
    count = [0]

    def counter(state):
        count[0] += 1
        return 0.0 * unit.kilojoule_per_mole, np.zeros((N, 3)) * unit.kilojoule_per_mole / unit.nanometer

    system = toy_system()
    system.addForce(mm.PythonForce(counter))
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, toy_blocks(1.0))
    context = start(integrator, system, v0, vt0)
    integrator.step(5)
    count[0] = 0
    integrator.step(100)
    assert count[0] == 100
    assert context.getParameter("lambda1") == pytest.approx(integrator.getLambdas()[0], abs=1e-14)


def _theta_velocity_after(change):
    """vtheta after one step, in a run where `change` touches the Context just before that step."""
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 0.0, 0.001, toy_blocks(0.0), watch_parameters=["biascheck"])
    system = toy_system()
    bias = mm.CustomCompoundBondForce(1, "biascheck*lambda1")
    bias.addGlobalParameter("biascheck", 0.0)
    bias.addGlobalParameter("lambda1", 1.0)
    bias.addEnergyParameterDerivative("lambda1")
    bias.addBond([0], [])
    bias.setForceGroup(1)
    system.addForce(bias)
    ctx = start(integrator, system, v0, vt0)
    integrator.step(10)
    change(ctx, integrator)
    state = ctx.getState(getPositions=True, getVelocities=True, getParameters=True, getIntegratorParameters=True)
    integrator.step(1)
    after = integrator.getThetaVelocities()
    # the same state, forced to recompute dU/dlambda: the reference answer
    ctx.setState(state)
    integrator.invalidate()
    integrator.step(1)
    return after, integrator.getThetaVelocities()


@pytest.mark.parametrize("name,change", [
    ("positions", lambda ctx, i: ctx.setPositions(
        ctx.getState(getPositions=True).getPositions(asNumpy=True)._value + 1e-3)),
    ("watched parameter", lambda ctx, i: ctx.setParameter("biascheck", 50.0)),
    ("theta", lambda ctx, i: i.setTheta(THETA0)),
], ids=lambda v: v if isinstance(v, str) else "")
def test_outside_changes_refresh_the_cached_derivative(name, change):
    after, reference = _theta_velocity_after(change)
    assert after == pytest.approx(reference, abs=1e-12)


def test_theta_survives_get_and_set_state():
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, toy_blocks(1.0))
    ctx = start(integrator, toy_system(), v0, vt0)
    integrator.step(50)
    state = ctx.getState(getPositions=True, getVelocities=True, getParameters=True, getIntegratorParameters=True)
    theta, vtheta = integrator.getTheta(), integrator.getThetaVelocities()
    lambda1 = ctx.getParameter("lambda1")
    integrator.step(50)
    assert integrator.getTheta() != theta
    ctx.setState(state)
    assert integrator.getTheta() == theta
    assert integrator.getThetaVelocities() == vtheta
    assert ctx.getParameter("lambda1") == lambda1


def test_lambdas_follow_theta_and_the_sites_sum_to_one():
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, toy_blocks(1.0))
    ctx = start(integrator, toy_system(), v0, vt0)
    integrator.step(20)
    lam = integrator.getLambdas()
    for b in range(1, 6):
        assert ctx.getParameter(f"lambda{b}") == pytest.approx(lam[b - 1], abs=1e-14)
    assert sum(lam[:3]) == pytest.approx(1.0)
    assert sum(lam[3:]) == pytest.approx(1.0)


def test_fixed_block_keeps_its_theta():
    blocks = toy_blocks(1.0)
    blocks[0] = blocks[0]._replace(fixed=True)
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, blocks)
    ctx = start(integrator, toy_system(), v0, vt0)
    integrator.step(20)
    assert ctx.getParameter("lambda2") != 0.0
    assert integrator.getTheta()[0] == THETA0[0]
    assert integrator.getTheta()[1] != THETA0[1]


def test_temperature_interface():
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, toy_blocks(1.0))
    assert integrator.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(300.0)
    integrator.setThetaVelocities([1.0] * 5)
    integrator.setTemperature(310.0 * unit.kelvin)
    assert integrator.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(310.0)
    assert integrator.getThetaVelocities()[0] == pytest.approx(np.sqrt(310.0 / 300.0))
    integrator.setTemperature(300.0, rescale_theta_velocities=False)
    assert integrator.getThetaVelocities()[0] == pytest.approx(np.sqrt(310.0 / 300.0))


def test_nvt_temperatures():
    """Short version of docs/msld_integrator.md's long run: within 10 % after 200 ps on the toy."""
    v0, vt0 = seeded_velocities()
    integrator = MSLDIntegrator(300.0, 5.0, 0.001, toy_blocks(5.0))
    integrator.setRandomNumberSeed(1)
    ctx = start(integrator, toy_system(), v0, vt0)
    integrator.step(20000)
    theta_t, atom_t = [], []
    for _ in range(4000):
        integrator.step(50)
        theta_t.append(integrator.getThetaTemperature())
        atom_t.append(ctx.getState(getEnergy=True).getKineticEnergy()._value)
    atom_t = np.array(atom_t) * 2.0 / (3 * N * 0.00831446261815324)
    assert np.mean(theta_t) == pytest.approx(300.0, rel=0.1)
    assert np.mean(atom_t) == pytest.approx(300.0, rel=0.1)


def test_barostat_changes_the_box():
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    rng = np.random.default_rng(0)
    f = msldplugin.MSLDNonbondedForce()
    f.setNonbondedMethod(f.PME)
    f.setCutoffDistance(1.0)
    f.addBlock(1)
    f.addBlock(1)
    positions = []
    for i, j, k in np.ndindex(5, 5, 5):
        system.addParticle(40.0)
        f.addParticle(0.0, 0.33, 0.5, 1 if (i, j, k) == (0, 0, 0) else 2 if (i, j, k) == (1, 0, 0) else 0)
        positions.append((np.array([i, j, k]) + 0.5) * 0.6 + 0.02 * rng.standard_normal(3))
    system.addForce(f)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 5))
    integrator = MSLDIntegrator(300.0, 5.0, 0.002, [Block(site=1), Block(site=1)])
    ctx = mm.Context(system, integrator, REFERENCE)
    ctx.setPositions(np.array(positions))
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    integrator.setThetaVelocitiesToTemperature(0)
    v0 = ctx.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
    integrator.step(200)
    v1 = ctx.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
    assert v1 != v0


def test_runs_under_replica_exchange():
    integrator = MSLDIntegrator(300.0, 1.0, 0.001, toy_blocks(1.0))
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("TOY", chain)
    for i in range(N):
        topology.addAtom(f"C{i}", app.Element.getBySymbol("C"), residue)
    simulation = app.Simulation(topology, toy_system(), integrator, REFERENCE)
    simulation.context.setPositions(X0)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    integrator.setTheta(THETA0)
    states = [{"temperature": 300.0 * unit.kelvin}, {"temperature": 320.0 * unit.kelvin}]
    sampler = app.ReplicaExchangeSampler(states, simulation, 20)
    sampler.simulate(5)
    assert sampler.currentIteration == 5
    assert len(sampler.replicaConformation) == 2

    # a temperature swap rescales vtheta with the atom velocities
    sampler.simulateReplica(0, steps=0)
    before = integrator.getThetaVelocities()
    t_before = integrator.getTemperature()
    sampler.replicaStateIndex[0] = 1 - sampler.replicaStateIndex[0]
    sampler.simulateReplica(0, steps=0)
    t_after = integrator.getTemperature()
    assert t_after != t_before
    scale = np.sqrt(t_after / t_before)
    assert integrator.getThetaVelocities() == pytest.approx([scale * v for v in before], rel=1e-12)
