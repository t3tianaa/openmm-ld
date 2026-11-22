"""Published ALF on openmm-ld: bias convention against the engine, and a toy flattening run."""
import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmm_ld import LambdaDynamics, MSLDModel, set_lambdas
from openmm_ld import alf_published as alfp

G = [0.0, 8.0, -6.0, 15.0, 3.0]          # hidden endpoint free energies (kJ/mol)


def _toy(seed=1, coupling=20.0):
    nb_ = len(G)
    n = nb_ + 1
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for _ in range(n):
        nb.addParticle(0.0, 0.3, 0.0)
    system.addForce(nb)
    hidden = mm.CustomBondForce("+".join(f"({g})*lambda{b + 1}" for b, g in enumerate(G))
                                + f"+{coupling}*lambda1*lambda2")
    for b in range(nb_):
        hidden.addGlobalParameter(f"lambda{b + 1}", 1.0)
        hidden.addEnergyParameterDerivative(f"lambda{b + 1}")
    hidden.addBond(0, 1, [])
    hidden.setForceGroup(1)
    system.addForce(hidden)
    model = MSLDModel(n, fnex=5.5)
    for b in range(nb_):
        model.add_block(site=1, atoms=[b + 1])
    model.finalize()
    alfp.add_alf_biases(model)
    model.scale_system(system)
    pos = np.array([[i * 0.5, 0, 0] for i in range(n)]) * unit.nanometer
    return LambdaDynamics(system, model, pos, platform="Reference", seed=seed, timestep=0.002)


def test_engine_bias_energy_equals_alf_energy():
    """Engine bias (kJ/mol) == 4.184 * ALF's GetEnergy expression (kcal/mol) at random lambdas."""
    dyn = _toy()
    rng = np.random.default_rng(3)
    nb_ = len(G)
    biases = alfp.Biases(nb_, rng.normal(size=nb_), rng.normal(size=(nb_, nb_)),
                         rng.normal(size=(nb_, nb_)), rng.normal(size=(nb_, nb_)))
    for m in (biases.c, biases.x, biases.s):
        np.fill_diagonal(m, 0.0)
    biases.c = np.triu(biases.c)              # c is stored for i < j only, as in ALF
    alfp.apply_biases(dyn.model, biases, dyn.context)
    bias_group = {f.getForceGroup() for f in dyn.context.getSystem().getForces() if f.getName() == "MSLDBias"}
    assert len(bias_group) == 1
    for _ in range(5):
        lam = rng.dirichlet(np.ones(nb_))
        set_lambdas(dyn.context, [0.0] + list(lam))
        e = dyn.context.getState(getEnergy=True, groups=bias_group).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        assert e == pytest.approx(alfp.KCAL * biases.energy(lam)[0], abs=1e-8)


def test_flattening_recovers_known_biases(tmp_path):
    """A few short cycles move the biases toward the hidden free energies (ALF sign: b ~ G / 4.184)."""
    dyn = _toy()
    b = alfp.flatten(dyn, str(tmp_path), cycles=6, steps=4000, sample_every=10, log=lambda m: None)
    expected = (np.array(G) - G[0]) / alfp.KCAL
    assert np.sign(b.b[1:]).tolist() == np.sign(expected[1:]).tolist()
    assert np.abs(b.b - expected).max() < 2.0
