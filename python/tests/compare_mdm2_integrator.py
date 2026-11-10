"""MSLDIntegrator on MDM2 against LambdaDynamics (OpenMM-MSLD, use_plugin), CPU platform, zero friction.

Takes the inputs of OpenMM-MSLD's test_systems/mdm2/dynamics_check_plugin.py (run its `prep` first) and
starts both integrators from the same positions and velocities. Reports differences and energy drift.

Usage: python compare_mdm2_integrator.py MSLD_CHECKOUT PREP_DIR NSTEPS
"""
import os
import sys
import time

import numpy as np
import openmm as mm
from openmm import unit

from openmm_ld import Block, MSLDIntegrator

engine, prep, nsteps = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path[:0] = [engine, os.path.join(engine, "test_systems", "mdm2")]
os.environ.setdefault("MIN_ITERS", "0")
from dynamics_check_plugin import prepare_model  # noqa: E402
from msld.integrator import LambdaDynamics  # noqa: E402

NM = unit.nanometer
_, model = prepare_model()
start = np.load(os.path.join(prep, "start.npz"))
base = open(os.path.join(prep, "base_system.xml")).read()
for bl in model.blocks[1:]:
    bl.theta_mass, bl.theta_friction = 0.05, 0.0
platform = mm.Platform.getPlatformByName("CPU")
props = {"Threads": "4"}
box = [mm.Vec3(*v) * NM for v in start["box"]]


def system():
    s = mm.XmlSerializer.deserialize(base)
    model.scale_system(s, use_plugin=True)
    return s


c1 = mm.Context(system(), mm.VerletIntegrator(0.001), platform, props)
c1.setPeriodicBoxVectors(*box)
c1.setPositions(start["x"] * NM)
dyn = LambdaDynamics(c1, model, temperature=300.0, timestep=0.001, atom_friction=0.0, seed=0)

blocks = [Block(site=model.blocks[b].site, theta_friction=0.0) for b in range(1, model.n_blocks)]
integrator = MSLDIntegrator(300.0, 0.0, 0.001, blocks, fnex=model.fnex)
c2 = mm.Context(system(), integrator, platform, props)
c2.setPeriodicBoxVectors(*box)
c2.setPositions(dyn.x * NM)
c2.setVelocities(dyn.vx * NM / unit.picosecond)
integrator.setTheta(dyn.theta[1:])
integrator.setThetaVelocities(dyn.vtheta[1:])

t_engine = t_integrator = 0.0
energies = []
for step in range(0, nsteps, 5):
    t = time.time(); dyn.step(5); t_engine += time.time() - t
    t = time.time(); integrator.step(5); t_integrator += time.time() - t
    st = c2.getState(getPositions=True, getEnergy=True, getVelocities=True)
    x = st.getPositions(asNumpy=True).value_in_unit(NM)
    energies.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    + st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                    + integrator.getThetaKineticEnergy())
    if step % 50 == 0 or step + 5 >= nsteps:
        print(f"step {step + 5}: max |dx| {np.max(np.abs(x - dyn.x)):.2e} nm, max |dtheta| "
              f"{np.max(np.abs(np.array(integrator.getTheta()) - np.array(dyn.theta[1:]))):.2e}, "
              f"max |dlambda| {np.max(np.abs(np.array(integrator.getLambdas()) - np.array(dyn.get_lambdas()[1:]))):.2e}",
              flush=True)
energies = np.array(energies)
t_axis = 0.005 * np.arange(1, len(energies) + 1)
tail = t_axis >= 0.1
slope = np.polyfit(t_axis[tail], energies[tail], 1)[0]
print(f"MSLDIntegrator: energy rise to max {energies.max() - energies[0]:.1f} kJ/mol, drift after 0.1 ps {slope:.1f} kJ/mol/ps")
print(f"time per step: LambdaDynamics {t_engine / nsteps:.3f} s, MSLDIntegrator {t_integrator / nsteps:.3f} s")
