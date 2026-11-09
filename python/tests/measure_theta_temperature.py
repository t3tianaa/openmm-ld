"""Long NVT runs of MSLDIntegrator on the oracle toy: mean theta and atom temperatures with standard
errors over independent runs. The numbers are in docs/msld_integrator.md.

Usage: python measure_theta_temperature.py [runs] [steps] [dt in ps]
"""
import sys
import time

import numpy as np
import openmm as mm

from openmm_ld import MSLDIntegrator
from test_msld_integrator import N, REFERENCE, X0, toy_blocks, toy_system
import msldplugin, os

KB = 0.00831446261815324
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
every = 50
theta_means, atom_means = [], []
t0 = time.time()
for run in range(runs):
    integrator = MSLDIntegrator(300.0, 1.0, dt, toy_blocks(1.0))
    integrator.setRandomNumberSeed(run + 1)
    ctx = mm.Context(toy_system(), integrator, REFERENCE)
    ctx.setPositions(X0)
    ctx.setVelocitiesToTemperature(300.0, run + 1)
    integrator.setThetaVelocitiesToTemperature(run + 1)
    integrator.step(steps // 10)         # equilibration
    theta, atoms = [], []
    for _ in range(steps // every):
        integrator.step(every)
        theta.append(integrator.getThetaTemperature())
        atoms.append(ctx.getState(getEnergy=True).getKineticEnergy()._value * 2.0 / (3 * N * KB))
    theta_means.append(np.mean(theta))
    atom_means.append(np.mean(atoms))
    print(f"run {run}: T_theta {theta_means[-1]:.1f} K, T_atoms {atom_means[-1]:.1f} K", flush=True)
sem = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))
print(f"dt {dt} ps, {runs} runs of {steps} steps: T_theta {np.mean(theta_means):.1f} +- {sem(theta_means):.1f} K, "
      f"T_atoms {np.mean(atom_means):.1f} +- {sem(atom_means):.1f} K ({time.time() - t0:.0f} s)")
