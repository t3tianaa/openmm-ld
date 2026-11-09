"""An MDM2 MSLD run from a prepared system, using only openmm_ld: scale, minimize, ALF, production.

Usage: python run_mdm2.py PREPARED_DIR OUTDIR [ALF_ITERS] [STEPS_PER_ITER] [PRODUCTION_STEPS]
Env: THREADS (default 4), MIN_ITERS (default 200). Needs msldplugin and its kernels, on the CPU platform.
"""
import os
import sys
import time

import openmm as mm

from openmm_ld import LambdaDynamics, load_prepared
from openmm_ld.alf import add_flattening_biases, estimate_ddg, flatten
from openmm_ld.io import populations

prepared_dir, out = sys.argv[1], sys.argv[2]
alf_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 2
steps_per_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 100
production = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
threads = os.environ.get("THREADS", "4")
min_iters = int(os.environ.get("MIN_ITERS", 200))
os.makedirs(out, exist_ok=True)

t0 = time.time()
prepared = load_prepared(prepared_dir)
model = prepared.to_model()
model.scale_bond = False          # BLaDE practice, as in OpenMM-MSLD's MDM2 script
model.scale_angle = False
add_flattening_biases(model)
info = model.scale_system(prepared.system)
print(f"scaled: {prepared.system.getNumForces()} forces, {info['scaled_14_exceptions']} alchemical 1-4s", flush=True)

dyn = LambdaDynamics(prepared.system, model, prepared.positions, temperature=300.0, timestep=0.001, atom_friction=1.0,
                     platform="CPU", properties={"Threads": threads}, box=prepared.box, seed=0)
print(f"PE before minimize {dyn.potential_energy():.1f} kJ/mol", flush=True)
t = time.time()
dyn.minimize(max_iterations=min_iters)
print(f"PE after minimize {dyn.potential_energy():.1f} kJ/mol ({time.time() - t:.0f} s); "
      f"endpoint gaps {({b: round(g, 1) for b, g in dyn.endpoint_gaps.items()})}", flush=True)

t = time.time()
report = flatten(dyn, n_iters=alf_iters, steps_per_iter=steps_per_iter, sample_every=10, max_step=8.0, tol=0.0)
print(f"ALF: {report['iterations']} iterations ({time.time() - t:.0f} s), biases "
      f"{({b: round(v, 2) for b, v in report['biases'].items()})}", flush=True)
print("DDG estimate (kJ/mol, not converged in a run this short):",
      {b: round(v[0], 2) for b, v in estimate_ddg(report, model).items()}, flush=True)

t = time.time()
metrics, trajectory = os.path.join(out, "metrics.csv"), os.path.join(out, "traj.dcd")
dyn.run(production, metrics=metrics, metrics_every=100, mapping=prepared.mapping,
        trajectory=trajectory, trajectory_every=200, structure=prepared)
print(f"production: {production} steps ({time.time() - t:.0f} s, {(time.time() - t) / production:.3f} s/step); "
      f"T_theta {dyn.theta_temperature():.0f} K, T_atoms {dyn.atom_temperature():.0f} K", flush=True)
print("mean lambda:", {k: round(v, 3) for k, v in populations(metrics).items()}, flush=True)
assert not any(m == "msld" or m.startswith("msld.") for m in sys.modules), "imported OpenMM-MSLD"
print(f"done in {time.time() - t0:.0f} s, without importing OpenMM-MSLD", flush=True)
