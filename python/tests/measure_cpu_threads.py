"""Thread scaling of MSLDNonbondedForce on MDM2 (CPU platform), against the same system on stock OpenMM.

Prints results and timings as JSON lines. Run with MSLD_KERNEL_TIMING=1 for the kernel's own split on stderr.

Usage: python measure_cpu_threads.py PREPARED_DIR OUT.json [threads...]
"""
import hashlib
import json
import sys
import time

import numpy as np
import openmm as mm
from openmm import unit

from openmm_ld import LambdaDynamics, load_prepared, set_lambdas

prepared_dir, out = sys.argv[1], sys.argv[2]
thread_counts = sys.argv[3:] or ["1", "2", "4", "10"]
STEPS = 100
results = {}


def scaled():
    prepared = load_prepared(prepared_dir)
    model = prepared.to_model()
    model.scale_bond = model.scale_angle = False
    model.scale_system(prepared.system)
    return prepared, model


LAMBDA = [1.0, 0.1, 0.05, 0.7, 0.05, 0.1]
prepared, model = scaled()
platform = mm.Platform.getPlatformByName("CPU")

# results against one thread, at the prepared positions and one step further
reference = None
for threads in thread_counts + [thread_counts[-1]]:
    ctx = mm.Context(prepared.system, mm.VerletIntegrator(0.001), platform, {"Threads": threads})
    ctx.setPeriodicBoxVectors(*prepared.box)
    ctx.setPositions(prepared.positions)
    set_lambdas(ctx, LAMBDA)
    st = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    d = np.array([st.getEnergyParameterDerivatives()[f"lambda{b}"] for b in range(1, 6)])
    digest = hashlib.md5(np.array(f).tobytes()).hexdigest()[:10] + f"{e:.17g}"
    if reference is None:
        reference = (e, np.array(f), d)
    row = {"threads": threads, "E rel err vs first": abs(e - reference[0]) / abs(reference[0]),
           "dU/dlambda max abs err": float(np.max(np.abs(d - reference[2]))),
           "F max abs err": float(np.max(np.abs(np.array(f) - reference[1]))), "digest": digest}
    results.setdefault("agreement", []).append(row)
    print(json.dumps(row), flush=True)
    del ctx

# a relaxed start for the timings: the prepared positions have clashes
prepared, model = scaled()
dyn = LambdaDynamics(prepared.system, model, prepared.positions, platform="CPU", properties={"Threads": thread_counts[-1]},
                     box=prepared.box, seed=0)
dyn.minimize(max_iterations=200, relax_blocks=False, check_endpoints=False)
start = dyn.x * unit.nanometer
del dyn

# a dynamics step with MSLDIntegrator; the kernel's own split goes to stderr when the Context goes away
results["dynamics"] = []
for threads in thread_counts:
    prepared, model = scaled()
    dyn = LambdaDynamics(prepared.system, model, start, platform="CPU", properties={"Threads": threads},
                         box=prepared.box, seed=0)
    dyn.step(20)          # past the first list build and force evaluation
    t = time.perf_counter()
    dyn.step(STEPS)
    row = {"threads": threads, "s/step": (time.perf_counter() - t) / STEPS}
    results["dynamics"].append(row)
    print(json.dumps(row), flush=True)
    del dyn

# stock OpenMM: the prepared system's own NonbondedForce alone, at the same positions, stepped by a small random walk.
# Not a dynamics step: unscaled, the competing blocks of a site overlap, and the system blows up.
results["stock"] = []
for threads in thread_counts:
    system = load_prepared(prepared_dir).system
    for i in reversed(range(system.getNumForces())):
        if not isinstance(system.getForce(i), mm.NonbondedForce):
            system.removeForce(i)
    for i in reversed(range(system.getNumConstraints())):
        system.removeConstraint(i)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), platform, {"Threads": threads})
    ctx.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    x = np.array(start.value_in_unit(unit.nanometer))
    rng = np.random.default_rng(0)
    ctx.setPositions(x)
    ctx.getState(getForces=True)
    t = time.perf_counter()
    for _ in range(50):
        x = x + rng.normal(0.0, 0.0005, x.shape)
        ctx.setPositions(x)
        ctx.getState(getEnergy=True, getForces=True)
    row = {"threads": threads, "stock NonbondedForce s/eval": (time.perf_counter() - t) / 50}
    results["stock"].append(row)
    print(json.dumps(row), flush=True)
    del ctx
with open(out, "w") as fh:
    json.dump(results, fh, indent=1)
