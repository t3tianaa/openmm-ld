"""Where does the MDM2 NVE energy drift come from? Diagnostic only: it changes no physics.

  python drift_mdm2.py equilibrate PREPARED OUTDIR [STEPS]     NVT from a relaxed start, saves the state
  python drift_mdm2.py VARIANT PREPARED OUTDIR [STEPS]         one NVE run from that state, appends to OUTDIR/drift.jsonl
  python drift_mdm2.py check PREPARED OUTDIR                   finite-difference check at the saved frame

Variants are in VARIANTS below. Drift is the slope of total energy against time, after the first 0.2 ps.
"""
import json
import os
import sys
import time

import numpy as np
import openmm as mm
from openmm import unit

import msldplugin
from openmm_ld import Block, LambdaDynamics, MSLDIntegrator, load_prepared, set_lambdas

NM, KJ = unit.nanometer, unit.kilojoule_per_mole
THREADS = os.environ.get("THREADS", "10")
PROPS = {"Threads": THREADS}
CPU = mm.Platform.getPlatformByName("CPU")
ENERGY_EVERY_PS = 0.01


def scaled(prepared_dir):
    prepared = load_prepared(prepared_dir)
    model = prepared.to_model()
    model.scale_bond = model.scale_angle = False
    model.scale_system(prepared.system)
    return prepared, model


def plugin_force(system):
    for i in range(system.getNumForces()):
        if system.getForce(i).getName() == "MSLDNonbondedForce":
            return i, msldplugin.MSLDNonbondedForce.cast(system.getForce(i))
    raise ValueError("no plugin force")


def equilibrate(prepared_dir, out, steps):
    prepared, model = scaled(prepared_dir)
    for bl in model.blocks[1:]:
        bl.theta_friction = 5.0
    dyn = LambdaDynamics(prepared.system, model, prepared.positions, atom_friction=5.0, platform="CPU", properties=PROPS,
                         box=prepared.box, seed=0)
    dyn.minimize(max_iterations=500, relax_blocks=True, check_endpoints=False)
    st = dyn.context.getState(getPositions=True, getVelocities=True)
    np.savez(os.path.join(out, "minimized.npz"), x=st.getPositions(asNumpy=True).value_in_unit(NM),
             v=st.getVelocities(asNumpy=True).value_in_unit(NM / unit.picosecond),
             theta=dyn.theta, vtheta=dyn.vtheta, lam=dyn.get_lambdas(),
             box=st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM))
    t = time.time()
    for k in range(steps // 500):
        dyn.step(500)
        print(f"step {(k + 1) * 500}: T_atoms {dyn.atom_temperature():.1f} K, T_theta {dyn.theta_temperature():.0f} K, "
              f"PE {dyn.potential_energy():.0f}, lambda {np.round(dyn.get_lambdas()[1:], 3).tolist()} ({time.time() - t:.0f} s)", flush=True)
    st = dyn.context.getState(getPositions=True, getVelocities=True)
    np.savez(os.path.join(out, "equilibrated.npz"), x=st.getPositions(asNumpy=True).value_in_unit(NM),
             v=st.getVelocities(asNumpy=True).value_in_unit(NM / unit.picosecond),
             theta=dyn.theta, vtheta=dyn.vtheta, lam=dyn.get_lambdas(),
             box=st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM))


def run_nve(label, prepared_dir, out, *, dt=0.001, ps=1.5, theta="free", tolerance=1e-6, integrator="msld", endpoint=False,
            nonbonded="plugin", switching=None, ewald=None, frozen_atoms=False, start_file="equilibrated.npz"):
    start = np.load(os.path.join(out, start_file))
    prepared, model = scaled(prepared_dir)
    system = prepared.system
    index, force = plugin_force(system)
    lam = list(start["lam"])
    if endpoint:
        best = int(np.argmax(lam[1:])) + 1
        lam = [1.0] + [1.0 if b == best else 0.0 for b in range(1, len(lam))]
    if switching is not None:
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(switching)
    if ewald is not None:
        force.setEwaldErrorTolerance(ewald)
    if nonbonded == "stock":
        system = stock_endpoint_system(prepared_dir, system, index, model, best)
    if frozen_atoms:
        for i in range(system.getNumParticles()):
            system.setParticleMass(i, 0.0)
        for k in reversed(range(system.getNumConstraints())):
            system.removeConstraint(k)
    if integrator == "verlet":
        integ = mm.VerletIntegrator(dt)
        integ.setConstraintTolerance(tolerance)
    else:
        blocks = [Block(site=model.blocks[b].site, theta_friction=0.0, fixed=(theta == "frozen")) for b in range(1, model.n_blocks)]
        integ = MSLDIntegrator(300.0, 0.0, dt, blocks, fnex=model.fnex, constraint_tolerance=tolerance)
    ctx = mm.Context(system, integ, CPU, PROPS)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*v) * NM for v in start["box"]])
    ctx.setPositions(start["x"] * NM)
    if not frozen_atoms:
        ctx.setVelocities(start["v"] * NM / unit.picosecond)
    if integrator == "verlet":
        set_lambdas(ctx, lam)
    else:
        integ.setTheta(list(start["theta"][1:]))
        integ.setThetaVelocities(list(start["vtheta"][1:]) if theta == "free" else [0.0] * (model.n_blocks - 1))
        integ.initializeLambdas(ctx)
    ctx.applyConstraints(tolerance)
    ctx.applyVelocityConstraints(tolerance)
    every = int(round(ENERGY_EVERY_PS / dt))
    times, energies, potentials, theta_k = [], [], [], []
    t0 = time.time()
    for k in range(int(round(ps / (every * dt))) + 1):
        if k:
            integ.step(every)
        st = ctx.getState(getEnergy=True)
        tk = integ.getThetaKineticEnergy() if integrator != "verlet" else 0.0
        pe = st.getPotentialEnergy().value_in_unit(KJ)
        times.append(k * every * dt)
        energies.append(pe + st.getKineticEnergy().value_in_unit(KJ) + tk)
        potentials.append(pe)
        theta_k.append(tk)
    times, energies = np.array(times), np.array(energies)
    use = times >= 0.2
    slope, intercept = np.polyfit(times[use], energies[use], 1)
    residual = energies[use] - (slope * times[use] + intercept)
    half = len(residual) // 2
    row = {"variant": label, "dt_fs": dt * 1000, "ps": ps, "tolerance": tolerance, "drift kJ/mol/ps": float(slope),
           "residual rms": float(residual.std()), "E range": float(energies.max() - energies.min()),
           "E(0.2ps) - E(0)": float(energies[use][0] - energies[0]), "PE range": float(np.ptp(potentials)),
           "theta KE mean": float(np.mean(theta_k)), "wall s": round(time.time() - t0, 1)}
    print(json.dumps(row), flush=True)
    with open(os.path.join(out, "drift.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    np.savez(os.path.join(out, f"trace_{label}.npz"), t=times, E=energies, PE=np.array(potentials), theta_ke=np.array(theta_k))


def stock_endpoint_system(prepared_dir, scaled_system, plugin_index, model, on_block):
    """The endpoint with block `on_block` alone switched on, as a stock NonbondedForce: same bonded, same everything."""
    plain = load_prepared(prepared_dir).system
    nb = next(f for f in plain.getForces() if isinstance(f, mm.NonbondedForce))
    nb = mm.XmlSerializer.clone(nb)
    off = {i for i, b in enumerate(model.atom_block) if b not in (0, on_block)}
    for i in off:
        q, s, e = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, s, 0.0)
    for k in range(nb.getNumExceptions()):
        i, j, q, s, e = nb.getExceptionParameters(k)
        if i in off or j in off:
            nb.setExceptionParameters(k, i, j, 0.0, s, 0.0)
    scaled_system.removeForce(plugin_index)
    scaled_system.addForce(nb)
    return scaled_system


VARIANTS = {
    "free_theta": {},
    "free_theta_dt0.5": {"dt": 0.0005},
    "frozen_theta": {"theta": "frozen"},
    "endpoint_verlet": {"integrator": "verlet", "endpoint": True},
    "endpoint_stock_nb": {"integrator": "verlet", "endpoint": True, "nonbonded": "stock"},
    "endpoint_dt0.5": {"integrator": "verlet", "endpoint": True, "dt": 0.0005},
    "endpoint_dt2": {"integrator": "verlet", "endpoint": True, "dt": 0.002},
    "endpoint_tol1e-10": {"integrator": "verlet", "endpoint": True, "tolerance": 1e-10},
    "endpoint_switch0.9": {"integrator": "verlet", "endpoint": True, "switching": 0.9},
    "endpoint_ewald1e-5": {"integrator": "verlet", "endpoint": True, "ewald": 1e-5},
    "endpoint_switch_ewald": {"integrator": "verlet", "endpoint": True, "switching": 0.9, "ewald": 1e-5},
    "frozen_theta_switch_ewald": {"theta": "frozen", "switching": 0.9, "ewald": 1e-5},
    "free_theta_switch_ewald": {"switching": 0.9, "ewald": 1e-5},
    "endpoint_padding0": {"integrator": "verlet", "endpoint": True, "ps": 1.0},     # with MSLD_NEIGHBOR_PADDING=0
    "free_theta_long": {"ps": 10.0},
    "from_minimized": {"ps": 4.0, "start_file": "minimized.npz"},
    "endpoint_long": {"integrator": "verlet", "endpoint": True, "ps": 10.0},
    "theta_only": {"frozen_atoms": True, "ps": 1.0},
    "theta_only_dt0.25": {"frozen_atoms": True, "dt": 0.00025, "ps": 0.25},
}


def check(prepared_dir, out):
    """Forces and dU/dlambda against central differences of the energy, at the saved MD frame."""
    start = np.load(os.path.join(out, "equilibrated.npz"))
    prepared, model = scaled(prepared_dir)
    ctx = mm.Context(prepared.system, mm.VerletIntegrator(0.001), CPU, PROPS)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*v) * NM for v in start["box"]])
    x = start["x"]
    energy = lambda: ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    lam = list(start["lam"])
    # partial lambda, so soft-core and every block count
    for label, values in (("MD lambda", lam), ("uniform", [1.0] + [0.2] * 5)):
        set_lambdas(ctx, values)
        ctx.setPositions(x * NM)
        st = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
        f = st.getForces(asNumpy=True).value_in_unit(KJ / NM)
        derivs = st.getEnergyParameterDerivatives()
        rng = np.random.default_rng(0)
        rows = []
        for h in (1e-3, 1e-4):
            for direction in (f, rng.normal(size=f.shape)):
                u = direction / np.linalg.norm(direction)
                ctx.setPositions((x + h * u) * NM); ep = energy()
                ctx.setPositions((x - h * u) * NM); em = energy()
                exact = -np.sum(np.array(f) * u)
                rows.append({"h": h, "FD": (ep - em) / (2 * h), "analytic": float(exact)})
        ctx.setPositions(x * NM)
        lam_rows = []
        for b in range(1, 6):
            h = 1e-5
            lp, lm = list(values), list(values)
            lp[b] += h
            lm[b] -= h
            set_lambdas(ctx, lp); ep = energy()
            set_lambdas(ctx, lm); em = energy()
            lam_rows.append({"b": b, "FD": (ep - em) / (2 * h), "analytic": derivs[f"lambda{b}"]})
        set_lambdas(ctx, values)
        print(json.dumps({"lambda": label, "directional": rows, "dU/dlambda": lam_rows}), flush=True)


if __name__ == "__main__":
    command, prepared_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    if command == "equilibrate":
        equilibrate(prepared_dir, out, int(sys.argv[4]) if len(sys.argv) > 4 else 8000)
    elif command == "check":
        check(prepared_dir, out)
    else:
        run_nve(command, prepared_dir, out, **VARIANTS[command])
