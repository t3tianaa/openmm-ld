"""LambdaDynamics: the run driver, with the same methods as OpenMM-MSLD msld/integrator.py so scripts port.

The difference is the constructor: it builds the Context itself, because the integrator is now MSLDIntegrator
and lives in the Context. Atoms and theta stay in the Context and the integrator, and `x`, `vx`, `theta`,
`vtheta` read and write them there.
"""
from __future__ import annotations

import math

import numpy as np
import openmm as mm
from openmm import unit

from .builder import bias_parameter_names, set_lambdas
from .forces import exception_pairs, lambda_name
from .integrator import MOLAR_GAS_CONSTANT, Block, MSLDIntegrator

_NM = unit.nanometer
_KJ = unit.kilojoule_per_mole
_F_UNIT = _KJ / _NM
_VEL = _NM / unit.picosecond
KB = MOLAR_GAS_CONSTANT


class LambdaDynamics:
    """Lambda dynamics of a System scaled by scale_msld_system.

    system, model : the scaled System and its finalized model
    positions     : initial positions (nm); box : optional periodic box vectors for the Context
    platform      : platform name; properties : its properties
    seed          : velocities and noise. The velocity draw has the same order as OpenMM-MSLD's, so one
                    seed gives one start in both
    """

    def __init__(self, system, model, positions, temperature=300.0, timestep=0.001, atom_friction=1.0,
                 platform="CPU", properties=None, box=None, seed=None, constraint_tolerance=1e-6):
        self.model = model
        self.kT = KB * temperature
        self.dt = timestep
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        blocks = [Block(site=bl.site, theta_mass=bl.theta_mass, theta_friction=bl.theta_friction,
                        fixed=bl.fixed, theta0=bl.theta0) for bl in model.blocks[1:]]
        self.integrator = MSLDIntegrator(temperature, atom_friction, timestep, blocks, fnex=model.fnex,
                                         constraint_tolerance=constraint_tolerance,
                                         watch_parameters=bias_parameter_names(system))
        if seed is not None:
            self.integrator.setRandomNumberSeed(int(seed) + 1)
        platform = mm.Platform.getPlatformByName(platform) if isinstance(platform, str) else platform
        self.context = mm.Context(system, self.integrator, platform, properties or {})
        if box is not None:
            self.context.setPeriodicBoxVectors(*box)
        self.context.setPositions(positions)

        self.n_atoms = system.getNumParticles()
        self.mass_x = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                                for i in range(self.n_atoms)])
        self.mass = [bl.theta_mass for bl in model.blocks]
        self.dof = [b for b in range(1, model.n_blocks) if not model.blocks[b].fixed]
        self.n_constraints = system.getNumConstraints()
        self.has_constraints = self.n_constraints > 0
        self.constraint_tol = constraint_tolerance

        safe_mass = np.where(self.mass_x > 0, self.mass_x, 1.0)
        vx = self.rng.standard_normal((self.n_atoms, 3)) * np.sqrt(self.kT / safe_mass)[:, None]
        vx[self.mass_x == 0] = 0.0
        vtheta = [bl.theta_velocity0 for bl in model.blocks]
        for b in self.dof:
            vtheta[b] = self.rng.standard_normal() * math.sqrt(self.kT / self.mass[b])
        self.vx = vx
        self.vtheta = vtheta
        self.project()
        if self.has_constraints:
            self.context.applyVelocityConstraints(self.constraint_tol)

    # ---- state ------------------------------------------------------------------------------------
    @property
    def x(self):
        return np.array(self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(_NM))

    @x.setter
    def x(self, value):
        self.context.setPositions(np.asarray(value) * _NM)

    @property
    def vx(self):
        return np.array(self.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(_VEL))

    @vx.setter
    def vx(self, value):
        self.context.setVelocities(np.asarray(value) * _VEL)

    @property
    def theta(self):
        return [0.0] + self.integrator.getTheta()

    @theta.setter
    def theta(self, value):
        self.integrator.setTheta(list(value)[1:])

    @property
    def vtheta(self):
        return [0.0] + self.integrator.getThetaVelocities()

    @vtheta.setter
    def vtheta(self, value):
        self.integrator.setThetaVelocities(list(value)[1:])

    # ---- lambda and energy ----------------------------------------------------------------------------
    def project(self):
        """Project theta -> lambda, push to the Context."""
        self.integrator.initializeLambdas(self.context)
        return self.get_lambdas()

    def get_lambdas(self):
        return self.model.lambda_from_theta(self.theta)

    def potential_energy(self):
        return self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(_KJ)

    def invalidate_forces(self):
        """Call after changing, from outside, anything dU/dlambda depends on that the integrator does not
        detect (see MSLDIntegrator)."""
        self.integrator.invalidate()

    def dUdlambda(self, analytic_deriv=None, lam=None):
        """dU/dlambda_b per block, all analytic. Returns (derivatives, lambdas)."""
        if lam is None:
            lam = self.project()
        if analytic_deriv is None:
            analytic_deriv = self.context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
        d = [0.0] * self.model.n_blocks
        for b in self.dof:
            name = lambda_name(b)
            d[b] = analytic_deriv[name] if name in analytic_deriv else 0.0
        return d, lam

    def dUdtheta(self, analytic_deriv=None, lam=None):
        """Softmax chain rule: dU/dtheta_i from dU/dlambda (per site)."""
        d, lam = self.dUdlambda(analytic_deriv, lam)
        g = [0.0] * self.model.n_blocks
        fnex = self.model.fnex
        theta = self.theta
        for s in range(1, self.model.n_sites):
            js = self.model.blocks_in_site(s)
            wmean = sum(lam[j] * d[j] for j in js)
            for i in js:
                if self.model.blocks[i].fixed:
                    continue
                g[i] = lam[i] * fnex * math.cos(theta[i]) * (d[i] - wmean)
        return g

    # ---- minimization ---------------------------------------------------------------------------------
    def minimize(self, max_iterations=0, tolerance=10.0, relax_blocks=True, check_endpoints=True):
        """Energy-minimize the atoms at the current lambda, on the lambda-scaled surface; theta stays fixed.
        relax_blocks: then relax each block alone at its own endpoint. check_endpoints: then warn, or raise,
        if a block's endpoint is still far above its site-mates. Returns self."""
        self.project()
        mm.LocalEnergyMinimizer.minimize(self.context, tolerance * _F_UNIT, max_iterations)
        if relax_blocks:
            self.relax_blocks(max_iterations=max_iterations, tolerance=tolerance)
        self.invalidate_forces()
        if check_endpoints:
            self.check_endpoints()
        return self

    def worst_contact(self, b):
        """(distance nm, block atom, env atom) of block b's closest non-excluded environment atom."""
        if getattr(self, "_nb_exceptions", None) is None:
            self._nb_exceptions = exception_pairs(self.context.getSystem())
            self._env_idx = np.array([i for i, blk in enumerate(self.model.atom_block) if blk == 0])
        box = self.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(_NM)
        periodic = self.context.getSystem().usesPeriodicBoundaryConditions()
        inv = np.linalg.inv(box) if periodic else None
        env = self._env_idx
        x = self.x
        best = (math.inf, -1, -1)
        for a in self.model.blocks[b].atoms:
            dr = x[env] - x[a]
            if periodic:
                s = dr @ inv
                dr = (s - np.round(s)) @ box
            d = np.linalg.norm(dr, axis=1)
            for j in np.argsort(d)[:16]:
                c = int(env[j])
                if (min(a, c), max(a, c)) in self._nb_exceptions:
                    continue
                if d[j] < best[0]:
                    best = (float(d[j]), a, c)
                break
        return best

    def check_endpoints(self, warn_gap=1.0e3, max_gap=1.0e4):
        """Each block's endpoint energy against the lowest of its site. Above `warn_gap` kJ/mol warns, above
        `max_gap` raises RuntimeError, both naming the closest contact. Returns {block: gap}."""
        import warnings
        e = self.endpoint_energies()
        gaps, bad = {}, []
        for s in range(1, self.model.n_sites):
            js = self.model.blocks_in_site(s)
            lo = min(e[b] for b in js)
            for b in js:
                gaps[b] = e[b] - lo
                if gaps[b] > warn_gap:
                    d, a, c = self.worst_contact(b)
                    bad.append((gaps[b], f"site {s} block {b}: endpoint {gaps[b]:.0f} kJ/mol above the "
                                         f"lowest of its site; closest non-excluded env contact "
                                         f"atom {a}-{c} at {d:.3f} nm"))
        self.endpoint_gaps = gaps
        if bad:
            msg = "; ".join(m for _, m in bad)
            if max(g for g, _ in bad) > max_gap:
                raise RuntimeError("endpoint check failed (clash not removed by minimize): " + msg)
            warnings.warn("large endpoint gap(s): " + msg, stacklevel=2)
        return gaps

    def endpoint_lambdas(self, b):
        """Current lambdas with block b's site switched to b alone (other sites unchanged)."""
        lam = list(self.get_lambdas())
        site = self.model.blocks[b].site
        for c in self.model.blocks_in_site(site):
            lam[c] = 1.0 if c == b else 0.0
        return lam

    def endpoint_energies(self):
        """{block: potential energy (kJ/mol) at endpoint_lambdas(block)} at the current positions."""
        out = {}
        for b in range(1, self.model.n_blocks):
            set_lambdas(self.context, self.endpoint_lambdas(b))
            out[b] = self.potential_energy()
        self.project()
        return out

    def relax_blocks(self, max_iterations=0, tolerance=10.0):
        """Minimize each block's atoms alone, at that block's endpoint, everything else frozen (zero mass on a
        copy of the System; constraints touching a frozen atom are dropped from the copy). Returns self."""
        xml = mm.XmlSerializer.serialize(self.context.getSystem())
        platform = self.context.getPlatform()
        box = self.context.getState().getPeriodicBoxVectors()
        x = self.x
        for b in range(1, self.model.n_blocks):
            atoms = list(self.model.blocks[b].atoms)
            if not atoms:
                continue
            mov = set(atoms)
            sysb = mm.XmlSerializer.deserialize(xml)
            for i in range(sysb.getNumParticles()):
                if i not in mov:
                    sysb.setParticleMass(i, 0.0)
            for k in reversed(range(sysb.getNumConstraints())):
                i, j, _ = sysb.getConstraintParameters(k)
                if i not in mov or j not in mov:
                    sysb.removeConstraint(k)
            ctx = mm.Context(sysb, mm.VerletIntegrator(0.001), platform)
            ctx.setPeriodicBoxVectors(*box)
            ctx.setPositions(x * _NM)
            for name, value in self.context.getParameters().items():
                ctx.setParameter(name, value)          # biases etc.
            set_lambdas(ctx, self.endpoint_lambdas(b))
            mm.LocalEnergyMinimizer.minimize(ctx, tolerance * _F_UNIT, max_iterations)
            new = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(_NM)
            x[atoms] = new[atoms]
            del ctx
        self.context.setPositions(x * _NM)
        if self.has_constraints:
            self.context.applyConstraints(self.constraint_tol)
        self.project()
        self.invalidate_forces()
        return self

    # ---- dynamics ---------------------------------------------------------------------------------------
    def step(self, nsteps=1):
        self.integrator.step(nsteps)

    def run(self, nsteps, *, minimize=False, metrics=None, metrics_every: int = 100, mapping=None,
            trajectory=None, trajectory_every: int = 1000, structure=None, reporters=()):
        """Integrate `nsteps`, saving metrics / a trajectory at their own step intervals.

          minimize   : energy-minimize once before stepping; True runs to convergence, an int caps iterations
          metrics    : CSV path for per-block lambda + energy (`mapping` names the columns; optional)
          trajectory : DCD path (needs `structure`, with .topology and .positions)
          reporters  : extra reporters, each with `.every`, `.report(dyn, step)`, `.close()`

        `run` steps only as finely as needed and reports each reporter at multiples of its `every`.
        Returns self.
        """
        if minimize:
            self.minimize(max_iterations=(0 if minimize is True else int(minimize)))

        reps = list(reporters)
        if metrics is not None:
            from .io import LambdaTrajectoryWriter
            reps.append(LambdaTrajectoryWriter(metrics, self.model, every=metrics_every, mapping=mapping))
        if trajectory is not None:
            if structure is None:
                raise ValueError("trajectory= needs structure= for its topology")
            from .io import TrajectoryWriter
            reps.append(TrajectoryWriter(trajectory, structure, every=trajectory_every, mapping=mapping))

        if not reps:
            self.step(nsteps)
            return self

        intervals = [r.every for r in reps if getattr(r, "every", 0)]
        stride = math.gcd(*intervals, nsteps) if intervals else nsteps
        done = 0
        try:
            while done < nsteps:
                chunk = min(stride, nsteps - done)
                self.step(chunk)
                done += chunk
                for r in reps:
                    if getattr(r, "every", 0) and done % r.every == 0:
                        r.report(self, done)
        finally:
            for r in reps:
                close = getattr(r, "close", None)
                if callable(close):
                    close()
        return self

    # ---- diagnostics --------------------------------------------------------------------------------------
    def theta_temperature(self):
        return self.integrator.getThetaTemperature()

    def atom_temperature(self):
        """Instantaneous atom kinetic temperature over 3N - n_constraints degrees of freedom."""
        ndof = 3 * int(np.count_nonzero(self.mass_x)) - self.n_constraints
        ke = 0.5 * float(np.sum(self.mass_x[:, None] * self.vx ** 2))
        return 2.0 * ke / (ndof * KB)
