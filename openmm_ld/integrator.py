"""MSLDIntegrator: atoms and theta under one VRORV Langevin step, as an OpenMM CustomIntegrator.

Design note: docs/msld_integrator.md. Runs on any platform that has kernels for the System's forces. Every
lambda-dependent force must report dU/dlambda (MSLDNonbondedForce does; stock custom forces do through
addEnergyParameterDerivative). A force that does not, such as a NonbondedForce with charge offsets, would
act on the atoms but exert no force on theta.
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import openmm as mm
from openmm import unit

MOLAR_GAS_CONSTANT = 0.00831446261815324         # kJ/mol/K

Block = namedtuple("Block", "site theta_mass theta_friction fixed theta0", defaults=(0.05, 1.0, False, 0.0))
Block.__doc__ = """One substituent. site >= 1; theta_mass in amu nm^2 (theta in rad); theta_friction in 1/ps."""


def _value(quantity, units):
    return quantity.value_in_unit(units) if unit.is_quantity(quantity) else quantity


class MSLDIntegrator(mm.CustomIntegrator):
    """Multisite lambda dynamics in one integrator.

    blocks[k] is block k + 1, coupled by the Context parameter "lambda<k+1>". theta and its velocity are
    global variables of the integrator, so they are in State.IntegratorParameters and survive
    context.getState()/setState() and checkpoints.

    dU/dlambda is kept from the end of one step for the start of the next, because OpenMM would otherwise
    evaluate the forces twice per step. Moved atoms (setPositions, setState, a barostat) and a changed lambda are
    detected. Anything else that changes dU/dlambda from outside needs one of these: name the Context
    parameters that may change (biases, for instance) in watch_parameters, or call invalidate().
    """

    def __init__(self, temperature, atom_friction, timestep, blocks, fnex=5.5, constraint_tolerance=1e-6,
                 watch_parameters=()):
        super().__init__(_value(timestep, unit.picoseconds))
        self._blocks = list(blocks)
        self._fnex = float(fnex)
        self.setConstraintTolerance(constraint_tolerance)
        self.addGlobalVariable("kT", MOLAR_GAS_CONSTANT * _value(temperature, unit.kelvin))
        self.addGlobalVariable("gamma_x", _value(atom_friction, unit.picoseconds ** -1))
        self.addGlobalVariable("fnex", self._fnex)
        self.addGlobalVariable("project_lambdas", 1.0)
        self.addGlobalVariable("stale", 0.0)
        self.addGlobalVariable("moved", 0.0)
        self.addPerDofVariable("x_free", 0.0)
        self.addPerDofVariable("x_last", 0.0)
        self._watched = list(watch_parameters)
        for name in self._watched:
            self.addGlobalVariable(f"last_{name}", 0.0)
        self._sites = {}
        for b, block in self._numbered():
            self._sites.setdefault(block.site, []).append(b)
            self.addGlobalVariable(f"theta{b}", block.theta0)
            self.addGlobalVariable(f"vtheta{b}", 0.0)
            self.addGlobalVariable(f"mtheta{b}", block.theta_mass)
            self.addGlobalVariable(f"gamma_theta{b}", block.theta_friction)
            self.addGlobalVariable(f"dUdlambda{b}", 0.0)
            self.addGlobalVariable(f"last_lambda{b}", 0.0)
        for site in self._sites:
            self.addGlobalVariable(f"wmean{site}", 0.0)
        self._build_step()

    def _numbered(self):
        return list(enumerate(self._blocks, start=1))

    def _movable(self):
        return [b for b, block in self._numbered() if not block.fixed]

    def _push_lambdas(self):
        for js in self._sites.values():
            z = "+".join(f"exp(fnex*sin(theta{j}))" for j in js)
            for b in js:
                self.addComputeGlobal(f"lambda{b}", f"exp(fnex*sin(theta{b}))/({z})")

    def _read_derivatives(self):
        # One deriv() per expression: with several in one expression, OpenMM 8.6's Reference
        # implementation gives them all the value of the first (found against LambdaDynamics).
        for site, js in self._sites.items():
            for b in js:
                self.addComputeGlobal(f"dUdlambda{b}", f"deriv(energy, lambda{b})")
            self.addComputeGlobal(f"wmean{site}", "+".join(f"lambda{j}*dUdlambda{j}" for j in js))

    def _half_kick(self):
        """B: v += dt/2 F/m for atoms and theta, with dU/dtheta from the softmax chain rule."""
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addConstrainVelocities()
        for site, js in self._sites.items():
            for b in js:
                if not self._blocks[b - 1].fixed:
                    self.addComputeGlobal(
                        f"vtheta{b}", f"vtheta{b}-0.5*dt*lambda{b}*fnex*cos(theta{b})"
                                      f"*(dUdlambda{b}-wmean{site})/mtheta{b}")

    def _half_drift(self):
        """A: x += dt/2 v with SHAKE, the constraint impulse going into v; theta += dt/2 vtheta."""
        self.addComputePerDof("x_free", "x+0.5*dt*v")
        self.addComputePerDof("x", "x_free")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+(x-x_free)/(0.5*dt)")
        for b in self._movable():
            self.addComputeGlobal(f"theta{b}", f"theta{b}+0.5*dt*vtheta{b}")

    def _build_step(self):
        self.addUpdateContextState()
        self.beginIfBlock("project_lambdas > 0.5")
        self._push_lambdas()
        self.endBlock()
        # dU/dlambda from the end of the previous step, unless something changed it since
        self.addComputeSum("moved", "abs(x-x_last)")
        watched = [f"lambda{b}" for b, _ in self._numbered()] + self._watched
        self.addComputeGlobal("stale", "project_lambdas+moved+" + "+".join(f"abs({n}-last_{n})" for n in watched))
        self.addComputeGlobal("project_lambdas", "0")
        self.beginIfBlock("stale > 0")
        self._read_derivatives()
        self.endBlock()
        self._half_kick()
        self._half_drift()
        # O: friction and noise
        self.addComputePerDof("v", "exp(-gamma_x*dt)*v+sqrt((1-exp(-2*gamma_x*dt))*kT/m)*gaussian")
        self.addConstrainVelocities()
        for b in self._movable():
            self.addComputeGlobal(f"vtheta{b}", f"exp(-gamma_theta{b}*dt)*vtheta{b}"
                                                f"+sqrt((1-exp(-2*gamma_theta{b}*dt))*kT/mtheta{b})*gaussian")
        self._half_drift()
        self._push_lambdas()
        self._read_derivatives()
        self._half_kick()
        self.addComputePerDof("x_last", "x")
        for name in watched:
            self.addComputeGlobal(f"last_{name}", name)

    # ---- temperature (getTemperature is what OpenMM's multistate samplers look for) ----------------
    def getTemperature(self):
        return self.getGlobalVariableByName("kT") / MOLAR_GAS_CONSTANT * unit.kelvin

    def setTemperature(self, temperature, rescale_theta_velocities=True):
        """Set the temperature, rescaling vtheta to it. OpenMM's replica-exchange sampler rescales the atom
        velocities on a temperature swap, restores this integrator's state first and then calls this method,
        so kT still holds the replica's old temperature here and theta gets the same scaling."""
        kT = MOLAR_GAS_CONSTANT * _value(temperature, unit.kelvin)
        if rescale_theta_velocities:
            scale = math.sqrt(kT / self.getGlobalVariableByName("kT"))
            for b, _ in self._numbered():
                name = f"vtheta{b}"
                self.setGlobalVariableByName(name, scale * self.getGlobalVariableByName(name))
        self.setGlobalVariableByName("kT", kT)

    # ---- theta state ----------------------------------------------------------------------------------
    def getTheta(self):
        return [self.getGlobalVariableByName(f"theta{b}") for b, _ in self._numbered()]

    def setTheta(self, theta):
        for (b, _), value in zip(self._numbered(), theta):
            self.setGlobalVariableByName(f"theta{b}", float(value))
        self.setGlobalVariableByName("project_lambdas", 1.0)

    def getThetaVelocities(self):
        return [self.getGlobalVariableByName(f"vtheta{b}") for b, _ in self._numbered()]

    def setThetaVelocities(self, velocities):
        for (b, _), value in zip(self._numbered(), velocities):
            self.setGlobalVariableByName(f"vtheta{b}", float(value))

    def setThetaVelocitiesToTemperature(self, seed=None):
        rng = np.random.default_rng(seed)
        kT = self.getGlobalVariableByName("kT")
        for b, block in self._numbered():
            v = 0.0 if block.fixed else rng.standard_normal() * math.sqrt(kT / block.theta_mass)
            self.setGlobalVariableByName(f"vtheta{b}", v)

    def getLambdas(self):
        """Lambda per block from the current theta; index 0 is block 1."""
        theta = self.getTheta()
        out = [0.0] * len(theta)
        for js in self._sites.values():
            w = [math.exp(self._fnex * math.sin(theta[j - 1])) for j in js]
            for j, wj in zip(js, w):
                out[j - 1] = wj / sum(w)
        return out

    def initializeLambdas(self, context):
        """Push lambda(theta) into the Context now instead of at the next step."""
        for b, value in enumerate(self.getLambdas(), start=1):
            context.setParameter(f"lambda{b}", value)
        self.setGlobalVariableByName("project_lambdas", 0.0)

    def invalidate(self):
        """Recompute dU/dlambda at the next step. Call after changing, from outside, anything it depends on
        that the step does not detect: a Context parameter not in watch_parameters, or the box alone."""
        self.setGlobalVariableByName("project_lambdas", 1.0)

    def getThetaKineticEnergy(self):
        return sum(0.5 * block.theta_mass * self.getGlobalVariableByName(f"vtheta{b}") ** 2
                   for b, block in self._numbered() if not block.fixed)

    def getThetaTemperature(self):
        n = len(self._movable())
        return 2.0 * self.getThetaKineticEnergy() / (n * MOLAR_GAS_CONSTANT) if n else 0.0
