# MSLDIntegrator: design note

Stage S3 of [ROADMAP.md](../ROADMAP.md). Code: `openmm_ld/integrator.py`. Tests: `python/tests/test_msld_integrator.py`.

## CustomIntegrator, not a new C++ class

OpenMM's `CustomIntegrator` can express the whole step, so S3 needs no new C++ for now. It also works on every
platform that has kernels for the System's forces, so it comes along when the force reaches the GPU (S2). What it
needed, each confirmed by the tests:

| Need | How | |
|---|---|---|
| θ, vθ, masses, frictions as state | Global variables `theta<b>`, `vtheta<b>`, `mtheta<b>`, `gamma_theta<b>` | Part of `State.IntegratorParameters`, so `getState`/`setState` keep them |
| λ = softmax(fnex·sin θ) each step | `addComputeGlobal("lambda<b>", ...)`: a CustomIntegrator may assign Context parameters | Forces are recomputed once after the assignment |
| dU/dλ | `deriv(energy, lambda<b>)` | Comes from `MSLDNonbondedForce` and the stock forces' `addEnergyParameterDerivative` |
| Constraints | `addConstrainPositions`, `addConstrainVelocities` | OpenMM's own solver instead of the Python SHAKE |
| Barostat | `addUpdateContextState` | Fires like with any integrator |
| Replica exchange | `getTemperature()`, `setTemperature()` on the Python class | What `app.ReplicaExchangeSampler` looks for |

A C++ class would still be justified if the expression evaluator becomes a bottleneck on the GPU; there's no sign of that yet.

**A limit of OpenMM found on the way:** with several `deriv()` calls in one expression, OpenMM 8.6's Reference
implementation gives them all the value of the first. Each derivative therefore goes into its own global variable first.

**A requirement this creates:** every λ-dependent force must report dU/dλ. A force that doesn't, such as a
`NonbondedForce` with charge offsets, would act on the atoms but exert no force on θ, with no error. So this integrator
goes with the plugin's force and not with the stock nonbonded build of OpenMM-MSLD.

## The step

VRORV Langevin over atoms and θ together, matching `LambdaDynamics` in `msld/integrator.py`:

1. **B:** v += dt/2·f/m, constrain velocities; vθ += dt/2·Fθ/mθ
2. **A:** x += dt/2·v with SHAKE and its impulse into v; θ += dt/2·vθ
3. **O:** v = a·v + b·ξ, constrain velocities; vθ likewise, with its own friction
4. **A:** as step 2, then λ = softmax(fnex·sin θ) per site
5. **B:** as step 1, at the new state

Here Fθ = −λ_b·fnex·cos θ_b·(dU/dλ_b − Σ_{j∈site} λ_j·dU/dλ_j). Biases and restraints stay ordinary forces. Fixed blocks
keep their θ but stay in the softmax.

**One force evaluation per step (counted by a test), which takes a cache.** OpenMM keeps forces valid between steps, but
apparently not the energy, so asking for `deriv(energy, ...)` at the start of a step evaluates everything again. So dU/dλ
from the end of one step is stored in global variables and reused at the start of the next. The step recomputes it if
atoms moved since (`setPositions`, `setState`, a barostat) or a λ changed. Anything else needs `watch_parameters=[...]`
at construction or a call to `invalidate()`. The risk: changing a bias parameter with neither would give one stale half
kick on θ, with no error. `LambdaDynamics.invalidate_forces` has the same contract.

## Validation

All on the oracle toy (24 atoms, 5 blocks, 2 sites, no cutoff) unless noted.

| Check | Result |
|---|---|
| Against `LambdaDynamics`, friction 0, same start, 100 steps | Positions and θ within 1e-10 at every step; with constraints, within 2e-5 (different constraint solvers, both at tolerance 1e-10) |
| Energy conservation, friction 0, 0.1 ps | Spread (std) of total energy 1.76, 0.31, 0.094 kJ/mol at dt 1, 0.5, 0.25 fs, out of about 930: second order in dt |
| Force evaluations | Exactly one per step, counted |
| Outside changes | `setPositions`, a watched parameter, `setTheta`: the next step matches a forced recompute to 1e-12 |
| `getState`/`setState` | θ, vθ and λ restored exactly |
| Barostat | Volume changes over 200 steps. That it fires, not that the pressure is right |
| `ReplicaExchangeSampler` | Runs 5 iterations with two temperatures. Exchange statistics not checked |
| NVT at friction 1 ps⁻¹, 5 runs of 2 ns at 1 fs, or 1 ns at 0.5 fs | See below |

| dt | T_θ (K) | T_atoms (K) |
|---|---|---|
| 1 fs | 301.6 ± 1.3 | 298.2 ± 0.4 |
| 1 fs, repeat | 297.8 ± 2.8 | 299.0 ± 0.6 |
| 0.5 fs | 303.3 ± 1.4 | 299.7 ± 0.9 |

(mean ± standard error over the 5 runs; `python/tests/measure_theta_temperature.py`.) T_θ is 300 K within about 1%. The
0.5 fs value is 2.4 standard errors high, which more runs would have to settle. T_atoms is low by about 0.5% at 1 fs and
not at 0.5 fs, which looks like the usual timestep effect.

**MDM2 (26,305 atoms, PME, CPU platform, friction 0, 400 steps), against `LambdaDynamics` with `use_plugin`:** λ agrees to
1.4e-7, θ to 2.1e-6, positions to 7.9e-4 nm (constraint tolerance 1e-6, different solvers). The energy drift is the
same as the old integrator's: 22.4 kJ/mol/ps after 0.1 ps. Time per step: 0.396 s against 0.429 s. Almost all of that is the
force evaluation (about 0.39 s), so the integrator's own cost is now small but the force still runs on one thread.

## Seeds and energy reads

Asking the Context for the energy advances the integrator's random stream, in this CustomIntegrator on the Reference and
CPU platforms. (Measured: one read before a run changes the noisy trajectory, and at zero friction it changes nothing.
Probably the kinetic-energy evaluation drawing unused random numbers; not confirmed in OpenMM's source.) So it doesn't
affect the physics, but one seed gives one trajectory only if the energy is read at the same points. A reporter that reads the
energy more often gives a different, equally valid, trajectory.

## What it doesn't do yet

- A `setState` from a temperature swap rescales the atom velocities in the sampler, but not vθ.
- The drift above is unexplained in both integrators, and this one inherits it. It needs a check from an equilibrated start.
- The Python class's extra methods are lost through `XmlSerializer`, which returns a plain `CustomIntegrator`. The step
  itself survives.
