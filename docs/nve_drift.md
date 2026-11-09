# NVE energy drift on MDM2

**Conclusion: at equilibrium there is no meaningful drift.** Over 10 ps, plugin force, `MSLDIntegrator`, θ free, 1 fs, zero friction, the
slope of total energy is −0.10 ± 0.14 kJ/mol/ps, out of about −268,000 kJ/mol. At one endpoint, λ fixed, it's +0.22 ± 0.13. The 22–25 kJ/mol/ps measured earlier was a start-up transient: those runs started from a
minimized geometry and fitted only the first fractions of a picosecond.

Script: `diagnostics/drift_mdm2.py`. Everything is from one frame equilibrated for 8 ps (friction 5 ps⁻¹, T_atoms about 300 K).

| Run (1.5 ps, drift fitted after 0.2 ps) | dt (fs) | Drift (kJ/mol/ps) | Fluctuation rms (kJ/mol) |
|---|---|---|---|
| Endpoint (λ exactly 1 and 0), plugin, Verlet | 1 | +2.0 | 4.2 |
| Same endpoint, stock `NonbondedForce` | 1 | −3.1 | 4.4 |
| MSLDIntegrator, θ free | 1 | +0.8 | 4.6 |
| MSLDIntegrator, θ frozen | 1 | +1.7 | 4.7 |
| Endpoint, LJ switching from 0.9 nm | 1 | −5.2 | 5.3 |
| Endpoint, Ewald tolerance 1e-5 | 1 | −1.4 | 2.7 |
| Endpoint, switching and Ewald 1e-5 | 1 | −1.4 | 3.0 |
| θ free, switching and Ewald 1e-5 | 1 | −2.4 | 3.0 |
| θ frozen, switching and Ewald 1e-5 | 1 | −1.6 | 2.9 |
| Endpoint | 0.5 | +3.4 | 3.6 |
| θ free | 0.5 | +4.7 | 4.3 |
| Endpoint | 2 | +1.4 | 21.7 |
| Endpoint, constraint tolerance 1e-10 | 1 | +2.0 | 4.2 |

Estimated errors on these slopes are about ±1 to ±2.5 kJ/mol/ps, so they don't separate from zero or from each other. The 10 ps runs are what settle it.

## What each comparison shows

- **Plugin against stock OpenMM at one endpoint:** same behaviour, so nothing here is specific to the plugin. Stock runs on the CPU platform in single precision.
- **θ:** with atoms frozen and θ moving, total energy stays within 0.002 kJ/mol rms while the potential energy swings by 7. At 0.25 fs it's 0.0001. The chain rule and θ's integration are right.
- **Neighbour list:** with no padding, rebuilt at every step, total energy differs from the padded run by at most 0.01 kJ/mol over 1 ps.
- **Constraint tolerance:** no visible effect.
- **Time step:** the fluctuation barely changes from 1 to 0.5 fs but is about 5 times larger at 2 fs. So at 1 fs integration error isn't what limits it.
- **Ewald tolerance:** 1e-5 brings the fluctuation from about 4.5 to about 3 kJ/mol. LJ switching alone doesn't help. So the cutoff in PME real space seems to set the fluctuation, at the default tolerance of 5e-4.
- **Forces and dU/dλ:** dU/dλ matches a central difference of the energy to about 5 significant figures at an MD frame. The force check along a direction is noisy at about 0.005 kJ/mol in the energy. A rough estimate of pairs crossing the unswitched 1.0 nm cutoff gives that size (not measured directly).

## The start-up transient

From a just-minimized start (4 ps, same settings):

| Time (ps) | 0–0.05 | 0.05–0.2 | 0.2–0.5 | 0.5–1 | 1–2 | 2–3 | 3–4 |
|---|---|---|---|---|---|---|---|
| Slope (kJ/mol/ps) | +1,253 | +82 | +2.9 | −14.9 | −5.8 | +3.9 | −0.7 |

Total energy jumps about 120 kJ/mol in the first 0.1 ps and then settles. This happens in both integrators and with both nonbonded builds. The mechanism isn't pinned down: it happens while
the system leaves the minimized geometry and λ moves fast away from uniform. Measure after equilibrating.

## Limits

One equilibrated frame, 10 ps. Slopes of successive 2.5 ps windows in those runs range from −1.3 to +1.4 kJ/mol/ps. A drift much below 0.3 kJ/mol/ps can't be ruled out, and a slow one would show only in longer runs. I didn't check how often the variants switched during these runs.
