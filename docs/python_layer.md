# The Python layer (stage P1)

`openmm_ld/` runs MSLD from a prepared system without OpenMM-MSLD. Example: `examples/run_mdm2.py`.

```python
from openmm_ld import LambdaDynamics, load_prepared
from openmm_ld.alf import add_flattening_biases, flatten

prepared = load_prepared("prepared_dir")      # hybrid_system.xml, _mapping.json, _start.pdb, _positions.npy
model = prepared.to_model()
add_flattening_biases(model)
model.scale_system(prepared.system)
dyn = LambdaDynamics(prepared.system, model, prepared.positions, box=prepared.box, seed=0)
dyn.minimize(max_iterations=200)
report = flatten(dyn, n_iters=30, steps_per_iter=20000)
dyn.run(100000, metrics="metrics.csv", mapping=prepared.mapping, trajectory="traj.dcd", structure=prepared)
```

## Porting a script from OpenMM-MSLD

Module and function names are the same (`msld.alf` becomes `openmm_ld.alf`, and so on). What changes:

| | OpenMM-MSLD | openmm_ld |
|---|---|---|
| Starting a run | Build a `Context`, then `LambdaDynamics(ctx, model, ...)` | `LambdaDynamics(system, model, positions, ...)` builds it; the Context is `dyn.context` |
| Nonbonded | Stock forces, or `use_plugin` | Always `MSLDNonbondedForce`; the switch is gone |
| dU/dλ | Analytic plus a finite difference | All analytic |
| Force groups | Split by derivative type | Not used |
| Platform | Any | Reference and CPU until the GPU kernels exist (stage S2) |
| `dyn.x`, `dyn.vx`, `dyn.theta` | Python arrays updated in place | Read from and written to the Context; assign a whole array, don't modify one in place |

`dyn.x` and `dyn.vx` copy every atom on each read, so read them at reporter intervals, not per step.

## Checked against OpenMM-MSLD

| What | Result |
|---|---|
| Scaling on the toy: no cutoff, PME, reaction field, NBFIX, impropers, bonded scaling on and off | Energy, dU/dλ, forces within 1e-10 |
| Scaling on MDM2, 4 partial-λ points | Exactly equal |
| Against the NumPy oracle, without OpenMM-MSLD | Within 5e-8 (the oracle's rounded Coulomb constant) |
| ALF on a fixed set of λ trajectories | Identical bias and coupling histories, ΔΔG estimates, Context parameters |
| Metrics writer | Identical files |
| Dynamics from one seed, zero friction, 50 steps | Identical start; positions and θ within 1e-9 |

These show that the port is faithful, not that the energy is right. See ROADMAP.md, rule 2.

## End-to-end run on MDM2

`examples/run_mdm2.py`, CPU platform, 26,305 atoms: minimization capped at 200 iterations (452 s), 2 ALF iterations of
100 steps (82 s), 1,000 production steps (0.40 s/step), metrics and DCD written. The script asserts OpenMM-MSLD was not
imported. This shows that the pieces work together; it's far too short for any science. The prepared input comes from
`examples/prepare_mdm2.py` ([prep.md](prep.md)).
