# The openmm-ld command (stage P3)

One command, one config file, from input PDBs to a production run. OpenMM-MSLD had no command line or config file to port;
its runs were Python scripts. So this is new, and it wraps the library calls those scripts made.

```bash
openmm-ld prepare examples/mdm2/mdm2.toml     # input files -> prepared/
openmm-ld alf     examples/mdm2/mdm2.toml     # minimize, ALF -> alf/biases.json, alf/state.xml
openmm-ld run     examples/mdm2/mdm2.toml     # production -> run/metrics.csv, run/traj.dcd
openmm-ld all     examples/mdm2/mdm2.toml     # alf and run in one process
```

Outputs go next to the config file, or under `--workdir DIR`. `--set run.steps=100` overrides any value.

**Install:** build and install the plugin with CMake (see the README), then `pip install --no-deps .` in the same conda
environment. Dependencies come from conda. Without installing, `python -m openmm_ld.cli` does the same, with `PYTHONPATH`
including the repository and `build/python`, and `MSLD_REFERENCE_PLUGIN` pointing at the kernel library.

## Config (TOML)

| Section | Keys (default) |
|---|---|
| `[prepare]` | `forcefield`, `output` ("prepared"), `seed` (0), `renormalize` (true), `solvate` (true) |
| `[[prepare.environment]]` | One per input: `file`, `add_hydrogens` (false), `chain`, `forcefield` |
| `[[prepare.site]]` | One per site: `residue` (from 0, within environment file `part`), `part` (0), `variants` (name = file), `mapper` ("backbone") |
| `[system]` | `scale_bond`, `scale_angle`, `scale_torsion` (true), `softcore_alpha` (0.5), `softcore_sigma_q` (0.3), `fnex` (5.5) |
| `[dynamics]` | `temperature` (300), `timestep` (0.001), `atom_friction` (1), `theta_mass` (0.05), `theta_friction` (1), `platform` ("CPU"), `threads` (4), `seed` (0) |
| `[minimize]` | `max_iterations` (0: to convergence) |
| `[alf]` | `output` ("alf"), `iterations` (20), `steps_per_iter` (20000), `sample_every` (100), `rate` (1), `couple` (true), `max_step` (none), `tol` (0.05) |
| `[run]` | `output` ("run"), `steps` (10000), `start` ("alf" or "prepared"), `metrics_every` (100), `trajectory_every` (1000) |

An unknown section or key is an error, so a typo can't silently fall back to a default.
`examples/mdm2/mdm2.toml` is the worked example, with settings far too short for science.

`run` with `start = "alf"` continues from the positions, velocities, θ and biases `alf` saved, with a new random stream. So
`alf` then `run` is as valid as `all`, but gives a different trajectory from it after the restart.

## Checked on MDM2

`python/tests/compare_cli_mdm2.sh`, same seed, CPU platform:

| Check | Result |
|---|---|
| `prepare` against `examples/prepare_mdm2.py` | All 6 files byte-identical |
| `all` against `examples/run_mdm2.py` (20 minimization iterations, 2 ALF iterations of 50 steps, 200 production steps) | `metrics.csv` byte-identical; DCD identical apart from its header's creation time |
| `alf` alone against `all` | `biases.json` byte-identical |
| `alf` then `run` | Completes, at 0.41 s/step, from the restored state |

The command was run as the installed console script, from a `pip install --prefix` into a scratch folder.

`python/tests/test_cli.py` does the same on the peptide alone, and checks that θ and the biases come back on restart.
