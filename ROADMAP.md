# Roadmap: openmm-ld as the standalone MSLD package

Decision: one package, a C++ core plus a Python layer, covering everything from system preparation to
production. [OpenMM-MSLD](https://github.com/t3tianaa/OpenMM-MSLD) is the reference implementation during the
port and then retires (ADR-0046 in its vault).

## Target layout

```
openmm-ld/
  openmmapi/            C++ API: MSLDNonbondedForce (exists)
  platforms/reference/  double-precision kernels, also serving the CPU platform (exists)
  platforms/common/     GPU kernels: OpenCL, CUDA, HIP (S2)
  serialization/        XML proxies (exists)
  python/               SWIG module msldplugin (exists)
  openmm_ld/            Python layer (P1-P3)
    integrator.py       MSLDIntegrator: atoms + theta, a CustomIntegrator (exists)
    model.py            blocks, sites, theta -> lambda
    forces.py           lambda-scaled bonded terms, ALF biases, restraints (stock OpenMM custom forces)
    alf.py              adaptive landscape flattening
    io.py               metrics, trajectories, populations
    prep/               ingest, hybrid building, solvation, clash handling
    cli.py              command line: PDB to running system
```

## Port order

| Stage | What | Done when | State |
|---|---|---|---|
| S0, S1 | Nonbonded force, Reference and CPU | Hand calculation, NumPy oracle, `NonbondedForce` at lambda = 1, engine, Ewald sum; BLaDE single points still to do | Done except BLaDE |
| S3 | `MSLDIntegrator` as a `CustomIntegrator` in `openmm_ld/integrator.py` ([design and results](docs/msld_integrator.md)) | T_theta = 300 K, barostat fires, energy conserved at zero friction, state survives `setState`; matches `LambdaDynamics` | Done on the toy; pressure and exchange statistics not checked |
| P1 | `model`, `forces`, `alf`, `io` | Same forces and same ALF updates as OpenMM-MSLD on the same inputs | Not started |
| P2 | `prep` | Same hybrid System as OpenMM-MSLD from the same inputs | Not started |
| P3 | `cli` | MDM2 runs from one command with OpenMM-MSLD uninstalled | Not started |
| S2 | GPU kernels | Equal to Reference within single-precision tolerance | Not started; can run alongside P1-P3 |

## Rules

1. **A module retires from OpenMM-MSLD only after its port here is checked against it on the same inputs.**
2. **Matching OpenMM-MSLD does not validate the physics.** The energy spec was written from that engine, so
   anything wrong in the spec passes that comparison. Independent references stay required: the Ewald sum in
   `python/tests/`, and BLaDE single points (still to do).
3. Until the integrator and Python layer exist, `use_plugin` in OpenMM-MSLD is the way to run the force in
   dynamics. It is a bridge, not the final design.
