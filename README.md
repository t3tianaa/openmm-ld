# openmm-ld
OpenMM pugin for lambda dynamics and multisite lambda dynamics

## Building (Reference/CPU platform)

```bash
conda activate msld            # OpenMM 8.6.1, cmake, swig from conda-forge
cmake -S . -B build            # OPENMM_DIR defaults to $CONDA_PREFIX
cmake --build build -j8
(cd build && ctest --output-on-failure)
cmake --install build          # libOpenMMMSLD -> $OPENMM_DIR/lib, libOpenMMMSLDReference -> $OPENMM_DIR/lib/plugins
```

`MSLDPlugin::MSLDNonbondedForce` implements the MSLD nonbonded energy (soft-core LJ + Coulomb, PME with
lambda-scaled reciprocal charges, 1-4 exceptions) and reports dU/dlambda_b for the global parameters
`lambda1`, `lambda2`, ... as energy parameter derivatives.
