# openmm-ld
OpenMM pugin for lambda dynamics and multisite lambda dynamics. See [ROADMAP.md](ROADMAP.md) for the plan.

## Building (Reference/CPU, CUDA and OpenCL platforms)

```bash
conda activate msld            # OpenMM 8.6.1, cmake, swig from conda-forge
cmake -S . -B build            # OPENMM_DIR defaults to $CONDA_PREFIX
cmake --build build -j8
(cd build && ctest --output-on-failure)
cmake --install build          # libOpenMMMSLD -> $OPENMM_DIR/lib, libOpenMMMSLD{Reference,CUDA,OpenCL} -> $OPENMM_DIR/lib/plugins
```

The CUDA library is built when `cuda.h` and `libcuda` are found (set `CUDA_HOME`; nvcc is not needed, OpenMM compiles
the kernels at run time), the OpenCL library when an OpenCL SDK is found. Force either with
`-DMSLD_BUILD_CUDA_LIB=ON|OFF`, `-DMSLD_BUILD_OPENCL_LIB=ON|OFF`. `openmm-ld check-gpu <config>` compares the GPU
kernel with the Reference kernel on a prepared system and times it against the stock-force route.

To install the `openmm-ld` command as well, run `pip install --no-deps .` in the same environment, then
`openmm-ld all examples/mdm2/mdm2.toml`. See [docs/cli.md](docs/cli.md).

`MSLDPlugin::MSLDNonbondedForce` implements the MSLD nonbonded energy (soft-core LJ + Coulomb, PME with
lambda-scaled reciprocal charges, 1-4 exceptions) and reports dU/dlambda_b for the global parameters
`lambda1`, `lambda2`, ... as energy parameter derivatives.
