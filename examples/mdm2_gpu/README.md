# MDM2 peptide site, GPU run

Two legs, same site and variants: `complex` (MDM2 + peptide) and `free` (peptide alone).
ΔΔG_bind(WT→X) = ΔG_complex(WT→X) − ΔG_free(WT→X).

```bash
conda activate msld                     # OpenMM 8.6.1 with CUDA (conda-forge), numpy, scipy, parmed, pdbfixer
pip install --no-deps -e ../..          # the openmm-ld Python layer (enough for the stock route; see below for the fast kernel)
./run_gpu.sh free    > free.out 2>&1 &
./run_gpu.sh complex > complex.out 2>&1 &
# progress: grep '\[ALF\]' out_*/alf.log
# after production of both legs:
openmm-ld ddg out_complex/run/dG.json out_free/run/dG.json ddg.json
```

## Fast route: build the MSLD GPU kernel (once, ~2 minutes)

The plugin now has a CUDA/OpenCL kernel for `MSLDNonbondedForce`. On the same device it ran 14x faster than
the stock-force route in our tests, with energies, forces and dU/dλ matching the Reference kernel
(mixed precision: energy within 0.13 kJ/mol of −74000, dU/dλ within 1e-6 relative). Kernels are compiled at run time
by OpenMM, so only `cuda.h` and `libcuda` are needed (your CUDA 13.1 toolkit has both).

```bash
conda activate msld
conda install -c conda-forge cmake swig make cxx-compiler     # build tools, if not already there
cd openmm-ld && mkdir -p build && cd build
CUDA_HOME=/usr/local/cuda cmake ..                           # look for "MSLD CUDA kernels: cuda.h in ..."
make -j20 install                                            # libOpenMMMSLD{,Reference,CUDA,OpenCL} + msldplugin
cd .. && pip install --no-deps .                             # the openmm-ld command
python -c "import openmm; print(openmm.Platform.getPluginLoadFailures())"   # should not list libOpenMMMSLDCUDA
```

Then, after `prepare`, check the kernel on your GPU (a few minutes; it changes no files):

```bash
cd out_free && openmm-ld check-gpu free.toml
```

It prints the energy, force and dU/dλ differences against the Reference kernel at three λ sets, the speed of both
routes, and `kernel AGREES` or `DISAGREES`. `run_gpu.sh` runs this check itself for a new leg and uses the kernel
when it agrees; otherwise it uses the stock route. The choice is stored in `out_<leg>/nonbonded.choice`, and a leg
that already started ALF on the stock route stays on it (do not switch a leg half way). Force a route with
`NONBONDED=plugin ./run_gpu.sh free` or `NONBONDED=stock ...`.

After pulling new code, run `make -j20 install` in `build/` again.

What to watch in `alf.log`: `scaling` should approach 1, `frac(l>0.99)` should become non-zero for every
variant, and `b` should stop drifting. T_theta should stay near 300 K.
