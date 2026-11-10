# Python wrappers (SWIG)

Module `msldplugin`, built by the top-level CMake (`MSLD_BUILD_PYTHON_WRAPPERS=ON`, the default) into
`build/python`. `cmake --install build` puts it in the active Python's site-packages.

Without installing, point Python at the build tree and load the Reference/CPU kernels explicitly:

```python
import msldplugin                     # PYTHONPATH=build/python
msldplugin.loadReferencePlugin("build/platforms/reference/libOpenMMMSLDReference.dylib")
```

`tests/test_oracle_toy.py` rebuilds the OpenMM-MSLD oracle toy systems (`tests/data/oracle_toy.json`,
from OpenMM-MSLD PR #6) with `MSLDNonbondedForce` plus stock custom forces for bonded terms, biases and
restraints, and checks energy and dU/dlambda for every case. It runs as the `PythonTests` ctest.

`tests/test_python_engine.py` compares the plugin with the OpenMM-MSLD Python engine (`msld/forces.py`,
PR #6 branch) at partial lambda for PME, reaction field (periodic and non-periodic, with switching)
and CHARMM NBFIX tables, on the oracle toy in a periodic box. It needs an OpenMM-MSLD checkout:
`cmake -DMSLD_ENGINE_PATH=/path/to/OpenMM-MSLD build` (skipped otherwise). The engine's within-block
reaction-field correction uses the wrong screening function (erfc for CutoffPeriodic, bare 1/r for
CutoffNonPeriodic), so for those methods the test removes that known engine error before comparing.

`tests/test_ewald_reference.py` compares the plugin's PME at partial lambda with a plain Ewald sum written
from the energy spec (`tests/ewald_reference.py`), which shares no code with OpenMM. The reference is tested
on its own first: the NaCl Madelung constant, independence of alpha where theory requires it, and
convergence of its k-space sum.

`tests/test_msld_integrator.py` covers `openmm_ld.MSLDIntegrator` (see `docs/msld_integrator.md`). Its step-by-step
comparison with `LambdaDynamics` needs `MSLD_ENGINE_PATH`. `tests/measure_theta_temperature.py` and
`tests/compare_mdm2_integrator.py` are the longer measurements behind that note.

`tests/test_python_layer.py` checks the `openmm_ld` Python layer against the NumPy oracle and, with `MSLD_ENGINE_PATH`,
against OpenMM-MSLD. `tests/compare_mdm2_layer.py` does the same comparison on MDM2.

`tests/prep/` holds the prep tests. `tests/test_prep_matches_msld.py` and `tests/compare_prep.py` compare `openmm_ld.prep`
with OpenMM-MSLD's prep.
