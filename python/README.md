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
