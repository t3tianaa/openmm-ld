# System preparation (stage P2)

`openmm_ld.prep` turns input files into an OpenMM System and a λ model. It is ported from OpenMM-MSLD's `msld/prep` with
the code unchanged, apart from `save()` also writing what `openmm_ld.load_prepared` reads. It needs ParmEd, RDKit and
SciPy in the environment, as it did there.

```bash
python examples/prepare_mdm2.py prepared/     # from the raw inputs in examples/mdm2
python examples/run_mdm2.py prepared/ out/    # minimize, ALF, production
```

```python
from openmm_ld.prep import Site, build_msld_system, ingest

environment = ingest("protein.pdb", forcefield=["amber14-all.xml"], add_hydrogens=True)
bundle = build_msld_system(environment=environment,
                           sites=[Site(residue=3, variants={"SER": "SER_var.pdb"}, mapper="backbone")],
                           forcefield=["amber14-all.xml"], renormalize=True, solvate=True)
bundle.save("prepared/")
```

## Seed it if you need the same start twice

Prep isn't reproducible without a seed. OpenMM's `Modeller.addHydrogens` places some hydrogens at random before minimizing
them, using Python's `random` module. On MDM2 only two atoms are affected, THR86:H2 and H3 on the peptide's N-terminus. Across
repeated builds they land in one of two positions about 0.14 nm apart, and within one of those vary by up to about 2e-4 nm.
That is probably the same geometry with the two hydrogens' labels swapped; I haven't confirmed it. OpenMM-MSLD behaves the same way.

Call `random.seed(n)` before the build for a repeatable result, as `examples/prepare_mdm2.py` does.

## Checked against OpenMM-MSLD's prep

`python/tests/compare_prep.py`, each build with `random.seed(0)`, on MDM2 (three times, each in a new process) and on the
peptide alone, dry (`test_prep_matches_msld.py`):

| What | Result |
|---|---|
| System XML: forces, parameters, exceptions, constraints, box | Byte-identical |
| Positions | Exactly equal |
| Mapping, λ model | Equal |
| Every file `save()` writes in OpenMM-MSLD | Byte-identical |

MDM2: 26,305 atoms, 33,934 exceptions, 24,624 constraints. Prep takes about 5 s on either.

The 11 prep test files came along with the code (`python/tests/prep/`, 101 tests). One changed in substance: same-site exclusion
is now checked through the plugin's blocks and sites, since there is no `NonbondedForce` after scaling. Others changed only
imports, the path to the inputs and the `LambdaDynamics` constructor. The ligand path (`MCSMapper`, SDF inputs) is covered by
those tests, not by a comparison with OpenMM-MSLD; the code is unchanged.

## Limits

This shows the port is faithful, not that prep is right. Any problem in OpenMM-MSLD's prep is here too.
