"""openmm_ld.prep against OpenMM-MSLD's msld.prep: the same inputs through both must give the same result.

Compares the System's XML byte for byte (forces, parameters, exceptions, constraints, box), positions, mapping,
lambda model and every file save() writes.

Each build starts with random.seed(0), because OpenMM's Modeller.addHydrogens places hydrogens at random before it
minimizes them, so prep isn't reproducible without a seed (see docs/prep.md).

Usage (MDM2): python compare_prep.py MSLD_CHECKOUT
"""
import filecmp
import json
import os
import random
import sys
import tempfile
import time
import warnings

import numpy as np
import openmm as mm
from openmm import unit

MDM2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "mdm2")


def build_mdm2(prep, directory=MDM2):
    random.seed(0)
    path = lambda name: os.path.join(directory, name)
    protein = prep.ingest(path("mdm2_protein.pdb"), forcefield=["amber14-all.xml"], add_hydrogens=True)
    peptide = prep.ingest(path("mdm2_peptide.pdb"), forcefield=["amber14-all.xml"], add_hydrogens=True)
    for r in peptide.residues:
        r.chain = "B"
    return prep.build_msld_system(
        environment=protein + peptide,
        sites=[prep.Site(residue=len(protein.residues) + 3,
                         variants={v: path(f"{v}_var.pdb") for v in ("ILE", "LEU", "SER", "VAL")}, mapper="backbone")],
        forcefield=["amber14-all.xml"], renormalize=True, solvate=True)


def build_peptide(prep, directory=MDM2):
    """The peptide alone, dry, with the same site and variants: the small case."""
    random.seed(0)
    path = lambda name: os.path.join(directory, name)
    peptide = prep.ingest(path("mdm2_peptide.pdb"), forcefield=["amber14-all.xml"], add_hydrogens=True)
    return prep.build_msld_system(
        environment=peptide,
        sites=[prep.Site(residue=3, variants={v: path(f"{v}_var.pdb") for v in ("ILE", "LEU", "SER", "VAL")}, mapper="backbone")],
        forcefield=["amber14-all.xml"], renormalize=True, solvate=False)


def compare(new, old):
    """Every difference between two bundles, as a list of strings (empty = identical)."""
    differences = []
    if mm.XmlSerializer.serialize(new.system) != mm.XmlSerializer.serialize(old.system):
        differences.append("System XML differs")
    xn = np.array(new.structure.positions.value_in_unit(unit.nanometer))
    xo = np.array(old.structure.positions.value_in_unit(unit.nanometer))
    if xn.shape != xo.shape or not np.array_equal(xn, xo):
        differences.append(f"positions differ, max {np.max(np.abs(xn - xo)) if xn.shape == xo.shape else 'shape'}")
    if tuple(new.structure.box if new.structure.box is not None else ()) != tuple(old.structure.box if old.structure.box is not None else ()):
        differences.append("box differs")
    if json.loads(new.mapping.to_json()) != json.loads(old.mapping.to_json()):
        differences.append("mapping differs")
    mn, mo = new.to_model(), old.to_model()
    if mn.atom_block != mo.atom_block or [(b.site, b.atoms) for b in mn.blocks] != [(b.site, b.atoms) for b in mo.blocks]:
        differences.append("lambda model differs")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        pn, po = new.save(d1), old.save(d2)
        for key in po:
            if not filecmp.cmp(pn[key], po[key], shallow=False):
                differences.append(f"saved file differs: {key}")
    return differences


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    import msld.prep as old_prep
    import openmm_ld.prep as new_prep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.time(); new = build_mdm2(new_prep); t_new = time.time() - t
        t = time.time(); old = build_mdm2(old_prep, os.path.join(sys.argv[1], "test_systems", "mdm2")); t_old = time.time() - t
    print(f"atoms {new.system.getNumParticles()}, forces {new.system.getNumForces()}, "
          f"exceptions {[f.getNumExceptions() for f in new.system.getForces() if isinstance(f, mm.NonbondedForce)]}, "
          f"constraints {new.system.getNumConstraints()}; build {t_new:.1f} s (openmm_ld), {t_old:.1f} s (msld)")
    differences = compare(new, old)
    print("identical" if not differences else differences)
