"""Prepare MDM2 from the raw inputs in examples/mdm2 with openmm_ld.prep; run it with run_mdm2.py.

Usage: python prepare_mdm2.py OUTDIR
"""
import os
import random
import sys
import time

from openmm_ld.prep import Site, build_msld_system, ingest

MDM2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdm2")
out = sys.argv[1]
path = lambda name: os.path.join(MDM2, name)

random.seed(0)     # addHydrogens places hydrogens at random: this makes the result reproducible
t = time.time()
protein = ingest(path("mdm2_protein.pdb"), forcefield=["amber14-all.xml"], add_hydrogens=True)
peptide = ingest(path("mdm2_peptide.pdb"), forcefield=["amber14-all.xml"], add_hydrogens=True)
# Both input PDBs are chain A, but the peptide is a separate binding partner: put it on its own chain, or
# viewers draw a spurious peptide bond across the gap between the protein C-terminus and the peptide.
for r in peptide.residues:
    r.chain = "B"
bundle = build_msld_system(
    environment=protein + peptide,
    sites=[Site(residue=len(protein.residues) + 3,
                variants={v: path(f"{v}_var.pdb") for v in ("ILE", "LEU", "SER", "VAL")}, mapper="backbone")],
    forcefield=["amber14-all.xml"], renormalize=True, solvate=True)
paths = bundle.save(out)
assert not any(m == "msld" or m.startswith("msld.") for m in sys.modules), "imported OpenMM-MSLD"
print(f"prepared in {time.time() - t:.1f} s: {bundle.system.getNumParticles()} atoms, "
      f"blocks {bundle.mapping.sites[0].block_names}, without importing OpenMM-MSLD; wrote {sorted(paths)} to {out}")
