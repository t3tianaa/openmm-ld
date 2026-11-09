"""Write MDM2 as a prepared system for openmm_ld.load_prepared, using OpenMM-MSLD's prep.

A stopgap until prep is ported (stage P2). It is the only part of an MDM2 run here that still needs OpenMM-MSLD.

Usage: python export_prepared_from_msld.py MSLD_CHECKOUT OUTDIR
"""
import os
import sys
from pathlib import Path

engine, out = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, engine)
import numpy as np
import openmm as mm
from openmm import app, unit
from msld.prep import ingest, build_msld_system, Site

MDM2 = Path(engine) / "test_systems" / "mdm2"
out.mkdir(parents=True, exist_ok=True)
protein = ingest(MDM2 / "mdm2_protein.pdb", forcefield=["amber14-all.xml"], add_hydrogens=True)
peptide = ingest(MDM2 / "mdm2_peptide.pdb", forcefield=["amber14-all.xml"], add_hydrogens=True)
for r in peptide.residues:
    r.chain = "B"
bundle = build_msld_system(
    environment=protein + peptide,
    sites=[Site(residue=len(protein.residues) + 3,
                variants={v: MDM2 / f"{v}_var.pdb" for v in ("ILE", "LEU", "SER", "VAL")}, mapper="backbone")],
    forcefield=["amber14-all.xml"], renormalize=True, solvate=True)
(out / "hybrid_system.xml").write_text(mm.XmlSerializer.serialize(bundle.system))
(out / "hybrid_mapping.json").write_text(bundle.mapping.to_json(indent=1))
positions = bundle.structure.positions
np.save(out / "hybrid_positions.npy", np.array(positions.value_in_unit(unit.nanometer)))
with open(out / "hybrid_start.pdb", "w") as fh:
    app.PDBFile.writeFile(bundle.structure.topology, positions, fh)
print(f"wrote {out}: {bundle.system.getNumParticles()} atoms, blocks {bundle.mapping.sites[0].block_names}")
