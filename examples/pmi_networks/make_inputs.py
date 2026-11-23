"""Build the extra inputs the two PMI networks need, from files already in this directory.

    python make_inputs.py

Writes, next to this script:
  ALA_var.pdb, ASN_var.pdb, GLN_var.pdb,
  MET_var.pdb, THR_var.pdb                side-chain donor residues for [[prepare.site]], in the same
                                          format as the ILE/LEU/SER/VAL files already here: one internal
                                          residue, Amber atom names, N/H/CA/HA/C/O then the side chain.
                                          Taken from complete residues of mdm2_protein.pdb, so the naming
                                          and geometry match what ff14SB expects.
  pmi_L10A_peptide.pdb                    the exploration network's background peptide, TSFAEYWNLASP
                                          (PMI-L10A), made from ../mdm2_gpu/mdm2_peptide.pdb (TSFAEYWALLSP, PMI-N8A)
                                          by A8N and L10A with PDBFixer.

Re-running it overwrites the four files. Nothing else reads this script; the configs read its output.
"""
import os

from pdbfixer import PDBFixer
from openmm.app import PDBFile

HERE = os.path.dirname(os.path.abspath(__file__))
# The two input structures live in examples/mdm2_gpu; this directory does not copy them.
MDM2 = os.path.join(HERE, os.pardir, "mdm2_gpu")
# A complete residue of each type in mdm2_protein.pdb (chain A, author numbering of that file).
DONORS = {"ALA": 19, "ASN": 55, "GLN": 20, "MET": 26, "THR": 39}
# Atom order of the existing *_var.pdb files: backbone first, then the side chain from CB.
BACKBONE = ["N", "H", "CA", "HA", "C", "O"]


def read_residue(path, resname, resnum):
    atoms = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[17:20].strip() == resname and int(line[22:26]) == resnum:
                atoms.append((line[12:16], line[30:38], line[38:46], line[46:54], line[76:78]))
    if not atoms:
        raise SystemExit(f"{path}: no {resname} {resnum}")
    return atoms


def write_var(resname, atoms, path):
    order = {name: i for i, name in enumerate(BACKBONE)}
    atoms = sorted(atoms, key=lambda a: (order.get(a[0].strip(), len(BACKBONE)),))
    with open(path, "w") as fh:
        fh.write(f"HEADER    INTERNAL L-{resname} RESIDUE\n")
        fh.write(f"TITLE     {resname} RESIDUE EXTRACTED FROM mdm2_protein.pdb\n")
        fh.write("REMARK    NO OXT OR TERMINAL NH2/NH3 HYDROGENS ARE PRESENT.\n")
        fh.write("REMARK    BACKBONE N AND C HAVE OPEN VALENCES FOR PEPTIDE BONDS.\n")
        for i, (name, x, y, z, element) in enumerate(atoms, start=1):
            # Columns 13-16 as the PDB wants them, and as the ILE/LEU/SER/VAL files here already are:
            # a 4-character name stays put, a shorter one leaves column 13 to the element symbol.
            name = name.strip()
            name = name if len(name) == 4 else f" {name:<3}"
            fh.write(f"ATOM  {i:5d} {name} {resname} A   4    {x}{y}{z}  1.00  0.00          {element}\n")
        fh.write("END\n")
    print(f"wrote {os.path.basename(path)}: {len(atoms)} atoms")


def main():
    protein = os.path.join(MDM2, "mdm2_protein.pdb")
    for resname, resnum in DONORS.items():
        atoms = read_residue(protein, resname, resnum)
        write_var(resname, atoms, os.path.join(HERE, f"{resname}_var.pdb"))

    # PMI-N8A (TSFAEYWALLSP) -> PMI-L10A (TSFAEYWNLASP): put Asn back at 8, take Leu10 down to Ala.
    fixer = PDBFixer(filename=os.path.join(MDM2, "mdm2_peptide.pdb"))
    fixer.applyMutations(["ALA-8-ASN", "LEU-10-ALA"], "A")
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # a 12-mer with no gaps: never add residues
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    out = os.path.join(HERE, "pmi_L10A_peptide.pdb")
    with open(out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    code = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
            "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
            "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    seq = "".join(code.get(r.name, "X") for r in fixer.topology.residues())
    print(f"wrote {os.path.basename(out)}: {seq} ({fixer.topology.getNumAtoms()} atoms)")
    if seq != "TSFAEYWNLASP":
        raise SystemExit(f"expected TSFAEYWNLASP (PMI-L10A), got {seq}")


if __name__ == "__main__":
    main()
