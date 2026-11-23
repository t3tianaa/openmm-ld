"""Build the side-chain donor residues the two PMI networks need.

    python make_inputs.py

Writes ALA_var.pdb, ASN_var.pdb, GLN_var.pdb, MET_var.pdb and THR_var.pdb next to this script, in the
same format as the LEU_var.pdb already here: one internal residue, Amber atom names, N/H/CA/HA/C/O and
then the side chain from CB. Each is lifted from a complete residue of ../mdm2_gpu/mdm2_protein.pdb, so
the naming and the geometry are already what ff14SB expects. The site mapper superimposes a donor on the
target residue's backbone, so the coordinates only have to be a valid rotamer, not a placed one.

Both networks use ../mdm2_gpu/mdm2_peptide.pdb as their peptide, so there is nothing else to build.
Re-running this overwrites the five files.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# The input structures live in examples/mdm2_gpu; this directory does not copy them.
MDM2 = os.path.join(HERE, os.pardir, "mdm2_gpu")
# A complete residue of each type in mdm2_protein.pdb (chain A, author numbering of that file).
DONORS = {"ALA": 19, "ASN": 55, "GLN": 20, "MET": 26, "THR": 39}
# Atom order of the existing LEU_var.pdb: backbone first, then the side chain from CB.
BACKBONE = ["N", "H", "CA", "HA", "C", "O"]
# Atoms a donor must not carry: a terminal residue's extra hydrogens or its C-terminal oxygen.
FORBIDDEN = {"OXT", "H1", "H2", "H3"}


def read_residue(path, resname, resnum):
    atoms = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[17:20].strip() == resname and int(line[22:26]) == resnum:
                atoms.append((line[12:16], line[30:38], line[38:46], line[46:54], line[76:78]))
    if not atoms:
        raise SystemExit(f"{path}: no {resname} {resnum}")
    names = {a[0].strip() for a in atoms}
    if names & FORBIDDEN or "H" not in names:
        raise SystemExit(f"{path}: {resname} {resnum} is not an internal residue (atoms {sorted(names)})")
    return atoms


def write_var(resname, atoms, path):
    order = {name: i for i, name in enumerate(BACKBONE)}
    atoms = sorted(atoms, key=lambda a: (order.get(a[0].strip(), len(BACKBONE)),))
    with open(path, "w") as fh:
        fh.write(f"HEADER    INTERNAL L-{resname} RESIDUE\n")
        fh.write(f"TITLE     {resname} RESIDUE EXTRACTED FROM ../mdm2_gpu/mdm2_protein.pdb\n")
        fh.write("REMARK    NO OXT OR TERMINAL NH2/NH3 HYDROGENS ARE PRESENT.\n")
        fh.write("REMARK    BACKBONE N AND C HAVE OPEN VALENCES FOR PEPTIDE BONDS.\n")
        for i, (name, x, y, z, element) in enumerate(atoms, start=1):
            # Columns 13-16 as the PDB wants them, and as LEU_var.pdb here already is: a 4-character
            # name stays put, a shorter one leaves column 13 to the element symbol.
            name = name.strip()
            name = name if len(name) == 4 else f" {name:<3}"
            fh.write(f"ATOM  {i:5d} {name} {resname} A   4    {x}{y}{z}  1.00  0.00          {element}\n")
        fh.write("END\n")
    print(f"wrote {os.path.basename(path)}: {len(atoms)} atoms")


def main():
    protein = os.path.join(MDM2, "mdm2_protein.pdb")
    expected = {"ALA": 10, "ASN": 14, "GLN": 17, "MET": 17, "THR": 14}
    for resname, resnum in sorted(DONORS.items()):
        atoms = read_residue(protein, resname, resnum)
        if len(atoms) != expected[resname]:
            raise SystemExit(f"{resname} {resnum}: expected {expected[resname]} atoms, got {len(atoms)}")
        write_var(resname, atoms, os.path.join(HERE, f"{resname}_var.pdb"))


if __name__ == "__main__":
    main()
