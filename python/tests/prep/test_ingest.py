"""Phase H: DataIngestPipeline — file(s) -> parametrized ParmEd Structure.

Verified on examples/mdm2_peptide.pdb (standard residues):
  - Mode A (ForceFieldInput, apply an ffxml to a PDB) is FAITHFUL: the ingested structure's
    energy matches a direct OpenMM createSystem to ~1e-10 at the same coordinates (Amber),
  - Mode B (AmberInput, prmtop+inpcrd) round-trips the Mode-A structure exactly (charges +
    energy), which is the ParmEd-native lossless path,
  - Mode A CHARMM loads and is createSystem-able (NBFIX recovery is a no-op for this
    ion-free peptide; ion-water NBFIX preservation is covered in test_prep_solvate),
  - the pipeline raises a precise error when an input is insufficient.
"""
import os
import sys
import tempfile

import numpy as np
import openmm as mm
from openmm import app, unit

from openmm_ld.prep import (DataIngestPipeline, ingest, resolve_source, load_structure,
                       apply_forcefield, AmberInput, CharmmInput, ForceFieldInput)

_REF = mm.Platform.getPlatform("Reference")
_KJ = unit.kilojoule_per_mole
_PDB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "mdm2", "mdm2_peptide.pdb")
_cache = {}


def _sdf():
    """A self-contained neutral L-serine SDF (C3H7NO3, 14 atoms), molecule title 'SER'.

    Generated with RDKit (a hard dep) so the SDF-route tests don't depend on a checked-in example
    file: elements + bond orders + a 3-D conformer is all that _sdf_to_openmm and GAFF need.
    """
    if "sdf" not in _cache:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles("NC(CO)C(=O)O"))    # serine; heavy atom 0 = N
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", "SER")                             # -> residue name in _sdf_to_openmm
        path = os.path.join(tempfile.mkdtemp(), "SER.sdf")
        w = Chem.SDWriter(path); w.write(mol); w.close()
        _cache["sdf"] = path
    return _cache["sdf"]


def _energy(system, coords_ang):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), _REF)
    ctx.setPositions((np.array(coords_ang) / 10.0).tolist() * unit.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(_KJ)


def _st_energy(st):
    return _energy(st.createSystem(nonbondedMethod=app.NoCutoff), st.coordinates)


# --------------------------------------------------------------- Mode A faithfulness
def test_mode_a_matches_direct_openmm():
    # hydrogenate ONCE and share the file: addHydrogens randomizes rotatable-H orientation, so
    # comparing two independent addHydrogens runs is not bit-exact. Same file -> same coords ->
    # any energy difference is a parametrization difference (there is none).
    ff = app.ForceField("amber14-all.xml")
    pdb0 = app.PDBFile(_PDB)
    mod = app.Modeller(pdb0.topology, pdb0.positions)
    mod.addHydrogens(ff)
    with tempfile.TemporaryDirectory() as d:
        hpdb = os.path.join(d, "h.pdb")
        with open(hpdb, "w") as f:
            app.PDBFile.writeFile(mod.topology, mod.positions, f)
        p = app.PDBFile(hpdb)
        e_direct = _energy(ff.createSystem(p.topology, nonbondedMethod=app.NoCutoff),
                           np.array(p.positions.value_in_unit(unit.nanometer)) * 10.0)
        st = load_structure(ForceFieldInput(hpdb, ["amber14-all.xml"], add_hydrogens=False))
        e_ing = _st_energy(st)
    assert abs(e_ing - e_direct) / max(1.0, abs(e_direct)) < 1e-6, (e_ing, e_direct)


# --------------------------------------------------------------- Mode B Amber round-trip
def test_amber_prmtop_round_trip():
    st = apply_forcefield(_PDB, ["amber14-all.xml"], add_hydrogens=True)
    pipe = DataIngestPipeline()
    with tempfile.TemporaryDirectory() as d:
        prm, crd = os.path.join(d, "p.prmtop"), os.path.join(d, "p.inpcrd")
        st.save(prm, overwrite=True)
        st.save(crd, overwrite=True)
        back = pipe.load(AmberInput(prm, crd))
    assert len(back.atoms) == len(st.atoms)
    assert np.allclose([a.charge for a in back.atoms], [a.charge for a in st.atoms], atol=1e-5)
    rel = abs(_st_energy(back) - _st_energy(st)) / max(1.0, abs(_st_energy(st)))
    assert rel < 1e-4, rel


# --------------------------------------------------------------- Mode A CHARMM
def test_mode_a_charmm_loads():
    st = load_structure(ForceFieldInput(_PDB, ["charmm36.xml"], add_hydrogens=True))
    assert len(st.atoms) == 191
    e = _st_energy(st)                                # createSystem must succeed (NBFIX-safe)
    assert np.isfinite(e)


def test_ingest_accepts_pathlib_path():
    # a pathlib.Path (not just str) works: coerced to str at the loader boundary, so OpenMM's
    # PDBFile/RDKit -- which only accept str -- do not choke ('PosixPath' is not iterable).
    from pathlib import Path
    st = ingest(Path(_PDB), forcefield=["amber14-all.xml"], add_hydrogens=True)
    assert len(st.atoms) > 0 and any(abs(a.charge) > 1e-6 for a in st.atoms)


# --------------------------------------------------------------- sufficiency errors
def test_missing_amber_prmtop_raises():
    try:
        DataIngestPipeline().load(AmberInput("does_not_exist.prmtop"))
        assert False
    except FileNotFoundError as ex:
        assert "prmtop" in str(ex)


def test_charmm_without_parameters_raises():
    try:
        DataIngestPipeline().load(CharmmInput("x.psf", parameters=[]))
        assert False
    except ValueError as ex:
        assert "parameter" in str(ex)


def test_forcefield_without_ff_raises():
    try:
        DataIngestPipeline().load(ForceFieldInput(_PDB, forcefield=[]))
        assert False
    except ValueError as ex:
        assert "force field" in str(ex)


def test_unknown_source_type_raises():
    try:
        DataIngestPipeline().load("not a source spec")
        assert False
    except TypeError as ex:
        assert "unknown ingest source" in str(ex)


# --------------------------------------------------------------- the one door: routing
# resolve_source is PURE (no file I/O), so it can be checked on made-up names: it decides the
# route from extensions only. The user never picks a source class.
def test_router_amber_prmtop_inpcrd():
    src = resolve_source("sys.prmtop", "sys.inpcrd")
    assert isinstance(src, AmberInput)
    assert src.prmtop == "sys.prmtop" and src.coordinates == "sys.inpcrd"


def test_router_amber_pdb_as_coords():
    # a .pdb next to a prmtop is used as the coordinate source (not a Mode-A route)
    src = resolve_source("sys.prmtop", "sys.pdb")
    assert isinstance(src, AmberInput) and src.coordinates == "sys.pdb"


def test_router_charmm_psf_params_coords():
    src = resolve_source("mol.psf", "par_all36.prm", "top_all36.rtf", "mol.crd")
    assert isinstance(src, CharmmInput)
    assert src.psf == "mol.psf" and src.coordinates == "mol.crd"
    assert src.parameters == ["par_all36.prm", "top_all36.rtf"]


def test_router_pdb_with_forcefield():
    src = resolve_source("pep.pdb", forcefield=["amber14-all.xml"], add_hydrogens=True)
    assert isinstance(src, ForceFieldInput)
    assert src.structure == "pep.pdb" and src.forcefield == ["amber14-all.xml"] and src.add_hydrogens


def test_router_sdf_with_forcefield():
    src = resolve_source("mol.sdf", forcefield=["ligand.xml"])
    assert isinstance(src, ForceFieldInput)
    assert src.structure == "mol.sdf" and src.forcefield == ["ligand.xml"]


def test_router_lone_sdf_needs_forcefield():
    try:
        resolve_source("mol.sdf")
        assert False
    except ValueError as ex:
        assert "force field" in str(ex)


def test_router_sdf_cannot_accompany_topology():
    try:
        resolve_source("sys.prmtop", "mol.sdf")
        assert False
    except ValueError as ex:
        assert ".sdf" in str(ex) and "ffxml route" in str(ex)


def test_router_explicit_coordinates_wins():
    src = resolve_source("sys.prmtop", "sys.inpcrd", coordinates="other.rst7")
    assert src.coordinates == "other.rst7"


def test_router_charmm_without_params_raises():
    try:
        resolve_source("mol.psf")
        assert False
    except ValueError as ex:
        assert "parameter" in str(ex)


def test_router_lone_pdb_needs_forcefield():
    try:
        resolve_source("pep.pdb")
        assert False
    except ValueError as ex:
        assert "force field" in str(ex)


def test_router_conflicting_topologies_raises():
    try:
        resolve_source("sys.prmtop", "mol.psf")
        assert False
    except ValueError as ex:
        assert "one system" in str(ex)


def test_router_forcefield_redundant_with_prmtop_raises():
    try:
        resolve_source("sys.prmtop", forcefield=["amber14-all.xml"])
        assert False
    except ValueError as ex:
        assert "only to a structure file" in str(ex)


def test_router_unknown_extension_raises():
    try:
        resolve_source("mystery.xyz")
        assert False
    except ValueError as ex:
        assert "unrecognized" in str(ex)


def test_router_no_paths_raises():
    try:
        resolve_source()
        assert False
    except ValueError as ex:
        assert "at least one" in str(ex)


def test_ingest_door_matches_explicit_spec():
    # the door and the hand-built ForceFieldInput must reach the SAME parametrized structure.
    # addHydrogens randomizes H orientation, so coords/energy are not bit-exact between runs, but
    # atom count and FF-assigned charges are deterministic.
    st_door = ingest(_PDB, forcefield=["amber14-all.xml"], add_hydrogens=True)
    st_spec = load_structure(ForceFieldInput(_PDB, ["amber14-all.xml"], add_hydrogens=True))
    assert len(st_door.atoms) == len(st_spec.atoms)
    assert np.allclose([a.charge for a in st_door.atoms],
                       [a.charge for a in st_spec.atoms], atol=1e-6)


# --------------------------------------------------------------- SDF structure input (Mode A)
def test_sdf_to_openmm_reads_geometry_and_bonds():
    # the SDF->Topology loader: elements + bonds + coordinates come across faithfully
    from collections import Counter
    from openmm import unit
    from openmm_ld.prep.ingest.pipeline import _sdf_to_openmm
    top, pos = _sdf_to_openmm(_sdf())
    assert top.getNumAtoms() == 14 and top.getNumBonds() == 13
    assert next(top.residues()).name == "SER"                       # SDF molecule title
    assert Counter(a.element.symbol for a in top.atoms()) == Counter(N=1, C=3, O=3, H=7)
    assert list(top.atoms())[0].element.symbol == "N"              # first atom is N
    assert np.isfinite(np.array(pos.value_in_unit(unit.angstrom))).all()   # 3-D coords came across


def test_sdf_with_standard_ff_is_refused():
    # policy: SDF is for ligands ONLY. An SDF paired with a standard (amino-acid/biopolymer) force
    # field is refused up front -- amino acids, standard or non-standard, must be provided as PDB.
    try:
        ingest(_sdf(), forcefield=["amber14-all.xml"])
        assert False
    except ValueError as ex:
        assert "ligand" in str(ex).lower() and "gaff" in str(ex).lower()


def test_smallmol_ff_requires_sdf():
    # a small-molecule FF (GAFF/OpenFF) needs bond orders, so it is rejected for a PDB
    try:
        apply_forcefield(_PDB, ["gaff-2.11"])
        assert False
    except ValueError as ex:
        assert "needs an SDF" in str(ex)


def test_sdf_parametrized_with_gaff():
    # THE feature: read structure + coords from an SDF, parametrize it with a force field (GAFF).
    # Works for ANY organic SDF (here a standard amino acid). GAFF shells out to AmberTools
    # antechamber; apply_forcefield ensures it is on PATH.
    st = ingest(_sdf(), forcefield=["gaff-2.11"])
    assert len(st.atoms) == 14
    assert any(abs(a.charge) > 1e-6 for a in st.atoms)          # real partial charges assigned
    assert abs(sum(a.charge for a in st.atoms)) < 1e-3          # neutral molecule
    assert st.bonds and st.angles and st.dihedrals              # full bonded parameter set
    sysm = st.createSystem(nonbondedMethod=app.NoCutoff)        # fully parametrized -> buildable
    assert sysm.getNumParticles() == 14


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nPhase H ingest tests passed")
